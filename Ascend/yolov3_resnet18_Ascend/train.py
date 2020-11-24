# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
######################## train YOLOv3 example ########################
train YOLOv3 and get network model files(.ckpt) :
python train.py --image_dir /data --anno_path /data/coco/train_coco.txt --mindrecord_dir=/data/Mindrecord_train

If the mindrecord_dir is empty, it wil generate mindrecord file by image_dir and anno_path.
Note if mindrecord_dir isn't empty, it will use mindrecord_dir rather than image_dir and anno_path.
"""

import os
import argparse
import numpy as np
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.communication.management import init
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor
from mindspore.train import Model, ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common.initializer import initializer

from src.yolov3 import yolov3_resnet18, YoloWithLossCell, TrainingWrapper
from src.dataset import create_yolo_dataset, data_to_mindrecord_byte_image
from src.config import ConfigYOLOV3ResNet18

import moxing as mox


def get_lr(learning_rate, start_step, global_step, decay_step, decay_rate, steps=False):
    """Set learning rate."""
    lr_each_step = []
    for i in range(global_step):
        if steps:
            lr_each_step.append(learning_rate * (decay_rate ** (i // decay_step)))
        else:
            lr_each_step.append(learning_rate * (decay_rate ** (i / decay_step)))
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    lr_each_step = lr_each_step[start_step:]
    return lr_each_step


def init_net_param(network, init_value='ones'):
    """Init:wq the parameters in network."""
    params = network.trainable_params()
    for p in params:
        if isinstance(p.data, Tensor) and 'beta' not in p.name and 'gamma' not in p.name and 'bias' not in p.name:
            p.set_parameter_data(initializer(init_value, p.data.shape, p.data.dtype))


def main():
    parser = argparse.ArgumentParser(description="YOLOv3 train")
    parser.add_argument("--only_create_dataset", type=bool, default=False, help="If set it true, only create "
                                                                                "Mindrecord, default is false.")
    parser.add_argument("--distribute", type=bool, default=False, help="Run distribute, default is false.")
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
    parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate, default is 0.001.")
    parser.add_argument("--mode", type=str, default="sink", help="Run sink mode or not, default is sink")
    parser.add_argument("--epoch_size", type=int, default=10, help="Epoch size, default is 10")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size, default is 32.")
    parser.add_argument("--pre_trained", type=str, default=None, help="Pretrained checkpoint file path")
    parser.add_argument("--pre_trained_epoch_size", type=int, default=0, help="Pretrained epoch size")
    parser.add_argument("--save_checkpoint_epochs", type=int, default=5, help="Save checkpoint epochs, default is 5.")
    parser.add_argument("--loss_scale", type=int, default=1024, help="Loss scale, default is 1024.")
    parser.add_argument("--mindrecord_dir", type=str, default="./Mindrecord",
                        help="Mindrecord directory. If the mindrecord_dir is empty, it wil generate mindrecord file by"
                             "image_dir and anno_path. Note if mindrecord_dir isn't empty, it will use mindrecord_dir "
                             "rather than image_dir and anno_path. Default is ./Mindrecord_train")
    parser.add_argument('--data_url', type=str, default=None, help='Dataset path')
    parser.add_argument('--train_url', type=str, default=None, help='Train output path')
    parser.add_argument("--anno_path", type=str, default="", help="Annotation path.")
    args_opt = parser.parse_args()

    device_id = int(os.getenv('DEVICE_ID'))
    device_num = int(os.getenv('RANK_SIZE'))
    rankid = int(os.getenv('RANK_ID'))

    local_data_url = '/cache/data'
    local_train_url = '/cache/ckpt'
    local_anno_url = '/cache/anno'
    local_mindrecord_url = '/cache/mindrecord'
    mox.file.copy_parallel(args_opt.mindrecord_dir,local_mindrecord_url)


    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=device_id)
    if args_opt.distribute:
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, mirror_mean=True,
                                          device_num=device_num)
        init()
        rank = rankid
        local_train_url = os.path.join(local_train_url,str(device_id))
    else:
        rank = 0
        device_num = 1

    print("Start create dataset!")

    # It will generate mindrecord file in args_opt.mindrecord_dir,
    # and the file name is yolo.mindrecord0, 1, ... file_num.
    if not os.path.isdir(local_mindrecord_url):
        os.makedirs(local_mindrecord_url)

    prefix = "train.mindrecord"
    mindrecord_file = os.path.join(local_mindrecord_url, prefix + "0")
    if not os.path.exists(mindrecord_file):
        mox.file.copy_parallel(args_opt.data_url,local_data_url)
        if args_opt.anno_path:
            anno_file=os.path.join(local_anno_url,os.path.split(args_opt.anno_path)[1])
        mox.file.copy_parallel(args_opt.anno_path,anno_file)
        if os.path.isdir(local_data_url) or os.path.exists(anno_file):
            print("Create Mindrecord.")
            data_to_mindrecord_byte_image(local_data_url,
                                          anno_file,
                                          local_mindrecord_url,
                                          prefix=prefix,
                                          file_num=8)
            print("Create Mindrecord Done, at {}".format(args_opt.mindrecord_dir))
            mox.file.copy_parallel(local_mindrecord_url,args_opt.mindrecord_dir)
        else:
            print("image_dir or anno_path not exits.")

    if not args_opt.only_create_dataset:
        loss_scale = float(args_opt.loss_scale)

        # When create MindDataset, using the fitst mindrecord file, such as yolo.mindrecord0.
        dataset = create_yolo_dataset(mindrecord_file, repeat_num=args_opt.epoch_size,
                                      batch_size=args_opt.batch_size, device_num=device_num, rank=rank)
        dataset_size = dataset.get_dataset_size()
        print("Create dataset done!")

        net = yolov3_resnet18(ConfigYOLOV3ResNet18())
        net = YoloWithLossCell(net, ConfigYOLOV3ResNet18())
        init_net_param(net, "XavierUniform")

        # checkpoint
        ckpt_config = CheckpointConfig(save_checkpoint_steps=dataset_size * args_opt.save_checkpoint_epochs)
        ckpoint_cb = ModelCheckpoint(prefix="yolov3", directory=local_train_url, config=ckpt_config)

        if args_opt.pre_trained:
            if args_opt.pre_trained_epoch_size <= 0:
                raise KeyError("pre_trained_epoch_size must be greater than 0.")
            param_dict = load_checkpoint(args_opt.pre_trained)
            load_param_into_net(net, param_dict)
        total_epoch_size = 60
        if args_opt.distribute:
            total_epoch_size = 160
        lr = Tensor(get_lr(learning_rate=args_opt.lr, start_step=args_opt.pre_trained_epoch_size * dataset_size,
                           global_step=total_epoch_size * dataset_size,
                           decay_step=1000, decay_rate=0.95, steps=True))
        opt = nn.Adam(filter(lambda x: x.requires_grad, net.get_parameters()), lr, loss_scale=loss_scale)
        net = TrainingWrapper(net, opt, loss_scale)

        callback = [TimeMonitor(data_size=dataset_size), LossMonitor(), ckpoint_cb]

        model = Model(net)
        dataset_sink_mode = False
        if args_opt.mode == "sink":
            print("In sink mode, one epoch return a loss.")
            dataset_sink_mode = True
        print("Start train YOLOv3, the first epoch will be slower because of the graph compilation.")
        model.train(args_opt.epoch_size, dataset, callbacks=callback, dataset_sink_mode=dataset_sink_mode)
        if device_id ==1:
            mox.file.copy_parallel(local_train_url,args_opt.train_url)
        
if __name__ == '__main__':
    main()
