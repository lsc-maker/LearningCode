# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
######################## train alexnet example ########################
train alexnet and get network model files(.ckpt) :
python train.py --data_path /YourDataPath
"""

import os
import argparse
from config import alexnet_cfg as cfg
from dataset import create_dataset
import mindspore.nn as nn
from mindspore import context
from mindspore.communication.management import init
from mindspore.train.model import Model, ParallelMode
from mindspore.nn.metrics import Accuracy
from mindspore.model_zoo.alexnet import AlexNet
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
import moxing as mox


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MindSpore AlexNet Example')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented (default: Ascend)')
    parser.add_argument('--data_url', type=str, default="./", help='path where the dataset is saved')
    parser.add_argument('--train_url', type=str, default=None, help='Train output path')
    parser.add_argument('--dataset_sink_mode', type=bool, default=False, help='dataset_sink_mode is False or True')
    args = parser.parse_args()

    device_id = int(os.getenv('DEVICE_ID'))
    device_num = int(os.getenv('RANK_SIZE'))

    local_data_url = '/cache/data'
    local_train_url = '/cache/ckpt'

    if device_num > 1:
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, enable_mem_reuse=False)
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                    mirror_mean=True)
        init()
        local_data_url = os.path.join(local_data_url,str(device_id))
    else:
        context.set_context(enable_hccl=False)

    network = AlexNet(cfg.num_classes)
    loss = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True, reduction="mean")
    loss_scale = FixedLossScaleManager(cfg.loss_scale, drop_overflow_update=False)
    opt = nn.Momentum(network.trainable_params(), cfg.learning_rate, cfg.momentum)
    model = Model(network, loss, opt, metrics={"Accuracy": Accuracy()})  # test

    mox.file.copy_parallel(args.data_url,local_data_url)

    print("============== Starting Training ==============")
    ds_train = create_dataset(local_data_url,
                              cfg.batch_size,
                              cfg.epoch_size,
                              "train")
    config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                                 keep_checkpoint_max=cfg.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_alexnet", directory=local_train_url, config=config_ck)
    model.train(cfg.epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor()],
                dataset_sink_mode=args.dataset_sink_mode)
    mox.file.copy_parallel(local_train_url,args.train_url)