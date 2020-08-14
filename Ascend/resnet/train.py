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
"""train resnet."""
import os
import random
import argparse
import numpy as np
from mindspore import context
from mindspore import Tensor
from mindspore import dataset as de
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.model import Model, ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init, get_rank, get_group_size
import mindspore.nn as nn
import mindspore.common.initializer as weight_init
from src.lr_generator import get_lr, warmup_cosine_annealing_lr
from src.crossentropy import CrossEntropy

import moxing as mox

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--net', type=str, default=None, help='Resnet Model, either resnet50 or resnet101')
parser.add_argument('--dataset', type=str, default=None, help='Dataset, either cifar10 or imagenet2012')
parser.add_argument('--run_distribute', type=bool, default=False, help='Run distribute')
#parser.add_argument('--device_num', type=int, default=1, help='Device num.')

#parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
parser.add_argument('--device_target', type=str, default='Ascend', help='Device target')
parser.add_argument('--pre_trained', type=str, default=None, help='Pretrained checkpoint path')
parser.add_argument('--data_url', type=str, default=None, help='Dataset path')
parser.add_argument('--train_url', type=str, default=None, help='Train output path')

args_opt = parser.parse_args()

device_id = int(os.getenv('DEVICE_ID'))
device_num = int(os.getenv('RANK_SIZE'))

local_data_url = '/cache/data'
local_train_url = '/cache/ckpt'

mox.file.copy_parallel(args_opt.data_url,local_data_url)

random.seed(1)
np.random.seed(1)
de.config.set_seed(1)

if args_opt.net == "resnet50":
    from src.resnet import resnet50 as resnet

    if args_opt.dataset == "cifar10":
        from src.config import config1 as config
        from src.dataset import create_dataset1 as create_dataset
    else:
        from src.config import config2 as config
        from src.dataset import create_dataset2 as create_dataset
else:
    from src.resnet import resnet101 as resnet
    from src.config import config3 as config
    from src.dataset import create_dataset3 as create_dataset

if __name__ == '__main__':
    target = args_opt.device_target
    ckpt_save_dir = config.save_checkpoint_path

    # init context
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)
    if args_opt.run_distribute:
        if target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(device_id=device_id, enable_auto_mixed_precision=True)
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              mirror_mean=True)
            if args_opt.net == "resnet50":
                auto_parallel_context().set_all_reduce_fusion_split_indices([107, 160])
            else:
                auto_parallel_context().set_all_reduce_fusion_split_indices([180, 313])
            init()
            local_train_url = os.path.join(local_train_url,str(device_id))
        # GPU target
        else:
            init("nccl")
            context.set_auto_parallel_context(device_num=get_group_size(), parallel_mode=ParallelMode.DATA_PARALLEL,
                                              mirror_mean=True)
            ckpt_save_dir = config.save_checkpoint_path + "ckpt_" + str(get_rank()) + "/"

    # create dataset
    if args_opt.net == "resnet50":
        dataset = create_dataset(dataset_path=local_data_url, do_train=True, repeat_num=config.epoch_size,
                                 batch_size=config.batch_size, target=target)
    else:
        dataset = create_dataset(dataset_path=local_data_url, do_train=True, repeat_num=config.epoch_size,
                                 batch_size=config.batch_size)
    step_size = dataset.get_dataset_size()

    # define net
    net = resnet(class_num=config.class_num)

    # init weight
    if args_opt.pre_trained:
        param_dict = load_checkpoint(args_opt.pre_trained)
        load_param_into_net(net, param_dict)
    else:
        for _, cell in net.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.default_input = weight_init.initializer(weight_init.XavierUniform(),
                                                                    cell.weight.default_input.shape,
                                                                    cell.weight.default_input.dtype).to_tensor()
            if isinstance(cell, nn.Dense):
                cell.weight.default_input = weight_init.initializer(weight_init.TruncatedNormal(),
                                                                    cell.weight.default_input.shape,
                                                                    cell.weight.default_input.dtype).to_tensor()

    # init lr
    if args_opt.net == "resnet50":
        if args_opt.dataset == "cifar10":
            lr = get_lr(lr_init=config.lr_init, lr_end=config.lr_end, lr_max=config.lr_max,
                        warmup_epochs=config.warmup_epochs, total_epochs=config.epoch_size, steps_per_epoch=step_size,
                        lr_decay_mode='poly')
        else:
            lr = get_lr(lr_init=config.lr_init, lr_end=0.0, lr_max=config.lr_max, warmup_epochs=config.warmup_epochs,
                        total_epochs=config.epoch_size, steps_per_epoch=step_size, lr_decay_mode='cosine')
    else:
        lr = warmup_cosine_annealing_lr(config.lr, step_size, config.warmup_epochs, 120,
                                        config.pretrain_epoch_size * step_size)
    lr = Tensor(lr)

    # define opt
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, config.momentum,
                   config.weight_decay, config.loss_scale)

    # define loss, model
    if target == "Ascend":
        if args_opt.dataset == "imagenet2012":
            if not config.use_label_smooth:
                config.label_smooth_factor = 0.0
            loss = CrossEntropy(smooth_factor=config.label_smooth_factor, num_classes=config.class_num)
        else:
            loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
        model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics={'acc'},
                      amp_level="O2", keep_batchnorm_fp32=False)
    else:
        # GPU target
        loss = SoftmaxCrossEntropyWithLogits(sparse=True, is_grad=False, reduction='mean')
        opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, config.momentum)
        model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'})

    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    if config.save_checkpoint:
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix="resnet", directory=local_train_url, config=config_ck)
        cb += [ckpt_cb]

    # train model
    model.train(config.epoch_size, dataset, callbacks=cb)
    if config.save_checkpoint and (device_num == 1 or device_id == 0):
            mox.file.copy_parallel(local_train_url,args_opt.train_url)
