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
#################train vgg16 example on cifar10########################
python train.py --data_path=$DATA_HOME --device_id=$DEVICE_ID
"""
import argparse
import os
import random
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.communication.management import init
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.model import Model, ParallelMode
from mindspore import context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.model_zoo.vgg import vgg16
from dataset import create_dataset
from config import cifar_cfg as cfg
import moxing as mox

random.seed(1)
np.random.seed(1)

def lr_steps(global_step, lr_max=None, total_epochs=None, steps_per_epoch=None):
    """Set learning rate."""
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    decay_epoch_index = [0.3 * total_steps, 0.6 * total_steps, 0.8 * total_steps]
    for i in range(total_steps):
        if i < decay_epoch_index[0]:
            lr_each_step.append(lr_max)
        elif i < decay_epoch_index[1]:
            lr_each_step.append(lr_max * 0.1)
        elif i < decay_epoch_index[2]:
            lr_each_step.append(lr_max * 0.01)
        else:
            lr_each_step.append(lr_max * 0.001)
    current_step = global_step
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]

    return learning_rate



parser = argparse.ArgumentParser(description='Cifar10 classification')
parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'GPU'],
                    help='device where the code will be implemented. (Default: Ascend)')
parser.add_argument('--device_id', type=int, default=None, help='device id of GPU or Ascend. (Default: None)')
parser.add_argument('--data_url', type=str, default=None, help='Dataset path')
parser.add_argument('--train_url', type=str, default=None, help='Train output path')

args_opt = parser.parse_args()
device_id = int(os.getenv('DEVICE_ID'))
device_num = int(os.getenv('RANK_SIZE'))

context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)
context.set_context(device_id=device_id)
context.set_context(enable_task_sink=True)
context.set_context(enable_loop_sink=True)
context.set_context(enable_mem_reuse=True)

local_data_url = '/cache/data'
local_train_url = '/cache/ckpt'

device_num = int(os.environ.get("DEVICE_NUM", 1))
if device_num > 1:
  context.set_context(enable_hccl=True)
  context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                    mirror_mean=True)
  init()
  local_data_url = os.path.join(local_data_url,str(device_id))
else:
  context.set_context(enable_hccl=False)

mox.file.copy_parallel(args_opt.data_url,local_data_url)
dataset = create_dataset(local_data_url, cfg.epoch_size)
batch_num = dataset.get_dataset_size()

net = vgg16(num_classes=cfg.num_classes)
lr = lr_steps(0, lr_max=cfg.lr_init, total_epochs=cfg.epoch_size, steps_per_epoch=batch_num)
opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), Tensor(lr), cfg.momentum, weight_decay=cfg.weight_decay)
loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean', is_grad=False)
'''
model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'},
              amp_level="O2", keep_batchnorm_fp32=False, loss_scale_manager=None)
'''
model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'})

config_ck = CheckpointConfig(save_checkpoint_steps=batch_num * 5, keep_checkpoint_max=cfg.keep_checkpoint_max)
time_cb = TimeMonitor(data_size=batch_num)
ckpoint_cb = ModelCheckpoint(prefix="train_vgg_cifar10", directory=local_train_url, config=config_ck)
loss_cb = LossMonitor()
model.train(cfg.epoch_size, dataset, callbacks=[time_cb, ckpoint_cb, loss_cb])
mox.file.copy_parallel(local_train_url,args_opt.train_url)
print("train success")
