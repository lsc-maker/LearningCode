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
##############test vgg16 example on cifar10#################
python eval.py --data_path=$DATA_HOME --device_id=$DEVICE_ID
"""
import os
import argparse
import mindspore.nn as nn
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.model import Model
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.model_zoo.vgg import vgg16
from config import cifar_cfg as cfg
import dataset
import moxing as mox

parser = argparse.ArgumentParser(description='Cifar10 classification')
parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'GPU'],
                    help='device where the code will be implemented. (Default: Ascend)')
parser.add_argument('--checkpoint_path', type=str, default=None, help='checkpoint file path.')
parser.add_argument('--device_id', type=int, default=None, help='device id of GPU or Ascend. (Default: None)')
parser.add_argument('--data_url', type=str, default=None, help='Dataset path')
parser.add_argument('--train_url', type=str, default=None, help='Train output path')
args_opt = parser.parse_args()

device_id = int(os.getenv('DEVICE_ID'))

context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)
context.set_context(device_id=device_id)
context.set_context(enable_mem_reuse=True)

local_data_url = '/cache/data'
local_ckpt_url = '/cache/ckpt'


if args_opt.checkpoint_path:
    checkpoint_file=os.path.join(local_ckpt_url,os.path.split(args_opt.checkpoint_path)[1])
mox.file.copy_parallel(args_opt.data_url,local_data_url)
mox.file.copy_parallel(args_opt.checkpoint_path,checkpoint_file)

net = vgg16(num_classes=cfg.num_classes)
opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, cfg.momentum,
               weight_decay=cfg.weight_decay)
loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean', is_grad=False)
model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'})

param_dict = load_checkpoint(checkpoint_file)
load_param_into_net(net, param_dict)
net.set_train(False)
dataset = dataset.create_dataset(local_data_url, 1, False)
res = model.eval(dataset)
print("result: ", res)
