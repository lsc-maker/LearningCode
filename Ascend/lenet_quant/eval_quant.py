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
######################## eval lenet example ########################
eval lenet according to model file:
python eval.py --data_path /YourDataPath --ckpt_path Your.ckpt
"""

import os
import argparse
import mindspore.nn as nn
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy
from mindspore.train.quant import quant
from src.dataset import create_dataset
from src.config import mnist_cfg as cfg
from src.lenet_fusion import LeNet5 as LeNet5Fusion

import moxing as mox

parser = argparse.ArgumentParser(description='MindSpore MNIST Example')
parser.add_argument('--device_target', type=str, default="Ascend",
                    choices=['Ascend', 'GPU'],
                    help='device where the code will be implemented (default: Ascend)')
#parser.add_argument('--data_path', type=str, default="./MNIST_Data",
#                    help='path where the dataset is saved')
#parser.add_argument('--ckpt_path', type=str, default="",
#                    help='if mode is test, must provide path where the trained ckpt file')
parser.add_argument('--dataset_sink_mode', type=bool, default=True,
                    help='dataset_sink_mode is False or True')
parser.add_argument('--data_url', type=str, default=None, help='Dataset path')
parser.add_argument('--train_url', type=str, default=None, help='Train output path')
parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint file path')

args = parser.parse_args()

device_id = int(os.getenv('DEVICE_ID'))
device_num = int(os.getenv('RANK_SIZE'))

if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

    local_data_url = '/cache/data'
    local_ckpt_url = '/cache/ckpt'

    if args.checkpoint_path:
        checkpoint_file=os.path.join(local_ckpt_url,os.path.split(args.checkpoint_path)[1])

    mox.file.copy_parallel(args.data_url,local_data_url)
    mox.file.copy_parallel(args.checkpoint_path,checkpoint_file)


    ds_eval = create_dataset(os.path.join(local_data_url, "eval"), cfg.batch_size, 1)
    step_size = ds_eval.get_dataset_size()

    # define fusion network
    network = LeNet5Fusion(cfg.num_classes)
    
    # define loss
    net_loss = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True, reduction="mean")
    # define network optimization
    net_opt = nn.Momentum(network.trainable_params(), cfg.lr, cfg.momentum)

    # convert fusion netwrok to quantization aware network
    network = quant.convert_quant_network(network, quant_delay=0, bn_fold=False, freeze_bn=10000)

    # call back and monitor
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    # load quantization aware network checkpoint
    param_dict = load_checkpoint(checkpoint_file)
    load_param_into_net(network, param_dict)

    print("============== Starting Testing ==============")
    acc = model.eval(ds_eval, dataset_sink_mode=args.dataset_sink_mode)
    print("============== {} ==============".format(acc))
