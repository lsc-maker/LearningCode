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
######################## eval alexnet example ########################
eval alexnet according to model file:
python eval.py --data_path /YourDataPath --ckpt_path Your.ckpt
"""

import os
import argparse
from config import alexnet_cfg as cfg
from dataset import create_dataset
import mindspore.nn as nn
from mindspore import context
from mindspore.model_zoo.alexnet import AlexNet
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy
import moxing as mox

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MindSpore AlexNet Example')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented (default: Ascend)')
    parser.add_argument('--dataset_sink_mode', type=bool, default=False, help='dataset_sink_mode is False or True')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint file path')
    parser.add_argument('--data_url', type=str, default=None, help='Dataset path')
    parser.add_argument('--train_url', type=str, default=None, help='Train output path')
    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, enable_mem_reuse=False)

    local_data_url = '/cache/data'
    local_ckpt_url = '/cache/ckpt'

    network = AlexNet(cfg.num_classes)
    loss = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True, reduction="mean")
    repeat_size = cfg.epoch_size
    opt = nn.Momentum(network.trainable_params(), cfg.learning_rate, cfg.momentum)
    model = Model(network, loss, opt, metrics={"Accuracy": Accuracy()})  # test

    if args.checkpoint_path:
        checkpoint_file=os.path.join(local_ckpt_url,os.path.split(args.checkpoint_path)[1])
    mox.file.copy_parallel(args.data_url,local_data_url)
    mox.file.copy_parallel(args.checkpoint_path,checkpoint_file)

    print("============== Starting Testing ==============")
    param_dict = load_checkpoint(checkpoint_file)
    load_param_into_net(network, param_dict)
    ds_eval = create_dataset(local_data_url,
                             cfg.batch_size,
                             1,
                             "test")
    acc = model.eval(ds_eval, dataset_sink_mode=args.dataset_sink_mode)
    print("============== Accuracy:{} ==============".format(acc))
