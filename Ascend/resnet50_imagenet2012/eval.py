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
eval.
"""
import os
import argparse
from dataset import create_dataset
from config import config
from mindspore import context
from mindspore.model_zoo.resnet import resnet50
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from crossentropy import CrossEntropy
import moxing as mox

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--run_distribute', type=bool, default=False, help='Run distribute')
#parser.add_argument('--device_num', type=int, default=1, help='Device num.')
parser.add_argument('--do_train', type=bool, default=False, help='Do train or not.')
parser.add_argument('--do_eval', type=bool, default=True, help='Do eval or not.')
parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint file path')
#parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
parser.add_argument('--data_url', type=str, default=None, help='Dataset path')
parser.add_argument('--train_url', type=str, default=None, help='Train output path')
parser.add_argument('--device_target', type=str, default='Ascend', help='Device target')
args_opt = parser.parse_args()

local_data_url = '/cache/data'
local_ckpt_url = '/cache/ckpt'

target = args_opt.device_target
context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)
if target == "Ascend":
    device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(device_id=device_id)

if __name__ == '__main__':

    net = resnet50(class_num=config.class_num)
    if args_opt.checkpoint_path:
        checkpoint_file=os.path.join(local_ckpt_url,os.path.split(args_opt.checkpoint_path)[1])
    mox.file.copy_parallel(args_opt.data_url,local_data_url)
    mox.file.copy_parallel(args_opt.checkpoint_path,checkpoint_file)
    if not config.use_label_smooth:
        config.label_smooth_factor = 0.0
    loss = CrossEntropy(smooth_factor=config.label_smooth_factor, num_classes=config.class_num)

    if args_opt.do_eval:
        dataset = create_dataset(dataset_path=local_data_url,do_train=False, batch_size=config.batch_size,
                                 target=target)
        step_size = dataset.get_dataset_size()

        if args_opt.checkpoint_path:
            param_dict = load_checkpoint(checkpoint_file)
            load_param_into_net(net, param_dict)
        net.set_train(False)

        model = Model(net, loss_fn=loss, metrics={'acc'})
        res = model.eval(dataset)
        print("result:", res, "ckpt=", args_opt.checkpoint_path)
