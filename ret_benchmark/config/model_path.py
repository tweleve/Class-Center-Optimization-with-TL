# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

# -----------------------------------------------------------------------------
# Config definition of imagenet pretrained model path
# -----------------------------------------------------------------------------


from yacs.config import CfgNode as CN

MODEL_PATH = dict()
MODEL_PATH = {"bninception": "/home/hzj/Pycharms/xbm/configs/pth/bn_inception.pth",
              # "resnet50": "~/.torch/models/pytorch_resnet50.pths_sets",
              "resnet50": '/home/hzj/Pycharms/xbm/configs/pth/resnet50.pth',
              # "googlenet": "~/.torch/models/googlenet-1378be20.pths_sets",
              # "googlenet": '~/Pycharms/xbm/pths_sets/googlenet-1378be20.pth',
              "googlenet": '/home/hzj/Pycharms/xbm/configs/pth/googlenet.pth'}

MODEL_PATH = CN(MODEL_PATH)
