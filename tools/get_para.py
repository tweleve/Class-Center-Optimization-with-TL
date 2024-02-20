import torch
import torchvision.models as models

import argparse
import torch
import os
import torch.nn as nn

from ret_benchmark.config import cfg
from ret_benchmark.data import build_data
from ret_benchmark.engine.trainer import do_train
from ret_benchmark.losses import build_loss
from ret_benchmark.modeling import build_model
from ret_benchmark.solver import build_lr_scheduler, build_optimizer
from ret_benchmark.utils.logger import setup_logger
from ret_benchmark.utils.checkpoint import Checkpointer
from tensorboardX import SummaryWriter

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

from center_loss import CenterLoss


def center_loss1():
    # 中心损失
    use_gpu = torch.cuda.is_available()
    criterion_cent = CenterLoss(num_classes=11, feat_dim=2, use_gpu=use_gpu)
    return criterion_cent


















def get_parameter_count(model):
    return sum(p.numel() for p in model.parameters())
def train(cfg):
    logger = setup_logger(name="Train", level=cfg.LOGGER.LEVEL)
    logger.info(cfg)
    model = build_model(cfg)
    model = models.resnet50(pretrained=True)
    total_params = get_parameter_count(model)
    print(f"Total parameters: {total_params}")

def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description="Train a retrieval network")
    parser.add_argument("--cfg", dest="cfg_file", help="config file", type=str,
                        # default='../configs/sample_config_MS.yaml'
                        # default='../configs/sample_config.yaml'
                        # default='../configs/AID_resnet50_ms.yaml'
                        # default='../configs/PatternNet_resnet50_ms.yaml'
                        # default='../configs/OPTIMAL-31_resnet50_ms.yaml'
                        default='../configs/UCMD-21_resnet50_ms.yaml',
                        # default='../configs/AID-30_resnet50_ms.yaml'
                        )
    return parser.parse_args()


if __name__ == "__main__":
    # UCMD   39:54,
    # AID    49:04
    args = parse_args()
    cfg.merge_from_file(args.cfg_file)
    train(cfg)
    num = input("手动输入整数：实际是为了暂停")
    num = int(num)
    print(num)
