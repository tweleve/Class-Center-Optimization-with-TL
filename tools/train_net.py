# encoding: utf-8


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


def train(cfg):
    logger = setup_logger(name="Train", level=cfg.LOGGER.LEVEL)
    logger.info(cfg)
    model = build_model(cfg)
    # print(model)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    if True:
        model = nn.DataParallel(model).cuda()

    criterion = build_loss(cfg)
    # 原来
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)
    # criterion_cent = 0
    # 修改添加中心损失
    criterion_cent = center_loss1()
    optimizer_centloss = torch.optim.SGD(criterion_cent.parameters(), lr=0.5)
    # scheduler = build_lr_scheduler(cfg, optimizer)

    train_loader = build_data(cfg, is_train=True)
    val_loader = build_data(cfg, is_train=False)

    logger.info(train_loader.dataset)
    # train_dataset_num_classes = len(train_loader.dataset.label_index_dict)
    # test_dataset_num_classes = len(val_loader[0].dataset.label_index_dict)
    # print(test_dataset_num_classes)
    for x in val_loader:
        logger.info(x.dataset)

    arguments = dict()
    arguments["iteration"] = 0

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    ckp_save_path = os.path.join(cfg.SAVE_DIR, cfg.NAME)
    os.makedirs(ckp_save_path, exist_ok=True)
    checkpointer = Checkpointer(model, optimizer, scheduler, ckp_save_path)

    tb_save_path = os.path.join(cfg.TB_SAVE_DIR, cfg.NAME)
    os.makedirs(tb_save_path, exist_ok=True)
    writer = SummaryWriter(tb_save_path)

    do_train(cfg, model, train_loader, val_loader,scheduler,
             criterion, criterion_cent,
             optimizer, optimizer_centloss,
             checkpointer, writer, device,
             checkpoint_period, arguments, logger)


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
