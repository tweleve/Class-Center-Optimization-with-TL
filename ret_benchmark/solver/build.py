import torch

from torch.optim.lr_scheduler import ReduceLROnPlateau
from center_loss import CenterLoss


def center_loss1():
    use_gpu = torch.cuda.is_available()
    center_loss = CenterLoss(num_classes=10, feat_dim=2, use_gpu=use_gpu)
    return center_loss


def build_optimizer(cfg, model, center_loss=None):
    # center_loss = center_loss1()
    K = center_loss
    if center_loss:
        parameters = list(model.parameters) + list(center_loss.parameters())
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(
            # list(model.parameters()),
            parameters,
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(
            list(model.parameters()),
            # list(model.parameters) + list(center_loss.parameters()),
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    return optimizer


def build_lr_scheduler(cfg, optimizer):
    return ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=4,
        threshold=0.001,
        cooldown=2,
        min_lr=cfg.SOLVER.BASE_LR / (10 * cfg.SOLVER.STEPS),
    )
