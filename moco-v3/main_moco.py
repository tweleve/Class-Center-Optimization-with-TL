import argparse
import math
import os
import shutil
import time
from functools import partial
import torch
import torch.optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models
from torch.utils.tensorboard import SummaryWriter
import moco.builder
import moco.loader
import moco.optimizer
import vits
import torch.distributed as dist
import torch.multiprocessing as mp
import warnings
import builtins
import random
import torch.nn as nn


def main_worker(args):

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    print("=> creating model '{}'".format(args.arch))
    # create model
    if args.arch.startswith('vit'):
        model = moco.builder.MoCo_ViT(partial(vits.__dict__[args.arch], stop_grad_conv1=args.stop_grad_conv1),
                                      args.moco_dim, args.moco_mlp_dim, args.moco_t, add_bf=args.add_bf)
    else:
        model = moco.builder.MoCo_ResNet(
            partial(torchvision_models.__dict__[args.arch], zero_init_residual=True),
            args.moco_dim, args.moco_mlp_dim, args.moco_t, add_bf=args.add_bf)

    # 在更改批量大小之前推断学习率。infer learning rate before changing batch size
    args.lr = args.lr * args.batch_size / 256

    if not torch.cuda.is_available():
        print('GPU not used, now use CPU, this will be slow')
    elif args.gpu is not None:  # Run
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # 注释掉以下行以进行调试.comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported--1.")
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported--2.")  # AllGather/rank 实现仅支持 DDP

    # 此处的optimizer可以修改为SGD + Momentum
    if args.optimizer == 'lars':
        optimizer = moco.optimizer.LARS(model.parameters(), args.lr, weight_decay=args.weight_decay,
                                        momentum=args.momentum)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler()
    summary_writer = SummaryWriter() if args.rank == 0 else None

    # 可选地从检查点恢复.optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # 要加载到指定的单个 gpu 的 Map 模型。Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # 数据加载代码,Data loading code
    # args.data = '/home/hzj/Pycharms/xbm/datasets/CUB_200_2011_1'
    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
    # 遵循 BYOL 的增强配方：https://arxiv.org/abs/2006.07733
    augmentation1 = [
        transforms.RandomResizedCrop(224, scale=(args.crop_min, 1.)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),  # not strengthened，未加强
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=1.0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    augmentation2 = [
        transforms.RandomResizedCrop(224, scale=(args.crop_min, 1.)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),  # not strengthened，未加强
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.1),
        transforms.RandomApply([moco.loader.Solarize()], p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]
    # print(traindir)
    train_dataset = datasets.ImageFolder(traindir,
                                         moco.loader.TwoCropsTransform(transforms.Compose(augmentation1),
                                                                       transforms.Compose(augmentation2)))

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    for epoch in range(args.start_epoch, args.epochs):
        # 训练一个epoch,train for one epoch
        train(train_loader, model, optimizer, scaler, summary_writer, epoch, args)
        save_checkpoint({'epoch': epoch + 1, 'arch': args.arch, 'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict()}, is_best=False,
                        filename='./cp/checkpoint_%04d.pth.tar' % epoch)

    if args.rank == 0:
        summary_writer.close()


def train(train_loader, model, optimizer, scaler, summary_writer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    learning_rates = AverageMeter('LR', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time, learning_rates, losses],
                             prefix="Epoch: [{}]".format(epoch))
    # 切换到训练模式
    model.train()
    end = time.time()
    iters_per_epoch = len(train_loader)
    moco_m = args.moco_m
    for i, (images, _) in enumerate(train_loader):
        # 对于训练集中的每个batch，测量数据加载时间，并根据当前迭代次数调整学习率和动量系数。
        data_time.update(time.time() - end)

        # adjust learning rate and momentum coefficient per iteration. 每次迭代调整学习率和动量系数
        lr = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, args)
        learning_rates.update(lr)
        if args.moco_m_cos:
            moco_m = adjust_moco_momentum(epoch + i / iters_per_epoch, args)

        images[0] = images[0].cuda(args.gpu, non_blocking=True)
        images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output计算损失
        with torch.cuda.amp.autocast(True):
            loss = model(images[0], images[1], moco_m)
        # 更新损失
        losses.update(loss.item(), images[0].size(0))
        if args.rank == 0:
            # scaler: 用于自动混合精度训练的缩放器。
            summary_writer.add_scalar("loss", loss.item(), epoch * iters_per_epoch + i)

        # compute gradient and do SGD step计算梯度并执行 SGD 步骤
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 测量经过的时间
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    # 计算并存储平均值和当前值

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    # 预热后使用半周期余弦衰减学习率
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.lr * 0.5 * (
                1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    # 根据当前epoch调整 moco 动量
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)
    return m


def creat_parser():
    parser = argparse.ArgumentParser(description='MoCo ImageNet Pre-Training')
    parser.add_argument('--data', metavar='DIR', default='/home/hzj/Pycharms/xbm/datasets/UCMD-21', type=str,
                        required=False,
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N',
                        help='mini-batch size (default: 4096), '
                             'this is the total batch size of all GPUs on the current node '
                             'when using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.6, type=float, metavar='LR',
                        help='initial (base) learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float, metavar='W',
                        help='weight decay (default: 1e-6)', dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N',
                        help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',  # ./pth/checkpoint_0099.pth.tar
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--rank', default=0, type=int,
                        help='分布式训练的节点等级，node rank for distributed training，default=-1, ')

    parser.add_argument('--seed', default=None, type=int,
                        help='用于初始化训练的种子，seed for initializing training. ')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')

    # moco specific configs:
    parser.add_argument('--moco-dim', default=256, type=int,
                        help='feature dimension (default: 256)')
    parser.add_argument('--moco-mlp-dim', default=4096, type=int,
                        help='hidden dimension in MLPs (default: 4096)')
    parser.add_argument('--moco-m', default=0.99, type=float,
                        help='moco momentum of updating momentum encoder (default: 0.99)')
    parser.add_argument('--moco-m-cos', action='store_true',
                        help='使用半周期余弦计划逐渐将 moco 动量增加到 1。'
                             'gradually increase moco momentum to 1 with a half-cycle cosine schedule')
    parser.add_argument('--moco-t', default=1.0, type=float,
                        help='softmax temperature (default: 1.0)')

    # vit 特定配置: vit specific configs:
    parser.add_argument('--stop-grad-conv1', action='store_true',
                        help='stop-grad after first conv, or patch embedding')

    # other upgrades
    parser.add_argument('--optimizer', default='lars', type=str, choices=['lars', 'adamw'],
                        help='optimizer used (default: lars)')
    parser.add_argument('--warmup-epochs', default=10, type=int, metavar='N',
                        help='number of warmup epochs')
    parser.add_argument('--crop-min', default=0.08, type=float,
                        help='随机裁剪的最小比例。minimum scale for random cropping (default: 0.08)')
    parser.add_argument('--add_bf', default=1, type=int,
                        help='添加BatchFormer 和不添加BatchFormer的区别。add_bf')

    return parser.parse_args()


if __name__ == '__main__':
    torchvision_model_names = sorted(name for name in torchvision_models.__dict__
                                     if name.islower() and not name.startswith("__")
                                     and callable(torchvision_models.__dict__[name]))

    model_names = ['vit_small', 'vit_base',
                   'vit_conv_small', 'vit_conv_base'] + torchvision_model_names
    args = creat_parser()
    # args.data = '/home/hzj/Pycharms/xbm/datasets/CUB_200_2011_1'
    # args.data = '/home/hzj/Pycharms/xbm/datasets/UCMD-21'

    main_worker(args)
