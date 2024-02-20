import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim * 256
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        # dismat 计算x中每个样本与中心点之间的距离。
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        # dismat = 1*dismat +  (-2)*(  x*self.centers.t()  ), 即 beta* dismat + alpha * (mat1*mat2);
        # admm_(beta,alpha,mat1,mat2)
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()  # 创建类别张量
        if self.use_gpu:
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)  # 真实标签扩展成（batch_size,num_classes）
        mask = labels.eq(classes.expand(batch_size, self.num_classes))  # 创建二值掩码mask

        dist = distmat * mask.float()  # 仅保留正确类别，其余为0
        # loss: 首先将dist矩阵中的元素夹紧到一个极小的非零值和一个较大值，然后将所有非零距离的和除以批次大小batch_size
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
