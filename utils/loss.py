import torch
from torch.nn import functional as F


def criterion_R(R_pred, R_gt, alpha=1):
    # 将 [batchsize, 9] 转换回 [batchsize, 3, 3] 的旋转矩阵
    R_pred = R_pred.view(-1, 3, 3)
    R_gt = R_gt.view(-1, 3, 3)

    R_pred = R_pred.to(torch.float32)  # 将 R_pred 转换为 float32
    R_gt = R_gt.to(torch.float32)  
    # 计算 Geodesic Loss
    R_diff = torch.matmul(R_pred.transpose(-1, -2), R_gt)
    
    # 计算旋转差异的迹（trace）
    trace = torch.sum(torch.diagonal(R_diff, dim1=-2, dim2=-1), dim=-1)
    
    # 通过 arccos 计算角度差异
    theta = torch.acos(torch.clamp((trace - 1) / 2, -1 + 1e-7, 1 - 1e-7))
    
    # Geodesic Loss
    geodesic_loss = torch.mean(theta)
    
    # 计算 L1 Loss
    l1_loss = F.l1_loss(R_pred, R_gt)
    
    # 组合 Geodesic Loss 和 L1 Loss
    combined_loss = alpha * geodesic_loss + (1 - alpha) * l1_loss


    
    return combined_loss

def L1_loss(me, gt):

    if isinstance(me, list):
        me = torch.tensor(me)
    if isinstance(gt, list):
        gt = torch.tensor(gt)

    # Compute the L1 loss (mean absolute error) between the reconstructed and target images
    loss = F.l1_loss(me, gt, reduction='mean')
    
    return loss