# -*- coding: utf-8 -*-
# @Time    : 2024/3/11 0011 15:06
# @Author  : Ronin
# @File    : loss.py
# @Software: PyCharm
import torch
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):

    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, x1, x2):
        # 目标1：让y等于0时，x[0]大的损失函数小,也就是x[1]小的损失函数小
        # 目标2：让y等于1时，x[1]大的损失函数小,也就是x[0]小的损失函数小
        # 目标3：不能完全相同，避免过拟合
        dist = F.pairwise_distance(x1, x2)
        return dist.mean()
