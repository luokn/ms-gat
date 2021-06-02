#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Kun Luo
# @Email   : olooook@outlook.com
# @File    : loss.py
# @Date    : 2021/06/02
# @Time    : 17:09:21


import torch
from torch import nn


class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        return huber_loss(pred, target, self.delta)


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta=1.0) -> torch.Tensor:
    l1, l2 = delta * torch.abs(pred - target) - delta**2 / 2,  (pred - target)**2 / 2
    return torch.where(torch.abs(pred - target) <= delta, l2, l1).mean()
