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
    """
    Implement of huber loss
    """

    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        return huber_loss(output, target, self.delta)


def huber_loss(output: torch.Tensor, target: torch.Tensor, delta=1.0) -> torch.Tensor:
    l1, l2 = delta * torch.abs(output - target) - delta**2 / 2, (output - target)**2 / 2
    return torch.where(torch.abs(output - target) <= delta, l2, l1).mean()
