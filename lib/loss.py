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
    Pytorch Implement of huber loss.

    Args:
        delta (float, optional): Defaults to 1.0.
    """

    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        return huber_loss(output, target, self.delta)


def huber_loss(output: torch.Tensor, target: torch.Tensor, delta=1.0) -> torch.Tensor:
    r"""
    Huber loss function.

    .. math::
    \begin{equation}
        \mathcal{L}_{\delta}(\mathcal{Y}, \hat{\mathcal{Y}}) =
        \begin{aligned}
            \begin{cases}
                \frac{1}{2} \left ( \mathcal{Y} - \hat{\mathcal{Y}} \right ) ^2                 & if \ \left | \mathcal{Y} - \hat{\mathcal{Y}} \right | \leq \delta \\
                \delta \left | \mathcal{Y} - \hat{\mathcal{Y}} \right | - \frac{1}{2} \delta ^2 & otherwise
            \end{cases}
        \end{aligned}
    \end{equation}

    Args:
        output (torch.Tensor): Network output.
        target (torch.Tensor): Ground truth.
        delta (float, optional): Defaults to 1.0.

    Returns:
        torch.Tensor: [description]
    """
    nn.L1Loss
    l1, l2 = delta * torch.abs(output - target) - delta**2 / 2, (output - target)**2 / 2
    return torch.where(torch.abs(output - target) <= delta, l2, l1).mean()
