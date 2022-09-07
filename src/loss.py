#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @File   : loss.py
# @Data   : 2021/06/02
# @Author : Luo Kun
# @Contact: luokun485@gmail.com

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
        torch.Tensor: loss
    """
    l1, l2 = delta * torch.abs(output - target) - delta**2 / 2, (output - target)**2 / 2
    return torch.where(torch.abs(output - target) <= delta, l2, l1).mean()


class GaussLoss(nn.Module):
    """
    Pytorch Implement of gauss loss.

    Args:
        sigma (float, optional): Defaults to 1.0.
        delta (float, optional): Defaults to 5e-2.
    """

    def __init__(self, sigma=1.0, delta=5e-2):
        super(GaussLoss, self).__init__()
        self.sigma, self.delta = sigma, delta

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return gauss_loss(output, target, self.sigma, self.delta)

    def extra_repr(self) -> str:
        return f"sigma={self.sigma}, delta={self.delta}"


def gauss_loss(output: torch.Tensor, target: torch.Tensor, sigma=1.0, delta=5e-2) -> torch.Tensor:
    r"""
    Gauss loss function.

    .. math::
    \begin{equation}
        \mathcal{L}_{\sigma,\delta}(\mathcal{Y}, \hat{\mathcal{Y}}) =
        \sigma^2 \left( 1 - \exp \{-\frac{(\mathcal{Y} - \hat{\mathcal{Y}}) ^2}{2 \sigma^2} \} \right) + \delta \left | \mathcal{Y} - \hat{\mathcal{Y}} \right |
    \end{equation}

    Args:
        output (torch.Tensor): Network output.
        target (torch.Tensor): Ground truth.
        sigma (float, optional): Defaults to 1.0.
        delta (float, optional): Defaults to 5e-2.

    Returns:
        torch.Tensor: loss
    """
    abs = torch.abs(output - target)
    return sigma**2 * torch.mean(1 - torch.exp(-(abs**2) / (2 * sigma**2))) + delta * torch.mean(abs)
