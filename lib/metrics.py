#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Kun Luo
# @Email   : olooook@outlook.com
# @File    : metrics.py
# @Date    : 2021/06/02
# @Time    : 17:11:49

import torch


class Metrics:
    """
    Calculate and update ``MAE/MAPE/RMSE`` with the training batch.
    """

    def __init__(self, mask_value=.0):
        self.n, self.mask_value = 0, mask_value
        self.AE, self.APE, self.SE = .0, .0, .0

    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> dict:
        self.n += target.nelement()
        # MAE
        self.AE += torch.abs(output - target).sum().item()
        MAE = self.AE / self.n
        # MAPE
        mask = target > self.mask_value
        masked_output, masked_target = output[mask], target[mask]
        self.APE += 100 * torch.abs((masked_output - masked_target) / masked_target).sum().item()
        MAPE = self.APE / self.n
        # RMSE
        self.SE += torch.square(output - target).sum().item()
        RMSE = (self.SE / self.n)**.5
        # stats
        return {'MAE': MAE, 'MAPE': MAPE, 'RMSE': RMSE}
