#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Kun Luo
# @Email   : olooook@outlook.com
# @File    : metrics.py
# @Date    : 2021/06/02
# @Time    : 17:11:49

import torch


class Metrics:
    def __init__(self, mask_value=.0):
        self.n, self.mask_value = 0, mask_value
        self.AE, self.APE, self.SE = .0, .0, .0
        self.MAE, self.MAPE, self.RMSE = .0, .0, .0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        self.n += target.nelement()
        # MAE
        self.AE += torch.abs(pred - target).sum().item()
        self.MAE = self.AE / self.n
        # MAPE
        mask = target > self.mask_value
        masked_pred, masked_target = pred[mask], target[mask]
        self.APE += 100 * torch.abs((masked_pred - masked_target) / masked_target).sum().item()
        self.MAPE = self.APE / self.n
        # RMSE
        self.SE += torch.square(pred - target).sum().item()
        self.RMSE = (self.SE / self.n)**.5

    def stats(self):
        return dict(MAE=self.MAE, MAPE=self.MAPE, RMSE=self.RMSE)