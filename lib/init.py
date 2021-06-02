#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Kun Luo
# @Email   : olooook@outlook.com
# @File    : init.py
# @Date    : 2021/06/02
# @Time    : 17:14:22

from torch import nn


def init_network(net: nn.Module) -> nn.Module:
    for param in net.parameters():
        if param.requires_grad:
            if param.ndim >= 2:
                nn.init.xavier_normal_(param)
            else:
                nn.init.uniform_(param)
    return net
