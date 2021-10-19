#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Kun Luo
# @Email   : olooook@outlook.com
# @File    : init.py
# @Date    : 2021/06/02
# @Time    : 17:14:22

from torch import nn


def init_network(net: nn.Module) -> nn.Module:
    """
    Use ``xavier_normal_`` or ``uniform_`` to initialize the parameters of the network

    Args:
        net (nn.Module): Neural network to initialize.

    Returns:
        nn.Module: net.
    """
    for param in net.parameters():
        if param.requires_grad:
            if param.ndim >= 2:
                nn.init.xavier_normal_(param)
            else:
                f_out = param.size(0)
                nn.init.uniform_(param, -f_out**-.5, f_out**-.5)
    return net
