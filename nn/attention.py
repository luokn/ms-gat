#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Kun Luo
# @Email   : olooook@outlook.com
# @File    : attention.py
# @Date    : 2021/07/02
# @Time    : 16:29:55


import torch
from torch import nn


class GAttention(nn.Module):
    def __init__(self, n_channels, n_timesteps):
        super(GAttention, self).__init__()
        self.Wg = nn.Parameter(torch.zeros(n_timesteps, n_timesteps), requires_grad=True)
        self.alpha = nn.Parameter(torch.zeros(n_channels), requires_grad=True)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        k = q = torch.einsum('bint,i->bnt', x, self.alpha)  # -> [batch_size, n_nodes, n_timesteps]
        att = torch.softmax(k @ self.Wg @ q.transpose(1, 2), dim=-1)  # -> [batch_size, n_nodes, n_nodes]
        return torch.einsum('bni,bcit->bcnt', att * adj, x)  # -> [batch_size, in_channels, n_nodes, n_timesteps]


class TAttention(nn.Module):
    def __init__(self, n_channels, n_nodes):
        super(TAttention, self).__init__()
        self.Wt1 = nn.Parameter(torch.zeros(10, n_nodes), requires_grad=True)
        self.Wt2 = nn.Parameter(torch.zeros(10, n_nodes), requires_grad=True)
        self.alpha = nn.Parameter(torch.zeros(n_channels), requires_grad=True)

    def forward(self, x: torch.Tensor):
        k = q = torch.einsum('bint,i->btn', x, self.alpha)  # -> [batch_size, n_timesteps, n_nodes]
        # -> [batch_size, n_timesteps, n_timesteps]
        att = torch.softmax((k @ self.Wt1.T) @ (q @ self.Wt2.T).transpose(1, 2), dim=-1)
        return torch.einsum('bti,bcni->bcnt', att, x)  # -> [batch_size, n_channels, n_nodes, n_timesteps]


class CAttention(nn.Module):
    def __init__(self, n_nodes, n_timesteps):
        super(CAttention, self).__init__()
        self.Wc = nn.Parameter(torch.zeros(n_timesteps, n_timesteps), requires_grad=True)
        self.alpha = nn.Parameter(torch.zeros(n_nodes), requires_grad=True)

    def forward(self, x: torch.Tensor):
        k = q = torch.einsum('bcit,i->bct', x, self.alpha)  # -> [batch_size, n_channels, n_timesteps]
        att = torch.softmax(k @ self.Wc @ q.transpose(1, 2), dim=-1)  # -> [batch_size, n_channels, n_channels]
        return torch.einsum('bci,bint->bcnt', att, x)  # -> [batch_size, n_channels, n_nodes, n_timesteps]
