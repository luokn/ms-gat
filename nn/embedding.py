#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Kun Luo
# @Email   : olooook@outlook.com
# @File    : embedding.py
# @Date    : 2021/07/02
# @Time    : 16:32:30


import torch
from torch import nn


class TE(nn.Module):
    def __init__(self, n_components, n_nodes, n_timesteps):
        super(TE, self).__init__()
        self.sizes = [n_components, n_nodes, n_timesteps]
        self.d_ebd = nn.Embedding(7, n_components * n_nodes * n_timesteps)
        self.h_ebd = nn.Embedding(24, n_components * n_nodes * n_timesteps)

    def forward(self, H: torch.Tensor, D: torch.Tensor):
        G = self.h_ebd(H) + self.d_ebd(D)  # -> [(batch_size * n_components * n_nodes * n_timesteps)]
        return G.view(len(G), *self.sizes)  # -> [batch_size, n_components, n_nodes, n_timesteps]
