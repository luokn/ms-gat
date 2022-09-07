#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @File   : embeddings.py
# @Data   : 2021/07/02
# @Author : Luo Kun
# @Contact: luokun485@gmail.com

import torch
from torch import nn


class TimeEmbedding(nn.Module):
    """
    Time Embedding.

    Args:
        n_components (int): Number of TPC.
        n_nodes (int): Number of nodes in the graph.
        n_timesteps (int): Number of output timesteps.

    Shape:
        H: ``[batch_size]``
        D: ``[batch_size]``
        output: ``[batch_size, n_components, n_nodes, n_timesteps]``
    """

    def __init__(self, n_components: int, n_nodes: int, n_timesteps: int):
        super(TimeEmbedding, self).__init__()
        self.n_components, self.n_nodes, self.n_timesteps = n_components, n_nodes, n_timesteps
        self.d_ebd = nn.Embedding(7, n_components * n_nodes * n_timesteps)
        self.h_ebd = nn.Embedding(24, n_components * n_nodes * n_timesteps)

    def forward(self, H: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
        G = self.h_ebd(H) + self.d_ebd(D)  # -> [(batch_size * n_components * n_nodes * n_timesteps)]
        # -> [batch_size, n_components, n_nodes, n_timesteps]
        return G.view(len(G), self.n_components, self.n_nodes, self.n_timesteps)

    def extra_repr(self) -> str:
        return f"n_components={self.n_components}, n_nodes={self.n_nodes}, n_timesteps={self.n_timesteps}"
