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
    """
    Graph Attention.

    Args:
        n_channels (int): Number of channels.
        n_timesteps (int): Number of timesteps.

    Shape:
        - signals: ``[batch_size, n_channels, n_nodes, n_timesteps]``
        - adjacency: ``[... , n_nodes, n_nodes]``
        - output: ``[batch_size, n_channels, n_nodes, n_timesteps]``
    """

    def __init__(self, n_channels: int, n_timesteps: int):
        super(GAttention, self).__init__()
        self.n_channels, self.n_timesteps = n_channels, n_timesteps
        self.Wg = nn.Parameter(torch.Tensor(n_timesteps, n_timesteps))
        self.alpha = nn.Parameter(torch.Tensor(n_channels))

    def forward(self, signals: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        k = q = torch.einsum('bint,i->bnt', signals, self.alpha)  # -> [batch_size, n_nodes, n_timesteps]
        att = torch.softmax(k @ self.Wg @ q.transpose(1, 2), dim=-1)  # -> [batch_size, n_nodes, n_nodes]
        # -> [batch_size, in_channels, n_nodes, n_timesteps]
        return torch.einsum('bni,bcit->bcnt', att * adjacency, signals)

    def extra_repr(self) -> str:
        return f'n_channels={self.n_channels}, n_timesteps={self.n_timesteps}'


class TAttention(nn.Module):
    """
    Temporal Attention.

    Args:
        n_channels (int): Number of channels.
        n_nodes (int): Number of nodes.

    Shape:
        - signals: ``[batch_size, n_channels, n_nodes, n_timesteps]``
        - output: ``[batch_size, n_channels, n_nodes, n_timesteps]``
    """

    def __init__(self, n_channels: int, n_nodes: int):
        super(TAttention, self).__init__()
        self.n_channels, self.n_nodes = n_channels, n_nodes
        self.Wt1 = nn.Parameter(torch.Tensor(10, n_nodes))
        self.Wt2 = nn.Parameter(torch.Tensor(10, n_nodes))
        self.alpha = nn.Parameter(torch.Tensor(n_channels))

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        k = q = torch.einsum('bint,i->btn', signals, self.alpha)  # -> [batch_size, n_timesteps, n_nodes]
        # -> [batch_size, n_timesteps, n_timesteps]
        att = torch.softmax((k @ self.Wt1.T) @ (q @ self.Wt2.T).transpose(1, 2), dim=-1)
        return torch.einsum('bti,bcni->bcnt', att, signals)  # -> [batch_size, n_channels, n_nodes, n_timesteps]

    def extra_repr(self) -> str:
        return f'n_channels={self.n_channels}, n_nodes={self.n_nodes}'


class CAttention(nn.Module):
    """
    Channel Attention.

    Args:
        n_nodes (int): Number of nodes.
        n_timesteps (int): Number of timesteps.

    Shape:
        - signals: ``[batch_size, n_channels, n_nodes, n_timesteps]``
        - output: ``[batch_size, n_channels, n_nodes, n_timesteps]``
    """

    def __init__(self, n_nodes, n_timesteps):
        super(CAttention, self).__init__()
        self.n_nodes, self.n_timesteps = n_nodes, n_timesteps
        self.Wc = nn.Parameter(torch.Tensor(n_timesteps, n_timesteps))
        self.alpha = nn.Parameter(torch.Tensor(n_nodes))

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        k = q = torch.einsum('bcit,i->bct', signals, self.alpha)  # -> [batch_size, n_channels, n_timesteps]
        att = torch.softmax(k @ self.Wc @ q.transpose(1, 2), dim=-1)  # -> [batch_size, n_channels, n_channels]
        return torch.einsum('bci,bint->bcnt', att, signals)  # -> [batch_size, n_channels, n_nodes, n_timesteps]

    def extra_repr(self) -> str:
        return f'n_nodes={self.n_nodes}, n_timesteps={self.n_timesteps}'
