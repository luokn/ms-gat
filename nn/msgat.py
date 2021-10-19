#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Kun Luo
# @Email   : olooook@outlook.com
# @File    : msgat.py
# @Date    : 2021/06/02
# @Time    : 20:56:11


import torch
from torch import nn

from .attention import CAttention, GAttention, TAttention
from .embedding import TE


class GACN(nn.Module):
    def __init__(self, in_channels, out_channels, n_timesteps):
        super(GACN, self).__init__()
        self.in_channels, self.out_channels, self.n_timesteps = in_channels, out_channels, n_timesteps
        self.gatt = GAttention(n_channels=in_channels, n_timesteps=n_timesteps)
        self.W = nn.Parameter(torch.Tensor(out_channels, in_channels), requires_grad=True)

    def forward(self, signals: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        output = self.gatt(signals, adjacency)  # -> [batch_size, in_channels, n_nodes, n_timesteps]
        output = output.transpose(1, -1) @ self.W.T  # -> [batch_size, n_timesteps, n_nodes, out_channels]
        return output.transpose(1, -1)  # -> [batch_size, out_channels, n_nodes, n_timesteps]

    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, n_timesteps={self.n_timesteps}'


class Chomp(nn.Module):
    """
    Crop a fixed length on the last dimension.

    Args:
        chomp_size (int): Length of cropping.

    Shape:
        - input: ``[..., n_features]``
        - output: ``[..., n_features - chomp_size]
    """

    def __init__(self, chomp_size: int):
        super(Chomp, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input[..., :-self.chomp_size]

    def extra_repr(self) -> str:
        return f'chomp_size={self.chomp_size}'


class TACN(nn.Module):
    def __init__(self, in_channels, out_channels, n_nodes, dilations):
        super(TACN, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.n_nodes, self.dilations = n_nodes, dilations
        channels = [in_channels] + [out_channels] * len(dilations)
        seq = [TAttention(n_channels=in_channels, n_nodes=n_nodes)]
        for i, dilation in enumerate(dilations):
            seq += [
                nn.Conv2d(channels[i], channels[i + 1], [1, 2], padding=[0, dilation], dilation=[1, dilation]),
                Chomp(chomp_size=dilation),
            ]
        self.seq = nn.Sequential(*seq)

    def forward(self, signals: torch.Tensor):
        return self.seq(signals)  # -> [batch_size, out_channels, n_nodes, n_timesteps]

    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, n_nodes={self.n_nodes}, dilations={self.dilations}'


class CACN(nn.Module):
    def __init__(self, in_channels, out_channels, n_nodes, n_timesteps):
        super(CACN, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.n_nodes, self.n_timesteps = n_nodes, n_timesteps
        self.seq = nn.Sequential(
            CAttention(n_nodes=n_nodes, n_timesteps=n_timesteps),
            nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, signals: torch.Tensor):
        return self.seq(signals)  # -> [batch_size, out_channels, n_nodes, n_timesteps]

    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, n_nodes={self.n_nodes}, n_timesteps={self.n_timesteps}'


class MEAM(nn.Module):
    def __init__(self, in_channels, out_channels, n_nodes, n_timesteps, dilations):
        assert out_channels % 3 == 0
        super(MEAM, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.n_nodes, self.n_timesteps, self.dilations = n_nodes, n_timesteps, dilations
        self.ln = nn.LayerNorm([n_timesteps])
        self.res = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.cacn = CACN(in_channels, out_channels // 3, n_nodes=n_nodes, n_timesteps=n_timesteps)
        self.tacn = TACN(in_channels, out_channels // 3, n_nodes=n_nodes, dilations=dilations)
        self.gacn = GACN(in_channels, out_channels // 3, n_timesteps=n_timesteps)

    def forward(self, signals: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        output = self.ln(signals)  # -> [batch_size, in_channels, n_nodes, n_timesteps]
        output = torch.cat([
            self.cacn(output),  # channel dimension
            self.tacn(output),  # time dimension
            self.gacn(output, adjacency)  # space dimension
        ], dim=1)  # -> [batch_size, out_channels, n_nodes, n_timesteps]
        return torch.relu(output + self.res(signals))  # -> [batch_size, out_channels, n_nodes, n_timesteps]

    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, n_nodes={self.n_nodes}, n_timesteps={self.n_timesteps}, dilations={self.dilations}'


class TPC(nn.Module):
    def __init__(self, channels, n_nodes, in_timesteps, out_timesteps, dilations):
        super(TPC, self).__init__()
        self.channels, self.n_nodes, self.in_timesteps, self.out_timesteps, self.dilations = channels, n_nodes, in_timesteps, out_timesteps, dilations
        self.tgacns = nn.ModuleList([
            MEAM(channels[i], channels[i + 1], n_nodes=n_nodes, n_timesteps=in_timesteps, dilations=d)
            for i, d in enumerate(dilations)
        ])
        self.ln = nn.LayerNorm([in_timesteps])
        self.fc = nn.Conv2d(in_timesteps, out_timesteps, kernel_size=[1, channels[-1]])

    def forward(self, signals: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        for tgacn in self.tgacns:
            signals = tgacn(signals, adjacency)
        signals = self.ln(signals)  # -> [batch_size, out_channels, n_nodes, in_timesteps]
        signals = self.fc(signals.transpose(1, 3))  # -> [batch_size, out_timesteps, n_nodes, 1]
        return signals[..., 0].transpose(1, 2)  # -> [batch_size, n_nodes, out_timesteps]

    def extra_repr(self) -> str:
        return f'channels={self.channels}, n_nodes={self.n_nodes}, in_timesteps={self.in_timesteps}, out_timesteps={self.out_timesteps}, dilations={self.dilations}'


class MSGAT(nn.Module):
    def __init__(self, components, in_timesteps, out_timesteps, adjacency, use_te=True):
        super(MSGAT, self).__init__()
        if use_te:
            self.te = TE(len(components), len(adjacency), out_timesteps)
        else:
            self.W = nn.Parameter(torch.Tensor(len(components), len(adjacency), out_timesteps), requires_grad=True)
        self.adj = nn.Parameter(adjacency, requires_grad=False)
        self.tpcs = nn.ModuleList([
            TPC(
                channels=c['channels'], n_nodes=len(adjacency), in_timesteps=in_timesteps, out_timesteps=out_timesteps, dilations=c['dilations']
            ) for c in components
        ])

    def forward(self, X: torch.Tensor, H: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
        G = self.te(H, D).unbind(1) if self.te is not None else self.W.unbind()
        return sum((tpc(x, self.adj) * g for tpc, x, g in zip(self.tpcs, X.unbind(1), G)))


def msgat96(
    n_components: int, in_channels: int, in_timesteps: int, out_timesteps: int, adjacency: torch.Tensor, use_te=True
):
    components = [{"channels": [in_channels, 48, 48], "dilations":[[1, 2], [2, 4]]}] * n_components
    return MSGAT(components, in_timesteps=in_timesteps, out_timesteps=out_timesteps, adjacency=adjacency, use_te=use_te)


def msgat72(
    n_components: int, in_channels: int, in_timesteps: int, out_timesteps: int, adjacency: torch.Tensor, use_te=True
):
    components = [{"channels": [in_channels, 72, 72], "dilations":[[1, 2], [2, 4]]}] * n_components
    return MSGAT(components, in_timesteps=in_timesteps, out_timesteps=out_timesteps, adjacency=adjacency, use_te=use_te)


def msgat48(
    n_components: int, in_channels: int, in_timesteps: int, out_timesteps: int, adjacency: torch.Tensor, use_te=True
):
    components = [{"channels": [in_channels, 96, 96], "dilations":[[1, 1, 2, 2], [4, 4]]}] * n_components
    return MSGAT(components, in_timesteps=in_timesteps, out_timesteps=out_timesteps, adjacency=adjacency, use_te=use_te)
