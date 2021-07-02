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
        self.gatt = GAttention(n_channels=in_channels, n_timesteps=n_timesteps)
        self.W = nn.Parameter(torch.zeros(out_channels, in_channels), requires_grad=True)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        out = self.gatt(x, adj)  # -> [batch_size, in_channels, n_nodes, n_timesteps]
        out = out.transpose(1, -1) @ self.W.T  # -> [batch_size, n_timesteps, n_nodes, out_channels]
        return out.transpose(1, -1)  # -> [batch_size, out_channels, n_nodes, n_timesteps]


class Chomp(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor):
        return x[..., :-self.chomp_size]


class TACN(nn.Module):
    def __init__(self, in_channels, out_channels, n_nodes, dilations):
        super(TACN, self).__init__()
        channels = [in_channels] + [out_channels] * len(dilations)
        seq = [TAttention(n_channels=in_channels, n_nodes=n_nodes)]
        for i, dilation in enumerate(dilations):
            seq += [
                nn.Conv2d(channels[i], channels[i + 1], [1, 2], padding=[0, dilation], dilation=[1, dilation]),
                Chomp(chomp_size=dilation),
            ]
        self.seq = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor):
        return self.seq(x)  # -> [batch_size, out_channels, n_nodes, n_timesteps]


class CACN(nn.Module):
    def __init__(self, in_channels, out_channels, n_nodes, n_timesteps):
        super(CACN, self).__init__()
        self.seq = nn.Sequential(
            CAttention(n_nodes=n_nodes, n_timesteps=n_timesteps),
            nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: torch.Tensor):
        return self.seq(x)  # -> [batch_size, out_channels, n_nodes, n_timesteps]


class MEAM(nn.Module):
    def __init__(self, in_channels, out_channels, n_nodes, n_timesteps, dilations):
        assert out_channels % 3 == 0
        super(MEAM, self).__init__()
        self.ln = nn.LayerNorm([n_timesteps])
        self.res = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.cacn = CACN(in_channels, out_channels // 3, n_nodes=n_nodes, n_timesteps=n_timesteps)
        self.tacn = TACN(in_channels, out_channels // 3, n_nodes=n_nodes, dilations=dilations)
        self.gacn = GACN(in_channels, out_channels // 3, n_timesteps=n_timesteps)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        out = self.ln(x)  # -> [batch_size, in_channels, n_nodes, n_timesteps]
        out = torch.cat([
            self.cacn(out),  # channel dimension
            self.tacn(out),  # time dimension
            self.gacn(out, adj)  # space dimension
        ], dim=1)  # -> [batch_size, out_channels, n_nodes, n_timesteps]
        return torch.relu(out + self.res(x))  # -> [batch_size, out_channels, n_nodes, n_timesteps]


class TPC(nn.Module):
    def __init__(self, channels, n_nodes, in_timesteps, out_timesteps, dilations):
        super(TPC, self).__init__()
        self.tgacns = nn.ModuleList([
            MEAM(channels[i], channels[i + 1], n_nodes=n_nodes, n_timesteps=in_timesteps, dilations=d)
            for i, d in enumerate(dilations)
        ])
        self.ln = nn.LayerNorm([in_timesteps])
        self.fc = nn.Conv2d(in_timesteps, out_timesteps, kernel_size=[1, channels[-1]])

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        for tgacn in self.tgacns:
            x = tgacn(x, adj)
        x = self.ln(x)  # -> [batch_size, out_channels, n_nodes, in_timesteps]
        x = self.fc(x.transpose(1, 3))  # -> [batch_size, out_timesteps, n_nodes, 1]
        return x[..., 0].transpose(1, 2)  # -> [batch_size, n_nodes, out_timesteps]


class MSGAT(nn.Module):
    def __init__(self, components, in_timesteps, out_timesteps, adj, te=True):
        super(MSGAT, self).__init__()
        if te:
            self.te = TE(len(components), len(adj), out_timesteps)
        else:
            self.W = nn.Parameter(torch.zeros(len(components), len(adj), out_timesteps), requires_grad=True)
        self.adj = nn.Parameter(adj, requires_grad=False)
        self.tpcs = nn.ModuleList([
            TPC(channels=c['channels'], n_nodes=len(adj), in_timesteps=in_timesteps,
                out_timesteps=out_timesteps, dilations=c['dilations'])
            for c in components
        ])

    def forward(self, X: torch.Tensor, H: torch.Tensor, D: torch.Tensor):
        G = self.te(H, D).unbind(1) if self.te else self.W.unbind()
        return sum((tpc(x, self.adj) * g for tpc, x, g in zip(self.tpcs, X.unbind(1), G)))


def msgat96(n_components: int, in_channels: int, in_timesteps: int, out_timesteps: int, adj: torch.Tensor, te=True):
    components = [{"channels": [in_channels, 48, 48], "dilations":[[1, 2], [2, 4]]}] * n_components
    net = MSGAT(components, in_timesteps=in_timesteps, out_timesteps=out_timesteps, adj=adj, te=te)
    return net


def msgat72(n_components: int, in_channels: int, in_timesteps: int, out_timesteps: int, adj: torch.Tensor, te=True):
    components = [{"channels": [in_channels, 72, 72], "dilations":[[1, 2], [2, 4]]}] * n_components
    net = MSGAT(components, in_timesteps=in_timesteps, out_timesteps=out_timesteps, adj=adj, te=te)
    return net


def msgat48(n_components: int, in_channels: int, in_timesteps: int, out_timesteps: int, adj: torch.Tensor, te=True):
    components = [{"channels": [in_channels, 96, 96], "dilations":[[1, 1, 2, 2], [4, 4]]}] * n_components
    net = MSGAT(components, in_timesteps=in_timesteps, out_timesteps=out_timesteps, adj=adj, te=te)
    return net
