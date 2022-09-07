#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @File   : msgat.py
# @Data   : 2021/06/02
# @Author : Luo Kun
# @Contact: luokun485@gmail.com

from typing import List

import torch
from torch import nn

from .attention import ChannelAttention, GraphAttention, TemporalAttention
from .embeddings import TimeEmbedding


class GACN(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, n_timesteps: int):
        super(GACN, self).__init__()
        self.in_channels, self.out_channels, self.n_timesteps = in_channels, out_channels, n_timesteps
        self.gatt = GraphAttention(n_channels=in_channels, n_timesteps=n_timesteps)
        self.W = nn.Parameter(torch.Tensor(out_channels, in_channels), requires_grad=True)

    def forward(self, signals: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        output = self.gatt(signals, adjacency)  # -> [batch_size, in_channels, n_nodes, n_timesteps]
        output = output.transpose(1, -1) @ self.W.T  # -> [batch_size, n_timesteps, n_nodes, out_channels]
        return output.transpose(1, -1)  # -> [batch_size, out_channels, n_nodes, n_timesteps]

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, n_timesteps={self.n_timesteps}"


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
        return f"chomp_size={self.chomp_size}"


class TACN(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, n_nodes: int, dilations: List[int]):
        super(TACN, self).__init__()
        self.in_channels, self.out_channels, self.n_nodes, self.dilations = (
            in_channels,
            out_channels,
            n_nodes,
            dilations,
        )
        channels = [in_channels] + [out_channels] * len(dilations)
        seq = [TemporalAttention(n_channels=in_channels, n_nodes=n_nodes)]
        for i, dilation in enumerate(dilations):
            seq += [
                nn.Conv2d(channels[i], channels[i + 1], [1, 2], padding=[0, dilation], dilation=[1, dilation]),
                Chomp(chomp_size=dilation),
            ]
        self.seq = nn.Sequential(*seq)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        return self.seq(signals)  # -> [batch_size, out_channels, n_nodes, n_timesteps]

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, n_nodes={self.n_nodes}, dilations={self.dilations}"


class CACN(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, n_nodes: int, n_timesteps: int):
        super(CACN, self).__init__()
        self.in_channels, self.out_channels, self.n_nodes, self.n_timesteps = (
            in_channels,
            out_channels,
            n_nodes,
            n_timesteps,
        )
        self.seq = nn.Sequential(ChannelAttention(n_nodes=n_nodes, n_timesteps=n_timesteps),
                                 nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        return self.seq(signals)  # -> [batch_size, out_channels, n_nodes, n_timesteps]

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, n_nodes={self.n_nodes}, n_timesteps={self.n_timesteps}"


class MEAM(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, n_nodes: int, n_timesteps: int, dilations: List[int]):
        assert out_channels % 3 == 0
        super(MEAM, self).__init__()
        self.in_channels, self.out_channels, self.n_nodes, self.n_timesteps, self.dilations = (
            in_channels,
            out_channels,
            n_nodes,
            n_timesteps,
            dilations,
        )
        self.ln = nn.LayerNorm([n_timesteps])
        self.res = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.cacn = CACN(in_channels, out_channels // 3, n_nodes=n_nodes, n_timesteps=n_timesteps)
        self.tacn = TACN(in_channels, out_channels // 3, n_nodes=n_nodes, dilations=dilations)
        self.gacn = GACN(in_channels, out_channels // 3, n_timesteps=n_timesteps)

    def forward(self, signals: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        output = self.ln(signals)  # -> [batch_size, in_channels, n_nodes, n_timesteps]
        output = torch.cat(
            [
                self.cacn(output),  # channel dimension
                self.tacn(output),  # temporal dimension
                self.gacn(output, adjacency),  # spatial dimension
            ],
            dim=1,
        )  # -> [batch_size, out_channels, n_nodes, n_timesteps]
        return torch.relu(output + self.res(signals))  # -> [batch_size, out_channels, n_nodes, n_timesteps]

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, n_nodes={self.n_nodes}, n_timesteps={self.n_timesteps}, dilations={self.dilations}"


class TPC(nn.Module):

    def __init__(self, channels: List[int], n_nodes: int, in_timesteps: int, out_timesteps: int, dilations: List[int]):
        super(TPC, self).__init__()
        self.channels, self.n_nodes, self.in_timesteps, self.out_timesteps, self.dilations = (
            channels,
            n_nodes,
            in_timesteps,
            out_timesteps,
            dilations,
        )
        self.tgacns = nn.ModuleList([
            MEAM(channels[i], channels[i + 1], n_nodes=n_nodes, n_timesteps=in_timesteps, dilations=d)
            for i, d in enumerate(dilations)
        ])
        self.ln = nn.LayerNorm([in_timesteps])
        self.fc = nn.Conv2d(in_timesteps, out_timesteps, kernel_size=[1, channels[-1]])

    def forward(self, signals: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        for tgacn in self.tgacns:
            signals = tgacn(signals, adjacency)
        output = self.ln(signals)  # -> [batch_size, out_channels, n_nodes, in_timesteps]
        output = self.fc(output.transpose(1, 3))  # -> [batch_size, out_timesteps, n_nodes, 1]
        return output[..., 0].transpose(1, 2)  # -> [batch_size, n_nodes, out_timesteps]

    def extra_repr(self) -> str:
        return f"channels={self.channels}, n_nodes={self.n_nodes}, in_timesteps={self.in_timesteps}, out_timesteps={self.out_timesteps}, dilations={self.dilations}"


class MSGAT(nn.Module):
    """
    The MS-GAT Model.

    Args:
        components (list): Configurations for the components.
        in_timesteps (int): Number of input timesteps.
        out_timesteps (int): Number of outpuy timesteps.
        use_te (bool, optional): Use TE. Defaults to True.
        adj (torch.Tensor): Adjacency matrix.

    Shape:
        X: ``[batch_size, n_channels, n_nodes, in_timesteps]``
        H: ``[batch_size]``
        D: ``[batch_size]``
        output: ``[batch_size, n_nodes, out_timesteps]``
    """

    def __init__(self, components: List[dict], in_timesteps: int, out_timesteps: int, use_te: bool, adj: torch.Tensor):
        super(MSGAT, self).__init__()
        if use_te:
            self.te = TimeEmbedding(len(components), len(adj), out_timesteps)
        else:
            self.W = nn.Parameter(torch.Tensor(len(components), len(adj), out_timesteps), requires_grad=True)
        self.adj = nn.Parameter(adj, requires_grad=False)
        self.tpcs = nn.ModuleList([
            TPC(
                channels=component["channels"],
                n_nodes=len(adj),
                in_timesteps=in_timesteps,
                out_timesteps=out_timesteps,
                dilations=component["dilations"],
            ) for component in components
        ])
        self.reset_parameters()

    def forward(self, X: torch.Tensor, H: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
        G = self.te(H, D).unbind(1) if self.te is not None else self.W.unbind()
        return sum((tpc(x, self.adj) * g for tpc, x, g in zip(self.tpcs, X.unbind(1), G)))

    def reset_parameters(self):
        """
        Use ``xavier_normal_`` or ``uniform_`` to initialize the parameters of the network
        """
        for param in self.parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2:
                nn.init.xavier_normal_(param)
            else:
                f_out = param.size(0)
                nn.init.uniform_(param, -(f_out**-0.5), f_out**-0.5)


def msgat48(n_components: int, in_channels: int, **kwargs):
    return MSGAT([{"channels": [in_channels, 48, 48], "dilations": [[1, 2], [2, 4]]}] * n_components, **kwargs)


def msgat72(n_components: int, in_channels: int, **kwargs):
    return MSGAT([{"channels": [in_channels, 72, 72], "dilations": [[1, 2], [2, 4]]}] * n_components, **kwargs)


def msgat96(n_components: int, in_channels: int, **kwargs):
    return MSGAT([{"channels": [in_channels, 96, 96], "dilations": [[1, 1, 2, 2], [4, 4]]}] * n_components, **kwargs)
