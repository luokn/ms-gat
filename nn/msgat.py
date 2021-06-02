#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Kun Luo
# @Email   : olooook@outlook.com
# @File    : msgat.py
# @Date    : 2021/06/02
# @Time    : 20:56:11

import torch
from torch import nn


class GAttention(nn.Module):
    def __init__(self, n_channels, n_timesteps):
        super(GAttention, self).__init__()
        self.W = nn.Parameter(torch.zeros(n_timesteps, n_timesteps), requires_grad=True)
        self.alpha = nn.Parameter(torch.zeros(n_channels), requires_grad=True)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        # k_{n,t} = q_{n,t} = x_{i,n,t} \alpha_{i}
        k = q = torch.einsum('bint,i->bnt', x, self.alpha)  # -> [batch_size, n_nodes, in_timesteps]
        att = torch.softmax(k @ self.W @ q.transpose(1, 2), dim=-1)  # -> [batch_size, n_nodes, n_nodes]
        return att * adj  # -> [batch_size, n_nodes, n_nodes]


class GACN(nn.Module):
    def __init__(self, in_channels, out_channels, n_timesteps):
        super(GACN, self).__init__()
        self.gatt = GAttention(n_channels=in_channels, n_timesteps=n_timesteps)
        self.W = nn.Parameter(torch.zeros(out_channels, in_channels), requires_grad=True)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        # [batch_size, n_nodes, n_nodes] @ [in_timesteps, batch_size, n_nodes, in_channels] @ [in_channels, out_channels]
        # -> [in_timesteps, batch_size, in_timesteps, out_channels]
        x_out = self.gatt(x, adj) @ x.permute(3, 0, 2, 1) @ self.W.T
        return x_out.permute(1, 3, 2, 0)  # -> [batch_size, out_channels, n_nodes, in_timesteps]


class TAttention(nn.Module):
    def __init__(self, n_channels, n_nodes):
        super(TAttention, self).__init__()
        self.W1 = nn.Parameter(torch.zeros(10, n_nodes), requires_grad=True)
        self.W2 = nn.Parameter(torch.zeros(10, n_nodes), requires_grad=True)
        self.alpha = nn.Parameter(torch.zeros(n_channels), requires_grad=True)

    def forward(self, x: torch.Tensor):
        # k_{t,n} = q_{t,n} = x_{i,n,t} \alpha_{i}
        k = q = torch.einsum('bint,i->btn', x, self.alpha)  # -> [batch_size, in_timesteps, n_nodes]
        # -> [batch_size, in_timesteps, in_timesteps]
        att = torch.softmax((k @ self.W1.T) @ (q @ self.W2.T).transpose(1, 2), dim=-1)
        # y_{c,n,t} = a_{t,i} x_{c,n,i}
        return torch.einsum('bti,bcni->bcnt', att, x)  # -> [batch_size, in_channels, n_nodes, in_timesteps]


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
                Chomp(dilation)
            ]
        self.seq = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor):
        return self.seq(x)  # -> [batch_size, out_channels, n_nodes, in_timesteps]


class CAttention(nn.Module):
    def __init__(self, n_nodes, n_timesteps):
        super(CAttention, self).__init__()
        self.W = nn.Parameter(torch.zeros(n_timesteps, n_timesteps), requires_grad=True)
        self.alpha = nn.Parameter(torch.zeros(n_nodes), requires_grad=True)

    def forward(self, x: torch.Tensor):
        # k_{c,t} = q_{c,t} = x_{c,i,t} \alpha_{i}
        k = q = torch.einsum('bcit,i->bct', x, self.alpha)  # -> [batch_size, in_channels, in_timesteps]
        att = torch.softmax(k @ self.W @ q.transpose(1, 2), dim=-1)  # [batch_size, in_channels, in_channels]
        # y_{c,n,t} = a_{c,i} x_{i,n,t}
        return torch.einsum('bci,bint->bcnt', att, x)  # -> [batch_size, in_channels, n_nodes, in_timesteps]


class CACN(nn.Module):
    def __init__(self, in_channels, out_channels, n_nodes, n_timesteps):
        super(CACN, self).__init__()
        self.catt = CAttention(n_nodes=n_nodes, n_timesteps=n_timesteps)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        out = self.catt(x)  # -> [batch_size, in_channels, n_nodes, in_timesteps]
        return self.conv(out)  # -> [batch_size, out_channels, n_nodes, in_timesteps]


class TGACN(nn.Module):
    def __init__(self, in_channels, out_channels, n_nodes, n_timesteps, dilations):
        assert in_channels % 3 == 0
        super(TGACN, self).__init__()
        self.ln = nn.LayerNorm([n_timesteps])
        self.res = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.cacn = CACN(in_channels, out_channels // 3, n_nodes=n_nodes, n_timesteps=n_timesteps)
        self.tacn = TACN(in_channels, out_channels // 3, n_nodes=n_nodes, dilations=dilations)
        self.gacn = GACN(in_channels, out_channels // 3, n_timesteps=n_timesteps)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        out = self.ln(x)  # -> [batch_size, in_channels, n_nodes, in_timesteps]
        # -> [batch_size, out_channels, n_nodes, in_timesteps]
        out = torch.cat([self.cacn(out),  self.tacn(out), self.gacn(out, adj)], dim=1)
        return torch.relu(out + self.res(x))  # -> [batch_size, out_channels, n_nodes, in_timesteps]


class TPC(nn.Module):
    def __init__(self, tgacns, in_timesteps, out_timesteps, n_nodes):
        super(TPC, self).__init__()
        self.tgacns = nn.ModuleList([
            TGACN(tgacn['in_channels'], tgacn['out_channels'],
                  n_nodes=n_nodes, n_timesteps=in_timesteps, dilations=tgacn['dilations'])
            for tgacn in tgacns
        ])
        self.ln = nn.LayerNorm([in_timesteps])
        self.fc = nn.Conv2d(in_channels=in_timesteps, out_channels=out_timesteps,
                            kernel_size=[1, tgacns[-1]['out_channels']])

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        for tgacn in self.tgacns:
            x = tgacn(x, adj)
        x = self.ln(x)  # -> [batch_size, out_channels, n_nodes, in_timesteps]
        x = self.fc(x.transpose(1, 3))  # -> [batch_size, out_timesteps, n_nodes, 1]
        return x[..., 0].transpose(1, 2)  # -> [batch_size, n_nodes, out_timesteps]


class TE(nn.Module):
    def __init__(self, n_components, n_nodes, n_timesteps):
        self.sizes = [n_components, n_nodes, n_timesteps]
        self.d_ebd = nn.Embedding(7, n_components * n_nodes * n_timesteps)
        self.h_ebd = nn.Embedding(24, n_components * n_nodes * n_timesteps)

    def forward(self, H: torch.Tensor, D: torch.Tensor):
        G = self.h_ebd(H) + self.d_ebd(D)  # [(batch_size * n_components * n_nodes * n_timesteps)]
        return G.view(len(G), *self.sizes)  # [batch_size, n_components, n_nodes, n_timesteps]


class MSGAT(nn.Module):
    def __init__(self, components, in_timesteps, out_timesteps, adj, te=True):
        super(MSGAT, self).__init__()
        if te:
            self.te = TE(len(components), len(adj), out_timesteps)
        else:
            self.W = nn.Parameter(torch.zeros(len(components), len(adj), out_timesteps))
        self.adj = nn.Parameter(adj, requires_grad=False)
        self.tpcs = nn.ModuleList([
            TPC(component, in_timesteps=in_timesteps, out_timesteps=out_timesteps, n_nodes=len(adj))
            for component in components
        ])

    def forward(self, X: torch.Tensor, H: torch.Tensor, D: torch.Tensor):
        if self.te:
            G = self.te(H, D)
            return sum((tpc(x, self.adj) * g for tpc, x, g in zip(self.tpcs, X.unbind(1), G.unbind(1))))
        else:
            return sum((tpc(x, self.adj) * w for tpc, x, w in zip(self.tpcs, X.unbind(1), self.W.unbind(0))))


def msgat(n_components, in_channels, in_timesteps, out_timesteps, adj, te=True):
    components = [[
        {
            'in_channels': in_channels,
            'out_channels': 72,
            'tcn_dilations': [1, 2]
        },
        {
            'in_channels': 72,
            'out_channels': 72,
            'tcn_dilations': [2, 4]
        },
    ]] * n_components
    net = MSGAT(components, in_timesteps=in_timesteps, out_timesteps=out_timesteps, adj=adj, te=te)
    return net
