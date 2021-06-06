#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Kun Luo
# @Email   : olooook@outlook.com
# @File    : data.py
# @Date    : 2021/06/02
# @Time    : 17:07:20

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_adj(file, n_nodes):  # load adjacency matrix
    r"""
    .. math::
        \tilde A = \tilde{D}^{-1/2} (A + I_n) \tilde{D}^{-1/2}
    """
    A = torch.eye(n_nodes)
    for line in open(file, 'r').readlines()[1:]:
        f, t, _ = line.split(',')
        f, t = int(f), int(t)
        A[f, t] = A[t, f] = 1

    D_rsqrt = A.sum(dim=1).rsqrt().diag()
    return D_rsqrt @ A @ D_rsqrt


def load_data(file, frequency, hours, out_timesteps, batch_size, num_workers=0, pin_memory=True):  # make data loaders
    timeseries = torch.from_numpy(np.load(file)['data'].astype(np.float32)).transpose(1, 2)
    X, H, D, Y = generate(timeseries, frequency, hours, out_timesteps)
    sizes = [int(.6 * len(X)), int(.8 * len(X))]
    normalize(X, dim=0, split=sizes[0])
    return [
        DataLoader(TensorDataset(*tensors), batch_size, shuffle=i == 0, num_workers=num_workers, pin_memory=pin_memory)
        for i, tensors in enumerate(zip(*[tensor.split(sizes) for tensor in [X, H, D, Y]]))
    ]


def generate(series: torch.Tensor, frequency, hours, out_timesteps):  # generate sliced datasets from sequence
    timesteps = [hour * frequency for hour in hours]
    max_timestep = max(timesteps)
    X = torch.stack([
        series[max_timestep - step:-step].unfold(0, size=frequency, step=1) for step in timesteps
    ], dim=0).transpose(0, 1)
    Y = series[max_timestep:, 0].unfold(0, size=frequency, step=1)
    T = torch.arange(len(Y)) // frequency
    H, D = T % 24, (T // 24) % 7
    return X, H, D, Y[..., :out_timesteps]


def normalize(x: torch.Tensor, dim: int, split: int):
    std, mean = torch.std_mean(x[:split], dim=dim, keepdim=True)
    x -= mean
    x /= std
    return dict(std=std, mean=mean)
