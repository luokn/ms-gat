#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Kun Luo
# @Email   : olooook@outlook.com
# @File    : data.py
# @Date    : 2021/06/02
# @Time    : 17:07:20

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor, hours: list, out_timesteps: int, frequency: int, start: int, end: int):
        self.X, self.Y, self.hours, self.out_timesteps, self.frequency, self.start, self.end = X, Y, hours, out_timesteps, frequency, start, end

    def __getitem__(self, index: int):
        t = index + self.start
        h = t // self.frequency
        d = h // 24
        x = torch.stack([
            self.X[..., (t - hour * self.frequency):(t - hour * self.frequency + self.frequency)] for hour in self.hours
        ])
        y = self.Y[0, :, t:(t + self.out_timesteps)]
        return x, h % 24, d % 7, y

    def __len__(self):
        return self.end - self.start


def load_data(file, hours, out_timesteps, frequency, batch_size, num_workers=0, pin_memory=True):  # load data
    in_timesteps = frequency * max(hours)
    data = torch.from_numpy(np.load(file)['data']).float().transpose(0, -1)  # -> [n_channels, n_nodes, n_timesteps]
    length = data.shape[-1] - in_timesteps - out_timesteps + 1
    split1, split2 = int(.6 * length), int(.8 * length)
    ranges = [
        [in_timesteps, in_timesteps + split1],
        [in_timesteps + split1, in_timesteps + split2],
        [in_timesteps + split2, in_timesteps + length]
    ]
    normalized_data = normalize(data, split=in_timesteps + split1)
    return [
        DataLoader(MyDataset(X=normalized_data, Y=data,
                             hours=hours, out_timesteps=out_timesteps, frequency=frequency, start=start, end=end),
                   batch_size=batch_size, shuffle=i == 0, num_workers=num_workers, pin_memory=pin_memory)
        for i, (start, end) in enumerate(ranges)
    ]


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


def normalize(tensor: torch.Tensor, split: int):
    std, mean = torch.std_mean(tensor[..., :split], dim=-1, keepdim=True)
    return (tensor - mean) / std
