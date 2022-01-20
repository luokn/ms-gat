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


class TimeSeriesSlice(Dataset):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor, interval, hours, q, tau):
        self.X, self.Y = X, Y
        self.interval, self.hours, self.q, self.tau = interval, hours, q, tau

    def __getitem__(self, i: int):
        t = torch.tensor(i + self.interval[0], dtype=torch.long)
        h = torch.div(t, self.tau, rounding_mode="trunc")
        d = torch.div(h, 24, rounding_mode="trunc")
        x = torch.stack([self.X[..., (t - h * self.tau) : (t - h * self.tau + self.tau)] for h in self.hours])
        y = self.Y[..., t : (t + self.q)]
        return x, h % 24, d % 7, y

    def __len__(self):
        return self.interval[1] - self.interval[0]


# load data
def load_data(data_file, **kwargs):
    """
    Create data loaders for training, validation and evaluation.

    Args:
        file (str): Data file.
        batch_size (int): Batch size.
        in_hours (list): Number of input hours.
        timesteps_per_hour (int): Timesteps per hour.
        out_timesteps (int): Number of output timesteps.
        num_workers (int): Number of workers. Defaults to 0.
        pin_memory (bool): Pin memory. Defaults to True.

    Returns:
        List[DataLoader]: Training, validation and evaluation data loader.
    """
    in_timesteps = kwargs["timesteps_per_hour"] * max(kwargs["in_hours"])
    data = (
        torch.from_numpy(np.load(data_file)["data"]).float().transpose(0, -1)
    )  # -> [n_channels, n_nodes, n_timesteps]
    length = data.size(-1) - in_timesteps - kwargs["out_timesteps"] + 1
    split1, split2 = int(0.6 * length), int(0.8 * length)
    intervals = [
        [in_timesteps, in_timesteps + split1],  # training.
        [in_timesteps + split1, in_timesteps + split2],  # validation.
        [in_timesteps + split2, in_timesteps + length],  # evaluation.
    ]
    normalized_data = normalize(data, split=in_timesteps + split1)
    return [
        DataLoader(
            TimeSeriesSlice(
                normalized_data,
                data[0],
                interval,
                kwargs["in_hours"],
                kwargs["out_timesteps"],
                kwargs["timesteps_per_hour"],
            ),
            kwargs["batch_size"],
            shuffle=i == 0,
            pin_memory=True,
            num_workers=kwargs["num_workers"],
        )
        for i, interval in enumerate(intervals)
    ]


def load_adj(adj_file: str, num_nodes: int) -> torch.Tensor:
    r"""
    Load adjacency matrix from adjacency file.

    .. math::
        \tilde A = \tilde{D}^{-1/2} (A + I_n) \tilde{D}^{-1/2}

    Args:
        file (str): Adjacency file.
        n_nodes (int): Number of nodes.

    Returns:
        torch.Tensor: Adjacency matrix.
    """
    A = torch.eye(num_nodes)
    for line in open(adj_file, "r").readlines()[1:]:
        f, t, _ = line.split(",")
        f, t = int(f), int(t)
        A[f, t] = A[t, f] = 1

    D_rsqrt = A.sum(dim=1).rsqrt().diag()
    return D_rsqrt @ A @ D_rsqrt


def normalize(tensor: torch.Tensor, split: int) -> torch.Tensor:
    std, mean = torch.std_mean(tensor[..., :split], dim=-1, keepdim=True)
    return (tensor - mean) / std
