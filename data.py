#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Kun Luo
# @Email   : olooook@outlook.com
# @File    : data.py
# @Date    : 2021/06/02
# @Time    : 17:07:20

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TimeSeriesSliceDataset(Dataset):
    def __init__(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        in_hours: List[int],
        out_timesteps: int,
        timesteps_per_hour: int,
        start: int,
        end: int,
    ):
        self.X, self.Y = X, Y
        self.in_hours = in_hours
        self.out_timesteps = out_timesteps
        self.timesteps_per_hour = timesteps_per_hour
        self.start, self.end = start, end

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t = torch.tensor(index + self.start).long()
        h = torch.floor(t / self.timesteps_per_hour).long()
        d = torch.floor(h / 24).long()
        x = torch.stack(
            [
                self.X[
                    ...,
                    (t - hour * self.timesteps_per_hour) : (
                        t - hour * self.timesteps_per_hour + self.timesteps_per_hour
                    ),
                ]
                for hour in self.in_hours
            ]
        )
        y = self.Y[..., t : (t + self.out_timesteps)]
        return x, h % 24, d % 7, y

    def __len__(self) -> int:
        return self.end - self.start


# load data
def load_data(
    data_file: str,
    timesteps_per_hour: int = 12,
    batch_size: int = 64,
    in_hours: List[int] = [1, 2, 3, 24, 7 * 24],
    out_timesteps: int = 12,
    num_workers=0,
) -> Tuple[DataLoader]:
    """
    Create data loaders for training, validation and evaluation.

    Args:
        file (str): Data file.
        batch_size (int): Batch size.
        in_hours (list): Number of input hours.
        out_timesteps (int): Number of output timesteps.
        timesteps_per_hour (int): Timesteps per hour.
        num_workers (int, optional): Number of workers. Defaults to 0.
        pin_memory (bool, optional): Pin memory. Defaults to True.

    Returns:
        List[DataLoader]: Training, validation and evaluation data loader.
    """
    in_timesteps = timesteps_per_hour * max(in_hours)
    data = (
        torch.from_numpy(np.load(data_file)["data"]).float().transpose(0, -1)
    )  # -> [n_channels, n_nodes, n_timesteps]
    length = data.shape[-1] - in_timesteps - out_timesteps + 1
    split1, split2 = int(0.6 * length), int(0.8 * length)
    ranges = [
        [in_timesteps, in_timesteps + split1],
        [in_timesteps + split1, in_timesteps + split2],
        [in_timesteps + split2, in_timesteps + length],
    ]
    normalized_data = normalize(data, split=in_timesteps + split1)
    return [
        DataLoader(
            TimeSeriesSliceDataset(
                X=normalized_data,
                Y=data[0],
                in_hours=in_hours,
                out_timesteps=out_timesteps,
                timesteps_per_hour=timesteps_per_hour,
                start=start,
                end=end,
            ),
            batch_size=batch_size,
            shuffle=i == 0,
            num_workers=num_workers,
            pin_memory=True,
        )
        for i, (start, end) in enumerate(ranges)
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
