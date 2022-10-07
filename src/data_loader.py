#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @File   : data_loader.py
# @Data   : 2021/06/02
# @Author : Luo Kun
# @Contact: luokun485@gmail.com

from typing import List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset


class DataLoaderForMSGAT:
    """
    Adjacency matrix and Training, validation, evaluation data loader for MS-GAT.

    Args:
        name (str): Dataset name.
        in_hours (List[int]): Number of input hours.
        timesteps_per_hour (int): Timesteps per hour.
        out_timesteps (int): Number of output timesteps.
        batch_size (int): Batch size.
        num_workers (int): Number of workers. Defaults to 0.
    """

    def __init__(
        self,
        name: str,
        in_hours: List[int],
        out_timesteps: int,
        batch_size: int,
        num_workers: int,
    ):
        with open("data/meta.yaml", "r") as f:
            metadata = yaml.safe_load(f)[name]
            self.adj_file = metadata["adj-file"]
            self.data_file = metadata["data-file"]
            self.num_nodes = metadata["num-nodes"]
            self.num_channels = metadata["num-channels"]
            self.timesteps_per_hour = metadata["timesteps-per-hour"]
        self.in_hours, self.out_timesteps = in_hours, out_timesteps
        self.batch_size, self.num_workers = batch_size, num_workers
        self.adj = self.__load_adj()
        self.training, self.validation, self.evaluation = self.__load_data()

    def __load_adj(self) -> torch.Tensor:
        r"""
        Load adjacency matrix from file.

        .. math::
            \tilde A = \tilde{D}^{-1/2} (A + I_n) \tilde{D}^{-1/2}

        Returns:
            torch.Tensor: Adjacency matrix.
        """
        A = torch.eye(self.num_nodes)
        for line in open(self.adj_file, "r").readlines()[1:]:
            src, dst, _ = line.split(",")
            src, dst = int(src), int(dst)
            A[src, dst] = A[dst, src] = 1

        D_rsqrt = A.sum(dim=1).rsqrt().diag()
        return D_rsqrt @ A @ D_rsqrt

    def __load_data(self) -> List[DataLoader]:
        in_timesteps = self.timesteps_per_hour * max(self.in_hours)
        # -> [n_channels, n_nodes, n_timesteps]
        data = torch.from_numpy(np.load(self.data_file)["data"]).float().transpose(0, -1)
        length = data.size(-1) - in_timesteps - self.out_timesteps + 1
        split1, split2 = int(0.6 * length), int(0.8 * length)
        intervals = [
            [in_timesteps, in_timesteps + split1],  # training.
            [in_timesteps + split1, in_timesteps + split2],  # validation.
            [in_timesteps + split2, in_timesteps + length],  # evaluation.
        ]
        normalized_data = normalize(data, split=in_timesteps + split1)
        return [
            DataLoader(
                TimeSeriesSlice(normalized_data, data[0], interval, self.in_hours, self.out_timesteps,
                                self.timesteps_per_hour),
                self.batch_size,
                shuffle=i == 0,
                pin_memory=True,
                num_workers=self.num_workers,
            ) for i, interval in enumerate(intervals)
        ]


class TimeSeriesSlice(Dataset):

    def __init__(
        self,
        inputs: torch.Tensor,
        target: torch.Tensor,
        interval: Tuple[int, int],
        hours: List[int],
        out_timesteps: int,
        timesteps_per_hour: int,
    ):
        self.inputs, self.target = inputs, target
        self.interval, self.hours, self.q, self.tau = interval, hours, out_timesteps, timesteps_per_hour

    def __getitem__(self, i: int):
        t = torch.tensor(i + self.interval[0], dtype=torch.long)
        h = torch.div(t, self.tau, rounding_mode="trunc")
        d = torch.div(h, 24, rounding_mode="trunc")
        x = torch.stack([self.inputs[..., (t - h * self.tau):(t - h * self.tau + self.tau)] for h in self.hours])
        y = self.target[..., t:(t + self.q)]
        return x, h % 24, d % 7, y

    def __len__(self):
        return self.interval[1] - self.interval[0]


def normalize(tensor: torch.Tensor, split: int) -> torch.Tensor:
    std, mean = torch.std_mean(tensor[..., :split], dim=-1, keepdim=True)
    return (tensor - mean) / std
