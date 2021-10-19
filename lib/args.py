#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Kun Luo
# @Email   : olooook@outlook.com
# @File    : args.py
# @Date    : 2021/06/08
# @Time    : 11:21:24

from argparse import ArgumentParser, Namespace


def parse_args() -> Namespace:
    """
    Parse command line arguments.
    """
    parser = ArgumentParser(description="Train MS-GAT")

    parser.add_argument('--data', type=str, help='Data file')
    parser.add_argument('--adj', type=str, help='Adjacency file')
    parser.add_argument('--nodes', type=int, help='Number of nodes')
    parser.add_argument('--channels', type=int, help='Number of channels')
    parser.add_argument('--checkpoints', type=str, help='Checkpoints path')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint name')

    parser.add_argument('--batch', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learn rate')
    parser.add_argument('--epochs', type=int, default=120, help='Number of epochs')
    parser.add_argument('--workers', type=int, default=0, help='Number of data loader workers')
    parser.add_argument('--gpu', type=int, default=None, help='GPU')
    parser.add_argument('--gpus', type=str, default=None, help='GPUs')

    parser.add_argument('--model', type=str, default='ms-gat', help='Model name')
    parser.add_argument('--use-te', type=bool, default=True, help='Use time embedding')
    parser.add_argument('--frequency', type=int, default=12, help='Time steps per hour')
    parser.add_argument('--in-hours', type=str, default='1,2,3,24,168', help='Hours of sampling')
    parser.add_argument('--out-timesteps', type=int, default=12, help='Number of output timesteps')
    parser.add_argument('--delta', type=float, default=60, help='Delta of huber loss')

    args = parser.parse_args()
    return args
