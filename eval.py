#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Kun Luo
# @Email   : olooook@outlook.com
# @File    : main.py
# @Date    : 2021/06/02
# @Time    : 17:05:00


import click
from torch.nn import DataParallel

from data import load_adj, load_data
from engine import Evaluator
from models import msgat

models = {"ms-gat": msgat.msgat72, "ms-gat48": msgat.msgat48, "ms-gat72": msgat.msgat72, "ms-gat96": msgat.msgat96}


@click.command(help="Evaluate.")
@click.option("-a", "--adj-file", type=str, help="Graph adjacency file.")
@click.option("-d", "--data-file", type=str, help="Time series data file.")
@click.option("-f", "--ckpt-file", type=str, help="Pre-saved checkpoint file.")
@click.option("-o", "--out-dir", type=str, help="Output directory for logs and records.")
@click.option("-n", "--num-nodes", type=int, help="The number of nodes in the graph.")
@click.option("-c", "--num-channels", type=int, help="Number of time series data channels.")
@click.option("-j", "--num-workers", type=int, help="Number of data loader workers.", default=0)
@click.option("-i", "--in-hours", type=str, help="Sampling hours.", default="1,2,3,24,168")
@click.option("-b", "--batch-size", type=int, help="Batch size.", default=64)
@click.option("--model", type=click.Choice(models.keys()), help="Model.", default="ms-gat")
@click.option("--gpus", type=str, help="GPUs.", default="0")
@click.option("--delta", type=float, help="Huber loss delta.", default=60)
@click.option("--te/--no-te", help="With/without time embedding.", default=True)
@click.option("--out-timesteps", type=int, help="Number of output timesteps.", default=12)
@click.option("--timesteps-per-hour", type=int, help="Timesteps per hour.", default=12)
def eval(
    adj_file,
    data_file,
    ckpt_file,
    out_dir,
    num_nodes,
    num_channels,
    num_workers,
    in_hours,
    batch_size,
    model,
    gpus,
    delta,
    te,
    out_timesteps,
    timesteps_per_hour,
):
    in_hours, gpus = [int(i) for i in in_hours.split(",")], [int(i) for i in gpus.split(",")]
    # data loaders
    data_loaders = load_data(
        data_file=data_file,
        batch_size=batch_size,
        in_hours=in_hours,
        out_timesteps=out_timesteps,
        timesteps_per_hour=timesteps_per_hour,
        num_workers=num_workers,
    )
    # network
    net = models[model](
        n_components=len(in_hours),
        in_channels=num_channels,
        in_timesteps=timesteps_per_hour,
        out_timesteps=out_timesteps,
        adjacency=load_adj(adj_file, num_nodes),
        use_te=te,
        init_params=False,
    )
    # enable GPU.
    if len(gpus) > 1:
        net = DataParallel(net, device_ids=gpus)
    net = net.cuda(gpus[0])
    # eval
    evaluator = Evaluator(
        model=net,
        out_dir=out_dir,
        ckpt_file=ckpt_file,
        delta=delta,
    )
    evaluator.eval(data_loaders[-1], gpu=gpus[0])


if __name__ == "__main__":
    eval()
