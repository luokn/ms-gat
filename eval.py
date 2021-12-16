#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Kun Luo
# @Email   : olooook@outlook.com
# @File    : main.py
# @Date    : 2021/06/02
# @Time    : 17:05:00


import click
import yaml
from torch.nn import DataParallel

from data import load_adj, load_data
from engine import Evaluator
from models import msgat

models = {"ms-gat": msgat.msgat72, "ms-gat48": msgat.msgat48, "ms-gat72": msgat.msgat72, "ms-gat96": msgat.msgat96}


@click.command(help="Evaluate.")
@click.argument("dataset", type=str)
@click.option("-c", "--ckpt", type=str, help="Checkpoint file.")
@click.option("-i", "--in-hours", type=str, help="Input hours.", default="1,2,3,24,168")
@click.option("-j", "--num-workers", type=int, help="Number of DataLoader workers.", default=0)
@click.option("-b", "--batch-size", type=int, help="Batch size.", default=64)
@click.option("--model", type=str, help="Model name.", default="ms-gat")
@click.option("--delta", type=float, help="HuberLoss delta.", default=50)
@click.option("--gpus", type=str, help="GPUs.", default="0")
@click.option("--out-timesteps", type=int, help="Number of output timesteps.", default=12)
@click.option("--te/--no-te", help="With/without TE.", default=True)
def eval(
    dataset,
    *,
    ckpt,
    in_hours,
    num_workers,
    batch_size,
    model,
    delta,
    gpus,
    out_timesteps,
    te,
):
    in_hours, gpus = [int(i) for i in in_hours.split(",")], [int(i) for i in gpus.split(",")]
    with open("data.yaml", "r") as f:
        dataset = yaml.load(f, Loader=yaml.CLoader)
    # data loaders
    data_loaders = load_data(
        data_file=dataset["data-file"],
        batch_size=batch_size,
        in_hours=in_hours,
        out_timesteps=out_timesteps,
        timesteps_per_hour=dataset["timesteps-per-hour"],
        num_workers=num_workers,
    )
    # model
    model = models[model](
        n_components=len(in_hours),
        in_channels=dataset["num_channels"],
        in_timesteps=dataset["timesteps-per-hour"],
        out_timesteps=out_timesteps,
        adjacency=load_adj(dataset["adj-file"], dataset["num-nodes"]),
        use_te=te,
    )
    # enable GPU.
    if len(gpus) > 1:
        model = DataParallel(model, device_ids=gpus)
    model = model.cuda(gpus[0])
    # eval
    Evaluator(
        model=model,
        ckpt_file=ckpt,
        delta=delta,
    ).eval(data_loaders[-1], gpu=gpus[0])


if __name__ == "__main__":
    eval()
