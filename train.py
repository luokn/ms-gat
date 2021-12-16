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
from engine import Evaluator, Trainer
from models import msgat

models = {"ms-gat": msgat.msgat72, "ms-gat48": msgat.msgat48, "ms-gat72": msgat.msgat72, "ms-gat96": msgat.msgat96}


@click.command()
@click.argument("dataset", type=str)
@click.option("-c", "--ckpt", type=str, help="checkpoint file.")
@click.option("-o", "--out-dir", type=str, help="output directory.")
@click.option("-i", "--in-hours", type=str, help="input hours.", default="1,2,3,24,168")
@click.option("-j", "--num-workers", type=int, help="data loader workers.", default=0)
@click.option("-b", "--batch-size", type=int, help="batch size.", default=64)
@click.option("--model", type=str, help="model name.", default="ms-gat")
@click.option("--delta", type=float, help="huber loss delta.", default=50)
@click.option("--gpus", type=str, help="GPUs.", default="0")
@click.option("--min-epochs", type=int, help="min epochs.", default=10)
@click.option("--max-epochs", type=int, help="max epochs.", default=100)
@click.option("--out-timesteps", type=int, help="number of output timesteps.", default=12)
@click.option("--te/--no-te", help="with/without TE.", default=True)
def train(
    dataset,
    *,
    ckpt,
    in_hours,
    out_dir,
    num_workers,
    batch_size,
    model,
    delta,
    gpus,
    min_epochs,
    max_epochs,
    out_timesteps,
    te,
):
    in_hours, gpus = [int(i) for i in in_hours.split(",")], [int(i) for i in gpus.split(",")]
    with open("data.yaml", "r") as f:
        dataset = yaml.load(f, Loader=yaml.CLoader)[dataset]
    # load data.
    data_loaders = load_data(
        dataset["data-file"],
        batch_size,
        in_hours=in_hours,
        out_timesteps=out_timesteps,
        timesteps_per_hour=dataset["timesteps-per-hour"],
        num_workers=num_workers,
    )
    # create model.
    model = models[model](
        n_components=len(in_hours),
        in_channels=dataset["num-channels"],
        in_timesteps=dataset["timesteps-per-hour"],
        out_timesteps=out_timesteps,
        adjacency=load_adj(dataset["adj-file"], dataset["num-nodes"]),
        use_te=te,
    )
    # enable cuda.
    if len(gpus) > 1:
        model = DataParallel(model, device_ids=gpus)
    model = model.cuda(gpus[0])
    # train.
    trainer = Trainer(
        model,
        out_dir,
        max_epochs,
        min_epochs,
        patience=20,
        min_delta=1e-4,
        delta=delta,
        lr=1e-3,
        weight_decay=1e-4,
        step_size=30,
        gamma=0.1,
    )
    if ckpt:
        trainer.load(ckpt_file=ckpt)
    trainer.fit((data_loaders[0], data_loaders[1]), gpu=gpus[0])
    print("Training completed!")
    # evaluate.
    Evaluator(
        model,
        out_dir,
        trainer.out_dir / trainer.best["ckpt"],
        delta=delta,
    ).eval(data_loaders[-1], gpu=gpus[0])


if __name__ == "__main__":
    train()
