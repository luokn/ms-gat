#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Kun Luo
# @Email   : olooook@outlook.com
# @File    : main.py
# @Date    : 2021/06/02
# @Time    : 17:05:00


import click
from click.utils import open_file
import yaml
from torch.nn import DataParallel

from data import load_adj, load_data
from engine import Evaluator, Trainer
from models import msgat

models = {"ms-gat": msgat.msgat72, "ms-gat48": msgat.msgat48, "ms-gat72": msgat.msgat72, "ms-gat96": msgat.msgat96}

to_list = lambda ctx, param, value: [int(i) for i in value.split(",")]


@click.command()
@click.option("-d", "--data", type=str, help="dataset name.", required=True)
@click.option("-c", "--ckpt", type=str, help="checkpoint file.", default=None)
@click.option("-o", "--out-dir", type=str, help="output directory.", required=True)
@click.option("-i", "--in-hours", type=str, callback=to_list, help="input hours.", default="1,2,3,24,168")
@click.option("-j", "--num-workers", type=int, help="data loader workers.", default=0)
@click.option("-b", "--batch-size", type=int, help="batch size.", default=64)
@click.option("--model", type=str, help="model name.", default="ms-gat")
@click.option("--delta", type=float, help="huber loss delta.", default=50)
@click.option("--gpu-ids", type=str, callback=to_list, help="GPU device id.", default="0")
@click.option("--min-epochs", type=int, help="min epochs.", default=10)
@click.option("--max-epochs", type=int, help="max epochs.", default=100)
@click.option("--out-timesteps", type=int, help="number of output timesteps.", default=12)
@click.option("--te/--no-te", type=bool, help="with/without TE.", default=True)
@click.option("--eval", type=bool, is_flag=True, help="evaluation mode.", default=False)
def main(data, ckpt, out_dir, **kwargs):
    with open("data.yaml", "r") as f:
        data = yaml.load(f, Loader=yaml.CLoader)[data]
    # load data.
    data_loaders = load_data(
        data_file=data["data-file"],
        timesteps_per_hour=data["timesteps-per-hour"],
        batch_size=kwargs["batch_size"],
        in_hours=kwargs["in_hours"],
        out_timesteps=kwargs["out_timesteps"],
        num_workers=kwargs["num_workers"],
    )
    # create model.
    model = models[kwargs["model"]](
        n_components=len(kwargs["in_hours"]),
        in_channels=data["num-channels"],
        in_timesteps=data["timesteps-per-hour"],
        out_timesteps=kwargs["out_timesteps"],
        use_te=kwargs["te"],
        adj=load_adj(data["adj-file"], data["num-nodes"]),
    )
    # enable cuda.
    if len(kwargs["gpu_ids"]) > 1:
        model = DataParallel(model, device_ids=kwargs["gpu_ids"])
    model.cuda(kwargs["gpu_ids"][0])
    if kwargs["eval"]:  # evaluate.
        evaluator = Evaluator(model, out_dir, ckpt=ckpt, delta=kwargs["delta"])
        evaluator.eval(data_loaders[-1], gpu_id=kwargs["gpu_ids"][0])
    else:  # train.
        trainer = Trainer(
            model,
            out_dir,
            delta=kwargs["delta"],
            max_epochs=kwargs["max_epochs"],
            min_epochs=kwargs["min_epochs"],
            lr=1e-3,
            weight_decay=1e-4,
            patience=20,
            min_delta=1e-4,
            gamma=0.1,
            step_size=30,
        )
        if ckpt:
            trainer.load(ckpt)
        trainer.fit((data_loaders[0], data_loaders[1]), gpu_id=kwargs["gpu_ids"][0])
        print("Training completed!")
        trainer.load(trainer.best["ckpt"])
        trainer.eval(data_loaders[-1], gpu_id=kwargs["gpu_ids"][0])


if __name__ == "__main__":
    main()
