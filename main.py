#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Kun Luo
# @Email   : olooook@outlook.com
# @File    : main.py
# @Date    : 2021/06/02
# @Time    : 17:05:00

import os

import click
from torch.functional import Tensor
from torch.nn import DataParallel

from data import load_adj, load_data
from engine import Evaluator, Trainer
from models import msgat

models = {"ms-gat": msgat.msgat72, "ms-gat48": msgat.msgat48, "ms-gat72": msgat.msgat72, "ms-gat96": msgat.msgat96}


@click.group()
def cli():
    ...


@cli.command(help="Train.")
@click.option("-a", "--adj-file", type=str, help="Graph adjacency file.")
@click.option("-d", "--data-file", type=str, help="Time series data file.")
@click.option("-f", "--ckpt-file", type=str, help="Pre-saved checkpoint file.")
@click.option("-o", "--out-dir", type=str, help="Output directory for logs and records.")
@click.option("-n", "--num-nodes", type=int, help="The number of nodes in the graph.")
@click.option("-c", "--num-channels", type=int, help="Number of time series data channels.")
@click.option("-j", "--num-workers", type=int, help="Number of data loader workers.", default=0)
@click.option("-i", "--in-hours", type=str, help="Sampling hours.", default="1,2,3,24,168")
@click.option("-b", "--batch-size", type=int, help="Batch size.", default=64)
@click.option("-e", "--epochs", type=int, help="Number of epochs.", default=100)
@click.option("--lr", type=float, help="Learn rate.", default=1e-3)
@click.option("--model", type=click.Choice(models.keys()), help="Model.", default="ms-gat")
@click.option("--gpus", type=str, help="GPUs.", default="0")
@click.option("--delta", type=float, help="Huber loss delta.", default=60)
@click.option("--weight-decay", type=float, help="Adam weight decay.", default=0)
@click.option("--te/--no-te", help="With/without time embedding.", default=True)
@click.option("--out-timesteps", type=int, help="Number of output timesteps.", default=12)
@click.option("--timesteps-per-hour", type=int, help="Timesteps per hour.", default=12)
def train(**args):
    # data loaders
    data_loaders = load_data(
        data_file=args["data_file"],
        batch_size=args["batch_size"],
        in_hours=[int(h) for h in args["in_hours"].split(",")],
        out_timesteps=args["out_timesteps"],
        timesteps_per_hour=args["timesteps_per_hour"],
        num_workers=args["num_workers"],
    )
    # model
    net = models[args["model"]](
        n_components=len(args["in_hours"].split(",")),
        in_channels=args["num_channels"],
        in_timesteps=args["timesteps_per_hour"],
        out_timesteps=args["out_timesteps"],
        adjacency=load_adj(args["adj_file"], args["num_nodes"]),
        use_te=args["te"],
        init_params=args["ckpt_file"] is None,
    )
    # enable GPU.
    if len(args["gpus"].split(",")) > 1:
        gpu_id = None
        net = DataParallel(net, device_ids=[int(i) for i in args["gpus"].split(",")]).cuda()
    else:
        gpu_id = int(args["gpus"])
        net = net.cuda(gpu_id)
    # train
    trainer = Trainer(
        model=net,
        out_dir=args["out_dir"],
        delta=args["delta"],
        lr=args["lr"],
        weight_decay=args["weight_decay"],
    )
    if args["ckpt_file"] is not None:
        trainer.load(ckpt_file=args["ckpt_file"])
    trainer.fit((data_loaders[0], data_loaders[1]), epochs=args["epochs"], gpu=gpu_id)
    print("Training completed!")
    # eval
    evaluator = Evaluator(
        model=net, ckpt_file=trainer.out_dir / trainer.best["ckpt"], out_dir=args["out_dir"], delta=args["delta"]
    )
    evaluator.eval(data_loaders[-1], gpu=gpu_id)


@cli.command(help="Evaluate.")
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
def eval(**args):
    # data loaders
    data_loaders = load_data(
        data_file=args["data_file"],
        batch_size=args["batch_size"],
        in_hours=[int(h) for h in args["in_hours"].split(",")],
        out_timesteps=args["out_timesteps"],
        timesteps_per_hour=args["timesteps_per_hour"],
        num_workers=args["num_workers"],
    )
    # network
    net = models[args["model"]](
        n_components=len(args["in_hours"].split(",")),
        in_channels=args["num_channels"],
        in_timesteps=args["timesteps_per_hour"],
        out_timesteps=args["out_timesteps"],
        adjacency=load_adj(args["adj_file"], args["num_nodes"]),
        use_te=args["te"],
        init_params=False,
    )
    # enable GPU.
    if len(args["gpus"].split(",")) > 1:
        gpu_id = None
        net = DataParallel(net, device_ids=[int(i) for i in args["gpus"].split(",")]).cuda()
    else:
        gpu_id = int(args["gpus"])
        net = net.cuda(gpu_id)
    # eval
    evaluator = Evaluator(model=net, ckpt_file=args["ckpt_file"], out_dir=args["out_dir"], delta=args["delta"])
    evaluator.eval(data_loaders[-1], gpu=gpu_id)


if __name__ == "__main__":
    cli()
