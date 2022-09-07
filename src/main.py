#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @File   : main.py
# @Data   : 2021/06/02
# @Author : Luo Kun
# @Contact: luokun485@gmail.com

import os

import click
from torch import cuda, nn

from data_loader import DataLoaderForMSGAT
from engine import Evaluator, Trainer
from models import msgat48, msgat72, msgat96

models = {"ms-gat": msgat72, "ms-gat48": msgat48, "ms-gat72": msgat72, "ms-gat96": msgat96}


def tolist(ctx, param, value):
    return [int(i) for i in value.split(",")]


@click.command()
@click.option("-d", "--data", type=str, help="Dataset name.", required=True)
@click.option("-c", "--ckpt", type=str, help="Checkpoint file.", default=None)
@click.option("-o", "--out-dir", type=str, help="Output directory.", default="checkpoints")
@click.option("-i", "--in-hours", type=str, callback=tolist, help="Input hours.", default="1,2,3,24,168")
@click.option("-b", "--batch-size", type=int, help="Batch size.", default=64)
@click.option("-w", "--num-workers", type=int, help="Number of 'DataLoader' workers.", default=0)
@click.option("--model", type=str, help="Model name.", default="ms-gat")
@click.option("--delta", type=float, help="Delta of 'HuberLoss'.", default=50)
@click.option("--gpu-ids", type=str, help="GPUs.", default="0")
@click.option("--out-timesteps", type=int, help="Length of output timesteps.", default=12)
@click.option("--no-te", type=bool, is_flag=True, help="Disable 'TE'.", default=False)
@click.option("--eval", type=bool, is_flag=True, help="Evaluate only.", default=False)
def main(data, ckpt, out_dir, in_hours, batch_size, num_workers, model, delta, gpu_ids, out_timesteps, no_te, eval):
    # load data.
    data = DataLoaderForMSGAT(data, in_hours, out_timesteps, batch_size, num_workers)

    # create model.
    model = models[model](
        n_components=len(in_hours),
        in_channels=data.num_channels,
        in_timesteps=data.timesteps_per_hour,
        out_timesteps=out_timesteps,
        use_te=not no_te,
        adj=data.adj,
    )

    # enable cuda.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    if cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.cuda()

    # train or eval.
    if not eval:
        # train.
        trainer = Trainer(model, delta, out_dir)
        if ckpt is not None:
            trainer.load(ckpt)
        trainer.fit((data.training, data.validation))
        click.echo("Training completed!")

    # evaluate.
    evaluator = Evaluator(model, delta, out_dir, ckpt if eval else trainer.best["ckpt"])
    evaluator.eval(data.evaluation)


if __name__ == "__main__":
    main()
