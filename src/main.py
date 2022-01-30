#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Kun Luo
# @Email   : olooook@outlook.com
# @File    : main.py
# @Date    : 2021/06/02
# @Time    : 17:05:00


import os

from click import command, echo, option
from torch import cuda, nn

from core import Evaluator, Trainer
from data import DataForMSGAT
from models import msgat48, msgat72, msgat96

models = {"ms-gat": msgat72, "ms-gat48": msgat48, "ms-gat72": msgat72, "ms-gat96": msgat96}


def to_list(ctx, param, value):
    return [int(i) for i in value.split(",")]


@command()
@option("-d", "--data", type=str, help="Dataset name.", required=True)
@option("-c", "--ckpt", type=str, help="Checkpoint file.", default=None)
@option("-o", "--out-dir", type=str, help="Output directory.", default="checkpoints")
@option("-i", "--in-hours", type=str, callback=to_list, help="Input hours.", default="1,2,3,24,168")
@option("-b", "--batch-size", type=int, help="Batch size.", default=64)
@option("-w", "--num-workers", type=int, help="Number of 'DataLoader' workers.", default=0)
@option("--model", type=str, help="Model name.", default="ms-gat")
@option("--delta", type=float, help="Delta of 'HuberLoss'.", default=50)
@option("--gpu-ids", type=str, help="GPUs.", default="0")
@option("--min-epochs", type=int, help="Min epochs.", default=10)
@option("--max-epochs", type=int, help="Max epochs.", default=100)
@option("--out-timesteps", type=int, help="Length of output timesteps.", default=12)
@option("--no-te", type=bool, is_flag=True, help="Disable 'TE'.", default=False)
@option("--eval", type=bool, is_flag=True, help="Evaluate only.", default=False)
def main(**kwargs):
    # load data.
    data = DataForMSGAT(
        name=kwargs["data"],
        in_hours=kwargs["in_hours"],
        out_timesteps=kwargs["out_timesteps"],
        batch_size=kwargs["batch_size"],
        num_workers=kwargs["num_workers"],
    )
    # create model.
    model = models[kwargs["model"]](
        n_components=len(kwargs["in_hours"]),
        in_channels=data.num_channels,
        in_timesteps=data.timesteps_per_hour,
        out_timesteps=kwargs["out_timesteps"],
        use_te=not kwargs["no_te"],
        adj=data.adj,
    )
    # enable cuda.
    os.environ["CUDA_VISIBLE_DEVICES"] = kwargs["gpu_ids"]
    if cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.cuda()
    # train.
    if not kwargs["eval"]:
        trainer = Trainer(
            model,
            out_dir=kwargs["out_dir"],
            max_epochs=kwargs["max_epochs"],
            min_epochs=kwargs["min_epochs"],
            delta=kwargs["delta"],
            lr=1e-3,
            weight_decay=1e-4,
            patience=20,
            min_delta=1e-4,
            gamma=0.1,
            step_size=30,
        )
        if kwargs["ckpt"] is not None:
            trainer.load(kwargs["ckpt"])
        trainer.fit((data.training, data.validation))
        echo("Training completed!")
    # evaluate.
    evaluator = Evaluator(
        model,
        out_dir=kwargs["out_dir"],
        ckpt=kwargs["ckpt"] if kwargs["eval"] else trainer.best["ckpt"],
        delta=kwargs["delta"],
    )
    evaluator.eval(data.evaluation)


if __name__ == "__main__":
    main()
