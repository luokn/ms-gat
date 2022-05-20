#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Kun Luo
# @Email   : olooook@outlook.com
# @File    : trainer.py
# @Date    : 2021/06/02
# @Time    : 17:41:37


from pathlib import Path
from time import localtime, strftime
from typing import Optional, Tuple

import click
import torch
from torch import nn, optim
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from loss import HuberLoss
from metrics import Metrics


class Engine:

    __labels__ = {
        "train": "[Train   ]",
        "validate": "[Validate]",
        "evaluate": "[Evaluate]",
    }

    def __init__(self, model: nn.Module, **kwargs):
        self.model = model
        self.loss_fn = HuberLoss(kwargs["delta"])
        self.out_dir = Path(kwargs["out_dir"])

        if not self.out_dir.exists():
            self.out_dir.mkdir(parents=True)
        self.log_file = self.out_dir / "run.log"

    def run_epoch(self, data: DataLoader, epoch: int, gpu_id: Optional[int], mode: str = "train"):
        self.model.train(mode == "train")
        with torch.set_grad_enabled(mode == "train"):
            loss_acc, loss_ave, metrics = 0.0, 0.0, Metrics()

            with click.progressbar(
                length=len(data), label=self.__labels__[mode], item_show_func=self.__show_item, width=25
            ) as pbar:
                for batch_idx, batch_data in enumerate(data):
                    batch_data = [tensor.cuda(gpu_id) for tensor in batch_data]
                    inputs, truth = batch_data[:-1], batch_data[-1]

                    with amp.autocast():
                        pred = self.model(*inputs)
                        loss = self.loss_fn(pred, truth)

                    if mode == "train":
                        self.optimizer.zero_grad()
                        self.grad_scaler.scale(loss).backward()
                        self.grad_scaler.step(self.optimizer)
                        self.grad_scaler.update()

                    # loss.
                    loss_acc += loss.item()
                    loss_ave = loss_acc / (batch_idx + 1)

                    # metrics.
                    metrics.update(pred, truth)
                    pbar.update(n_steps=1, current_item=(loss_ave, metrics))

            stats = {"loss": loss_ave, "MAE": metrics.MAE, "MAPE": metrics.MAPE, "RMSE": metrics.RMSE}

            # log to file.
            if mode == "evaluate":
                self.log_to_file(self.__labels__[mode], **stats)
            else:
                self.log_to_file(self.__labels__[mode], epoch=epoch, **stats)

            return loss_ave

    def log_to_file(self, *args, **kwargs):
        with open(self.log_file, "a") as f:
            f.write(strftime("%Y/%m/%d %H:%M:%S", localtime()))
            f.write(" - ")
            f.write(" - ".join([f"{i}" for i in args]))
            f.write(" - ")
            f.write(",".join([f"{k}={v}" for k, v in kwargs.items()]))
            f.write("\n")

    @staticmethod
    def __show_item(current_item):
        if current_item is None:
            return ""
        loss, metrics = current_item
        return f"loss={loss:.2f} MAE={metrics.MAE:.2f} MAPE={metrics.MAPE:.2f}% RMSE={metrics.RMSE:.2f}"


class Trainer(Engine):
    def __init__(self, model: nn.Module, **kwargs):
        super(Trainer, self).__init__(model, **kwargs)
        self.optimizer = optim.Adam(model.parameters(), lr=kwargs["lr"], weight_decay=kwargs["weight_decay"])
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=kwargs["step_size"], gamma=kwargs["gamma"])

        self.grad_scaler = amp.GradScaler()

        self.patience, self.min_delta = kwargs["patience"], kwargs["min_delta"]
        self.max_epochs, self.min_epochs = kwargs["max_epochs"], kwargs["min_epochs"]

        self.epoch = 1
        self.best = {"epoch": 0, "loss": float("inf"), "ckpt": ""}

    def fit(self, data_loaders: Tuple[DataLoader, DataLoader], gpu_id=None):
        while self.epoch <= self.max_epochs:
            click.echo(f"Epoch {self.epoch}")

            self.run_epoch(data_loaders[0], mode="train", epoch=self.epoch, gpu_id=gpu_id)
            loss = self.run_epoch(data_loaders[1], mode="validate", epoch=self.epoch, gpu_id=gpu_id)

            self.scheduler.step()
            if self.epoch > self.min_epochs:
                if loss < (1 - self.min_delta) * self.best["loss"]:
                    self.best = dict(epoch=self.epoch, loss=loss, ckpt=self.out_dir / f"{self.epoch}_{loss:.2f}.pkl")
                    self.save(ckpt=self.best["ckpt"])
                elif self.epoch > self.best["epoch"] + self.patience:
                    break  # early stop.
            self.epoch += 1

    def save(self, ckpt):
        states = dict(
            best=self.best,
            epoch=self.epoch,
            model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(),
            grad_scaler=self.grad_scaler.state_dict(),
        )
        torch.save(states, ckpt)

        click.echo(f"• Save checkpoint {ckpt}")

    def load(self, ckpt):
        states = torch.load(ckpt)
        self.best = states["best"]
        self.epoch = states["epoch"] + 1
        self.model.load_state_dict(states["model"])
        self.optimizer.load_state_dict(states["optimizer"])
        self.scheduler.load_state_dict(states["scheduler"])
        self.grad_scaler.load_state_dict(states["grad_scaler"])

        click.echo(f"• Load checkpoint {ckpt}")


class Evaluator(Engine):
    def __init__(self, model: nn.Module, **kwargs):
        super(Evaluator, self).__init__(model, **kwargs)
        states = torch.load(kwargs["ckpt"])
        model.load_state_dict(states["model"])

    def eval(self, data_loader: DataLoader, gpu_id=None):
        self.run_epoch(data_loader, mode="evaluate", epoch=None, gpu_id=gpu_id)
