#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Kun Luo
# @Email   : olooook@outlook.com
# @File    : trainer.py
# @Date    : 2021/06/02
# @Time    : 17:41:37


from pathlib import Path
from time import localtime, strftime
from typing import Tuple

import click
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import Module
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from models.loss import HuberLoss


class Engine:
    def __init__(self, model: Module, out_dir: str, **kwargs: dict) -> None:
        self.model = model
        self.criterion = HuberLoss(kwargs["delta"])
        self.out_dir = Path(out_dir)
        if not self.out_dir.exists():
            self.out_dir.mkdir(parents=True)
        self.log_file = self.out_dir / "run.log"

    def _run_epoch(self, data_loader: DataLoader, mode: str, gpu: int, epoch=None) -> dict:
        labels = {"train": "[Train   ]", "validate": "[Validate]", "evaluate": ["Evaluate"]}
        L_acc, L_ave, metrics = 0.0, 0.0, Metrics()
        self.model.train(mode == "train")
        with click.progressbar(
            length=len(data_loader), label=labels[mode], item_show_func=self.__show_item, width=25
        ) as bar:
            for batch_idx, batch in enumerate(data_loader):
                batch = [t.cuda(gpu) for t in batch]
                inputs, target = batch[:-1], batch[-1]
                if mode == "train":
                    with autocast():
                        output = self.model(*inputs)
                        loss = self.criterion(output, target)
                    self.optimizer.zero_grad()
                    self.grad_scaler.scale(loss).backward()
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    output = self.model(*inputs)
                    loss = self.criterion(output, target)
                # update loss.
                L_acc += loss.item()
                L_ave = L_acc / (batch_idx + 1)
                # update metrics.
                metrics.update(output, target)
                # statistics.
                stats = dict(loss=L_ave, **metrics.stats())
                # update progress bar.
                bar.update(n_steps=1, current_item=stats)
                # log to file.
        if mode == "train" or mode == "validate":
            self._log(labels[mode], epoch=epoch, **stats)
        else:
            self._log(labels[mode], **stats)
        return stats

    def _log(self, *args: list, **kwargs: dict):
        with open(self.log_file, "a") as f:
            f.write(strftime("%Y/%m/%d %H:%M:%S", localtime()))
            f.write(" - ")
            f.write(" - ".join([f"{arg}" for arg in args]))
            f.write(" - ")
            f.write(",".join([f"{k}={v}" for k, v in kwargs.items()]))
            f.write("\n")

    @staticmethod
    def __show_item(stats: dict):
        if stats is None:
            return ""
        return f"loss={stats['loss']:.2f} MAE={stats['MAE']:.2f} MAPE={stats['MAPE']:.2f}% RMSE={stats['RMSE']:.2f}"


class Trainer(Engine):
    def __init__(
        self, model: Module, out_dir: str, max_epochs: int, min_epochs: int, patience: int, min_delta: int, **kwargs
    ):
        super(Trainer, self).__init__(model, out_dir, delta=kwargs["delta"])
        self.optimizer = Adam(model.parameters(), lr=kwargs["lr"], weight_decay=kwargs["weight_decay"])
        self.scheduler = StepLR(self.optimizer, step_size=kwargs["step_size"], gamma=kwargs["gamma"])
        self.grad_scaler = GradScaler()
        #
        self.patience, self.min_delta = patience, min_delta
        self.max_epochs, self.min_epochs = max_epochs, min_epochs
        self.epoch = 1
        self.best = {"epoch": 0, "loss": float("inf"), "ckpt": ""}

    def fit(self, data_loaders: Tuple[DataLoader, DataLoader], gpu: int):
        while self.epoch <= self.max_epochs:
            click.echo(f"Epoch {self.epoch}")
            with torch.enable_grad():
                self._run_epoch(data_loaders[0], mode="train", gpu=gpu, epoch=self.epoch)
            with torch.no_grad():
                stats = self._run_epoch(data_loaders[1], mode="validate", gpu=gpu, epoch=self.epoch)
            self.scheduler.step()
            if self.epoch > self.min_epochs:
                if stats["loss"] < (1 - self.min_delta) * self.best["loss"]:
                    self.best = dict(
                        epoch=self.epoch,
                        loss=stats["loss"],
                        ckpt=f"{self.epoch}_{stats['loss']:.2f}.pkl",
                    )
                    self.save(ckpt_file=self.out_dir / self.best["ckpt"])
                elif self.epoch > self.best["epoch"] + self.patience:
                    break  # early stop.
            self.epoch += 1

    def save(self, ckpt_file):
        click.echo(f"• Save checkpoint {ckpt_file}")
        states = dict(
            best=self.best,
            epoch=self.epoch,
            model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(),
            grad_scaler=self.grad_scaler.state_dict(),
        )
        torch.save(states, ckpt_file)

    def load(self, ckpt_file):
        click.echo(f"• Load checkpoint {ckpt_file}")
        states = torch.load(ckpt_file)
        self.best = states["best"]
        self.epoch = states["epoch"] + 1
        self.model.load_state_dict(states["model"])
        self.optimizer.load_state_dict(states["optimizer"])
        self.scheduler.load_state_dict(states["scheduler"])
        self.grad_scaler.load_state_dict(states["grad_scaler"])


class Evaluator(Engine):
    def __init__(self, model: Module, out_dir: str, ckpt_file: str, **kwargs):
        super(Evaluator, self).__init__(model, out_dir, delta=kwargs["delta"])
        states = torch.load(ckpt_file)
        model.load_state_dict(states["model"])

    def eval(self, data_loader: DataLoader, gpu: int):
        with torch.no_grad():
            self._run_epoch(data_loader, mode="evaluate", gpu=gpu)


class Metrics:
    """
    Calculate ``MAE/MAPE/RMSE``.
    """

    def __init__(self, mask_value: float = 0.0):
        self.n, self.mask_value = 0, mask_value
        self.AE, self.APE, self.SE, self.MAE, self.MAPE, self.RMSE = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    def stats(self) -> dict:
        return {"MAE": self.MAE, "MAPE": self.MAPE, "RMSE": self.RMSE}

    def update(self, output: torch.Tensor, target: torch.Tensor):
        self.n += target.nelement()
        # MAE
        self.AE += torch.abs(output - target).sum().item()
        self.MAE = self.AE / self.n
        # MAPE
        mask = target > self.mask_value
        masked_output, masked_target = output[mask], target[mask]
        self.APE += 100 * torch.abs((masked_output - masked_target) / masked_target).sum().item()
        self.MAPE = self.APE / self.n
        # RMSE
        self.SE += torch.square(output - target).sum().item()
        self.RMSE = (self.SE / self.n) ** 0.5
