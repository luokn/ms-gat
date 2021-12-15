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
from torch.cuda.amp import GradScaler, autocast
from torch.nn import Module
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from models.loss import HuberLoss


class Trainer:
    def __init__(
        self,
        model: Module,
        out_dir: str,
        *,
        delta: float,
        lr: float,
        weight_decay: float,
        patience: int = 10,
        min_epochs: int = 10,
        min_delta: float = 5e-4,
    ):
        self.model = model
        self.criterion = HuberLoss(delta)
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=20, gamma=0.5)
        self.out_dir = Path(out_dir)
        if not self.out_dir.exists():
            self.out_dir.mkdir(parents=True)
        self.patience, self.min_epochs, self.min_delta = patience, min_epochs, min_delta
        self.best = {"epoch": 0, "loss": float("inf"), "ckpt": ""}
        self.epoch = 1
        self.logger = Logger(self.out_dir / ".log")
        self.grad_scaler = GradScaler()

    def fit(self, data_loaders: Tuple[DataLoader, DataLoader], epochs: int = 100, gpu: Optional[int] = None):
        while self.epoch <= epochs:
            click.echo(f"Epoch {self.epoch}/{epochs}")
            # train and validate.
            self.train_epoch(data_loaders[0], gpu)
            stats = self.validate_epoch(data_loaders[1], gpu)
            self.scheduler.step()
            if self.epoch > self.min_epochs:
                # optimality check
                if stats["loss"] < (1 - self.min_delta) * self.best["loss"]:
                    self.best = dict(
                        epoch=self.epoch,
                        loss=stats["loss"],
                        ckpt=f"{self.epoch}_loss={stats['loss']:.2f}.pkl",
                    )
                    # save the best checkpoint.
                    self.save(ckpt_file=self.out_dir / self.best["ckpt"])
                # early stop.
                elif self.epoch > self.best["epoch"] + self.patience:
                    break
            self.epoch += 1

    def save(self, ckpt_file: str):
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

    def load(self, ckpt_file: str):
        click.echo(f"• Load checkpoint {ckpt_file}")
        states = torch.load(ckpt_file)
        self.best = states["best"]
        self.epoch = states["epoch"] + 1
        self.model.load_state_dict(states["model"])
        self.optimizer.load_state_dict(states["optimizer"])
        self.scheduler.load_state_dict(states["scheduler"])
        self.grad_scaler.load_state_dict(states["grad_scaler"])

    @torch.enable_grad()
    def train_epoch(self, data_loader: DataLoader, gpu: Optional[int]) -> dict:
        self.model.train()
        return self.__run_epoch(data_loader, gpu, label="[Train   ]", train=True)

    @torch.no_grad()
    def validate_epoch(self, data_loader: DataLoader, gpu: Optional[int]) -> dict:
        self.model.eval()
        return self.__run_epoch(data_loader, gpu, label="[Validate]", train=False)

    def __run_epoch(self, data_loader: DataLoader, gpu: Optional[int], label: str, train: bool) -> dict:
        L_acc, L_ave, metrics = 0.0, 0.0, Metrics()
        with click.progressbar(length=len(data_loader), label=label, item_show_func=show_item, width=25) as bar:
            for batch_idx, batch in enumerate(data_loader):
                batch = [t.cuda(gpu) for t in batch]
                inputs, target = batch[:-1], batch[-1]
                if train:
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
                bar.update(1, stats)
                # log to file.
                self.logger.log(label, epoch=self.epoch, batch_idx=batch_idx, batch_size=len(target), **stats)
        return stats


class Evaluator:
    def __init__(self, model: Module, ckpt_file: str, out_dir: str, delta: float):
        states = torch.load(ckpt_file)
        model.load_state_dict(states["model"])
        self.model = model
        self.criterion = HuberLoss(delta)
        self.out_dir = Path(out_dir)
        if not self.out_dir.exists():
            self.out_dir.mkdir(parents=True)

    @torch.no_grad()
    def eval(self, data_loader: DataLoader, gpu: Optional[int] = None):
        self.model.eval()
        L_acc, L_ave, metrics = 0.0, 0.0, Metrics()
        with click.progressbar(length=len(data_loader), label="[Evaluate]", item_show_func=show_item, width=25) as bar:
            for batch_idx, batch in enumerate(data_loader):
                batch = [t.cuda(gpu) for t in batch]
                inputs, target = batch[:-1], batch[-1]
                output = self.model(*inputs)
                loss = self.criterion(output, target)
                # update loss.
                L_acc += loss.item()
                L_ave = L_acc / (batch_idx + 1)
                # update metrics.
                metrics.update(output, target)
                # update progress bar.
                bar.update(1, dict(loss=L_ave, **metrics.stats()))


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


class Logger:
    def __init__(self, log_file: Path):
        self.log_file = log_file

    def log(self, *args: list, **kwargs: dict):
        with open(self.log_file, "a") as f:
            f.write(strftime("%Y/%m/%d %H:%M:%S", localtime()))
            f.write("".join([f" - {i}" for i in args]))
            f.write(" - ")
            f.write(",".join([f"{k}={v}" for k, v in kwargs.items()]))
            f.write("\n")


def show_item(stats: dict):
    if stats is None:
        return ""
    return f"loss:{stats['loss']:.2f} MAE={stats['MAE']:.2f} MAPE={stats['MAPE']:.2f}% RMSE={stats['RMSE']:.2f}"
