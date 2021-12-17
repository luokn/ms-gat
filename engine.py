#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Kun Luo
# @Email   : olooook@outlook.com
# @File    : trainer.py
# @Date    : 2021/06/02
# @Time    : 17:41:37


from pathlib import Path
from time import localtime, strftime

import click
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from models.loss import HuberLoss


class Engine:
    def __init__(self, model, out_dir, **kwargs):
        self.model = model
        self.loss_fn = HuberLoss(kwargs["delta"])
        self.out_dir = Path(out_dir)
        if not self.out_dir.exists():
            self.out_dir.mkdir(parents=True)
        self.log_file = self.out_dir / "run.log"

    def _run_once(self, data_loader, mode, epoch, gpu_id):
        self.model.train(mode == "train")
        with torch.set_grad_enabled(mode == "train"):
            labels = {"train": "[Train   ]", "validate": "[Validate]", "evaluate": "[Evaluate]"}
            L_acc, L_ave, metrics = 0.0, 0.0, Metrics()
            with click.progressbar(
                length=len(data_loader), label=labels[mode], item_show_func=self.__show_item, width=25
            ) as bar:
                for batch_idx, batch_data in enumerate(data_loader):
                    batch_data = [tensor.cuda(gpu_id) for tensor in batch_data]
                    inputs, truth = batch_data[:-1], batch_data[-1]
                    with autocast():
                        output = self.model(*inputs)
                        loss = self.loss_fn(output, truth)
                    if mode == "train":
                        self.optimizer.zero_grad()
                        self.grad_scaler.scale(loss).backward()
                        self.grad_scaler.step(self.optimizer)
                        self.grad_scaler.update()
                    # else:
                    #     output = self.model(*inputs)
                    #     loss = self.criterion(output, target)
                    L_acc += loss.item()
                    L_ave = L_acc / (batch_idx + 1)  # loss.
                    metrics.update(output, truth)  # metrics.
                    bar.update(n_steps=1, current_item=(L_ave, metrics))
            stats = dict(loss=L_ave, MAE=metrics.MAE, MAPE=metrics.MAPE, RMSE=metrics.RMSE)
            # log to file.
            if mode == "evaluate":
                self._log(labels[mode], **stats)
            else:
                self._log(labels[mode], epoch=epoch, **stats)
            return L_ave

    def _log(self, *args, **kwargs):
        with open(self.log_file, "a") as f:
            f.write(strftime("%Y/%m/%d %H:%M:%S", localtime()))
            f.write(" - ")
            f.write(" - ".join([f"{arg}" for arg in args]))
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
    def __init__(self, model, out_dir, **kwargs):
        super(Trainer, self).__init__(model, out_dir, delta=kwargs["delta"])
        self.optimizer = Adam(model.parameters(), lr=kwargs["lr"], weight_decay=kwargs["weight_decay"])
        self.scheduler = StepLR(self.optimizer, step_size=kwargs["step_size"], gamma=kwargs["gamma"])
        self.grad_scaler = GradScaler()
        #
        self.patience, self.min_delta = kwargs["patience"], kwargs["min_delta"]
        self.max_epochs, self.min_epochs = kwargs["max_epochs"], kwargs["min_epochs"]
        self.epoch = 1
        self.best = {"epoch": 0, "loss": float("inf"), "ckpt": ""}

    def fit(self, data_loaders, gpu_id=None):
        while self.epoch <= self.max_epochs:
            click.echo(f"Epoch {self.epoch}")
            self._run_once(data_loaders[0], mode="train", epoch=self.epoch, gpu_id=gpu_id)
            loss = self._run_once(data_loaders[1], mode="validate", epoch=self.epoch, gpu_id=gpu_id)
            self.scheduler.step()
            if self.epoch > self.min_epochs:
                if loss < (1 - self.min_delta) * self.best["loss"]:
                    self.best = dict(epoch=self.epoch, loss=loss, ckpt=self.out_dir / f"{self.epoch}_{loss:.2f}.pkl")
                    self.save(ckpt=self.best["ckpt"])
                elif self.epoch > self.best["epoch"] + self.patience:
                    break  # early stop.
            self.epoch += 1

    def eval(self, data_loader, gpu_id=None):
        self._run_once(data_loader, mode="evaluate", epoch=None, gpu_id=gpu_id)

    def save(self, ckpt):
        click.echo(f"• Save checkpoint {ckpt}")
        states = dict(
            best=self.best,
            epoch=self.epoch,
            model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(),
            grad_scaler=self.grad_scaler.state_dict(),
        )
        torch.save(states, ckpt)

    def load(self, ckpt):
        click.echo(f"• Load checkpoint {ckpt}")
        states = torch.load(ckpt)
        self.best = states["best"]
        self.epoch = states["epoch"] + 1
        self.model.load_state_dict(states["model"])
        self.optimizer.load_state_dict(states["optimizer"])
        self.scheduler.load_state_dict(states["scheduler"])
        self.grad_scaler.load_state_dict(states["grad_scaler"])


class Evaluator(Engine):
    def __init__(self, model, out_dir, **kwargs):
        super(Evaluator, self).__init__(model, out_dir, delta=kwargs["delta"])
        states = torch.load(kwargs["ckpt"])
        model.load_state_dict(states["model"])

    def eval(self, data_loader, gpu_id=None):
        self._run_once(data_loader, mode="evaluate", epoch=None, gpu_id=gpu_id)


class Metrics:
    """
    Calculate ``MAE/MAPE/RMSE``.
    """

    def __init__(self, mask_value=0.0):
        self.n, self.mask_value = 0, mask_value
        self.AE, self.APE, self.SE, self.MAE, self.MAPE, self.RMSE = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

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
