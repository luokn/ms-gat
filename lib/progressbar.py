#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Kun Luo
# @Email   : olooook@outlook.com
# @File    : progressbar.py
# @Date    : 2021/06/02
# @Time    : 17:14:59

from datetime import timedelta
from time import time


class ProgressBar:
    def __init__(self, total: int):
        self.stage, self.total = 0, total
        self.time = time()

    def update(self, postfix='', n=1):
        self.stage += n
        progress = "●" * (30 * self.stage // self.total)
        time_delta = timedelta(seconds=int(time() - self.time))
        print(f'\r[{progress:30}] {self.stage:3d}/{self.total:<3d} {time_delta} {postfix}', end='')

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        print('')
