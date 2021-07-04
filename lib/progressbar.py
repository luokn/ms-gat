#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Kun Luo
# @Email   : olooook@outlook.com
# @File    : progressbar.py
# @Date    : 2021/06/02
# @Time    : 17:14:59


from time import time


class ProgressBar:
    def __init__(self, iterable):
        self.iter, self.total = iter(iterable), len(iterable)
        self.time, self.stage = time(), 0

    def update(self, postfix):
        self.stage += 1
        progress = '=' * int(25 * self.stage / self.total)
        minutes, seconds = divmod(int(time() - self.time), 60)
        print(f'\r[{progress:25}] '
              f'- {self.stage:3d}/{self.total:<3d} '
              f'- {minutes:02d}:{seconds:02d} '
              f'- {postfix}', end='')

    def __iter__(self):
        return self.iter

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        print('')
