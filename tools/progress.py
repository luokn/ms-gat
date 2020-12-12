from datetime import timedelta
from time import time


class ProgressBar:
    def __init__(self, total: int, n_circles=50):
        self.phase = 0
        self.total = total
        self.n_circles = n_circles
        self.t_start = time()
        self.line_len = -1

    def update(self, postfix='', n=1):
        self.phase += n
        t_delta = timedelta(seconds=int(time() - self.t_start))
        circles = '●' * (self.n_circles * self.phase // self.total)
        circles = circles.ljust(self.n_circles, '○')
        line = f'\r[32m[{circles}] [{self.phase}/{self.total} {t_delta}] {postfix}'
        if self.line_len > len(line):
            line = line.ljust(self.line_len)
        self.line_len = len(line)
        print(line, end='')

    def close(self):
        self.phase, self.line_len = 0, -1
        print('')

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()
