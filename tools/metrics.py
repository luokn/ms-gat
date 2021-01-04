from torch import FloatTensor


class Metrics:
    def __init__(self):
        self.n = 0
        self.AE, self.SE = .0, .0
        self.MAE, self.RMSE = .0, .0

    def update(self, pred: FloatTensor, y: FloatTensor):
        self.n += pred.nelement()
        self.AE += (pred - y).abs().sum().item()
        self.SE += (pred - y).square().sum().item()
        self.MAE = self.AE / self.n
        self.RMSE = (self.SE / self.n) ** .5

    def clear(self):
        self.n = 0
        self.AE, self.SE = .0, .0
