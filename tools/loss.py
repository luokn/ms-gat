import torch
from torch import FloatTensor
from torch.nn import Module


def huber_loss(pred: FloatTensor, y: FloatTensor, delta: float) -> FloatTensor:
    l2_loss = .5 * (pred - y).square()
    l1_loss = delta * (pred - y).abs() - .5 * (delta ** 2)
    loss = torch.where((pred - y).abs() <= delta, l2_loss, l1_loss)
    return torch.mean(loss)


class HuberLoss(Module):
    def __init__(self, delta: float):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, pred: FloatTensor, y: FloatTensor) -> FloatTensor:
        return huber_loss(pred, y, self.delta)
