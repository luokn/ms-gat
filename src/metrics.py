import torch


class Metrics:
    """
    Calculate ``MAE/MAPE/RMSE``.
    """

    def __init__(self, mask_value=0.0):
        self.n, self.mask_value = 0, mask_value
        self.AE, self.APE, self.SE, self.MAE, self.MAPE, self.RMSE = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    def update(self, prediction, truth):
        self.n += truth.nelement()
        # MAE
        self.AE += torch.abs(prediction - truth).sum().item()
        self.MAE = self.AE / self.n
        # MAPE
        mask = truth > self.mask_value
        masked_prediction, masked_truth = prediction[mask], truth[mask]
        self.APE += 100 * torch.abs((masked_prediction - masked_truth) / masked_truth).sum().item()
        self.MAPE = self.APE / self.n
        # RMSE
        self.SE += torch.square(prediction - truth).sum().item()
        self.RMSE = (self.SE / self.n) ** 0.5
