import torch


class Metrics:
    """
    Calculate ``MAE/MAPE/RMSE``.
    """

    def __init__(self, mask_value=0.0):
        self.n, self.mask_value = 0, mask_value
        self.AE, self.APE, self.SE, self.MAE, self.MAPE, self.RMSE = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    def update(self, y_pred, y_true):
        self.n += y_true.nelement()
        # MAE
        self.AE += torch.abs(y_pred - y_true).sum().item()
        self.MAE = self.AE / self.n
        # MAPE
        mask = y_true > self.mask_value
        masked_prediction, masked_truth = y_pred[mask], y_true[mask]
        self.APE += 100 * torch.abs((masked_prediction - masked_truth) / masked_truth).sum().item()
        self.MAPE = self.APE / self.n
        # RMSE
        self.SE += torch.square(y_pred - y_true).sum().item()
        self.RMSE = (self.SE / self.n) ** 0.5

    def todict(self):
        return {"MAE": self.MAE, "MAPE": self.MAPE, "RMSE": self.RMSE}
