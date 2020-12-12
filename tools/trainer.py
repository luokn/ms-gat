from functools import partial
from json import dumps

import torch
from tools.data import load_data
from tools.metrics import Metrics
from tools.model import make_msgat
from tools.progress import ProgressBar
from tools.utils import load_adj_matrix, log_to_file, make_out_dir
from torch.nn import MSELoss
from torch.optim import Adam


class Trainer:
    def __init__(self, *, lr: float, epochs: int, batch_size: int, data_split: float,
                 adj_file: str, data_file: str, out_dir: str, n_nodes: int,  points_per_hour: int,
                 device_for_data: str = 'cpu', device_for_model: str = 'cpu'):
        print('Loading...')
        # load data
        adj = load_adj_matrix(adj_file, n_nodes, device_for_model)
        loaders, statistics = load_data(data_file, batch_size, data_split, points_per_hour, device_for_data)
        self.t_loader, self.v_loader = loaders
        # create mdoel
        self.net = make_msgat(points_per_hour, points_per_hour, n_nodes, adj, device_for_model)
        self.optimizer = Adam(self.net.parameters(), lr=lr)
        self.criterion = MSELoss().to(device_for_model)
        self.out_dir = make_out_dir(out_dir)
        self.log_t = partial(log_to_file, f'{self.out_dir}/t.log')
        self.log_v = partial(log_to_file, f'{self.out_dir}/v.log')
        self.device, self.epochs = device_for_model, epochs
        # torch.save(statistics, f'{self.out_dir}/statistics.pth')

    def run(self):
        print('Training...')
        # train
        best = float('inf')
        history = []
        for epoch in range(self.epochs):
            print(f"Epoch: {epoch + 1}")
            loss_t = self.train_epoch(epoch)
            loss_v, MAE, RMSE = self.validate_epoch(epoch)
            if epoch >= int(.2 * self.epochs) and MAE < best:
                best = MAE
                torch.save(self.net.state_dict(), f'{self.out_dir}/MAE={best:.2f}.pkl')
            history.append(dict(loss_t=loss_t, loss_v=loss_v, MAE=MAE, RMSE=RMSE))
        open(f'{self.out_dir}/history.json', 'w').write(dumps(history))

    def train_epoch(self, epoch):
        self.net.train()
        loss_sum, loss_ave = .0, .0
        with ProgressBar(total=len(self.t_loader)) as bar:
            for i, batch in enumerate(self.t_loader):
                x, h, d, y = [t.to(self.device) for t in batch]
                self.optimizer.zero_grad()
                pred = self.net(x, h, d)
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()
                loss_sum += loss
                loss_ave = loss_sum / (i + 1)
                bar.update(postfix=f'loss_t={loss_ave:.2f}')
                self.log_t(epoch=epoch, batch=i, loss=loss)
        return loss_ave

    @torch.no_grad()
    def validate_epoch(self, epoch):
        metrics = Metrics()
        self.net.eval()
        loss_sum, loss_ave = .0, .0
        with ProgressBar(total=len(self.v_loader)) as bar:
            for i, batch in enumerate(self.v_loader):
                x, h, d, y = [t.to(self.device) for t in batch]
                pred = self.net(x, h, d)
                loss = self.criterion(pred, y)
                metrics.update(pred, y)
                loss_sum += loss
                loss_ave = loss_sum / (i + 1)
                bar.update(
                    postfix=f'loss_v={loss_ave:.2f} MAE={metrics.MAE:.2f} RMSE={metrics.RMSE:.2f}'
                )
                self.log_v(epoch=epoch, batch=i, loss=loss, MAE=metrics.MAE, RMSE=metrics.RMSE)
        return loss_ave, metrics.MAE, metrics.RMSE
