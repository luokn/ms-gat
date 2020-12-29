import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def generate_datasets(data: torch.Tensor, points_per_hour, offset_hours):
    steps = [hour * points_per_hour for hour in offset_hours]
    step_max = max(steps)
    X = torch.stack([
        data[step_max - step:-step].unfold(0, size=points_per_hour, step=1) for step in steps
    ], dim=0).transpose(0, 1)
    Y = data[step_max:, 0].unfold(0, size=points_per_hour, step=1)
    T = torch.arange(len(Y), dtype=torch.long, device=data.device) // points_per_hour
    H, D = T % 24, (T // 24) % 7
    return X, H, D, Y


def normalize_dataset(x: torch.Tensor, split):
    std, mean = torch.std_mean(x[:split], dim=0, keepdim=True)
    x -= mean
    x /= std
    return dict(std=std, mean=mean)


def load_data(data_file, batch_size, points_per_hour, device='cpu'):
    data = torch.from_numpy(np.load(data_file)['data'])
    data = data.transpose(1, 2).float().to(device)
    X, H, D, Y = generate_datasets(data, points_per_hour, [1, 2, 3, 24, 7 * 24])
    t_split, v_split = int(X.size(0) * .6), int(X.size(0) * .8)
    statistics = normalize_dataset(X, t_split)
    datasets = [
        # train
        TensorDataset(X[:t_split], H[:t_split], D[:t_split], Y[:t_split]),
        # validate
        TensorDataset(X[t_split:v_split], H[t_split:v_split], D[t_split:v_split], Y[t_split:v_split]),
        # test
        TensorDataset(X[v_split:], H[v_split:], D[v_split:], Y[v_split:])
    ]
    data_loaders = [DataLoader(dataset, batch_size, shuffle=True) for dataset in datasets]
    return data_loaders, statistics


def load_adj(adj_file, n_nodes, device='cpu'):
    r"""
    .. math:: 
        \tilde A = \tilde{D}^{-1/2} (A + I_n) \tilde{D}^{-1/2}
    """
    A = torch.eye(n_nodes, device=device)
    for ln in open(adj_file, 'r').readlines()[1:]:
        i, j, _ = ln.split(',')
        i, j = int(i), int(j)
        A[i, j] = A[j, i] = 1

    D_rsqrt = A.sum(dim=1).rsqrt().diag()
    return D_rsqrt @ A @ D_rsqrt
