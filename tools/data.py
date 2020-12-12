import numpy
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
    data = torch.from_numpy(numpy.load(data_file)['data'])
    data = data.transpose(1, 2).float().to(device)
    X, H, D, Y = generate_datasets(data, points_per_hour, [1, 2, 3, 24, 7 * 24])
    split0, split1 = int(X.size(0) * .6), int(X.size(0) * .8)
    statistics = normalize_dataset(X, split0)
    datasets = [
        # train
        TensorDataset(X[:split0], H[:split0], D[:split0], Y[:split0]),
        # validate
        TensorDataset(X[split0:split1], H[split0:split1], D[split0:split1], Y[split0:split1]),
        # test
        TensorDataset(X[split1:], H[split1:], D[split1:], Y[split1:])
    ]
    data_loaders = [DataLoader(dataset, batch_size, shuffle=True) for dataset in datasets]
    return data_loaders, statistics
