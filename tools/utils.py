import os
from datetime import datetime

import torch


def load_adj_matrix(adj_file, n_nodes, device='cuda:0'):
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


def make_out_dir(out_dir):
    out_dir = os.path.join(out_dir, datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return out_dir


def log_to_file(file, **kwargs):
    with open(file, 'a') as f:
        f.write(','.join([f'{k}={v}' for k, v in kwargs.items()]))
        f.write('\n')
