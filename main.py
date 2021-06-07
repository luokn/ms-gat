#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Kun Luo
# @Email   : olooook@outlook.com
# @File    : main.py
# @Date    : 2021/06/02
# @Time    : 17:05:00

import os
from argparse import ArgumentParser

from torch import nn, optim

from lib import HuberLoss, Trainer, init_network, load_adj, load_data
from nn import msgat

parser = ArgumentParser(description="Train MS-GAT")

parser.add_argument('--data', type=str, help='Data file')
parser.add_argument('--adj', type=str, help='Adjacency file')
parser.add_argument('--nodes', type=int, help='Number of nodes')
parser.add_argument('--channels', type=int, help='Number of channels')
parser.add_argument('--checkpoints', type=str, help='Checkpoints path')
parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint name')

parser.add_argument('--batch', type=int, default=64, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='Learn rate')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--workers', type=int, default=0, help='Number of data loader workers')
parser.add_argument('--gpu', type=int, default=None, help='GPU')
parser.add_argument('--gpus', type=str, default=None, help='GPUs')

parser.add_argument('--model', type=str, default='msgat', help='Model name')
parser.add_argument('--no-te', type=bool, default=False, help='No time embedding')
parser.add_argument('--frequency', type=int, default=12, help='Time steps per hour')
parser.add_argument('--in-hours', type=str, default='1,2,3,24,168', help='Hours of sampling')
parser.add_argument('--out-timesteps', type=int, default=12, help='Number of output timesteps')
parser.add_argument('--delta', type=float, default=60, help='Delta of huber loss')

args = parser.parse_args()
print(args)

if __name__ == '__main__':
    if args.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    in_hours = [int(hour) for hour in args.in_hours.split(',')]
    # data loaders
    data_loaders = load_data(args.data, args.batch, in_hours, args.out_timesteps, args.frequency, args.workers)
    # adjacency matrix
    adj = load_adj(args.adj, args.nodes)
    # network
    net = msgat(len(in_hours), args.channels, args.frequency, args.out_timesteps, adj, te=not args.no_te)
    net = nn.DataParallel(net).cuda() if args.gpus else net.cuda(args.gpu)
    net = init_network(net)
    # optimizer
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    # loss function
    criterion = HuberLoss(args.delta)
    # trainer
    trainer = Trainer(net, optimizer, criterion, args.checkpoints)
    if args.checkpoint:
        # load checkpoint
        trainer.load_checkpoint(args.checkpoint)
    # train and validate
    trainer.fit(data_loaders[0], data_loaders[1], epochs=args.epochs, device=args.gpu)
    # load best checkpoint
    trainer.load_checkpoint(f'epoch={trainer.best_epoch}_loss={trainer.min_loss:.2f}.pkl')
    # evaluate
    trainer.evaluate(data_loaders[2], device=args.gpu)
    # save history
    trainer.save_history()
