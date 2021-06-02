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

parser = ArgumentParser()

parser.add_argument('--data', type=str)
parser.add_argument('--adj', type=str)
parser.add_argument('--nodes', type=int)
parser.add_argument('--channels', type=int)
parser.add_argument('--checkpoints', type=str)
parser.add_argument('--checkpoint', type=str, default=None)

parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--workers', type=int, default=0)
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--gpus', type=str, default=None)

parser.add_argument('--model', type=str, default='msgat')
parser.add_argument('--components', type=int, default=5)
parser.add_argument('--no-te', type=bool, default=False)
parser.add_argument('--frequency', type=int, default=12)
parser.add_argument('--in-timesteps', type=int, default=12)
parser.add_argument('--out-timesteps', type=int, default=12)
parser.add_argument('--hours', type=str, default='1,2,3,24,168')
parser.add_argument('--delta', type=float, default=60)

args = parser.parse_args()
print(args)

if __name__ == '__main__':
    if args.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    # adjacency matrix
    adj = load_adj(args.adj, args.nodes)
    # data loaders
    data_loaders = load_data(args.data,
                             frequency=args.frequency,
                             out_timesteps=args.out_timesteps,
                             hours=[int(h) for h in args.hours.split(',')],
                             batch_size=args.batch, num_workers=args.workers, pin_memory=True)
    # network
    net = msgat(args.components, in_channels=args.channels, in_timesteps=args.in_timesteps,
                out_timesteps=args.out_timesteps, adj=adj, te=not args.no_te)
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
