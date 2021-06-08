#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Kun Luo
# @Email   : olooook@outlook.com
# @File    : main.py
# @Date    : 2021/06/02
# @Time    : 17:05:00

import os

from torch import nn, optim

from lib import (HuberLoss, Trainer, init_network, load_adj, load_data,
                 parse_args)
from nn import msgat, msgat_l, msgat_m, msgat_s

models = {'msgat': msgat, 'msgat_l': msgat_l,  'msgat_m': msgat_m, 'msgat_s': msgat_s}

if __name__ == '__main__':
    # parser arguments
    args = parse_args()
    print(args)
    # set CUDA_VISIBLE_DEVICES
    if args.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    # data loaders
    in_hours = [int(hour) for hour in args.in_hours.split(',')]
    data_loaders = load_data(args.data, args.batch, in_hours, args.out_timesteps, args.frequency, args.workers)
    # adjacency matrix
    adj = load_adj(args.adj, args.nodes)
    # network
    net = models[args.model](len(in_hours), args.channels, args.frequency, args.out_timesteps, adj, te=not args.no_te)
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
