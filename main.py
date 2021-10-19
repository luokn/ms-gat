#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Kun Luo
# @Email   : olooook@outlook.com
# @File    : main.py
# @Date    : 2021/06/02
# @Time    : 17:05:00

import os

from torch.nn import DataParallel
from torch.optim import Adam

from lib import HuberLoss, Trainer, init_network, load_adj, load_data, parse_args
from nn import ms_gat48, ms_gat72, ms_gat96

models = {'ms-gat': ms_gat72, 'ms-gat48': ms_gat48, 'ms-gat72': ms_gat72, 'ms-gat96': ms_gat96}

if __name__ == '__main__':
    # parser arguments
    args = parse_args()
    print(args)
    # set CUDA_VISIBLE_DEVICES
    if args.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    # data loaders
    in_hours = tuple(map(int, args.in_hours.split(',')))
    data = load_data(args.data, args.batch, in_hours, args.out_timesteps, args.frequency, args.workers)
    # adjacency matrix
    adj = load_adj(args.adj, args.nodes)
    # network
    net = models[args.model](len(in_hours), args.channels, args.frequency, args.out_timesteps, adj, te=not args.no_te)
    net = DataParallel(net).cuda() if args.gpus else net.cuda(args.gpu)
    net = init_network(net)
    # optimizer
    optimizer = Adam(net.parameters(), lr=args.lr)
    # loss function
    criterion = HuberLoss(args.delta)
    # trainer
    trainer = Trainer(net, optimizer, criterion, args.checkpoints)
    # load checkpoint
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    # train and validate
    trainer.train(data[0], data[1], epochs=args.epochs, gpu=args.gpu)
    # load best checkpoint
    trainer.load_checkpoint(f'epoch={trainer.best_epoch}_loss={trainer.min_loss:.2f}.pkl')
    # evaluate
    trainer.evaluate(data[-1], gpu=args.gpu)
    # save history
    trainer.save_history()
