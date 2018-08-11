from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import sys
import json
import numpy as np
import random

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from reid import models, datasets
from reid.loss import TripletLoss
from reid.trainers import Trainer
from reid.evaluators import Evaluator 
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomIdentitySampler, caffeSampler
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint
from reid.optim.sgd_caffe import SGD_caffe

def get_data(name, split_id, data_dir,
             height, width, crop_height, crop_width, batch_size,
             caffe_sampler=False,
             workers=4):

    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root, split_id=split_id)
    train_set = dataset.trainval
    num_classes = dataset.num_trainval_ids 

    # transforms
    train_transformer = T.Compose([
        T.RectScale(height, width),
#        T.CenterCrop((crop_height, crop_width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.RGB_to_BGR(),
        T.NormalizeBy(255),
    ])
    test_transformer = T.Compose([
        T.RectScale(height, width),
#        T.CenterCrop((crop_height, crop_width)),
        T.ToTensor(),
        T.RGB_to_BGR(),
        T.NormalizeBy(255),
    ])

    # dataloaders
    sampler = caffeSampler(train_set, name, batch_size=batch_size, root=dataset.images_dir) if caffe_sampler else \
              RandomIdentitySampler(train_set, 10) #TODO
    train_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir,
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        sampler=sampler,
        pin_memory=True, drop_last=True)
    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, train_loader, test_loader


def main(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    # Redirect print to both console and log file
    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Log args
    print(args)
    args_dict = vars(args)
    with open(osp.join(args.logs_dir, 'args.json'), 'w') as f:
        json.dump(args_dict, f)

    # Create data loaders
    dataset, num_classes, train_loader, test_loader = \
        get_data(args.dataset, args.split, args.data_dir, args.height, \
                 args.width, args.crop_height, args.crop_width, args.batch_size, \
                 args.caffe_sampler, \
                 args.workers)

    # Create model
    valid_args = ['features', 'use_relu', 'dilation']
    model_kwargs = {k:v for k,v in args_dict.items() if k in valid_args}
    model = models.create(args.arch, **model_kwargs)
    model = nn.DataParallel(model).cuda()

    # Evaluator
    evaluator = Evaluator(model)

    # Criterion
    criterion = TripletLoss(margin=args.margin).cuda()

    # Optimizer
    params = []
    params_dict = dict(model.named_parameters())
    for key, value in params_dict.items():
        if value.requires_grad == False:
            continue
        if key[-4:]=='bias':
            params += [{'params': [value], 'lr':args.lr*2, 'weight_decay':args.weight_decay*0.0}]
        else:
            params += [{'params': [value], 'lr':args.lr*1, 'weight_decay':args.weight_decay*1.0}]
    optimizer = SGD_caffe(params,
                          lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    # Trainer
    trainer = Trainer(model, criterion)

    # Evaluate
    def evaluate(test_model):
        print('Test with model {}:'.format(test_model))
        checkpoint = load_checkpoint(osp.join(args.logs_dir, '{}.pth.tar'.format(test_model)))
        model.module.load(checkpoint)
        evaluator.evaluate(test_loader, dataset.query, dataset.gallery, msg='TEST')

    # Schedule learning rate
    def adjust_lr(epoch):
        lr = args.lr if epoch <= 200 else \
            args.lr * (0.2 ** ((epoch-200)//200 + 1))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    # Start training
    for epoch in range(1, args.epochs+1):

        adjust_lr(epoch)
        trainer.train(epoch, train_loader, optimizer)

        # Save
        if epoch%200 == 0 or epoch==args.epochs:
            save_dict = model.module.save_dict()
            save_dict.update({'epoch': epoch})
            save_checkpoint(save_dict, fpath=osp.join(args.logs_dir, 'epoch_{}.pth.tar'.format(epoch)))

        print('\n * Finished epoch {:3d}\n'.format(epoch))

    evaluate('epoch_750')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Triplet loss classification")

    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=180)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int, default=160)
    parser.add_argument('--width', type=int, default=80)
    parser.add_argument('--crop-height', type=int, default=160)
    parser.add_argument('--crop-width', type=int, default=80)
    # model
    parser.add_argument('-a', '--arch', type=str, default='inception_v1_cpm_pretrained',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=512)
    parser.add_argument('--use-relu', dest='use_relu', action='store_true', default=False)
    parser.add_argument('--dilation', type=int, default=2)
    # loss
    parser.add_argument('--margin', type=float, default=0.2,
                        help="margin of the triplet loss, default: 0.2")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate of all parameters")
    parser.add_argument('--weight-decay', type=float, default=2e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=750)
    parser.add_argument('--caffe-sampler', dest='caffe_sampler', action='store_true', default=False)
    # paths
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))

    main(parser.parse_args())
