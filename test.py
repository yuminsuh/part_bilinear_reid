import os
import os.path as osp
import json
import numpy as np
import argparse

import torch
from torch.utils.data import DataLoader

from reid import datasets, models
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.serialization import load_checkpoint
from reid.utils.osutils import set_paths

def evaluate(args):
    # Settings
    exp_dir = './logs/{}/{}'.format(args.dataset, args.exp)
    target_epoch = args.epoch
    batch_size = args.batchsize
    gpu_ids = args.gpus

    set_paths('paths')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    args = json.load(open(osp.join(exp_dir, "args.json"), "r"))

    # Load data
    t = T.Compose([
                    T.RectScale(args['height'], args['width']),
                    T.CenterCrop((args['crop_height'], args['crop_width'])),
                    T.ToTensor(),
                    T.RGB_to_BGR(),
                    T.NormalizeBy(255),
                  ])
    dataset = datasets.create(args['dataset'], 'data/{}'.format(args['dataset']))
    dataset_ = Preprocessor(list(set(dataset.query)|set(dataset.gallery)), root=dataset.images_dir, transform=t)
    dataloader = DataLoader(dataset_, batch_size=batch_size, shuffle=False)

    # Load model
    model = models.create(args['arch'], dilation=args['dilation'], use_relu=args['use_relu'], initialize=False).cuda()
    weight_file = osp.join(exp_dir, 'epoch_{}.pth.tar'.format(target_epoch))
    model.load(load_checkpoint(weight_file))
    model.eval()

    # Evaluate
    evaluator = Evaluator(model)
    evaluator.evaluate(dataloader, dataset.query, dataset.gallery)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument('-d', '--dataset', type=str, default='market1501')
    parser.add_argument('-e', '--exp', type=str, default='d2_b250')
    parser.add_argument('-b', '--batchsize', type=int, default=50)
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--epoch', type=int, default=750)
    
    evaluate(parser.parse_args())
