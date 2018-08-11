import os
import os.path as osp
import json
import numpy as np

import torch
from torch.utils.data import DataLoader

from reid import datasets, models
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.serialization import load_checkpoint
from reid.utils.osutils import set_paths

# Settings
exp_dir = 'logs/market1501/eccv18_reproduce'
target_epoch = 750
batch_size = 50
gpu_ids = '0'

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
model = models.create(args['arch'], dilation=args['dilation'], use_relu=args['use_relu']).cuda()
weight_file = osp.join(exp_dir, 'epoch_{}.pth.tar'.format(target_epoch))
model.load(load_checkpoint(weight_file))
model.eval()

# Evaluate
evaluator = Evaluator(model)
evaluator.evaluate(dataloader, dataset.query, dataset.gallery)
