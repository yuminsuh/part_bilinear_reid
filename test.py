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

os.environ['CUDA_VISIBLE_DEVICES']='0'
exp_dir = 'logs/market1501/eccv18_reproduce'

args = json.load(open(osp.join(exp_dir, "args.json"), "r"))
target_epoch = 740

weight_file = osp.join(exp_dir, 'epoch_{}.pth.tar'.format(target_epoch))

t = T.Compose([
                T.RectScale(args['height'], args['width']),
                T.CenterCrop((args['crop_height'], args['crop_width'])),
                T.ToTensor(),
                T.RGB_to_BGR(),
                T.NormalizeBy(255),
              ])
dataset = datasets.create(args['dataset'], 'data/{}'.format(args['dataset']))
dataset_ = Preprocessor(list(set(dataset.query)|set(dataset.gallery)), root=dataset.images_dir, transform=t)
dataloader = DataLoader(dataset_, batch_size=50, shuffle=False)

model = models.create(args['arch'], dilation=args['dilation'], use_relu=args['use_relu'], initialize=False)
print('Loading model from {}...'.format(weight_file))
checkpoint = load_checkpoint(weight_file)
model_app_dict = {l: torch.from_numpy(np.array(v)).view_as(p) for k,v in checkpoint['app_state_dict'].items() for l,p in model.app_feat_extractor.state_dict().items() if k in l}
model_part_dict = {l: torch.from_numpy(np.array(v)).view_as(p) for k,v in checkpoint['part_state_dict'].items() for l,p in model.part_feat_extractor.state_dict().items() if k in l}
model.app_feat_extractor.module.load_state_dict(checkpoint['app_state_dict'], strict=False)
model.part_feat_extractor.module.load_state_dict(checkpoint['part_state_dict'], strict=False)
model.eval()

evaluator = Evaluator(model)
evaluator.evaluate(dataloader, dataset.query, dataset.gallery)
