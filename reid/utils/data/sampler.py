from __future__ import absolute_import
from collections import defaultdict

import numpy as np
import os.path as osp
import os
import glob
import torch
from torch.utils.data.sampler import Sampler

class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances=1):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_samples)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        return iter(ret)

class caffeSampler(Sampler):
    def __init__(self, data_source, dataset, batch_size, iter_per_epoch=100, root=None):
        self.data_source = data_source
        self.index_dic = defaultdict(list)
        for index, (fname, _, _) in enumerate(data_source):
            self.index_dic[fname].append(index)
        self.train_list_path = os.environ['{}_TRAIN_LIST'.format(dataset.upper())]
        dataset_dir = os.environ['{}_DATA_ROOT'.format(dataset.upper())]
        if dataset in ['market1501', 'dukemtmc', 'cuhk03_np']:
            dataset_dir = osp.join(dataset_dir, 'bounding_box_train')
        ext = '*.jpg' if dataset in ['market1501', 'dukemtmc', 'mars'] else '*.png'
        orig_train_list = [v.rstrip().split()[0] for v in open(self.train_list_path, "r").readlines()]
        symlink_list = glob.glob(osp.join(root, ext))
        self.orig_to_link_dict = {osp.realpath(sym): osp.basename(sym) for sym in symlink_list}
        self.sym_train_list = [int(self.index_dic[self.orig_to_link_dict[osp.join(dataset_dir, path)]][0]) for path in orig_train_list]

        self.epoch = 0
        self.iter_per_epoch = iter_per_epoch
        self.batch_size = batch_size
    def __len__(self):
        return self.iter_per_epoch*self.batch_size
    def __iter__(self):
        sel_start = self.epoch * self.iter_per_epoch * self.batch_size
        sel_end = (self.epoch+1) * self.iter_per_epoch * self.batch_size
        self.epoch += 1
        return iter(self.sym_train_list[sel_start: sel_end])
