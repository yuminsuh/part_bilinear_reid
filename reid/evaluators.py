from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch

from .evaluation_metrics import cmc_meanap_fast, cmc, mean_ap
from .feature_extraction import extract_cnn_feature
from .utils.meters import AverageMeter
import numpy as np


def extract_features(model, data_loader, print_freq=10):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    for i, (imgs, fnames, pids, _) in enumerate(data_loader):
        data_time.update(time.time() - end)

        outputs = extract_cnn_feature(model, imgs.cuda())
        for fname, output, pid in zip(fnames, outputs, pids):
            features[fname] = output
            labels[fname] = pid

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))

    return features, labels

def pairwise_distance(features, query=None, gallery=None, metric=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        if metric is not None:
            x = metric.transform(x)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist = dist.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    if metric is not None:
        x = metric.transform(x)
        y = metric.transform(y)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist

class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, topk=1000, msg=''):

        # Extract query & gallery features
        features, _ = extract_features(self.model, data_loader)
        query_ids = [pid for _,pid,_ in query]
        query_cams = [cid for _,_,cid in query]
        gallery_ids = [pid for _,pid,_ in gallery]
        gallery_cams = [cid for _,_,cid in gallery]

        feat_query = torch.cat([features[f].unsqueeze(0) for f,_,_ in query], 0)
        feat_gallery = torch.cat([features[f].unsqueeze(0) for f,_,_ in gallery], 0)

        # Calculate CMC & mAP
#        result_cmc, result_meanap = cmc_meanap_fast(feat_query, feat_gallery,
#                                 query_ids, gallery_ids,
#                                 query_cams, gallery_cams, topk=topk)

        distmat = pairwise_distance(features, query, gallery)
        result_cmc = cmc(distmat, query_ids, gallery_ids, query_cams, gallery_cams,\
                         separate_camera_set=False,\
                         single_gallery_shot=False,\
                         first_match_break=True)
        result_meanap = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)

        print('CMC Scores')
        for k in [1,5,10]:
            print('  top-{:<4}{:12.1%}'.format(k, result_cmc[k - 1]))
        print('{} Mean AP: {:3.1%}'.format(msg, result_meanap))

        return result_cmc[0], result_meanap
