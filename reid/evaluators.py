from __future__ import print_function, absolute_import
import time
from collections import OrderedDict, defaultdict
import os.path as osp
import scipy.io
import sys

import torch

from .evaluation_metrics import cmc_meanap_fast, cmc, mean_ap
from .feature_extraction import extract_cnn_feature
from .utils.meters import AverageMeter
import numpy as np


def extract_features(model, data_loader, print_freq=10):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    index = OrderedDict()
    features = np.zeros((len(data_loader.dataset.dataset), 512)) #OrderedDict()
    labels = OrderedDict()

    end = time.time()
    for i, (imgs, fnames, pids, _) in enumerate(data_loader):
        data_time.update(time.time() - end)

        outputs = extract_cnn_feature(model, imgs.cuda())
        for j, (fname, output, pid) in enumerate(zip(fnames, outputs, pids)):
            idx = i*imgs.shape[0] + j
            index[fname] = idx
            features[idx,:] = output.detach().cpu().numpy()
#            features[fname] = output
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

        sys.stdout.flush()

    return features, labels, index

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
        mars_dir = osp.join('MARS-evaluation', 'info')
        images_dir = 'data/mars/images'

        # Extract query & gallery features
#        features_raw, _, index = extract_features(self.model, data_loader)
        import pickle
#        pickle.dump(features_raw, open('features_raw.pkl','wb'))
#        print('save done!')
        features_raw = pickle.load(open('features_raw.pkl','rb'))
        index = OrderedDict()
        for i, fnames in enumerate(data_loader.dataset.dataset):
            for j, fname in enumerate(fnames):
                index[fname] = i*50 + j
        print('load done!')
        if True:
            print('Only for MARS dataset!')
            """ read from original data """
            orig_track_info = scipy.io.loadmat(osp.join(mars_dir, 'tracks_test_info.mat'))['track_test_info']
            orig_query_idx = scipy.io.loadmat(osp.join(mars_dir, 'query_IDX.mat'))['query_IDX']
            orig_test_list = [line.rstrip() for line in open(osp.join(mars_dir, 'test_name.txt'),'r').readlines()]

            orig_to_sym_dict = {}
            for v in query:
                origpath = osp.basename(osp.realpath(osp.join(images_dir, v[0])))
                orig_to_sym_dict[origpath] = v[0]

            query_track_info = orig_track_info[orig_query_idx-1,:].squeeze()
            query_track_iids = []
            for q in query_track_info:
                sympath = orig_to_sym_dict[osp.basename(orig_test_list[q[0]-1])]
                query_track_iids.append(int(sympath.split('_')[0] + sympath.split('_')[1] + sympath.split('_')[2]))

            """ track info """
            def parse_trackid(trackid):
                pid = int('{:010d}'.format(trackid)[:4])
                cam = int('{:010d}'.format(trackid)[4:6])
                return trackid, pid, cam

            """ extract track features """
            target = list(set(query) | set(gallery))
            track_ids = [int(v[0].split('_')[0]+v[0].split('_')[1]+v[0].split('_')[2]) for v in target]
            unique_track_iids = np.unique(track_ids)
            trackid_to_feats_dict = defaultdict(list)
            for track_id, v in zip(track_ids, target):
                trackid_to_feats_dict[track_id].append(features_raw[index[v[0]],:])
            features_track = {}
            for track_id in unique_track_iids:
                features_track[track_id] = torch.stack(trackid_to_feats_dict[track_id]).mean(dim=0)
                features_track[track_id] = features_track[track_id] / torch.sqrt((features_track[track_id]**2).sum())

            query_track = [parse_trackid(q) for q in query_track_iids]
            gallery_track = [parse_trackid(q) for q in unique_track_iids]

            # Overwrite
            features = features_track
            query = query_track
            gallery = gallery_track
            print(len(query))
            print(len(gallery))
        else:
            features = features_raw

        query_ids = [pid for _,pid,_ in query]
        query_cams = [cid for _,_,cid in query]
        gallery_ids = [pid for _,pid,_ in gallery]
        gallery_cams = [cid for _,_,cid in gallery]

#        feat_query = torch.cat([features[f].unsqueeze(0) for f,_,_ in query], 0)
#        feat_gallery = torch.cat([features[f].unsqueeze(0) for f,_,_ in gallery], 0)

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
