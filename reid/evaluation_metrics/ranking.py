from __future__ import absolute_import
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import pairwise_distances

from ..utils import to_numpy

def _unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=np.bool)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask


def cmc(distmat, query_ids=None, gallery_ids=None,
        query_cams=None, gallery_cams=None, topk=100,
        separate_camera_set=False,
        single_gallery_shot=False,
        first_match_break=False):
    distmat = to_numpy(distmat)
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute CMC for each query
    ret = np.zeros(topk)
    num_valid_queries = 0
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        if separate_camera_set:
            # Filter out samples from same camera
            valid &= (gallery_cams[indices[i]] != query_cams[i])
        if not np.any(matches[i, valid]): continue
        if single_gallery_shot:
            repeat = 10
            gids = gallery_ids[indices[i][valid]]
            inds = np.where(valid)[0]
            ids_dict = defaultdict(list)
            for j, x in zip(inds, gids):
                ids_dict[x].append(j)
        else:
            repeat = 1
        for _ in range(repeat):
            if single_gallery_shot:
                # Randomly choose one instance for each id
                sampled = (valid & _unique_sample(ids_dict, len(valid)))
                index = np.nonzero(matches[i, sampled])[0]
            else:
                index = np.nonzero(matches[i, valid])[0]
            delta = 1. / (len(index) * repeat)
            for j, k in enumerate(index):
                if k - j >= topk: break
                if first_match_break:
                    ret[k - j] += 1
                    break
                ret[k - j] += delta
        num_valid_queries += 1
    if num_valid_queries == 0:
        raise RuntimeError("No valid query")
    return ret.cumsum() / num_valid_queries


def mean_ap(distmat, query_ids=None, gallery_ids=None,
            query_cams=None, gallery_cams=None):
    distmat = to_numpy(distmat)
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute AP for each query
    aps = []
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        y_true = matches[i, valid]
        y_score = -distmat[i][indices[i]][valid]
        if not np.any(y_true): continue
        aps.append(average_precision_score(y_true, y_score))
    if len(aps) == 0:
        raise RuntimeError("No valid query")
    return np.mean(aps)

def cmc_small(distmat, query_ids=None, gallery_ids=None,
        query_cams=None, gallery_cams=None, topk=100):
    distmat = to_numpy(distmat)
    num_query = distmat.shape[0]
    # Compute CMC, AP for each query
    ret = np.zeros(topk)
    aps = []
    num_valid_queries = 0
    for i in range(num_query):
        dists = distmat[i,:].squeeze()
        tmp = np.argpartition(dists, topk)[:topk]
        indices = tmp[np.argsort(dists[tmp])]
        matches = (gallery_ids[indices] == query_ids[i])
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices] != query_ids[i]) |
                 (gallery_cams[indices] != query_cams[i]))
        y_true = matches[valid]
        y_score = -dists[indices[valid]]
        if not np.any(y_true): continue
        # CMC
        index = np.nonzero(y_true)[0][0]
        if index < topk:
            ret[index] += 1
        # AP
        aps.append(average_precision_score(y_true, y_score))
        num_valid_queries += 1
    #
    if num_valid_queries == 0:
        raise RuntimeError("No valid query")
    print("num_valid_queries: {}".format(num_valid_queries))
    print("num_query: {}".format(num_query))

    return ret.cumsum() / num_query, np.asarray(aps).sum() / num_query

def cmc_meanap_fast(feat_query, feat_gallery,
                    query_ids=None, gallery_ids=None,
                    query_cams=None, gallery_cams=None,
                    topk=100):
    feat_query = to_numpy(feat_query)
    feat_gallery = to_numpy(feat_gallery)
    print(feat_query.shape)
    print(feat_gallery.shape)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    #
    num_query, num_subquery = feat_query.shape[0], 10000
    topk = min(topk, len(gallery_ids)-1)

    # Compute CMC, AP for each query
    cmcs, aps = [], []
    for i_start in range(0, num_query, num_subquery):
        print(i_start)
        i_end = min(i_start+num_subquery, num_query)
        distmat = pairwise_distances(feat_query[i_start:i_end,:], feat_gallery)
        print(distmat.shape)
        cmc_now, ap_now = cmc_small(distmat, query_ids[i_start:i_end], gallery_ids,
                                            query_cams[i_start:i_end], gallery_cams,
                                            topk=topk)
        cmcs += [cmc_now * np.floor(i_end-i_start) / num_query]
        aps += [ap_now * np.floor(i_end-i_start) / num_query]

    return np.array(cmcs).sum(axis=0), np.array(aps).sum()
