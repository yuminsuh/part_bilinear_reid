from __future__ import absolute_import
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import pairwise_distances

from ..utils import to_numpy

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
