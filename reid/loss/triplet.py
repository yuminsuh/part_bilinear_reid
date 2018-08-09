from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

import torch.nn.functional as F

def pdist(A, B = None, squared = False, eps = 1e-4):
    B = B if B is not None else A
    prod = torch.mm(A, B.t())
    normA = (A * A).sum(1).unsqueeze(1)
    normB = (B * B).sum(1).unsqueeze(1)
    D = (normA.expand_as(prod) + normB.t().expand_as(prod) - 2 * prod).clamp(min = 0)
    return D if squared else D.clamp(min = eps).sqrt().clamp(min = eps)

class TripletLoss(nn.Module):
    def __init__(self, margin=0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        d = pdist(inputs, squared=True)
#        print(d)
        # pos: binary indicator. if labels are same, 1 except for itself
        pos = torch.eq(*[targets.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(d) - torch.autograd.Variable(torch.eye(len(d))).type_as(d)
        neg = 1- torch.eq(*[targets.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(d)
        T = d.unsqueeze(1).expand(*(len(d),) * 3)
        M = pos.unsqueeze(1).expand_as(T) * neg.unsqueeze(2).expand_as(T)
        allloss = M * F.relu(T - T.transpose(1,2) + self.margin)
        allloss = allloss.cpu().detach().numpy()
        num_eff = (allloss>0).sum()
        num_wrong = (M * F.relu(T - T.transpose(1,2)) >0).sum()
        if num_eff > 0:
            loss = (M * F.relu(T - T.transpose(1,2) + self.margin)).sum() / float(num_eff) # same with caffe
        else:
            loss = Variable(torch.zeros(1), requires_grad=True).cuda()
        prec = 1.0 - float(num_wrong)/ M.sum()
        return loss, prec, num_eff, M.sum()
