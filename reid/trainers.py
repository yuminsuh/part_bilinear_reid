from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable

from .loss import TripletLoss
from .utils.meters import AverageMeter


class BaseTrainer(object):
    def __init__(self, model, criterion):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        num_effs= AverageMeter()
        num_alls = AverageMeter()
        eff_ratios = AverageMeter()

        end = time.time()

        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
#            pdb.set_trace()
            loss, prec1, num_eff, num_all = self._forward(inputs, targets)

            losses.update(loss.data[0], targets.size(0))
            precisions.update(prec1, targets.size(0))
            num_effs.update(float(num_eff), targets.size(0))
            num_alls.update(float(num_all), targets.size(0))
            eff_ratios.update(float(num_eff)/num_all, targets.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      'nEff {:.2f} ({:.2f})\t'
                      'nAll {:.2f} ({:.2f})\t'
                      'effRatio {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg,
                              num_effs.val, num_effs.avg,
                              num_alls.val, num_alls.avg,
                              eff_ratios.val, eff_ratios.avg))

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs).cuda()]
        targets = Variable(pids.cuda())
        return inputs, targets


    def _forward(self, inputs, targets):
        outputs = self.model(*inputs)
        loss, prec, num_eff, num_all = self.criterion(outputs, targets)
        return loss, prec, num_eff, num_all
