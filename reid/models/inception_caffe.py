"""
Original code from: https://github.com/antspy/inception_v1.pytorch
Hacked by Yumin Suh (n12345@snu.ac.kr)
"""

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Inception_base(nn.Module):
    def __init__(self, depth_dim, input_size, config, dilation=1):
        super(Inception_base, self).__init__()

        self.depth_dim = depth_dim

        self._1x1 = nn.Conv2d(input_size, out_channels=config[0][0], kernel_size=1, stride=1, padding=0)
        self._3x3_reduce = nn.Conv2d(input_size, out_channels=config[1][0], kernel_size=1, stride=1, padding=0)
        self._3x3 = nn.Conv2d(config[1][0], config[1][1], kernel_size=3, stride=1, padding=1*dilation, dilation=dilation)
        self._5x5_reduce = nn.Conv2d(input_size, out_channels=config[2][0], kernel_size=1, stride=1, padding=0)
        self._5x5 = nn.Conv2d(config[2][0], config[2][1], kernel_size=5, stride=1, padding=2*dilation, dilation=dilation)

        self.max_pool_1 = nn.MaxPool2d(kernel_size=config[3][0], stride=1, padding=1*dilation)
        self.pool_proj = nn.Conv2d(input_size, out_channels=config[3][1], kernel_size=1, stride=1, padding=0)

    def forward(self, input):

        output1 = F.relu(self._1x1(input))

        output2 = F.relu(self._3x3_reduce(input))
        output2 = F.relu(self._3x3(output2))

        output3 = F.relu(self._5x5_reduce(input))
        output3 = F.relu(self._5x5(output3))

        output4 = F.relu(self.pool_proj(self.max_pool_1(input)))

        return torch.cat([output1, output2, output3, output4], dim=self.depth_dim)

class Inception_v1(nn.Module):
    def __init__(self, num_features=512, dilation=1, initialize=True, feat_type='4e'):
        super(Inception_v1, self).__init__()
        self.dilation = dilation
        self.pretrained = os.environ['INCEPTION_V1_PRETRAINED']
        self.feat_type = feat_type

        #conv2d0
        self.conv1__7x7_s2 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.lrn1 = nn.CrossMapLRN2d(5, 0.0001, 0.75, 1)

        #conv2d1
        self.conv2__3x3_reduce = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)

        #conv2d2
        self.conv2__3x3  = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.lrn3 = nn.CrossMapLRN2d(5, 0.0001, 0.75, 1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.inception_3a = Inception_base(1, 192, [[64], [96,128], [16, 32], [3, 32]]) #3a
        self.inception_3b = Inception_base(1, 256, [[128], [128,192], [32, 96], [3, 64]]) #3b
        if self.dilation == 1:
            self.max_pool_inc3= nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        pool_kernel = self.dilation*2 + 1
        self.inception_4a = Inception_base(1, 480, [[192], [ 96,208], [16, 48], [pool_kernel, 64]], dilation=self.dilation) #4a
        self.inception_4b = Inception_base(1, 512, [[160], [112,224], [24, 64], [pool_kernel, 64]], dilation=self.dilation) #4b
        self.inception_4c = Inception_base(1, 512, [[128], [128,256], [24, 64], [pool_kernel, 64]], dilation=self.dilation) #4c
        self.inception_4d = Inception_base(1, 512, [[112], [144,288], [32, 64], [pool_kernel, 64]], dilation=self.dilation) #4d
        self.inception_4e = Inception_base(1, 528, [[256], [160,320], [32,128], [pool_kernel,128]], dilation=self.dilation) #4e

        if self.feat_type == '5b':
            assert self.dilation == 1
            pool_kernel = self.dilation*2 + 1

            self.inception_5a = Inception_base(1, 832, [[256], [160,320], [32,128], [pool_kernel,128]], dilation=self.dilation) #5a
            self.inception_5b = Inception_base(1, 832, [[384], [192,384], [48,128], [pool_kernel,128]], dilation=self.dilation) #5b
            self.input_feat = nn.Conv2d(1024, num_features, (1,1), (1,1), (0,0))
        else:
            self.input_feat = nn.Conv2d(832, num_features, (1,1), (1,1), (0,0))
            self.bn = nn.BatchNorm2d(num_features, affine=False)

        if initialize:
            self.init_pretrained()

    def forward(self, input):

        output = self.max_pool1(F.relu(self.conv1__7x7_s2(input)))
        output = self.lrn1(output)

        output = F.relu(self.conv2__3x3_reduce(output))
        output = F.relu(self.conv2__3x3(output))
        output = self.max_pool3(self.lrn3(output))

        output = self.inception_3a(output)
        output = self.inception_3b(output)
        if self.dilation == 1:
            output = self.max_pool_inc3(output)

        output = self.inception_4a(output)
        output = self.inception_4b(output)
        output = self.inception_4c(output)
        output = self.inception_4d(output)
        output = self.inception_4e(output)

        if self.feat_type == '5b':
            output = self.inception_5a(output)
            output = self.inception_5b(output)
            output = nn.AdaptiveAvgPool2d(1)(output)
            output = self.input_feat(output)
            output = F.normalize(output, dim=1)
            output = output.squeeze()
        else:
            output = self.input_feat(output)
            output = self.bn(output)

        return output

    def init_pretrained(self):
        state_dict = torch.load(self.pretrained)
        state_dict['input_feat.weight'] = nn.init.xavier_uniform_(self.input_feat.weight).detach()
        state_dict['input_feat.bias'] = torch.zeros(self.input_feat.bias.size())
        if not self.feat_type =='5b':
            #TODO
            state_dict['bn.running_mean'] = self.bn.running_mean
            state_dict['bn.running_var'] = self.bn.running_var

        model_dict = {}
        for k,v in state_dict.items():
            for l,p in self.state_dict().items():
                if k.replace("/",".") in l.replace("__",".").replace("._","."):
                    model_dict[l] = torch.from_numpy(np.array(v)).view_as(p) 

#        self.load_state_dict(model_dict, strict=False)
        self.load_state_dict(model_dict, strict=True)
        """
        print('inception v1 pretrained model loaded!')
        print('difference:--------------------------')
        print(set(self.state_dict().keys()).symmetric_difference(model_dict.keys()))
        print('inception v1 pretrained model loaded!')
        """

    def save_dict(self):
        return self.state_dict()

    def load(self, checkpoint):
        checkpoint.pop('epoch')
        if torch.__version__=='0.4.0':
            checkpoint.pop('bn.num_batches_tracked',None)
        self.load_state_dict(checkpoint)

def inception_v1(features=512, use_relu=False, dilation=1, initialize=True):
    model = Inception_v1(num_features=features, feat_type='5b', dilation=dilation, initialize=initialize)
    return model
