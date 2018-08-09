import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from reid.models.inception_caffe import Inception_v1
from reid.models.CompactBilinearPooling_dsybaik import CompactBilinearPooling

class CPM(nn.Module):
    def __init__(self, depth_dim, num_features_part=128, use_relu=False, dilation=1):
        super(CPM, self).__init__()

        self.depth_dim = depth_dim
        self.use_relu = use_relu
        self.dilation = dilation

        self.conv1_1 = nn.Conv2d(3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1_stage1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

        self.conv2_1 = nn.Conv2d(64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2_stage1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

        self.conv3_1 = nn.Conv2d(128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_4 = nn.Conv2d(256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool3_stage1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

        self.conv4_1 = nn.Conv2d(256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_3_CPM = nn.Conv2d(512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4_4_CPM = nn.Conv2d(256, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Stage1
        # limbs
        self.conv5_1_CPM_L1 = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5_2_CPM_L1 = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5_3_CPM_L1 = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5_4_CPM_L1 = nn.Conv2d(128, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv5_5_CPM_L1 = nn.Conv2d(512, out_channels=38, kernel_size=1, stride=1, padding=0)
        # joints
        self.conv5_1_CPM_L2 = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5_2_CPM_L2 = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5_3_CPM_L2 = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5_4_CPM_L2 = nn.Conv2d(128, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv5_5_CPM_L2 = nn.Conv2d(512, out_channels=19, kernel_size=1, stride=1, padding=0)

        if self.dilation == 1:
            self.concat_stage3_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

        # Stage2
        # limbs
        self.Mconv1_stage2_L1 = nn.Conv2d(185, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv2_stage2_L1 = nn.Conv2d(128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv3_stage2_L1 = nn.Conv2d(128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv4_stage2_L1 = nn.Conv2d(128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv5_stage2_L1 = nn.Conv2d(128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv6_stage2_L1 = nn.Conv2d(128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.Mconv7_stage2_L1 = nn.Conv2d(128, out_channels=38, kernel_size=1, stride=1, padding=0)
        # joints
        self.Mconv1_stage2_L2 = nn.Conv2d(185, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv2_stage2_L2 = nn.Conv2d(128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv3_stage2_L2 = nn.Conv2d(128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv4_stage2_L2 = nn.Conv2d(128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv5_stage2_L2 = nn.Conv2d(128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv6_stage2_L2 = nn.Conv2d(128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.Mconv7_stage2_L2 = nn.Conv2d(128, out_channels=19, kernel_size=1, stride=1, padding=0)

        self.pose1 = nn.Conv2d(185, out_channels=num_features_part, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features_part, affine=False)

    def forward(self, inputs):
        # data rescale
        output1 = 0.0039 * inputs

        output1 = F.relu(self.conv1_1(output1))
        output1 = F.relu(self.conv1_2(output1))
        output1 = self.pool1_stage1(output1)

        output1 = F.relu(self.conv2_1(output1))
        output1 = F.relu(self.conv2_2(output1))
        output1 = self.pool2_stage1(output1)

        output1 = F.relu(self.conv3_1(output1))
        output1 = F.relu(self.conv3_2(output1))
        output1 = F.relu(self.conv3_3(output1))
        output1 = F.relu(self.conv3_4(output1))
        output1 = self.pool3_stage1(output1)

        output1 = F.relu(self.conv4_1(output1))
        output1 = F.relu(self.conv4_2(output1))
        output1 = F.relu(self.conv4_3_CPM(output1))
        output1 = F.relu(self.conv4_4_CPM(output1))

        output2_1 = F.relu(self.conv5_1_CPM_L1(output1))
        output2_1 = F.relu(self.conv5_2_CPM_L1(output2_1))
        output2_1 = F.relu(self.conv5_3_CPM_L1(output2_1))
        output2_1 = F.relu(self.conv5_4_CPM_L1(output2_1))
        output2_1 = self.conv5_5_CPM_L1(output2_1)

        output2_2 = F.relu(self.conv5_1_CPM_L2(output1))
        output2_2 = F.relu(self.conv5_2_CPM_L2(output2_2))
        output2_2 = F.relu(self.conv5_3_CPM_L2(output2_2))
        output2_2 = F.relu(self.conv5_4_CPM_L2(output2_2))
        output2_2 = self.conv5_5_CPM_L2(output2_2)

        output2 = torch.cat([output2_1, output2_2, output1], dim=self.depth_dim)

        output3_1 = F.relu(self.Mconv1_stage2_L1(output2))
        output3_1 = F.relu(self.Mconv2_stage2_L1(output3_1))
        output3_1 = F.relu(self.Mconv3_stage2_L1(output3_1))
        output3_1 = F.relu(self.Mconv4_stage2_L1(output3_1))
        output3_1 = F.relu(self.Mconv5_stage2_L1(output3_1))
        output3_1 = F.relu(self.Mconv6_stage2_L1(output3_1))
        output3_1 = self.Mconv7_stage2_L1(output3_1)

        output3_2 = F.relu(self.Mconv1_stage2_L2(output2))
        output3_2 = F.relu(self.Mconv2_stage2_L2(output3_2))
        output3_2 = F.relu(self.Mconv3_stage2_L2(output3_2))
        output3_2 = F.relu(self.Mconv4_stage2_L2(output3_2))
        output3_2 = F.relu(self.Mconv5_stage2_L2(output3_2))
        output3_2 = F.relu(self.Mconv6_stage2_L2(output3_2))
        output3_2 = self.Mconv7_stage2_L2(output3_2)

        output3 = torch.cat([output3_1, output3_2, output1], dim=self.depth_dim)
        if self.dilation == 1:
            output3 = self.concat_stage3_pool(output3)
        output3 = self.pose1(output3)
        output3 = self.bn(output3)
        if self.use_relu:
            output3 = F.relu(output3)

        return output3

class Bilinear_Pooling(nn.Module):

    def __init__(self, num_feat1=512, num_feat2=128, num_feat_out=512):
        super(Bilinear_Pooling, self).__init__()
        self.num_feat1 = num_feat1
        self.num_feat2 = num_feat2
        self.num_feat_out = num_feat_out

        self.cbp = CompactBilinearPooling(self.num_feat1, self.num_feat2, self.num_feat_out, sum_pool=True)

    def forward(self, input1, input2):
        assert(self.num_feat1 == input1.shape[1])
        assert(self.num_feat2 == input2.shape[1])
        output = self.cbp(input1, input2)
        output = F.normalize(output, dim=1)
        output = output.squeeze()

        return output

class Inception_v1_cpm(nn.Module):
    def __init__(self, num_features=512, use_bn=True, use_relu=False, dilation=1):
        super(Inception_v1_cpm, self).__init__()
        num_features_app, num_features_part = (512, 128)
        self.app_feat_extractor = Inception_v1(num_features=num_features_app, dilation=dilation)
        self.part_feat_extractor = CPM(1, num_features_part=num_features_part, dilation=dilation)
        self.pooling = Bilinear_Pooling(num_feat1=num_features_app, num_feat2=num_features_part, num_feat_out=num_features)

    def forward(self, inputs):
        output_app = self.app_feat_extractor(inputs)
        output_part = self.part_feat_extractor(inputs)
        return self.pooling(output_app, output_part)

def inception_v1_cpm_pretrained(features=512, use_relu=False, dilation=1):
    model = Inception_v1_cpm(num_features=features, use_relu=use_relu, dilation=dilation)

    app_stream_pretrained_weight = os.environ['INCEPTION_V1_PRETRAINED']
    part_stream_pretrained_weight = os.environ['CPM_PRETRAINED']
    #
    state_dict = torch.load(app_stream_pretrained_weight)
    state_dict = {'app_feat_extractor.'+k.replace("app_feat_extractor.",""):v for k,v in state_dict.items()}
    part_dict = torch.load(part_stream_pretrained_weight)
    part_dict = {'part_feat_extractor.'+k.replace("part_feat_extractor.","").replace("app_feat_extractor.","").replace("feat_extractor.","."):v for k,v in part_dict.items()}
    state_dict.update(part_dict)
    state_dict['app_feat_extractor.input_feat.weight'] = nn.init.xavier_uniform_(model.app_feat_extractor.input_feat.weight).detach()
    state_dict['app_feat_extractor.input_feat.bias'] = torch.zeros(model.app_feat_extractor.input_feat.bias.size())
    state_dict['part_feat_extractor.pose1.weight'] = nn.init.xavier_uniform_(model.part_feat_extractor.pose1.weight).detach()
    state_dict['part_feat_extractor.pose1.bias'] = torch.zeros(model.part_feat_extractor.pose1.bias.size())
    #TODO
    state_dict['app_feat_extractor.bn.running_mean'] = model.app_feat_extractor.bn.running_mean
    state_dict['app_feat_extractor.bn.running_var'] = model.app_feat_extractor.bn.running_var
    state_dict['part_feat_extractor.bn.running_mean'] = model.part_feat_extractor.bn.running_mean
    state_dict['part_feat_extractor.bn.running_var'] = model.part_feat_extractor.bn.running_var

    model_dict = {l: torch.from_numpy(np.array(v)).view_as(p) for k,v in state_dict.items() for l,p in model.state_dict().items() if k.replace("/",".").replace("__",".").replace("._",".") in l.replace("__",".").replace("._",".")}
    model.load_state_dict(model_dict, strict=False)

    model.app_feat_extractor = nn.DataParallel(model.app_feat_extractor.cuda())
    model.part_feat_extractor = nn.DataParallel(model.part_feat_extractor.cuda())
    model.pooling = nn.DataParallel(model.pooling.cuda())

    return model
