# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.models.utils import *
import attr

import torch
import torch.nn.functional as F
from torch import nn
from GSCNN import network
from GSCNN.network.wider_resnet import wider_resnet38_a2
from GSCNN.network import SEresnext
from GSCNN.network import Resnet
from GSCNN.network.wider_resnet import wider_resnet38_a2
from GSCNN.config import cfg
from GSCNN.network.mynn import initialize_weights, Norm2d
from torch.autograd import Variable

from GSCNN.my_functionals import GatedSpatialConv as gsc

import cv2

from IPython import embed

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
    

class _AtrousSpatialPyramidPoolingModule(nn.Module):
    '''
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    '''

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=[6, 12, 18]):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          Norm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                Norm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            Norm2d(reduction_dim), nn.ReLU(inplace=True))
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, reduction_dim, kernel_size=1, bias=False),
            Norm2d(reduction_dim), nn.ReLU(inplace=True))
         

    def forward(self, x, edge):
        x_size = x.size()

        # print("\ninside of _AtrousSpatialPyramidPoolingModule")
        # print("x.size ", x.size())              # torch.Size([1, 19, 512, 1024])
        # print("edge.size ", edge.size())        # torch.Size([1, 1, 512, 1024])

        img_features = self.img_pooling(x)
        # print("img_features.size() ", img_features.size())  # torch.Size([1, 19, 1, 1])
        img_features = self.img_conv(img_features)
        img_features = F.interpolate(img_features, x_size[2:],
                                     mode='bilinear',align_corners=True)
        out = img_features

        edge_features = F.interpolate(edge, x_size[2:],
                                      mode='bilinear',align_corners=True)
        edge_features = self.edge_conv(edge_features)
        out = torch.cat((out, edge_features), 1)

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


@HEADS.register_module()
class SegFormerGSCNNHead(BaseDecodeHead):
    def __init__(self, feature_strides, **kwargs):
        super(SegFormerGSCNNHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        embedding_dim = kwargs['decoder_params']['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.gated_conv_c4 = gsc.GatedSpatialConv2d(c4_in_channels, embedding_dim, kernel_size=3, padding=1)
        self.gated_conv_c3 = gsc.GatedSpatialConv2d(c3_in_channels, embedding_dim, kernel_size=3, padding=1)
        self.gated_conv_c2 = gsc.GatedSpatialConv2d(c2_in_channels, embedding_dim, kernel_size=3, padding=1)
        self.gated_conv_c1 = gsc.GatedSpatialConv2d(c1_in_channels, embedding_dim, kernel_size=3, padding=1)

        self.dsn1 = nn.Conv2d(c4_in_channels, 1, 1)
        self.dsn2 = nn.Conv2d(c3_in_channels, 1, 1)
        self.dsn3 = nn.Conv2d(c2_in_channels, 1, 1)
        self.dsn4 = nn.Conv2d(c1_in_channels, 1, 1)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4*2,  # Multiplied by 2 for MLP and Gated outputs
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, for different scales
        c1, c2, c3, c4 = x

        n, _, h, w = c4.shape

        # Process with MLP
        _c4_mlp = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c3_mlp = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c2_mlp = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c1_mlp = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        # Process with GatedSpatialConv2d
        # print("c4.size() ", c4.size())  #torch.Size([1, 256, 16, 32])
        # print("c3.size() ", c3.size())  #torch.Size([1, 160, 32, 64])
        # print("c2.size() ", c2.size())  #torch.Size([1, 64, 64, 128])
        # print("c1.size() ", c1.size())  #torch.Size([1, 32, 128, 256])

        c4_resize = resize(c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
        c4_resize_1channel = self.dsn1(c4_resize)
        c3_resize = resize(c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
        c3_resize_1channel = self.dsn2(c3_resize)
        c2_resize = resize(c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
        c2_resize_1channel = self.dsn3(c2_resize)
        c1_resize = resize(c1, size=c1.size()[2:], mode='bilinear', align_corners=False)
        c1_resize_1channel = self.dsn4(c1_resize)

        _c4_gated = self.gated_conv_c4(c4_resize, c3_resize_1channel)  # Here using c4 as both input and gating features, adapt as 
        _c3_gated = self.gated_conv_c3(c3_resize, c2_resize_1channel)
        _c2_gated = self.gated_conv_c2(c2_resize, c1_resize_1channel)
        _c1_gated = self.gated_conv_c1(c1, c1_resize_1channel)

        # Resize and combine MLP and Gated features
        _c4 = torch.cat([resize(_c4_mlp, size=c1.size()[2:], mode='bilinear', align_corners=False), 
                         resize(_c4_gated, size=c1.size()[2:], mode='bilinear', align_corners=False)], dim=1)
        _c3 = torch.cat([resize(_c3_mlp, size=c1.size()[2:], mode='bilinear', align_corners=False),
                         resize(_c3_gated, size=c1.size()[2:], mode='bilinear', align_corners=False)], dim=1)
        _c2 = torch.cat([resize(_c2_mlp, size=c1.size()[2:], mode='bilinear', align_corners=False),
                         resize(_c2_gated, size=c1.size()[2:], mode='bilinear', align_corners=False)], dim=1)
        _c1 = torch.cat([_c1_mlp, _c1_gated], dim=1)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x