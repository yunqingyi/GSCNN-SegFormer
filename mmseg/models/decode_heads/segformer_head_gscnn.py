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

        print("\ninside of _AtrousSpatialPyramidPoolingModule")
        print("x.size ", x.size())              # torch.Size([1, 19, 512, 1024])
        print("edge.size ", edge.size())        # torch.Size([1, 1, 512, 1024])

        img_features = self.img_pooling(x)
        print("img_features.size() ", img_features.size())  # torch.Size([1, 19, 1, 1])
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
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, **kwargs):
        super(SegFormerGSCNNHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        # ------------------------------add gated SCNN layers to the MLP decoder------------------------------------
        # criterion = None
        # self.criterion = criterion

        wide_resnet = wider_resnet38_a2(classes=1000, dilation=True)
        wide_resnet = torch.nn.DataParallel(wide_resnet)
        
        wide_resnet = wide_resnet.module
        self.mod1 = wide_resnet.mod1
        self.mod2 = wide_resnet.mod2
        self.mod3 = wide_resnet.mod3
        self.mod4 = wide_resnet.mod4
        self.mod5 = wide_resnet.mod5
        self.mod6 = wide_resnet.mod6
        self.mod7 = wide_resnet.mod7
        self.pool2 = wide_resnet.pool2
        self.pool3 = wide_resnet.pool3
        self.interpolate = F.interpolate
        del wide_resnet

        self.dsn1 = nn.Conv2d(64, 1, 1)
        self.dsn3 = nn.Conv2d(256, 1, 1)
        self.dsn4 = nn.Conv2d(512, 1, 1)
        # self.dsn7 = nn.Conv2d(4096, 1, 1)

        self.res1 = Resnet.BasicBlock(int(c1_in_channels), int(c1_in_channels), stride=1, downsample=None)
        self.d1 = nn.Conv2d(int(c1_in_channels), 1, 1)
        self.d2 = nn.Conv2d(19, 512, 1, 1)

        # self.res2 = Resnet.BasicBlock(32, 32, stride=1, downsample=None)
        # self.d2 = nn.Conv2d(32, 16, 1)
        # self.res3 = Resnet.BasicBlock(16, 16, stride=1, downsample=None)
        # self.d3 = nn.Conv2d(16, 8, 1)
        self.fuse = nn.Conv2d(int(c1_in_channels/8), 1, kernel_size=1, padding=0, bias=False)

        self.cw = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)

        self.gate1 = gsc.GatedSpatialConv2d(32, 32)
        self.gate2 = gsc.GatedSpatialConv2d(16, 16)
        self.gate3 = gsc.GatedSpatialConv2d(8, 8)
         
        self.aspp = _AtrousSpatialPyramidPoolingModule(512, 256,
                                                       output_stride=8)

        self.bot_fine = nn.Conv2d(64, 48, kernel_size=1, bias=False)
        self.bot_aspp = nn.Conv2d(1280 + 256, 256, kernel_size=1, bias=False)

        self.final_seg = nn.Sequential(
            nn.Conv2d(560, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.num_classes, kernel_size=1, bias=False))

        self.sigmoid = nn.Sigmoid()
        initialize_weights(self.final_seg)
        # -------------------------------------------------------------------------------------------------------------

    def forward(self, inputs, img):
        # print(len(inputs))
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        # H*W*C1/4
        # H*W*C2/8
        # H*W*C3/16
        # H*W*C4/32
        c1, c2, c3, c4 = x

        # -----------------------------------add gated SCNN layers to the MLP decoder----------------------------------
        # N*C*H*W
        # [1, 32, 128, 256]
        # x_size = list(c1.size())
        # x_size[2] *= 4
        # x_size[3] *= 4
        x_size = img.size()
        # s1 = F.interpolate(self.dsn3(c1), x_size[2:], mode='bilinear', align_corners=True)
        # s2 = F.interpolate(self.dsn4(c2), x_size[2:], mode='bilinear', align_corners=True)
        # s3 = F.interpolate(self.dsn7(c3), x_size[2:], mode='bilinear', align_corners=True)
        
        # H*W*C1, this either enlarges or shrinks the image
        m1f = F.interpolate(c1, x_size[2:], mode='bilinear', align_corners=True)

        # edge detection stuff
        im_arr = img.cpu().numpy().transpose((0,2,3,1)).astype(np.uint8)
        # x_size[0], 1, x_size[2], x_size[3] = N, 1, H, W
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            canny[i] = cv2.Canny(im_arr[i],10,100)
        canny = torch.from_numpy(canny).cuda().float()

        # parallel GSCNN branch
        # H*W*C1 input dim, H*W*C1/2 output dim
        # [1, 32, 512, 1024]
        cs = self.res1(m1f)

        # cs = F.interpolate(cs, x_size[2:], mode='bilinear', align_corners=True)

        # [1, 1, 512, 1024]
        cs = self.d1(cs)

        # cs = self.gate1(cs, c4)
        # cs = self.res2(cs)
        # cs = F.interpolate(cs, x_size[2:], mode='bilinear', align_corners=True)
        # cs = self.d2(cs)
        # cs = self.gate2(cs, s4)
        # cs = self.res3(cs)
        # cs = F.interpolate(cs, x_size[2:], mode='bilinear', align_corners=True)
        # cs = self.d3(cs)
        # cs = self.gate3(cs, s7)
        # cs = self.fuse(cs)
        cs = F.interpolate(cs, x_size[2:], mode='bilinear', align_corners=True)

        # H*W*C1/2
        edge_out = self.sigmoid(cs)
        # concat H*W*C1/2 with N*1*H*W
        # print("edge_out.shape ", edge_out.shape)    # torch.Size([1, 1, 512, 1024])
        # print("canny.shape ", canny.shape)          # torch.Size([1, 1, 512, 1024])

        cat = torch.cat((edge_out, canny), dim=1)
        # print("cat.shape ", cat.shape)              # torch.Size([1, 2, 512, 1024])
        acts = self.cw(cat)
        acts = self.sigmoid(acts)
        # print("acts.size() ", acts.size())          # torch.Size([1, 1, 512, 1024])

        # -------------------------------------------------------------------------------------------------------------
        

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)
        # print("x.size() ", x.size()) # torch.Size([1, 19, 128, 256])
        
        # upsample? enlarge? from [1, 19, 128, 256] to [1, 19, 512, 1024]
        x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=True)
        x = self.d2(x)

        # return x

        # -------------------------------------------------------------------------------------------------------------
        # aspp
        # print("aspp")
        # print("x ", x.size())           # torch.Size([1, 19, 128, 256])
        # print("acts ", acts.size())     # torch.Size([1, 1, 512, 1024])

        # x = self.aspp(x, acts)
        dec0_up = x

        dec0_fine = self.bot_fine(c2)
        dec0_up = self.interpolate(dec0_up, c2.size()[2:], mode='bilinear',align_corners=True)
        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)

        dec1 = self.final_seg(dec0)  
        seg_out = self.interpolate(dec1, x_size[2:], mode='bilinear')            
        
        return seg_out
        # gts = None
        # if self.training:
        #     return self.criterion((seg_out, edge_out), gts)              
        # else:
        #     return seg_out, edge_out
        # -------------------------------------------------------------------------------------------------------------
        

    def forward_train(self, inputs, img, img_metas, gt_semantic_seg, train_cfg):
            """Forward function for training.
            Args:
                inputs (list[Tensor]): List of multi-level img features.
                img_metas (list[dict]): List of image info dict where each dict
                    has: 'img_shape', 'scale_factor', 'flip', and may also contain
                    'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                    For details on the values of these keys see
                    `mmseg/datasets/pipelines/formatting.py:Collect`.
                gt_semantic_seg (Tensor): Semantic segmentation masks
                    used if the architecture supports semantic segmentation task.
                train_cfg (dict): The training config.

            Returns:
                dict[str, Tensor]: a dictionary of loss components
            """
            seg_logits = self.forward(inputs, img)
            losses = self.losses(seg_logits, gt_semantic_seg)
            return losses