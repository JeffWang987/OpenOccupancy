import math
from functools import partial
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.runner import BaseModule

import torch
import torch.nn as nn
import torch.nn.functional as F

import spconv.pytorch as spconv
from spconv.pytorch import functional as Fsp

from mmdet3d.models.builder import MIDDLE_ENCODERS

import copy

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_cfg=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        build_norm_layer(norm_cfg, out_channels)[1],
        nn.ReLU(inplace=True),
    )

    return m



class SparseBasicBlock(spconv.SparseModule):

    def __init__(self, inplanes, planes, stride=1, norm_cfg=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        self.net = spconv.SparseSequential(
            spconv.SubMConv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, indice_key=indice_key),
            build_norm_layer(norm_cfg, planes)[1],
            nn.ReLU(inplace=True),
            spconv.SubMConv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, indice_key=indice_key),
            build_norm_layer(norm_cfg, planes)[1],
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.net(x)
        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))

        return out



@MIDDLE_ENCODERS.register_module()
class SparseLiDAREnc4x(nn.Module):
    def __init__(self, input_channel, norm_cfg, base_channel, out_channel, 
                sparse_shape_xyz, **kwargs):
        super().__init__()

        block = post_act_block
        self.sparse_shape_xyz = sparse_shape_xyz

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channel, base_channel, 3),
            nn.GroupNorm(16, base_channel),
            nn.ReLU(inplace=True))

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(base_channel, base_channel, norm_cfg=norm_cfg, indice_key='res1'),
            SparseBasicBlock(base_channel, base_channel, norm_cfg=norm_cfg, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            block(base_channel, base_channel*2, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, indice_key='res2'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            block(base_channel*2, base_channel*4, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(base_channel*4, base_channel*4, norm_cfg=norm_cfg, indice_key='res3'),
            SparseBasicBlock(base_channel*4, base_channel*4, norm_cfg=norm_cfg, indice_key='res3'),
        )

        self.conv_out = spconv.SparseSequential(
            spconv.SubMConv3d(base_channel*4, out_channel, 3),
            nn.GroupNorm(16, out_channel),
            nn.ReLU(inplace=True))



    def forward(self, voxel_features, coors, batch_size):
        # spconv encoding
        coors = coors.int()  # zyx 与 points的输入(xyz)相反
        # FIXME bs=1 hardcode
        input_sp_tensor = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape_xyz[::-1], batch_size)
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        
        x = self.conv_out(x_conv3)

        return {'x': x.dense().permute(0,1,4,3,2), # B, C, W, H, D 
                'pts_feats': [x]}





@MIDDLE_ENCODERS.register_module()
class SparseLiDAREnc8x(nn.Module):
    def __init__(self, input_channel, norm_cfg, base_channel, out_channel, 
                sparse_shape_xyz, **kwargs):
        super().__init__()

        block = post_act_block
        self.sparse_shape_xyz = sparse_shape_xyz

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channel, base_channel, 3),
            nn.GroupNorm(16, base_channel),
            nn.ReLU(inplace=True))

        self.conv1 = spconv.SparseSequential(
            block(base_channel, base_channel*2, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv1', conv_type='spconv'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, indice_key='res1'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            block(base_channel*2, base_channel*4, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(base_channel*4, base_channel*4, norm_cfg=norm_cfg, indice_key='res2'),
            SparseBasicBlock(base_channel*4, base_channel*4, norm_cfg=norm_cfg, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            block(base_channel*4, base_channel*8, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(base_channel*8, base_channel*8, norm_cfg=norm_cfg, indice_key='res3'),
            SparseBasicBlock(base_channel*8, base_channel*8, norm_cfg=norm_cfg, indice_key='res3'),
        )

        self.conv_out = spconv.SparseSequential(
            spconv.SubMConv3d(base_channel*8, out_channel, 3),
            nn.GroupNorm(16, out_channel),
            nn.ReLU(inplace=True))



    def forward(self, voxel_features, coors, batch_size):
        # spconv encoding
        coors = coors.int()  # zyx 与 points的输入(xyz)相反
        # FIXME bs=1 hardcode
        input_sp_tensor = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape_xyz[::-1], batch_size)
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        
        x = self.conv_out(x_conv3)

        return {'x': x.dense().permute(0,1,4,3,2), # B, C, W, H, D 
                'pts_feats': [x]}

