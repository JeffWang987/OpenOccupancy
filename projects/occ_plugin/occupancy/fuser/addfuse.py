import random
from typing import List

import torch
from torch import nn

from mmdet3d.models.builder import FUSION_LAYERS


@FUSION_LAYERS.register_module()
class AddFuser(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, input_modality=None) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        if input_modality == None:
            input_modality = dict(
                use_lidar=True,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False)
        self.use_lidar = input_modality['use_lidar']
        self.use_img = input_modality['use_camera']

        if self.use_img:
            self.img_enc = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(True),
            )
        if self.use_lidar:
            self.pts_enc = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(True),
            )

    def forward(self, img_voxel_feats, pts_voxel_feats):
        features = []
        if self.use_img:
            img_voxel_feats = self.img_enc(img_voxel_feats)
            features.append(img_voxel_feats)
        if self.use_lidar:
            pts_voxel_feats = self.pts_enc(pts_voxel_feats)
            features.append(pts_voxel_feats)

        weights = [1] * len(features)
        if self.training and random.random() < self.dropout:
            index = random.randint(0, len(features) - 1)
            weights[index] = 0

        return sum(w * f for w, f in zip(weights, features)) / sum(weights)
