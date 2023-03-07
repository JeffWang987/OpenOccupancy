import random
from typing import List

import torch
from torch import nn

from mmdet3d.models.builder import FUSION_LAYERS


@FUSION_LAYERS.register_module()
class ConvFuser(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.occ_enc = nn.Sequential(
            nn.Conv3d(in_channels*2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(True),
        )
    def forward(self, img_voxel_feats, pts_voxel_feats):
        return self.occ_enc(torch.cat([img_voxel_feats, pts_voxel_feats], dim=1))
