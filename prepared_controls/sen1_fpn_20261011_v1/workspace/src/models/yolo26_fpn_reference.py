# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""One prespecified additive top-down output reference; no width/val search."""
import torch
from torch import nn
from torch.nn import functional as F
from .yolo26_unet_decoder import Yolo26UNetDecoder
from .yolo26_semantic import _infer_feature_channels


class Yolo26FPNReference(Yolo26UNetDecoder):
    def __init__(self, *, common_channels=16, num_classes=2):
        super().__init__(common_channels=common_channels,num_classes=num_classes,decoder_channels=32)
        # Remove all replaced decoder parameters, rather than register unused padding.
        for name in ('p3_lateral','p2_lateral','p1_lateral','input_lateral','up_to_p2',
                     'up_to_p1','up_to_input','fuse_p2','fuse_p1','fuse_input','decoder_head'):
            delattr(self,name)
        channels=_infer_feature_channels(self.core,(0,2,16))
        # 36*w*w + 174*w + 2 parameters with this actual feature graph.
        # Nearest integer width to the historical 47,810-parameter output path.
        width=34
        self.fpn_laterals=nn.ModuleDict({k:nn.Conv2d(c,width,1) for k,c in
            [('neck_p3',channels[16]),('p2',channels[2]),('p1',channels[0]),('input',common_channels)]})
        self.fpn_smooth=nn.ModuleDict({k:nn.Sequential(nn.Conv2d(width,width,3,padding=1,bias=False),
             nn.BatchNorm2d(width),nn.SiLU()) for k in ('neck_p3','p2','p1','input')})
        self.fpn_head=nn.Conv2d(width,num_classes,1)

    def forward_native(self,image,sensor):
        name=self._sensor_name(sensor)
        common=self.input_adapters(image,name)
        features=self._forward_pyramid(common,name); features['input']=common
        x=self.fpn_smooth['neck_p3'](self.fpn_laterals['neck_p3'](features['neck_p3']))
        for key in ('p2','p1','input'):
            lateral=self.fpn_laterals[key](features[key])
            x=self.fpn_smooth[key](F.interpolate(x,size=lateral.shape[-2:],mode='bilinear',align_corners=False)+lateral)
        return self.fpn_head(x)
