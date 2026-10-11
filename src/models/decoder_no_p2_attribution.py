# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Separate no-P2 structural deletion; frozen nine-arm sources stay unchanged."""
import copy
import torch
from torch import nn
from .decoder_attribution import DecoderAttribution as OriginalAttribution

ARMS=('full','no_p2')

class DecoderAttribution(OriginalAttribution):
    def __init__(self,base,arm):
        if arm not in ARMS:raise ValueError(arm)
        super().__init__(base,'full')
        self.arm=arm
        if arm=='no_p2':
            old=base.fuse_p2;first=old.block[0];width=first.in_channels//2
            block=copy.deepcopy(old)
            conv=nn.Conv2d(width,first.out_channels,3,padding=1,bias=False)
            with torch.no_grad():conv.weight.copy_(first.weight[:,:width])
            block.block[0]=conv;base.fuse_p2=block;base.p2_lateral=nn.Identity()

    def forward_native(self,image,sensor):
        if self.arm=='full':return self.base.forward_native(image,sensor)
        b=self.base;name=b._sensor_name(sensor);common=b.input_adapters(image,name)
        pyramid=b._forward_pyramid(common,name)
        x=b._resize_like(b.up_to_p2(b.p3_lateral(pyramid['neck_p3'])),pyramid['p2'])
        x=b.fuse_p2(x)
        x=b._resize_like(b.up_to_p1(x),pyramid['p1'])
        x=b.fuse_p1(torch.cat((x,b.p1_lateral(pyramid['p1'])),1))
        x=b._resize_like(b.up_to_input(x),common)
        return b.decoder_head(b.fuse_input(torch.cat((x,b.input_lateral(common)),1)))
