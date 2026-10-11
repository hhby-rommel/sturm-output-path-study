# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Separate post-hoc decoder attribution variants; original model source unchanged."""
from collections.abc import Mapping
import itertools,copy
import torch
from torch import nn
from .yolo26_unet_decoder import _DecoderBlock

ARMS=('full','no_skip_capacity','no_input','no_p1')
DECODER_NAMES=('p3_lateral','p2_lateral','p1_lateral','input_lateral','up_to_p2','fuse_p2','up_to_p1','fuse_p1','up_to_input','fuse_input','decoder_head')

def decoder_parameter_count(base):
    return sum(p.numel() for n in DECODER_NAMES for p in getattr(base,n).parameters())

def refinement(in_channels,out_channels,hidden):
    return nn.Sequential(nn.Conv2d(in_channels,hidden,3,padding=1,bias=False),nn.BatchNorm2d(hidden),nn.SiLU(inplace=True),
                         nn.Conv2d(hidden,out_channels,3,padding=1,bias=False),nn.BatchNorm2d(out_channels),nn.SiLU(inplace=True),nn.Identity())

def matching_widths(base):
    """Parameter-only deterministic search. No images or evaluation scores accepted."""
    widths=(base.up_to_p2.out_channels,base.up_to_p1.out_channels,base.up_to_input.out_channels)
    target=decoder_parameter_count(base)
    fixed=sum(p.numel() for n in ('p3_lateral','up_to_p2','up_to_p1','up_to_input','decoder_head') for p in getattr(base,n).parameters())
    def count(w,h):return 18*w*h+2*h+2*w
    choices=itertools.product(*(range(w,2*w+1) for w in widths))
    h=min(choices,key=lambda h:(abs(fixed+sum(count(w,z) for w,z in zip(widths,h))-target),sum(abs(z/w-1.5) for w,z in zip(widths,h)),h))
    total=fixed+sum(count(w,z) for w,z in zip(widths,h))
    return dict(hidden_widths=h,output_widths=widths,full_decoder_parameters=target,no_skip_decoder_parameters=total,delta=total-target,relative_delta_percent=100*(total-target)/target,search='integer hidden widths w..2w; minimum absolute parameter difference, then deviation from1.5w, then lexicographic')

class DecoderAttribution(nn.Module):
    def __init__(self,base,arm):
        super().__init__()
        if arm not in ARMS:raise ValueError(arm)
        self.base=base;self.arm=arm;self.capacity_design=None
        if arm=='no_skip_capacity':
            self.capacity_design=matching_widths(base)
            for name,w,h in zip(('fuse_p2','fuse_p1','fuse_input'),self.capacity_design['output_widths'],self.capacity_design['hidden_widths']):
                block=_DecoderBlock(w,w);block.block=refinement(w,w,h);setattr(base,name,block)
            for name in ('p2_lateral','p1_lateral','input_lateral'):setattr(base,name,nn.Identity())
        elif arm in ('no_input','no_p1'):
            name='fuse_input' if arm=='no_input' else 'fuse_p1'
            old=getattr(base,name);first=old.block[0];w=first.in_channels//2
            block=copy.deepcopy(old);conv=nn.Conv2d(w,first.out_channels,3,padding=1,bias=False)
            with torch.no_grad():conv.weight.copy_(first.weight[:,:w])
            block.block[0]=conv;setattr(base,name,block)
            setattr(base,'input_lateral' if arm=='no_input' else 'p1_lateral',nn.Identity())

    def forward_native(self,image,sensor):
        if self.arm=='full':return self.base.forward_native(image,sensor)
        name=self.base._sensor_name(sensor);b=self.base;c=b.input_adapters(image,name);p=b._forward_pyramid(c,name)
        x=b.up_to_p2(b.p3_lateral(p['neck_p3']));x=b._resize_like(x,p['p2'])
        if self.arm!='no_skip_capacity':x=torch.cat((x,b.p2_lateral(p['p2'])),1)
        x=b.fuse_p2(x);x=b.up_to_p1(x);x=b._resize_like(x,p['p1'])
        if self.arm not in ('no_skip_capacity','no_p1'):x=torch.cat((x,b.p1_lateral(p['p1'])),1)
        x=b.fuse_p1(x);x=b.up_to_input(x);x=b._resize_like(x,c)
        if self.arm not in ('no_skip_capacity','no_input'):x=torch.cat((x,b.input_lateral(c)),1)
        return b.decoder_head(b.fuse_input(x))

    def forward(self,inputs,sensor=None):
        if isinstance(inputs,Mapping):
            if 'image' not in inputs:return {s:self.forward(inputs[s],s) for s in ('S1','S2') if s in inputs}
            sensor=inputs.get('sensor') if sensor is None else sensor
            if isinstance(sensor,list):
                if len(set(sensor))!=1:raise ValueError('one sensor per subgroup')
                sensor=sensor[0]
            return self.forward_native(inputs['image'],sensor)
        if sensor is None:raise ValueError('sensor required')
        return self.forward_native(inputs,sensor)

def static_route_count(model,sensor):
    base=model.base;other='S2' if sensor=='S1' else 'S1'
    return sum(p.numel() for n,p in base.named_parameters() if not n.startswith(('core.model.17.',f'input_adapters.projections.{other}.',f'feature_adapters.P3_{other}.',f'feature_adapters.P4_{other}.')))
