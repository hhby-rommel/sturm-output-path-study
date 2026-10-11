# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Local-only spatial-context hypothesis; never registered in formal runner."""
import torch
from torch import nn


class SmallContext(nn.Module):
    def __init__(self,channels=32,conditional=False):
        super().__init__();self.conditional=conditional
        self.spatial=nn.ModuleList(nn.Conv2d(channels,channels,3,padding=d,dilation=d,groups=channels,bias=False) for d in [1,2,3])
        self.embedding=nn.Parameter(torch.zeros(2,channels))
        self.output=nn.Conv2d(channels,channels,1,bias=False)
        nn.init.zeros_(self.output.weight)  # New branch identity; no old parameter reset.
    def forward(self,x,sensor):
        e=self.embedding[0 if sensor=='S1' else 1] if self.conditional else self.embedding.mean(0)
        feature=sum(layer(x) for layer in self.spatial)/3
        feature=torch.nn.functional.silu(feature)*(1+torch.tanh(e)[None,:,None,None])
        return x+self.output(feature)


class ContextWrapper(nn.Module):
    """Preserve original forward graph and inject only after P3 lateral."""
    def __init__(self,base,conditional):
        super().__init__();self.base=base;self.context=SmallContext(base.p3_lateral.out_channels,conditional)
        self.sensor=None
        self.base.p3_lateral.register_forward_hook(self._inject)
    def _inject(self,module,inputs,output):
        if self.sensor not in ['S1','S2']:raise RuntimeError('single-sensor route required')
        return self.context(output,self.sensor)
    def forward(self,batch):
        sensor=batch['sensor'];sensor=sensor[0] if isinstance(sensor,list) else sensor
        self.sensor=self.base._sensor_name(sensor)
        return self.base(batch)
