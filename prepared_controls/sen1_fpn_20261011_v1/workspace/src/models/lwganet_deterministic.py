# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Deterministic equivalent of official nonoverlapping 2x2 max-unpool."""
import torch
from torch import nn
from .lwganet_comparator import LWGANetSTURM

class UniqueIndexMaxUnpool2d(nn.Module):
    def forward(self,x,indices,output_size=None):
        if x.shape!=indices.shape:raise ValueError('unpool index shape mismatch')
        b,c,h,w=x.shape
        expected=(2*h,2*w)
        if output_size is not None and tuple(output_size[-2:])!=expected:
            raise ValueError('only official even-grid nonoverlapping 2x2 unpool is supported')
        flat=indices.flatten(2)
        # Nonoverlapping 2x2/stride2 pool guarantees unique indices per plane.
        # Reject other callers rather than choose an arbitrary duplicate winner.
        ordered=torch.sort(flat,dim=-1).values
        if torch.any(ordered[...,1:]==ordered[...,:-1]):
            raise ValueError('duplicate indices violate nonoverlapping-pool contract')
        result=x.new_zeros(b,c,expected[0]*expected[1]).scatter(2,flat,x.flatten(2))
        return result.reshape(b,c,*expected)

class LWGANetDeterministicSTURM(LWGANetSTURM):
    architecture_name='LWGANet_L0_UNetFormer_shared_STURM_window4_deterministic_unpool_v2'
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        replaced=[]
        for name,module in list(self.named_modules()):
            if isinstance(module,nn.MaxUnpool2d):
                if (module.kernel_size!=(2,2) or module.stride!=(2,2) or module.padding!=(0,0)):
                    raise ValueError('unexpected upstream unpool configuration')
                parent,_,child=name.rpartition('.')
                setattr(self.get_submodule(parent),child,UniqueIndexMaxUnpool2d())
                replaced.append(name)
        if not replaced:raise RuntimeError('expected official MaxUnpool modules missing')
        self.initialization_report['unpool_compatibility']=dict(replaced_modules=replaced,
            scope='unique nonoverlapping 2x2 indices only; zero-initialized scatter equivalent')
