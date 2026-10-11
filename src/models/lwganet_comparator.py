# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Pinned LWGANet-L0 + official UNetFormer, native STURM adaptation."""
import argparse
import ast
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import List
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .optional_comparison import _GroupedSensorModel

ROOT = Path(__file__).resolve().parents[2]
COMMIT = 'a5ba000fa72f224eb4669383d8daf0beea6f85e7'
WEIGHT_SHA = 'dcc8b97df07e02edfc373d26e8065c9dc0fbd1ef507715bc23df8b32e890fd45'

def bn_only_build_norm(cfg, channels, postfix=''):
    """MMCV 1.7.2 BN branch semantics; reject every other normalization type."""
    config = cfg.copy()
    if config.pop('type') != 'BN':
        raise ValueError('local compatibility builder permits official BN only')
    requires_grad = config.pop('requires_grad', True)
    config.setdefault('eps', 1e-5)
    layer = nn.BatchNorm2d(channels, **config)
    for parameter in layer.parameters():
        parameter.requires_grad = requires_grad
    return 'bn' + str(postfix), layer

def official_namespace():
    from timm.layers import DropPath, trunc_normal_, to_2tuple
    from einops import rearrange, repeat
    dependency_path = ROOT/'results/lwganet_public_asset_audit_20261001_v1/dependencies'
    sys.path.insert(0, str(dependency_path))
    import antialiased_cnns
    lock = json.loads((ROOT/'configs/lwganet_official_audit_source_lock.json').read_text())
    if lock['commit'] != COMMIT:
        raise RuntimeError('source commit changed')
    namespace = dict(__name__='lwganet_pinned_segmentation', torch=torch, nn=nn, F=F,
        copy=copy, os=os, Tensor=Tensor, List=List, DropPath=DropPath,
        trunc_normal_=trunc_normal_, to_2tuple=to_2tuple, rearrange=rearrange,
        repeat=repeat, antialiased_cnns=antialiased_cnns, build_norm_layer=bn_only_build_norm)
    for relative in ('segmentation/geoseg/models/lwganet.py',
                     'segmentation/geoseg/models/UNetFormer_lwganet.py'):
        path = ROOT/'external/lwganet-official-audit'/relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != lock['files'][relative]:
            raise RuntimeError('official source changed: ' + relative)
        nodes = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef)]
        # Original class bodies unchanged. Imports and factories (including
        # the empty checkpoint path / ignored embed_dim) are not executed.
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace

class LWGANetSTURM(_GroupedSensorModel):
    architecture_name = 'LWGANet_L0_UNetFormer_shared_STURM_window4_v1'
    def __init__(self, *, window_size=4):
        super().__init__(common_channels=3)
        namespace = official_namespace()
        weight = ROOT/'models/pretrained/research_comparators/lwganet_l0_e299.pth'
        if hashlib.sha256(weight.read_bytes()).hexdigest() != WEIGHT_SHA:
            raise RuntimeError('pretrained asset changed')
        tensor_lock_path = ROOT/'configs/lwganet_tensor_asset_lock_v1.json'
        if tensor_lock_path.exists():
            tensor_lock = json.loads(tensor_lock_path.read_text())
            tensor_path = ROOT/tensor_lock['local_tensor_asset']
            if (tensor_lock['origin_sha256'] != WEIGHT_SHA or
                hashlib.sha256(tensor_path.read_bytes()).hexdigest() != tensor_lock['sha256']):
                raise RuntimeError('tensor-only asset identity changed')
            state = torch.load(tensor_path, map_location='cpu', weights_only=True)
        else:
            # Legacy v1 diagnostic loading retained for reproducibility only.
            with torch.serialization.safe_globals([argparse.Namespace]):
                state = torch.load(weight, map_location='cpu', weights_only=True)['model']
        backbone = namespace['LWGANet'](in_chans=3, stem_dim=32, depths=(1,2,4,2),
            att_kernel=(11,11,11,11), norm_layer=dict(type='BN', requires_grad=True),
            act_layer=nn.GELU, drop_path_rate=0., fork_feat=True)
        discarded = {k: v for k, v in state.items() if k.startswith(('head.', 'avgpool_pre_head.'))}
        retained = {k: v for k, v in state.items() if k not in discarded}
        expected_new = {f'norm{i}.{suffix}' for i in (0,2,4,6)
            for suffix in ('weight','bias','running_mean','running_var','num_batches_tracked')}
        target = backbone.state_dict()
        if set(target) - set(retained) != expected_new or set(retained) - set(target):
            raise RuntimeError('unexpected classification-to-segmentation mapping')
        # Fill only the explicitly declared new output BN with its initialization;
        # then strict load the entire resulting backbone state dictionary.
        merged = {**target, **retained}
        backbone.load_state_dict(merged, strict=True)
        self.core = nn.Module()
        self.core.backbone = backbone
        self.core.decoder = namespace['Decoder']([32,64,128,256], 64, .1, window_size, 2)
        self.initialization_report = dict(source_commit=COMMIT, weight_sha256=WEIGHT_SHA,
            pretrained_tensor_count=len(retained), discarded_classifier_keys=sorted(discarded),
            new_output_norm_keys=sorted(expected_new),
            pretrained_tensors_exact=all(torch.equal(backbone.state_dict()[k], v) for k,v in retained.items()),
            decoder_initialization='random official decoder including aux head',
            window_size=window_size, upstream_window_size=8,
            normalization_compatibility='MMCV BN-only semantics; no global module monkey patch',
            source_class_bodies_modified=False)

    def forward_native(self, image, sensor):
        if image.shape[-2:] != (128,128):
            raise ValueError('local pretest requires native 128x128')
        features = self.core.backbone(self.input_adapters(image, self._sensor_name(sensor)))
        return self.core.decoder(*features, *image.shape[-2:])
