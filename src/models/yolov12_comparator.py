# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Pinned official YOLOv12-turbo trunk plus STURM semantic decoder.

Loaded by file in an isolated official-Ultralytics process, because the project's
installed YOLO26 API and official YOLOv12 API are different. No upstream patches.
"""
import copy
from pathlib import Path
import torch
from torch import nn
from ultralytics.nn.tasks import DetectionModel


class DecoderBlock(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False),
            nn.BatchNorm2d(co), nn.SiLU(inplace=True),
            nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co),
            nn.SiLU(inplace=True), nn.Identity())
    def forward(self, x):
        return self.block(x)


class Residual(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(c, max(8, int(c*.25)), 1), nn.SiLU(),
                                   nn.Conv2d(max(8, int(c*.25)), c, 1))
        nn.init.zeros_(self.block[-1].weight)
        nn.init.zeros_(self.block[-1].bias)
    def forward(self, x):
        return x + self.block(x)


class YoloV12SemanticComparator(nn.Module):
    def __init__(self, source_root, weights_path, common_channels=16):
        super().__init__()
        source_root = Path(source_root)
        import yaml
        cfg = yaml.safe_load((source_root/'ultralytics/cfg/models/v12/yolov12.yaml').read_text())
        cfg['scale'] = 'n'
        original = DetectionModel(copy.deepcopy(cfg), ch=3, nc=80, verbose=False)
        raw = torch.load(weights_path, map_location='cpu', weights_only=False)
        original.load_state_dict(raw['model'].float().state_dict(), strict=True)
        self.original_parameter_count = sum(p.numel() for p in original.parameters())
        # The full original detector is strictly verified BEFORE adaptation.
        adapted = DetectionModel(copy.deepcopy(cfg), ch=common_channels, nc=80, verbose=False)
        target = adapted.state_dict()
        matched = {k:v for k,v in original.state_dict().items()
                   if k in target and target[k].shape == v.shape}
        target.update(matched)
        adapted.load_state_dict(target, strict=True)
        self.load_report = dict(original_strict_load=True, matched_keys=len(matched),
             shape_skipped=[k for k,v in original.state_dict().items()
                            if k in target and target[k].shape != v.shape],
             pretraining='COCO detection', common_channels=common_channels,
             official_model_parameter_count=self.original_parameter_count)
        self.core = nn.ModuleList(list(adapted.model)[:15])
        self.save = adapted.save
        self.projections = nn.ModuleDict({s:nn.Sequential(nn.Conv2d(c, common_channels, 1), nn.SiLU())
                                         for s,c in [('S1',2),('S2',9)]})
        self.feature_adapters = nn.ModuleDict({f'{p}_{s}':Residual(128)
                                              for p in ['P3','P4'] for s in ['S1','S2']})
        # Official n-scale output widths: layer0=16, layer2=64, layer14=64.
        self.p3_lateral=nn.Conv2d(64,32,1)
        self.p2_lateral=nn.Conv2d(64,32,1)
        self.p1_lateral=nn.Conv2d(16,16,1)
        self.input_lateral=nn.Conv2d(common_channels,8,1)
        self.up_to_p2=nn.ConvTranspose2d(32,32,2,stride=2)
        self.fuse_p2=DecoderBlock(64,32)
        self.up_to_p1=nn.ConvTranspose2d(32,16,2,stride=2)
        self.fuse_p1=DecoderBlock(32,16)
        self.up_to_input=nn.ConvTranspose2d(16,8,2,stride=2)
        self.fuse_input=DecoderBlock(16,8)
        self.decoder_head=nn.Conv2d(8,2,1)
        self.reference = original  # Used only for init parity, then released.

    def trunk(self, common, sensor=None):
        saved=[]; captured={}; x=common
        for layer in self.core:
            if layer.f != -1:
                x = saved[layer.f] if isinstance(layer.f,int) else [x if i == -1 else saved[i] for i in layer.f]
            x=layer(x)
            if sensor is not None and layer.i in (4,6):
                x=self.feature_adapters[f'{"P3" if layer.i == 4 else "P4"}_{sensor}'](x)
            if layer.i in (0,2,14):captured[layer.i]=x
            saved.append(x if layer.i in self.save else None)
        return captured

    def release_reference(self):
        del self.reference

    def forward(self,batch):
        sensor=batch['sensor']
        if isinstance(sensor,list):
            if len(set(sensor)) != 1:raise ValueError('mixed native sensors')
            sensor=sensor[0]
        image=batch['image']
        if sensor not in ('S1','S2') or image.shape[1] != (2 if sensor=='S1' else 9):
            raise ValueError('native sensor channel mismatch')
        common=self.projections[sensor](image); p=self.trunk(common,sensor)
        x=self.fuse_p2(torch.cat([self.up_to_p2(self.p3_lateral(p[14])),self.p2_lateral(p[2])],1))
        x=self.fuse_p1(torch.cat([self.up_to_p1(x),self.p1_lateral(p[0])],1))
        x=self.fuse_input(torch.cat([self.up_to_input(x),self.input_lateral(common)],1))
        return self.decoder_head(x)
