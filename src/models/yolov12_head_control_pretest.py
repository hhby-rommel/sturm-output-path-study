# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Train/val control: retained official trunk plus fresh P3 semantic head.

This is an adapted semantic baseline, not the official native detector.
"""
import torch
from torch import nn
from torch.nn import functional as F


class YoloV12LightHead(nn.Module):
    def __init__(self, decoder, head_seed=10187):
        super().__init__()
        self.core = decoder.core
        self.save = decoder.save
        self.projections = decoder.projections
        self.feature_adapters = decoder.feature_adapters
        # Bind graph traversal without registering the discarded decoder.
        self._trunk_function = type(decoder).trunk
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(head_seed)
            self.semantic_head = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.SiLU(), nn.Conv2d(32, 2, 1))

    def forward(self, batch):
        sensor = batch['sensor']
        if isinstance(sensor, list):
            if len(set(sensor)) != 1:
                raise ValueError('mixed native sensors')
            sensor = sensor[0]
        image = batch['image']
        if sensor not in ('S1', 'S2') or image.shape[1] != (2 if sensor == 'S1' else 9):
            raise ValueError('native sensor channel mismatch')
        common = self.projections[sensor](image)
        p = self._trunk_function(self, common, sensor)
        return F.interpolate(self.semantic_head(p[14]), size=image.shape[-2:],
                             mode='bilinear', align_corners=False)
