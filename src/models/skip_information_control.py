# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Information-only skip interventions with the original decoder intact.

No modules, widths, parameters, buffers, initialization or BN policies are
changed. Projected P2/P1/input features are multiplied by zero immediately
before fusion. The neck path and all projection computations are retained.
AdamW decay can still change the zero-task-gradient projection parameters.
"""
from __future__ import annotations
import torch
from .yolo26_unet_decoder import Yolo26UNetDecoder

INFORMATION_MODES = ('full', 'zero_all_skips', 'zero_input')

class SkipInformationControl(Yolo26UNetDecoder):
    def __init__(self, *, information_mode: str, **kwargs):
        if information_mode not in INFORMATION_MODES:
            raise ValueError(f'Unknown information mode: {information_mode}')
        if kwargs.get('use_input_skip', True) is not True:
            raise ValueError('Information controls require all original modules')
        kwargs['use_input_skip'] = True
        super().__init__(**kwargs)
        self.information_mode = information_mode  # configuration, no learned gate

    def forward_native(self, image, sensor):
        if self.information_mode == 'full':
            return super().forward_native(image, sensor)
        name = self._sensor_name(sensor)
        common_image = self.input_adapters(image, name)
        pyramid = self._forward_pyramid(common_image, name)
        p1 = self.p1_lateral(pyramid['p1'])
        p2 = self.p2_lateral(pyramid['p2'])
        if self.information_mode == 'zero_all_skips':
            p1 = p1 * 0.0
            p2 = p2 * 0.0
        x = self.up_to_p2(self.p3_lateral(pyramid['neck_p3']))
        x = self._resize_like(x, p2)
        x = self.fuse_p2(torch.cat([x, p2], dim=1))
        x = self.up_to_p1(x)
        x = self._resize_like(x, p1)
        x = self.fuse_p1(torch.cat([x, p1], dim=1))
        x = self.up_to_input(x)
        input_skip = self.input_lateral(common_image) * 0.0
        x = self._resize_like(x, input_skip)
        x = self.fuse_input(torch.cat([x, input_skip], dim=1))
        return self.decoder_head(x)
