# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Single-level P2 skip gates for local, main-output-only architecture controls.

The existing core, decoder and P3/P4 adapters remain intact. No extra loss,
normalization, input, pairing or teacher is introduced.
"""
import torch
from torch import nn
from .yolo26_unet_decoder import Yolo26UNetDecoder


class Yolo26P2GateDecoder(Yolo26UNetDecoder):
    def __init__(self, *, sensor_specific: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.sensor_specific = sensor_specific
        # New-module construction must not consume the legacy data RNG stream.
        with torch.random.fork_rng(devices=[]):
            keys = ("S1", "S2") if sensor_specific else ("shared",)
            self.p2_skip_gates = nn.ModuleDict({
                key: nn.Conv2d(self.p2_lateral.in_channels, 1, 1) for key in keys
            })
            for gate in self.p2_skip_gates.values():
                nn.init.zeros_(gate.weight)
                nn.init.zeros_(gate.bias)

    def _forward_pyramid(self, common_image, sensor):
        pyramid = super()._forward_pyramid(common_image, sensor)
        key = sensor if self.sensor_specific else "shared"
        p2 = pyramid["p2"]
        # 2*sigmoid(0)=1: initial skip and logits exactly match the original.
        pyramid["p2"] = p2 * (2.0 * torch.sigmoid(self.p2_skip_gates[key](p2)))
        return pyramid
