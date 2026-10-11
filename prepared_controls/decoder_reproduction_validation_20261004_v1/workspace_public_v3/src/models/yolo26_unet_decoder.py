# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""YOLO26 semantic path with a lightweight U-Net-style high-resolution decoder."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from .yolo26_semantic import SharedYolo26, _infer_feature_channels


class _DecoderBlock(nn.Module):
    """Two-convolution refinement block used after each skip fusion."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Yolo26UNetDecoder(SharedYolo26):
    """Shared YOLO26 with U-Net-inspired P2/P1/input skip decoding.

    The native YOLO26 semantic head emits at a relatively coarse feature
    scale.  This variant keeps the pretrained-compatible YOLO backbone and
    neck, but stops before the native semantic head and decodes the 16x16
    neck feature to 128x128 through skips from the 32x32 and 64x64 backbone
    features plus the adapted input.  This transfers the most relevant U-Net
    inductive bias for flood boundaries while adding only a small decoder.

    ``common_channels=3`` isolates the decoder change.  ``common_channels=16``
    also tests whether a wider sensor bridge preserves more of the nine S2
    bands before the shared YOLO trunk; its first YOLO stem is freshly
    initialized because its input width differs from the 3-channel checkpoint.
    """

    HEAD_LAYER_INDEX = 17
    P1_LAYER_INDEX = 0  # 64x64
    P2_LAYER_INDEX = 2  # 32x32
    NECK_P3_LAYER_INDEX = 16  # 16x16

    def __init__(
        self,
        *,
        scale: str = "n",
        common_channels: int = 3,
        num_classes: int = 2,
        decoder_channels: int = 32,
        dropout: float = 0.0,
        use_attention: bool = False,
        use_input_skip: bool = True,
    ) -> None:
        if decoder_channels < 16 or decoder_channels % 4:
            raise ValueError("decoder_channels must be >=16 and divisible by four")
        super().__init__(
            scale=scale,
            common_channels=common_channels,
            num_classes=num_classes,
            use_attention=use_attention,
        )
        self.use_input_skip = bool(use_input_skip)
        pyramid_channels = _infer_feature_channels(
            self.core,
            (self.P1_LAYER_INDEX, self.P2_LAYER_INDEX, self.NECK_P3_LAYER_INDEX),
        )
        d = int(decoder_channels)
        self.p3_lateral = nn.Conv2d(pyramid_channels[self.NECK_P3_LAYER_INDEX], d, 1)
        self.p2_lateral = nn.Conv2d(pyramid_channels[self.P2_LAYER_INDEX], d, 1)
        self.p1_lateral = nn.Conv2d(pyramid_channels[self.P1_LAYER_INDEX], d // 2, 1)
        self.input_lateral = nn.Conv2d(common_channels, d // 4, 1)
        self.up_to_p2 = nn.ConvTranspose2d(d, d, 2, stride=2)
        self.fuse_p2 = _DecoderBlock(2 * d, d, dropout=dropout)
        self.up_to_p1 = nn.ConvTranspose2d(d, d // 2, 2, stride=2)
        self.fuse_p1 = _DecoderBlock(d, d // 2, dropout=dropout)
        self.up_to_input = nn.ConvTranspose2d(d // 2, d // 4, 2, stride=2)
        self.fuse_input = _DecoderBlock(
            d // 2 if self.use_input_skip else d // 4,
            d // 4,
            dropout=dropout,
        )
        self.decoder_head = nn.Conv2d(d // 4, num_classes, 1)

    def _forward_pyramid(self, common_image: torch.Tensor, sensor: str) -> dict[str, torch.Tensor]:
        x = common_image
        saved: list[object] = []
        captured: dict[str, torch.Tensor] = {}
        for layer in self.core.model:
            if layer.i == self.HEAD_LAYER_INDEX:
                break
            if layer.f != -1:
                x = (
                    saved[layer.f]
                    if isinstance(layer.f, int)
                    else [x if index == -1 else saved[index] for index in layer.f]
                )
            x = layer(x)
            if layer.i == self.P1_LAYER_INDEX:
                captured["p1"] = x
            if layer.i == self.P2_LAYER_INDEX:
                captured["p2"] = x
            if layer.i == self.P3_LAYER_INDEX:
                x = self.feature_adapters[f"P3_{sensor}"](x)
                if self.use_attention:
                    x = self.attention["P3"](x)
            elif layer.i == self.P4_LAYER_INDEX:
                x = self.feature_adapters[f"P4_{sensor}"](x)
                if self.use_attention:
                    x = self.attention["P4"](x)
            if layer.i == self.NECK_P3_LAYER_INDEX:
                captured["neck_p3"] = x
            saved.append(x if layer.i in self.core.save else None)
        missing = {name for name in ("p1", "p2", "neck_p3") if name not in captured}
        if missing:
            raise RuntimeError(f"YOLO26 U-Net decoder missed pyramid features: {sorted(missing)}")
        return captured

    @staticmethod
    def _resize_like(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != reference.shape[-2:]:
            x = torch.nn.functional.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward_native(
        self,
        image: torch.Tensor,
        sensor: str | int | torch.Tensor,
    ) -> torch.Tensor:
        name = self._sensor_name(sensor)
        common_image = self.input_adapters(image, name)
        pyramid = self._forward_pyramid(common_image, name)
        p1 = self.p1_lateral(pyramid["p1"])
        p2 = self.p2_lateral(pyramid["p2"])
        x = self.up_to_p2(self.p3_lateral(pyramid["neck_p3"]))
        x = self._resize_like(x, p2)
        x = self.fuse_p2(torch.cat([x, p2], dim=1))
        x = self.up_to_p1(x)
        x = self._resize_like(x, p1)
        x = self.fuse_p1(torch.cat([x, p1], dim=1))
        x = self.up_to_input(x)
        if self.use_input_skip:
            input_skip = self.input_lateral(common_image)
            x = self._resize_like(x, input_skip)
            x = self.fuse_input(torch.cat([x, input_skip], dim=1))
        else:
            x = self.fuse_input(x)
        return self.decoder_head(x)
