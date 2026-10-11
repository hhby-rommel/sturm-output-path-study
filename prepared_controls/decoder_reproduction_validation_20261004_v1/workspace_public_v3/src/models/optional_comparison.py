# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Explicitly optional semantic-segmentation comparison models.

These models are not part of the frozen formal matrix.  They are small
comparison controls that can be selected explicitly through ``server_main``.
Both use the project's native S1/S2 sensor adapter and return two-class
logits at the input spatial resolution.  No constructor downloads weights.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .yolo26_semantic import SensorInputAdapter


class _GroupedSensorModel(nn.Module):
    """Common native/grouped dispatch for optional comparison models."""

    def __init__(self, common_channels: int = 3) -> None:
        super().__init__()
        self.common_channels = int(common_channels)
        self.input_adapters = SensorInputAdapter(common_channels=self.common_channels)

    @staticmethod
    def _sensor_name(sensor: str | int | torch.Tensor) -> str:
        return SensorInputAdapter._sensor_name(sensor)

    def forward_grouped(self, groups: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
        outputs: dict[str, object] = {}
        for sensor in ("S1", "S2"):
            if sensor in groups:
                batch = groups[sensor]
                outputs[sensor] = self.forward_native(batch["image"], sensor)  # type: ignore[arg-type]
        if not outputs:
            raise ValueError("grouped batch contains neither S1 nor S2")
        return outputs

    def forward(
        self,
        inputs: torch.Tensor | Mapping[str, object],
        sensor: str | int | torch.Tensor | None = None,
    ) -> object:
        if isinstance(inputs, Mapping):
            if "image" not in inputs:
                return self.forward_grouped(inputs)  # type: ignore[arg-type]
            if sensor is None:
                sensor = inputs.get("sensor")  # type: ignore[assignment]
                if isinstance(sensor, list):
                    if not sensor or len(set(sensor)) != 1:
                        raise ValueError("a native mapping forward requires one sensor per batch")
                    sensor = sensor[0]
            if sensor is None:
                raise ValueError("sensor is required for a native mapping forward")
            return self.forward_native(inputs["image"], sensor)  # type: ignore[arg-type]
        if sensor is None:
            raise ValueError("sensor is required for a native tensor forward")
        return self.forward_native(inputs, sensor)


class TorchvisionDeepLabV3MobileNetV3(_GroupedSensorModel):
    """Torchvision DeepLabV3-MobileNetV3-Large comparison control.

    Torchvision does not expose a DeepLabV3+ MobileNetV3 implementation, so
    the model is deliberately named after the exact closest implementation.
    ``weights_backbone=None`` is important: importing/constructing it never
    attempts an online weight download.
    """

    architecture_name = "torchvision_deeplabv3_mobilenet_v3_large"

    def __init__(self, *, common_channels: int = 3, num_classes: int = 2) -> None:
        super().__init__(common_channels=common_channels)
        try:
            from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
        except Exception as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "Torchvision is required for the optional DeepLabV3-MobileNetV3 comparison model"
            ) from exc
        self.core = deeplabv3_mobilenet_v3_large(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
            aux_loss=False,
        )

    def forward_native(self, image: torch.Tensor, sensor: str | int | torch.Tensor) -> torch.Tensor:
        name = self._sensor_name(sensor)
        common_image = self.input_adapters(image, name)
        logits = self.core(common_image)["out"]
        if logits.shape[-2:] != image.shape[-2:]:
            logits = F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return logits


class _SegFormerStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SegFormerB0Like(_GroupedSensorModel):
    """Dependency-free SegFormer-B0-like hierarchical encoder control.

    The project environment does not provide ``transformers``.  This is a
    clearly named local approximation (overlap patch stages plus an MLP-style
    multi-scale decoder), not a claim of exact Hugging Face SegFormer-B0
    reproduction and not part of the registered matrix.
    """

    architecture_name = "segformer_b0_like_local"

    def __init__(self, *, common_channels: int = 3, num_classes: int = 2) -> None:
        super().__init__(common_channels=common_channels)
        widths = (32, 64, 160, 256)
        self.stages = nn.ModuleList(
            [
                _SegFormerStage(common_channels, widths[0], 4),
                _SegFormerStage(widths[0], widths[1], 2),
                _SegFormerStage(widths[1], widths[2], 2),
                _SegFormerStage(widths[2], widths[3], 2),
            ]
        )
        decoder_width = 64
        self.projections = nn.ModuleList(nn.Conv2d(width, decoder_width, 1) for width in widths)
        self.fuse = nn.Sequential(
            nn.Conv2d(4 * decoder_width, decoder_width, 1, bias=False),
            nn.BatchNorm2d(decoder_width),
            nn.GELU(),
        )
        self.head = nn.Conv2d(decoder_width, num_classes, 1)

    def forward_native(self, image: torch.Tensor, sensor: str | int | torch.Tensor) -> torch.Tensor:
        name = self._sensor_name(sensor)
        x = self.input_adapters(image, name)
        features: list[torch.Tensor] = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        reference_size = features[0].shape[-2:]
        projected = [
            F.interpolate(projection(feature), size=reference_size, mode="bilinear", align_corners=False)
            for projection, feature in zip(self.projections, features)
        ]
        logits = self.head(self.fuse(torch.cat(projected, dim=1)))
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
