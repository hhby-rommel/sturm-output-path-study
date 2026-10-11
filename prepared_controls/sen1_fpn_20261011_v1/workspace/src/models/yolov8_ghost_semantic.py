# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""YOLOv8-Ghost semantic models for a strict low-parameter screen."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from ultralytics.nn.tasks import SemanticSegmentationModel

from .yolo26_semantic import (
    EfficientChannelAttention,
    ResidualFeatureAdapter,
    SensorInputAdapter,
    _infer_feature_channels,
)


P3_LAYER_INDEX = 15
P4_LAYER_INDEX = 18


def _base_config_path() -> Path:
    import ultralytics

    return Path(ultralytics.__file__).resolve().parent / "cfg" / "models" / "v8" / "yolov8-ghost.yaml"


def build_yolov8_ghost_semantic(
    *,
    scale: str = "n",
    in_channels: int,
    num_classes: int = 2,
    verbose: bool = False,
) -> SemanticSegmentationModel:
    """Build YOLOv8-Ghost with a dense two-class semantic head."""
    scale = scale.lower()
    if scale not in {"n", "s", "m", "l", "x"}:
        raise ValueError(f"unsupported YOLOv8-Ghost scale: {scale!r}")
    if in_channels not in {2, 3, 9}:
        raise ValueError(f"expected STURM-Flood channels 2, 3, or 9, got {in_channels}")
    if num_classes != 2:
        raise ValueError("the locked STURM-Flood protocol uses two operational classes")

    config = copy.deepcopy(yaml.safe_load(_base_config_path().read_text(encoding="utf-8")))
    config["scale"] = scale
    config["nc"] = num_classes
    # The Ghost configuration is detection-only. P3/P4 are sufficient for the
    # project's dense semantic output and keep the contract used by YOLO26/11.
    config["head"][-1] = [[P3_LAYER_INDEX, P4_LAYER_INDEX], 1, "SemanticSegment", [num_classes]]
    return SemanticSegmentationModel(cfg=config, ch=in_channels, nc=num_classes, verbose=verbose)


def load_matching_yolov8_ghost_weights(
    model: SemanticSegmentationModel,
    checkpoint_path: str | Path,
) -> dict[str, int]:
    """Load only checkpoint tensors with matching names and shapes."""
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if isinstance(checkpoint, Mapping):
        source_model = checkpoint.get("model")
    else:
        source_model = None
    if isinstance(source_model, nn.Module):
        source_state = source_model.state_dict()
    elif isinstance(source_model, Mapping):
        source_state = source_model
    elif isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
        source_state = checkpoint["state_dict"]
    else:
        raise ValueError(f"unsupported YOLOv8-Ghost checkpoint structure: {checkpoint_path}")

    target_state = model.state_dict()
    matched: dict[str, Any] = {}
    skipped_shape = 0
    skipped_name = 0
    for key, value in source_state.items():
        if key not in target_state:
            skipped_name += 1
        elif tuple(value.shape) != tuple(target_state[key].shape):
            skipped_shape += 1
        else:
            matched[key] = value
    target_state.update(matched)
    model.load_state_dict(target_state, strict=True)
    return {
        "matched_keys": len(matched),
        "skipped_shape_keys": skipped_shape,
        "skipped_name_keys": skipped_name,
        "target_keys": len(target_state),
    }


class SharedYoloV8Ghost(nn.Module):
    """Shared YOLOv8-Ghost semantic model with native S1/S2 adapters."""

    def __init__(
        self,
        *,
        scale: str = "n",
        common_channels: int = 3,
        num_classes: int = 2,
        use_attention: bool = False,
    ) -> None:
        super().__init__()
        self.common_channels = common_channels
        self.num_classes = num_classes
        self.use_attention = bool(use_attention)
        self.core = build_yolov8_ghost_semantic(
            scale=scale,
            in_channels=common_channels,
            num_classes=num_classes,
        )
        self.input_adapters = SensorInputAdapter(common_channels=common_channels)
        feature_channels = _infer_feature_channels(self.core, (P3_LAYER_INDEX, P4_LAYER_INDEX))
        p3_channels = feature_channels[P3_LAYER_INDEX]
        p4_channels = feature_channels[P4_LAYER_INDEX]
        self.feature_channels = {"P3": p3_channels, "P4": p4_channels}
        self.feature_adapters = nn.ModuleDict(
            {
                "P3_S1": ResidualFeatureAdapter(p3_channels),
                "P3_S2": ResidualFeatureAdapter(p3_channels),
                "P4_S1": ResidualFeatureAdapter(p4_channels),
                "P4_S2": ResidualFeatureAdapter(p4_channels),
            }
        )
        self.attention = nn.ModuleDict(
            {"P3": EfficientChannelAttention(), "P4": EfficientChannelAttention()}
            if self.use_attention
            else {}
        )

    @staticmethod
    def _sensor_name(sensor: str | int | torch.Tensor) -> str:
        if isinstance(sensor, torch.Tensor):
            values = torch.unique(sensor.detach().cpu())
            if values.numel() != 1:
                raise ValueError("a native tensor forward requires one sensor per batch")
            sensor = int(values.item())
        return SensorInputAdapter._sensor_name(sensor)

    def _forward_core(self, common_image: torch.Tensor, sensor: str) -> torch.Tensor | tuple[torch.Tensor, ...]:
        x = common_image
        saved: list[object] = []
        for layer in self.core.model:
            if layer.f != -1:
                x = saved[layer.f] if isinstance(layer.f, int) else [
                    x if index == -1 else saved[index] for index in layer.f
                ]
            x = layer(x)
            if layer.i == P3_LAYER_INDEX:
                x = self.feature_adapters[f"P3_{sensor}"](x)
                if self.use_attention:
                    x = self.attention["P3"](x)
            elif layer.i == P4_LAYER_INDEX:
                x = self.feature_adapters[f"P4_{sensor}"](x)
                if self.use_attention:
                    x = self.attention["P4"](x)
            saved.append(x if layer.i in self.core.save else None)
        return x

    def forward_native(
        self,
        image: torch.Tensor,
        sensor: str | int | torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        name = self._sensor_name(sensor)
        return self._forward_core(self.input_adapters(image, name), name)

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
            if "image" in inputs:
                if sensor is None:
                    sensor = inputs.get("sensor")  # type: ignore[assignment]
                    if isinstance(sensor, list):
                        if not sensor or len(set(sensor)) != 1:
                            raise ValueError("a native mapping forward requires one sensor per batch")
                        sensor = sensor[0]
                if sensor is None:
                    raise ValueError("sensor is required for a native mapping forward")
                return self.forward_native(inputs["image"], sensor)  # type: ignore[arg-type]
            return self.forward_grouped(inputs)  # type: ignore[arg-type]
        if sensor is None:
            raise ValueError("sensor is required when forwarding a native tensor")
        return self.forward_native(inputs, sensor)
