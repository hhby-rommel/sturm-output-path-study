# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""YOLO26 semantic-segmentation model construction for native sensor inputs."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from ultralytics.nn.tasks import SemanticSegmentationModel


def _base_config_path() -> Path:
    import ultralytics

    return Path(ultralytics.__file__).resolve().parent / "cfg" / "models" / "26" / "yolo26-sem.yaml"


def build_yolo26_semantic(
    *,
    scale: str = "n",
    in_channels: int,
    num_classes: int = 2,
    verbose: bool = False,
) -> SemanticSegmentationModel:
    """Build a YOLO26 semantic model with an explicit channel/class contract.

    ``in_channels`` is 2 for native S1, 9 for native S2, or 3 for the shared
    adapter bridge.  Pretrained
    3-channel weights are intentionally not silently loaded into these custom
    stems; weight-transfer policy belongs to the later adapter experiment.
    """
    scale = scale.lower()
    if scale not in {"n", "s", "m", "l", "x"}:
        raise ValueError(f"unsupported YOLO26 scale: {scale!r}")
    if not isinstance(in_channels, int) or in_channels < 1:
        raise ValueError(f"expected a positive integer input channel count, got {in_channels}")
    if num_classes != 2:
        raise ValueError("the locked STURM-Flood protocol uses two operational classes")

    config = yaml.safe_load(_base_config_path().read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    config["scale"] = scale
    config["nc"] = num_classes
    return SemanticSegmentationModel(
        cfg=config,
        ch=in_channels,
        nc=num_classes,
        verbose=verbose,
    )


def first_conv_in_channels(model: nn.Module) -> int:
    """Return the first convolution's input channel count for a smoke check."""
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            return int(module.in_channels)
    raise RuntimeError("no Conv2d layer found in YOLO26 model")


def load_matching_yolo26_weights(
    model: SemanticSegmentationModel,
    checkpoint_path: str | Path,
) -> dict[str, int]:
    """Load only checkpoint tensors whose names and shapes match ``model``.

    The bundled YOLO26 semantic checkpoint is a Cityscapes model with a
    19-class head, whereas this project has a locked two-class head.  Matching
    backbone/neck tensors can still be used as initialization, but classifier
    tensors with incompatible shapes must remain freshly initialized.  The
    function is intentionally explicit so a class-head mismatch cannot be
    silently hidden by a broad ``strict=False`` load.
    """
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
        raise ValueError(f"unsupported YOLO26 checkpoint structure: {checkpoint_path}")

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


class SensorInputAdapter(nn.Module):
    """Map each native sensor tensor to the shared YOLO26 input contract."""

    def __init__(self, common_channels: int = 3) -> None:
        super().__init__()
        if common_channels < 1:
            raise ValueError("common_channels must be positive")
        self.common_channels = common_channels
        self.projections = nn.ModuleDict(
            {
                "S1": nn.Sequential(nn.Conv2d(2, common_channels, kernel_size=1), nn.SiLU()),
                "S2": nn.Sequential(nn.Conv2d(9, common_channels, kernel_size=1), nn.SiLU()),
            }
        )

    @staticmethod
    def _sensor_name(sensor: str | int) -> str:
        if isinstance(sensor, int):
            sensor = "S1" if sensor == 0 else "S2" if sensor == 1 else ""
        sensor = str(sensor).upper()
        if sensor not in {"S1", "S2"}:
            raise ValueError(f"sensor must be S1 or S2, got {sensor!r}")
        return sensor

    def forward(self, image: torch.Tensor, sensor: str | int) -> torch.Tensor:
        name = self._sensor_name(sensor)
        expected = 2 if name == "S1" else 9
        if image.ndim != 4 or image.shape[1] != expected:
            raise ValueError(
                f"{name} input adapter expects [B,{expected},H,W], got {tuple(image.shape)}"
            )
        return self.projections[name](image)


class ResidualFeatureAdapter(nn.Module):
    """Small zero-initialised residual adapter for one YOLO feature level."""

    def __init__(self, channels: int, bottleneck_ratio: float = 0.25) -> None:
        super().__init__()
        hidden = max(8, int(channels * bottleneck_ratio))
        self.block = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        # Start as an identity mapping so the shared backbone is not abruptly
        # perturbed before the adapters receive useful gradients.
        nn.init.zeros_(self.block[-1].weight)
        nn.init.zeros_(self.block[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.block(features)


class EfficientChannelAttention(nn.Module):
    """Very small ECA-style channel gate for controlled lightweight pilots."""

    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError("kernel_size must be a positive odd number")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)
        # Start as an identity gate so a pretrained backbone is not scaled by
        # roughly 0.5 before the attention branch has learned a signal.
        nn.init.zeros_(self.conv.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(features).squeeze(-1).transpose(1, 2)
        weights = (2.0 * torch.sigmoid(self.conv(pooled))).transpose(1, 2).unsqueeze(-1)
        return features * weights


def _last_conv_out_channels(module: nn.Module) -> int:
    convolutions = [layer for layer in module.modules() if isinstance(layer, nn.Conv2d)]
    if not convolutions:
        raise RuntimeError(f"cannot infer output channels for {type(module).__name__}")
    return int(convolutions[-1].out_channels)


def _infer_feature_channels(model: SemanticSegmentationModel, layer_indices: tuple[int, ...]) -> dict[int, int]:
    """Infer actual feature widths from the installed Ultralytics graph."""
    captured: dict[int, int] = {}
    hooks = []

    def capture(index: int):
        def hook(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
            if not isinstance(output, torch.Tensor) or output.ndim != 4:
                raise RuntimeError(f"YOLO26 layer {index} did not emit a 4-D feature tensor")
            captured[index] = int(output.shape[1])

        return hook

    for index in layer_indices:
        hooks.append(model.model[index].register_forward_hook(capture(index)))
    was_training = model.training
    model.eval()
    parameter = next(model.parameters())
    with torch.inference_mode():
        model(torch.zeros(1, parameter.shape[1] if parameter.ndim == 4 else 3, 128, 128, device=parameter.device, dtype=parameter.dtype))
    for hook in hooks:
        hook.remove()
    model.train(was_training)
    missing = set(layer_indices) - set(captured)
    if missing:
        raise RuntimeError(f"failed to capture YOLO26 feature layers: {sorted(missing)}")
    return captured


class SharedYolo26(nn.Module):
    """Shared YOLO26 semantic model with native S1/S2 adapter routing.

    The input projections are a channel bridge (S1 2->3, S2 9->3).  The
    sensor-specific feature adapters are inserted after the P3 and P4
    backbone features (Ultralytics layer indices 4 and 6); P5 is deliberately
    left untouched.  All groups then use the same YOLO26 backbone and head.
    """

    P3_LAYER_INDEX = 4
    P4_LAYER_INDEX = 6

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
        self.core = build_yolo26_semantic(
            scale=scale,
            in_channels=common_channels,
            num_classes=num_classes,
        )
        self.input_adapters = SensorInputAdapter(common_channels=common_channels)
        feature_channels = _infer_feature_channels(
            self.core,
            (self.P3_LAYER_INDEX, self.P4_LAYER_INDEX),
        )
        p3_channels = feature_channels[self.P3_LAYER_INDEX]
        p4_channels = feature_channels[self.P4_LAYER_INDEX]
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
            {
                "P3": EfficientChannelAttention(),
                "P4": EfficientChannelAttention(),
            }
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
                x = (
                    saved[layer.f]
                    if isinstance(layer.f, int)
                    else [x if index == -1 else saved[index] for index in layer.f]
                )
            x = layer(x)
            if layer.i == self.P3_LAYER_INDEX:
                x = self.feature_adapters[f"P3_{sensor}"](x)
                if self.use_attention:
                    x = self.attention["P3"](x)
            elif layer.i == self.P4_LAYER_INDEX:
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
        common_image = self.input_adapters(image, name)
        return self._forward_core(common_image, name)

    def forward_grouped(self, groups: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
        """Forward grouped native batches produced by ``grouped_sensor_collate``."""
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


class QualityRouter(nn.Module):
    """Predict clean/degraded adapter weights from quality values only."""

    def __init__(self, feature_dim: int = 4, hidden_dim: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.register_buffer("quality_center", torch.zeros(feature_dim))
        self.register_buffer("quality_scale", torch.ones(feature_dim))
        self.log_noise_index = True

    def set_quality_normalization(
        self,
        center: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        """Install train-only robust-normalization statistics."""
        if center.shape != self.quality_center.shape or scale.shape != self.quality_scale.shape:
            raise ValueError("quality normalization statistics must have shape [4]")
        if not torch.isfinite(center).all() or not torch.isfinite(scale).all():
            raise ValueError("quality normalization statistics must be finite")
        if (scale <= 0).any():
            raise ValueError("quality normalization scales must be positive")
        self.quality_center.copy_(center.to(device=self.quality_center.device, dtype=self.quality_center.dtype))
        self.quality_scale.copy_(scale.to(device=self.quality_scale.device, dtype=self.quality_scale.dtype))

    def forward(
        self,
        quality: torch.Tensor,
        quality_available: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if quality.ndim != 2 or quality.shape[1] != 4:
            raise ValueError(f"quality must have shape [B,4], got {tuple(quality.shape)}")
        if quality_available is None:
            quality_available = torch.ones_like(quality)
        if quality_available.shape != quality.shape:
            raise ValueError("quality_available must have the same shape as quality")
        transformed_quality = quality.to(dtype=torch.float32).clone()
        if self.log_noise_index:
            transformed_quality[:, 3] = torch.log1p(transformed_quality[:, 3].clamp_min(0.0))
        transformed_quality = (
            transformed_quality - self.quality_center.to(dtype=transformed_quality.dtype)
        ) / self.quality_scale.to(dtype=transformed_quality.dtype).clamp_min(1e-6)
        transformed_quality = torch.nan_to_num(
            transformed_quality,
            nan=0.0,
            posinf=5.0,
            neginf=-5.0,
        ).clamp(-5.0, 5.0)
        # An unavailable feature must not leak its fill value after
        # normalization.  The availability channel remains available to the
        # router, while the corresponding normalized feature is neutralized.
        transformed_quality = transformed_quality * quality_available.to(dtype=torch.float32)
        inputs = torch.cat(
            [transformed_quality, quality_available.to(dtype=torch.float32)],
            dim=1,
        )
        if not torch.isfinite(inputs).all():
            raise ValueError("quality router received non-finite quality values")
        return torch.softmax(self.network(inputs), dim=1)


class QualityRoutedYolo26(SharedYolo26):
    """M2/M3 shared YOLO26 with quality-conditioned clean/degraded adapters.

    Each sensor has a clean and degraded residual adapter at P3 and P4.  The
    router receives the four input-only quality features and their availability
    mask; ``sensor_index`` is represented by the sensor-specific router and
    adapter bank, never concatenated into the quality vector.
    """

    def __init__(
        self,
        *,
        scale: str = "n",
        common_channels: int = 3,
        num_classes: int = 2,
    ) -> None:
        super().__init__(
            scale=scale,
            common_channels=common_channels,
            num_classes=num_classes,
        )
        self.routers = nn.ModuleDict(
            {
                "S1": QualityRouter(),
                "S2": QualityRouter(),
            }
        )
        self.feature_adapters = nn.ModuleDict(
            {
                f"{level}_{sensor}_{branch}": ResidualFeatureAdapter(channels)
                for level, channels in self.feature_channels.items()
                for sensor in ("S1", "S2")
                for branch in ("clean", "degraded")
            }
        )

    def set_quality_normalizer(self, statistics: Mapping[str, object]) -> None:
        """Set per-sensor router statistics fitted on training events only."""
        for sensor in ("S1", "S2"):
            entry = statistics.get(sensor)
            if not isinstance(entry, Mapping):
                raise ValueError(f"quality statistics miss sensor {sensor}")
            center = torch.tensor(entry["center"], dtype=torch.float32)
            scale = torch.tensor(entry["scale"], dtype=torch.float32)
            self.routers[sensor].set_quality_normalization(center, scale)

    def _forward_routed_core(
        self,
        common_image: torch.Tensor,
        sensor: str,
        route_weights: torch.Tensor,
    ) -> tuple[torch.Tensor | tuple[torch.Tensor, ...], torch.Tensor]:
        x = common_image
        saved: list[object] = []
        adapter_deltas: list[torch.Tensor] = []
        weights = route_weights.to(dtype=common_image.dtype)
        for layer in self.core.model:
            if layer.f != -1:
                x = (
                    saved[layer.f]
                    if isinstance(layer.f, int)
                    else [x if index == -1 else saved[index] for index in layer.f]
                )
            x = layer(x)
            if layer.i in (self.P3_LAYER_INDEX, self.P4_LAYER_INDEX):
                level = "P3" if layer.i == self.P3_LAYER_INDEX else "P4"
                clean = self.feature_adapters[f"{level}_{sensor}_clean"](x)
                degraded = self.feature_adapters[f"{level}_{sensor}_degraded"](x)
                x = (
                    weights[:, 0].view(-1, 1, 1, 1) * clean
                    + weights[:, 1].view(-1, 1, 1, 1) * degraded
                )
                adapter_deltas.append((degraded - clean).abs().mean())
            saved.append(x if layer.i in self.core.save else None)
        if not adapter_deltas:
            raise RuntimeError("quality-routed core did not visit P3/P4")
        return x, torch.stack(adapter_deltas).mean()

    def forward_native(
        self,
        image: torch.Tensor,
        sensor: str | int | torch.Tensor,
        quality: torch.Tensor | None = None,
        quality_available: torch.Tensor | None = None,
        *,
        return_diagnostics: bool = True,
    ) -> object:
        name = self._sensor_name(sensor)
        if quality is None:
            raise ValueError("quality is required for a quality-routed forward")
        common_image = self.input_adapters(image, name)
        route_weights = self.routers[name](quality, quality_available)
        logits, adapter_delta = self._forward_routed_core(
            common_image,
            name,
            route_weights,
        )
        if not return_diagnostics:
            return logits
        return {
            "logits": logits,
            "route_weights": route_weights,
            "adapter_delta": adapter_delta,
        }

    def forward_grouped(self, groups: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
        outputs: dict[str, object] = {}
        for sensor in ("S1", "S2"):
            if sensor in groups:
                batch = groups[sensor]
                outputs[sensor] = self.forward_native(
                    batch["image"],
                    sensor,
                    batch["quality"],
                    batch["quality_available"],
                )  # type: ignore[arg-type]
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
                return self.forward_native(
                    inputs["image"],
                    sensor,
                    inputs.get("quality"),
                    inputs.get("quality_available"),
                )  # type: ignore[arg-type]
            return self.forward_grouped(inputs)  # type: ignore[arg-type]
        if sensor is None:
            raise ValueError("sensor is required when forwarding a native tensor")
        raise ValueError("quality is required when forwarding a native tensor directly")
