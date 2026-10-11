# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Sensor-aware, label-preserving synthetic degradations for M3 diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def _spatial_noise_index(
    image: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute a finite per-tile spatial coefficient-of-variation proxy."""
    valid = valid_mask.to(dtype=image.dtype).unsqueeze(1)
    count = valid.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    mean = (image * valid).sum(dim=(2, 3), keepdim=True) / count
    variance = (((image - mean) ** 2) * valid).sum(dim=(2, 3), keepdim=True) / count
    channel_cv = torch.sqrt(variance.clamp_min(0.0)) / (mean.abs() + eps)
    channel_cv = channel_cv.squeeze(-1).squeeze(-1)
    # CUDA median-with-indices is not deterministic on this environment.
    # Sorting is deterministic and the small fixed channel count is cheap.
    ordered = torch.sort(channel_cv, dim=1).values
    return ordered[:, ordered.shape[1] // 2].clamp_min(0.0)


def quality_badness_tensor(
    quality: torch.Tensor,
    quality_available: torch.Tensor,
) -> torch.Tensor:
    """Torch equivalent of the input-only quality badness scalar."""
    if quality.shape != quality_available.shape or quality.ndim != 2 or quality.shape[1] != 4:
        raise ValueError("quality tensors must both have shape [B,4]")
    available = quality_available.to(dtype=quality.dtype)
    values = torch.stack(
        [
            1.0 - quality[:, 0],
            quality[:, 1],
            quality[:, 2],
            quality[:, 3] / (1.0 + quality[:, 3].abs()),
        ],
        dim=1,
    )
    return (values * available).sum(dim=1) / available.sum(dim=1).clamp_min(1.0)


def degrade_sensor_batch(
    batch: Mapping[str, Any],
    sensor: str,
    *,
    severity: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Create one degraded view while preserving target/event identity.

    S1 receives multiplicative speckle-like noise; S2 receives channel-scaled
    additive noise.  Both receive 5--15% valid-pixel dropout, whose pixels are
    excluded from the degraded loss and consistency mask.  The operation uses
    no flood labels and keeps invalid source pixels invalid.
    """
    name = sensor.upper()
    if name not in {"S1", "S2"}:
        raise ValueError(f"sensor must be S1 or S2, got {sensor!r}")
    image = batch["image"]
    valid_mask = batch["valid_mask"].bool()
    if not isinstance(image, torch.Tensor) or not isinstance(valid_mask, torch.Tensor):
        raise TypeError("degradation requires tensor image and valid_mask")
    if image.ndim != 4 or valid_mask.ndim != 3 or image.shape[0] != valid_mask.shape[0]:
        raise ValueError("image must be BCHW and valid_mask must be BHW")
    if severity is None:
        severity = 0.5 + 0.5 * torch.rand(image.shape[0], device=image.device)
    severity = severity.to(device=image.device, dtype=image.dtype).flatten()
    if severity.shape != (image.shape[0],):
        raise ValueError("severity must have shape [B]")

    if name == "S1":
        noise = torch.randn_like(image)
        degraded_image = image * (1.0 + 0.12 * severity[:, None, None, None] * noise)
    else:
        scale = image.flatten(2).std(dim=2, unbiased=False).clamp_min(1e-3)
        noise = torch.randn_like(image)
        degraded_image = image + 0.20 * severity[:, None, None, None] * scale[:, :, None, None] * noise

    dropout_probability = 0.05 + 0.10 * severity[:, None, None]
    dropout = (torch.rand_like(valid_mask, dtype=image.dtype) < dropout_probability) & valid_mask
    degraded_valid = valid_mask & ~dropout
    degraded_image = torch.where(
        degraded_valid.unsqueeze(1),
        degraded_image,
        torch.zeros_like(degraded_image),
    )

    quality = batch["quality"].clone().to(dtype=torch.float32)
    quality_available = batch["quality_available"].clone().to(dtype=torch.float32)
    quality[:, 0] = degraded_valid.to(dtype=torch.float32).mean(dim=(1, 2))
    quality[:, 1] = 1.0 - quality[:, 0]
    quality[:, 3] = _spatial_noise_index(degraded_image.float(), degraded_valid)
    quality_score = quality_badness_tensor(quality, quality_available)

    degraded = dict(batch)
    degraded["image"] = degraded_image
    degraded["valid_mask"] = degraded_valid
    degraded["quality"] = quality
    degraded["quality_available"] = quality_available
    degraded["quality_score"] = quality_score
    degraded["degradation_severity"] = severity
    degraded["degradation_retention_rate"] = (
        degraded_valid.sum(dim=(1, 2)).float()
        / valid_mask.sum(dim=(1, 2)).float().clamp_min(1.0)
    )
    return degraded
