# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sensor-specific image-quality features for SQA-YOLO26-DCL.

The functions in this module deliberately use only input-image properties and
valid/nodata masks. They must not use flood labels when computing quality.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


QUALITY_FEATURE_NAMES = (
    "valid_pixel_ratio",
    "nodata_ratio",
    "saturation_ratio",
    "sensor_specific_noise_index",
)


def _as_float32(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"expected CHW or HW array, got shape={array.shape}")
    return array


def valid_pixel_ratio(image: np.ndarray, nodata_value: Optional[float] = None) -> float:
    """Return the fraction of pixels valid across all channels."""
    x = _as_float32(image)
    valid = np.isfinite(x).all(axis=0)
    if nodata_value is not None:
        valid &= (x != nodata_value).all(axis=0)
    return float(valid.mean())


def nodata_ratio(image: np.ndarray, nodata_value: Optional[float] = None) -> float:
    return 1.0 - valid_pixel_ratio(image, nodata_value=nodata_value)


def saturation_ratio(
    image: np.ndarray,
    low: float,
    high: float,
    valid_mask: Optional[np.ndarray] = None,
) -> float:
    """Fraction of finite channel values at a declared sensor range boundary."""
    x = _as_float32(image)
    finite = np.isfinite(x)
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != x.shape[1:]:
            raise ValueError(f"valid_mask must be HW, got {valid.shape}")
        finite &= valid[None, ...]
    saturated = finite & ((x <= low) | (x >= high))
    return float(saturated.sum() / max(finite.sum(), 1))


def speckle_cv_index(image: np.ndarray, window: int = 7, eps: float = 1e-6) -> float:
    """Median local coefficient of variation on linearised S1 intensities.

    The input is assumed to be in dB. The implementation uses a lightweight
    integral-image box filter and returns a scalar per tile.
    """
    x = _as_float32(image)
    linear = np.power(10.0, np.clip(x, -80.0, 40.0) / 10.0)
    values = []
    pad = window // 2
    for channel in linear:
        padded = np.pad(channel, pad, mode="reflect")
        # Add a leading zero row/column so half-open window indices can reach
        # the padded boundary without an off-by-one access.
        integral = np.pad(padded, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        integral2 = np.pad(padded * padded, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        h, w = channel.shape
        y0, y1 = np.arange(h), np.arange(h) + window
        x0, x1 = np.arange(w), np.arange(w) + window
        sums = (
            integral[y1[:, None], x1[None, :]]
            - integral[y0[:, None], x1[None, :]]
            - integral[y1[:, None], x0[None, :]]
            + integral[y0[:, None], x0[None, :]]
        )
        sums2 = (
            integral2[y1[:, None], x1[None, :]]
            - integral2[y0[:, None], x1[None, :]]
            - integral2[y1[:, None], x0[None, :]]
            + integral2[y0[:, None], x0[None, :]]
        )
        mean = sums / float(window * window)
        variance = np.maximum(sums2 / float(window * window) - mean * mean, 0.0)
        values.append(np.median(np.sqrt(variance) / (mean + eps)))
    return float(np.median(values))


def sensor_specific_noise_index(
    image: np.ndarray,
    sensor: str,
    valid_mask: Optional[np.ndarray] = None,
    eps: float = 1e-6,
) -> float:
    """Return an input-only sensor noise proxy for either sensor.

    S1 uses the local speckle coefficient of variation used by the project
    protocol.  S2 has no official per-tile noise flag in the supplied
    metadata, so it uses the median spatial coefficient of variation across
    channels.  This is a diagnostic proxy, not a physical uncertainty
    estimate, and never reads the flood label.
    """
    name = sensor.upper()
    x = _as_float32(image)
    if valid_mask is None:
        valid = np.isfinite(x).all(axis=0)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != x.shape[1:]:
            raise ValueError(f"valid_mask must be HW, got {valid.shape}")
        valid &= np.isfinite(x).all(axis=0)
    if name == "S1":
        return float(np.nan_to_num(speckle_cv_index(x), nan=0.0, posinf=0.0, neginf=0.0))
    if name != "S2":
        raise ValueError(f"sensor must be S1 or S2, got {sensor!r}")
    values = []
    for channel in x:
        finite_values = channel[valid]
        if finite_values.size < 2:
            continue
        mean = float(np.mean(finite_values))
        std = float(np.std(finite_values))
        values.append(std / (abs(mean) + eps))
    if not values:
        return 0.0
    return float(np.nan_to_num(np.median(values), nan=0.0, posinf=0.0, neginf=0.0))


def quality_vector(
    image: np.ndarray,
    sensor: str,
    nodata_value: Optional[float] = None,
    saturation_range: Optional[tuple[float, float]] = None,
) -> Dict[str, float]:
    """Compute the audit-time quality vector without label access."""
    x = _as_float32(image)
    valid = np.isfinite(x).all(axis=0)
    if nodata_value is not None and np.isfinite(nodata_value):
        valid &= (x != nodata_value).all(axis=0)
    result = {
        "valid_pixel_ratio": float(valid.mean()),
        "nodata_ratio": float(1.0 - valid.mean()),
    }
    if saturation_range is None:
        # No official clipping range is present in the supplied metadata.
        # Keep the tensor finite and expose availability explicitly.
        result["saturation_ratio"] = 0.0
        result["saturation_ratio_available"] = 0.0
    else:
        result["saturation_ratio"] = saturation_ratio(
            image, saturation_range[0], saturation_range[1], valid_mask=valid
        )
        result["saturation_ratio_available"] = 1.0
    result["sensor_specific_noise_index"] = sensor_specific_noise_index(
        x, sensor, valid_mask=valid
    )
    result["sensor_specific_noise_index_available"] = 1.0
    return result


def quality_badness(vector: Dict[str, float]) -> float:
    """Map the finite quality vector to a scalar where larger means worse."""
    available = np.asarray(
        [
            1.0,
            1.0,
            float(vector.get("saturation_ratio_available", 0.0)),
            float(vector.get("sensor_specific_noise_index_available", 1.0)),
        ],
        dtype=np.float32,
    )
    values = np.asarray(
        [
            1.0 - float(vector["valid_pixel_ratio"]),
            float(vector["nodata_ratio"]),
            float(vector["saturation_ratio"]),
            float(vector["sensor_specific_noise_index"])
            / (1.0 + abs(float(vector["sensor_specific_noise_index"]))),
        ],
        dtype=np.float32,
    )
    return float((values * available).sum() / max(float(available.sum()), 1.0))
