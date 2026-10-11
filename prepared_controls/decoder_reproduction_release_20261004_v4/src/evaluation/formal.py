# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Deterministic, event-aware evaluation for formal experiments."""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np
import torch

from src.training.degradation import degrade_sensor_batch
from src.training.masked_segmentation import MaskedSegmentationMeter, MaskedSemanticLoss, loss_from_batch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def fixed_rng(seed: int, tag: int = 1) -> Iterator[None]:
    """Temporarily fix every RNG and restore the caller's state afterwards."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        seed_everything(seed + 10000 + tag)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def is_better_checkpoint(candidate: Mapping[str, float], incumbent: Mapping[str, float] | None) -> bool:
    """Apply the preregistered degraded, clean, then earlier-epoch ordering."""
    if incumbent is None:
        return True
    candidate_key = (
        float(candidate["degraded_flood_f1"]),
        float(candidate["clean_flood_f1"]),
        -int(candidate["epoch"]),
    )
    incumbent_key = (
        float(incumbent["degraded_flood_f1"]),
        float(incumbent["clean_flood_f1"]),
        -int(incumbent["epoch"]),
    )
    return candidate_key > incumbent_key


def _move(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _metrics(meter: MaskedSegmentationMeter) -> dict[str, Any]:
    return meter.compute()


def _slice_outputs(outputs: Any, index: int) -> Any:
    if isinstance(outputs, torch.Tensor):
        return outputs[index : index + 1]
    if isinstance(outputs, tuple):
        return tuple(_slice_outputs(value, index) for value in outputs)
    if isinstance(outputs, list):
        return [_slice_outputs(value, index) for value in outputs]
    if isinstance(outputs, Mapping):
        return {key: _slice_outputs(value, index) if isinstance(value, (torch.Tensor, tuple, list, Mapping)) else value for key, value in outputs.items()}
    raise TypeError(f"unsupported model output type: {type(outputs).__name__}")


@torch.no_grad()
def _evaluate_view(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    criterion: MaskedSemanticLoss,
    device: torch.device,
    *,
    degraded: bool,
) -> dict[str, Any]:
    model.eval()
    overall = MaskedSegmentationMeter(2, criterion.ignore_index)
    sensor_meters: dict[str, MaskedSegmentationMeter] = {}
    event_meters: dict[str, MaskedSegmentationMeter] = {}
    losses: list[float] = []
    invalid_pixels = 0
    total_pixels = 0
    for grouped in loader:
        for sensor, native in grouped.items():
            batch = _move(native, device)
            if degraded:
                batch = degrade_sensor_batch(batch, sensor)
            outputs = model(batch)
            loss, _ = loss_from_batch(outputs, batch, criterion)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite evaluation loss for {sensor}")
            losses.append(float(loss.detach().cpu()))
            sensor_meter = sensor_meters.setdefault(sensor, MaskedSegmentationMeter(2, criterion.ignore_index))
            overall.update(outputs, batch["mask"], batch["valid_mask"])
            sensor_meter.update(outputs, batch["mask"], batch["valid_mask"])
            valid_mask = batch["valid_mask"].detach().cpu()
            total_pixels += valid_mask.numel()
            invalid_pixels += int((~valid_mask.bool()).sum())
            event_ids = list(batch["event_id"])
            for index, event_id in enumerate(event_ids):
                meter = event_meters.setdefault(event_id, MaskedSegmentationMeter(2, criterion.ignore_index))
                meter.update(
                    _slice_outputs(outputs, index),
                    batch["mask"][index : index + 1],
                    batch["valid_mask"][index : index + 1],
                )
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "overall": _metrics(overall),
        "per_sensor": {name: _metrics(meter) for name, meter in sorted(sensor_meters.items())},
        "per_event": {name: _metrics(meter) for name, meter in sorted(event_meters.items())},
        "invalid_pixel_count": invalid_pixels,
        "invalid_pixel_ratio": invalid_pixels / max(total_pixels, 1),
    }


def evaluate_clean_and_degraded(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    criterion: MaskedSemanticLoss,
    device: torch.device,
    *,
    seed: int,
    tag: int = 1,
) -> dict[str, Any]:
    """Evaluate clean and one fixed degraded view without perturbing training RNG."""
    clean = _evaluate_view(model, loader, criterion, device, degraded=False)
    with fixed_rng(seed, tag):
        degraded = _evaluate_view(model, loader, criterion, device, degraded=True)
    return {
        "fixed_degraded_tag": tag,
        "clean": clean,
        "degraded": degraded,
        "clean_degraded_f1_gap": float(clean["overall"]["flood_f1"] - degraded["overall"]["flood_f1"]),
    }
