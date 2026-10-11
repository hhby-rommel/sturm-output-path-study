# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Small, explicit training/validation loops for grouped S1/S2 batches."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .degradation import degrade_sensor_batch
from .dcl_losses import masked_consistency_mse, route_direction_hinge, route_quality_order_hinge
from .masked_segmentation import (
    MaskedSegmentationMeter,
    MaskedSemanticLoss,
    _main_logits,
    loss_from_batch,
)


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device=device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _as_sensor_groups(batch: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Normalize strict same-sensor or grouped-collate batches to one format."""
    if "image" in batch:
        sensors = batch.get("sensor")
        if isinstance(sensors, list):
            if not sensors or len(set(sensors)) != 1:
                raise ValueError("a native batch must contain one sensor")
            sensor = sensors[0]
        else:
            sensor = str(sensors)
        return {sensor: batch}
    groups = {sensor: value for sensor, value in batch.items() if sensor in {"S1", "S2"}}
    if not groups:
        raise ValueError("batch contains neither a native image nor S1/S2 groups")
    return groups  # type: ignore[return-value]


def _mean_or_zero(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _route_weights(outputs: object) -> torch.Tensor:
    if not isinstance(outputs, Mapping) or "route_weights" not in outputs:
        raise ValueError("M2/M3 model output must contain route_weights")
    weights = outputs["route_weights"]
    if not isinstance(weights, torch.Tensor) or weights.ndim != 2 or weights.shape[1] != 2:
        raise ValueError("route_weights must have shape [B,2]")
    return weights


def _flood_probability(outputs: object, target_size: tuple[int, int]) -> torch.Tensor:
    logits = _main_logits(outputs)  # type: ignore[arg-type]
    logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)
    return torch.softmax(logits, dim=1)[:, 1:2]


def _structured_diagnostics(outputs: object) -> dict[str, float]:
    if not isinstance(outputs, Mapping):
        return {}
    diagnostics: dict[str, float] = {}
    route = outputs.get("route_weights")
    if isinstance(route, torch.Tensor):
        diagnostics["clean_route_weight"] = float(route[:, 0].detach().mean().cpu())
        diagnostics["degraded_route_weight"] = float(route[:, 1].detach().mean().cpu())
    adapter_delta = outputs.get("adapter_delta")
    if isinstance(adapter_delta, torch.Tensor):
        diagnostics["adapter_delta"] = float(adapter_delta.detach().cpu())
    return diagnostics


def _freeze_batchnorm_running_stats(model: torch.nn.Module) -> None:
    """Keep BN affine parameters trainable while freezing running statistics."""
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def train_one_epoch(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    criterion: MaskedSemanticLoss,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    max_batches: int | None = None,
    grad_clip_norm: float | None = None,
    freeze_batchnorm: bool = False,
    model_variant: str = "m1",
    consistency_weight: float = 0.0,
    route_weight: float = 0.0,
    confidence_floor: float = 0.1,
    route_beta: float = 0.5,
    route_margin_min: float = 0.05,
    route_margin_max: float = 0.20,
    quality_order_weight: float = 0.0,
    quality_order_weight_by_sensor: Mapping[str, float] | None = None,
    quality_order_degraded_weight_by_sensor: Mapping[str, float] | None = None,
    quality_order_margin: float = 0.05,
    quality_order_min_gap: float = 0.0,
) -> dict[str, Any]:
    """Train one epoch with one optimizer step per grouped outer batch.

    ``freeze_batchnorm`` is useful for tiny overfit checks and very small
    sensor-group batches: affine BN parameters still learn, but running
    statistics cannot oscillate between S1 and S2.
    """
    model_variant = model_variant.lower()
    if model_variant not in {"m1", "m1_degraded", "m2", "m3", "fixed_route", "shared_dcl"}:
        raise ValueError(
            "model_variant must be m1, m1_degraded, m2, m3, fixed_route or shared_dcl, "
            f"got {model_variant!r}"
        )
    if model_variant in {"m1", "m1_degraded"} and (consistency_weight or route_weight):
        raise ValueError("M1 variants cannot receive consistency or route loss weights")
    model.train()
    if freeze_batchnorm:
        _freeze_batchnorm_running_stats(model)
    total_losses: list[float] = []
    sensor_losses: dict[str, list[float]] = defaultdict(list)
    sensor_valid_pixels: dict[str, int] = defaultdict(int)
    auxiliary_losses: dict[str, list[float]] = defaultdict(list)
    route_deltas: list[float] = []
    retention_rates: list[float] = []
    adapter_deltas: list[float] = []
    steps = 0

    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        groups = _as_sensor_groups(raw_batch)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for sensor, native_batch in groups.items():
            batch = _move_batch(native_batch, device)
            outputs = model(batch)
            clean_loss, items = loss_from_batch(outputs, batch, criterion)
            loss = clean_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss for sensor {sensor}")
            sensor_losses[sensor].append(float(clean_loss.detach().cpu()))
            sensor_valid_pixels[sensor] += int(items["valid_pixels"].detach().cpu())

            if model_variant in {"m1_degraded", "m2", "m3", "fixed_route", "shared_dcl"}:
                degraded_batch = degrade_sensor_batch(batch, sensor)
                degraded_outputs = model(degraded_batch)
                degraded_loss, degraded_items = loss_from_batch(
                    degraded_outputs,
                    degraded_batch,
                    criterion,
                )
                loss = loss + degraded_loss
                auxiliary_losses["degraded_segmentation"].append(
                    float(degraded_loss.detach().cpu())
                )
                sensor_valid_pixels[sensor] += int(degraded_items["valid_pixels"].detach().cpu())
                retention_rates.append(
                    float(degraded_batch["degradation_retention_rate"].detach().mean().cpu())
                )
                if model_variant in {"m2", "m3"}:
                    clean_weights = _route_weights(outputs)
                    degraded_weights = _route_weights(degraded_outputs)
                    route_loss, _margin = route_direction_hinge(
                        clean_weights[:, 1],
                        degraded_weights[:, 1],
                        degraded_batch["quality_score"] - batch["quality_score"],
                        beta=route_beta,
                        margin_min=route_margin_min,
                        margin_max=route_margin_max,
                    )
                    if route_weight:
                        loss = loss + float(route_weight) * route_loss
                    auxiliary_losses["route"].append(float(route_loss.detach().cpu()))
                    order_weight = float(quality_order_weight)
                    if quality_order_weight_by_sensor is not None:
                        order_weight = float(quality_order_weight_by_sensor.get(sensor, 0.0))
                    if order_weight:
                        clean_order_loss, clean_pair_count = route_quality_order_hinge(
                            clean_weights[:, 1],
                            batch["quality_score"],
                            margin=quality_order_margin,
                            min_quality_gap=quality_order_min_gap,
                        )
                        degraded_order_loss, degraded_pair_count = route_quality_order_hinge(
                            degraded_weights[:, 1],
                            degraded_batch["quality_score"],
                            margin=quality_order_margin,
                            min_quality_gap=quality_order_min_gap,
                        )
                        degraded_order_weight = order_weight
                        if quality_order_degraded_weight_by_sensor is not None:
                            degraded_order_weight = float(
                                quality_order_degraded_weight_by_sensor.get(sensor, 0.0)
                            )
                        order_loss = 0.5 * (clean_order_loss + degraded_order_loss)
                        weighted_order_loss = (
                            order_weight * clean_order_loss
                            + degraded_order_weight * degraded_order_loss
                        )
                        pair_count = clean_pair_count + degraded_pair_count
                        loss = loss + weighted_order_loss
                        auxiliary_losses["quality_order_clean"].append(
                            float(clean_order_loss.detach().cpu())
                        )
                        auxiliary_losses["quality_order_degraded"].append(
                            float(degraded_order_loss.detach().cpu())
                        )
                        auxiliary_losses["quality_order"].append(float(order_loss.detach().cpu()))
                        auxiliary_losses["quality_order_pairs"].append(float(pair_count.cpu()))
                    route_deltas.append(
                        float((degraded_weights[:, 1] - clean_weights[:, 1]).detach().mean().cpu())
                    )
                    adapter_deltas.extend(
                        value
                        for value in (
                            _structured_diagnostics(outputs).get("adapter_delta"),
                            _structured_diagnostics(degraded_outputs).get("adapter_delta"),
                        )
                        if value is not None
                    )
                if model_variant in {"m3", "shared_dcl"} and consistency_weight:
                    target_size = tuple(batch["mask"].shape[-2:])
                    consistency_loss = masked_consistency_mse(
                        _flood_probability(outputs, target_size),
                        _flood_probability(degraded_outputs, target_size),
                        batch["valid_mask"] & degraded_batch["valid_mask"],
                        floor=confidence_floor,
                    )
                    loss = loss + float(consistency_weight) * consistency_loss
                    auxiliary_losses["consistency"].append(
                        float(consistency_loss.detach().cpu())
                    )
            losses.append(loss)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite auxiliary training loss for sensor {sensor}")
        if not losses:
            continue
        batch_loss = torch.stack(losses).mean()
        batch_loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        total_losses.append(float(batch_loss.detach().cpu()))
        steps += 1

    return {
        "steps": steps,
        "loss": _mean_or_zero(total_losses),
        "per_sensor_loss": {sensor: _mean_or_zero(values) for sensor, values in sensor_losses.items()},
        "per_sensor_valid_pixels": dict(sensor_valid_pixels),
        "auxiliary_losses": {
            name: _mean_or_zero(values) for name, values in auxiliary_losses.items()
        },
        "mean_route_delta": _mean_or_zero(route_deltas),
        "mean_degraded_retention": _mean_or_zero(retention_rates),
        "mean_adapter_delta": _mean_or_zero(adapter_deltas),
    }


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    criterion: MaskedSemanticLoss,
    *,
    device: torch.device,
    max_batches: int | None = None,
    model_variant: str = "m1",
    evaluate_degraded: bool = False,
    confidence_floor: float = 0.1,
) -> dict[str, Any]:
    """Validate with the same masked loss and per-sensor confusion meters."""
    model_variant = model_variant.lower()
    if model_variant not in {"m1", "m2", "m3"}:
        raise ValueError(f"model_variant must be m1, m2 or m3, got {model_variant!r}")
    if evaluate_degraded and model_variant == "m1":
        raise ValueError("degraded evaluation requires an M2 or M3 model")
    model.eval()
    total_losses: list[float] = []
    sensor_losses: dict[str, list[float]] = defaultdict(list)
    meters: dict[str, MaskedSegmentationMeter] = {}
    overall_meter = MaskedSegmentationMeter(
        num_classes=criterion.num_classes,
        ignore_index=criterion.ignore_index,
    )
    degraded_meters: dict[str, MaskedSegmentationMeter] = {}
    degraded_overall_meter = MaskedSegmentationMeter(
        num_classes=criterion.num_classes,
        ignore_index=criterion.ignore_index,
    )
    degraded_losses: list[float] = []
    consistency_losses: list[float] = []
    route_deltas: list[float] = []
    retention_rates: list[float] = []
    adapter_deltas: list[float] = []
    steps = 0

    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        groups = _as_sensor_groups(raw_batch)
        batch_loss_values = []
        for sensor, native_batch in groups.items():
            batch = _move_batch(native_batch, device)
            outputs = model(batch)
            loss, _items = loss_from_batch(outputs, batch, criterion)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite validation loss for sensor {sensor}")
            batch_loss_values.append(float(loss.detach().cpu()))
            sensor_losses[sensor].append(float(loss.detach().cpu()))
            meter = meters.setdefault(
                sensor,
                MaskedSegmentationMeter(
                    num_classes=criterion.num_classes,
                    ignore_index=criterion.ignore_index,
                ),
            )
            meter.update(outputs, batch["mask"], batch["valid_mask"])
            overall_meter.update(outputs, batch["mask"], batch["valid_mask"])
            if evaluate_degraded:
                degraded_batch = degrade_sensor_batch(batch, sensor)
                degraded_outputs = model(degraded_batch)
                degraded_loss, _degraded_items = loss_from_batch(
                    degraded_outputs,
                    degraded_batch,
                    criterion,
                )
                if not torch.isfinite(degraded_loss):
                    raise FloatingPointError(f"non-finite degraded validation loss for sensor {sensor}")
                degraded_losses.append(float(degraded_loss.detach().cpu()))
                target_size = tuple(batch["mask"].shape[-2:])
                consistency_losses.append(
                    float(
                        masked_consistency_mse(
                            _flood_probability(outputs, target_size),
                            _flood_probability(degraded_outputs, target_size),
                            batch["valid_mask"] & degraded_batch["valid_mask"],
                            floor=confidence_floor,
                        )
                        .detach()
                        .cpu()
                    )
                )
                degraded_meter = degraded_meters.setdefault(
                    sensor,
                    MaskedSegmentationMeter(
                        num_classes=criterion.num_classes,
                        ignore_index=criterion.ignore_index,
                    ),
                )
                degraded_meter.update(
                    degraded_outputs,
                    degraded_batch["mask"],
                    degraded_batch["valid_mask"],
                )
                degraded_overall_meter.update(
                    degraded_outputs,
                    degraded_batch["mask"],
                    degraded_batch["valid_mask"],
                )
                clean_weights = _route_weights(outputs)
                degraded_weights = _route_weights(degraded_outputs)
                route_deltas.append(
                    float((degraded_weights[:, 1] - clean_weights[:, 1]).detach().mean().cpu())
                )
                retention_rates.append(
                    float(degraded_batch["degradation_retention_rate"].detach().mean().cpu())
                )
                adapter_deltas.extend(
                    value
                    for value in (
                        _structured_diagnostics(outputs).get("adapter_delta"),
                        _structured_diagnostics(degraded_outputs).get("adapter_delta"),
                    )
                    if value is not None
                )
        if batch_loss_values:
            total_losses.append(_mean_or_zero(batch_loss_values))
            steps += 1

    result = {
        "steps": steps,
        "loss": _mean_or_zero(total_losses),
        "per_sensor": {
            sensor: {"loss": _mean_or_zero(sensor_losses[sensor]), **meter.compute()}
            for sensor, meter in meters.items()
        },
        "overall": overall_meter.compute(),
    }
    if evaluate_degraded:
        result["degraded_loss"] = _mean_or_zero(degraded_losses)
        result["mean_consistency_loss"] = _mean_or_zero(consistency_losses)
        result["degraded_per_sensor"] = {
            sensor: meter.compute() for sensor, meter in degraded_meters.items()
        }
        result["degraded_overall"] = degraded_overall_meter.compute()
        result["mean_route_delta"] = _mean_or_zero(route_deltas)
        result["mean_degraded_retention"] = _mean_or_zero(retention_rates)
        result["mean_adapter_delta"] = _mean_or_zero(adapter_deltas)
    return result
