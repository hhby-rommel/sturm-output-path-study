# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class OptimizationSpec:
    """Versioned optimizer contract persisted with every run."""

    version: str
    optimizer: str
    learning_rate: float
    weight_decay: float
    scheduler: str
    warmup_fraction: float
    minimum_lr_ratio: float
    exclude_norm_and_bias_from_weight_decay: bool
    track_parameter_update: bool

    @classmethod
    def from_mapping(cls, version: str, values: Mapping[str, Any]) -> "OptimizationSpec":
        spec = cls(version=version, **dict(values))
        if spec.optimizer != "adamw":
            raise ValueError(f"unsupported optimizer: {spec.optimizer}")
        if spec.scheduler not in {"none", "warmup_cosine"}:
            raise ValueError(f"unsupported scheduler: {spec.scheduler}")
        if spec.learning_rate <= 0 or spec.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if not 0.0 <= spec.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if not 0.0 <= spec.minimum_lr_ratio <= 1.0:
            raise ValueError("minimum_lr_ratio must be in [0, 1]")
        if spec.scheduler == "none" and (spec.warmup_fraction or spec.minimum_lr_ratio != 1.0):
            raise ValueError("scheduler=none requires zero warmup and minimum_lr_ratio=1")
        return spec

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parameter_groups(
    model: torch.nn.Module,
    *,
    weight_decay: float,
    exclude_norm_and_bias: bool,
) -> list[dict[str, Any]]:
    if not exclude_norm_and_bias:
        return [{"params": list(model.parameters()), "weight_decay": weight_decay}]

    normalization_types = (
        torch.nn.modules.batchnorm._BatchNorm,
        torch.nn.modules.instancenorm._InstanceNorm,
        torch.nn.GroupNorm,
        torch.nn.LayerNorm,
    )
    normalized_parameter_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, normalization_types):
            normalized_parameter_ids.update(id(parameter) for parameter in module.parameters(recurse=False))
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith(".bias") or id(parameter) in normalized_parameter_ids:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(model: torch.nn.Module, spec: OptimizationSpec) -> torch.optim.Optimizer:
    groups = _parameter_groups(
        model,
        weight_decay=spec.weight_decay,
        exclude_norm_and_bias=spec.exclude_norm_and_bias_from_weight_decay,
    )
    return torch.optim.AdamW(groups, lr=spec.learning_rate)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    spec: OptimizationSpec,
    *,
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if spec.scheduler == "none":
        return None
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    warmup_steps = max(1, round(total_steps * spec.warmup_fraction))

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return max((step + 1) / warmup_steps, 1.0 / warmup_steps)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return spec.minimum_lr_ratio + (1.0 - spec.minimum_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)
