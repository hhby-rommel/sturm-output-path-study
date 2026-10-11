# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Losses for quality-conditioned routing and degradation consistency."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def confidence_weight(clean_probability: torch.Tensor, floor: float = 0.1) -> torch.Tensor:
    """Detached confidence weights; uncertain boundary pixels are downweighted."""
    confidence = 2.0 * torch.abs(clean_probability.detach() - 0.5)
    return torch.clamp(confidence, min=floor)


def masked_consistency_mse(
    clean_probability: torch.Tensor,
    degraded_probability: torch.Tensor,
    valid_mask: torch.Tensor,
    floor: float = 0.1,
) -> torch.Tensor:
    """Use the clean view as a detached teacher for the degraded view."""
    if clean_probability.shape != degraded_probability.shape:
        raise ValueError("clean and degraded probabilities must have the same shape")
    if valid_mask.ndim == clean_probability.ndim - 1:
        valid_mask = valid_mask.unsqueeze(1)
    valid_mask = valid_mask.to(dtype=clean_probability.dtype)
    weights = confidence_weight(clean_probability, floor=floor)
    loss = weights * (degraded_probability - clean_probability.detach()).pow(2)
    loss = loss * valid_mask
    denominator = (weights * valid_mask).sum().clamp_min(1.0)
    return loss.sum() / denominator


def route_direction_hinge(
    clean_degraded_weight: torch.Tensor,
    degraded_degraded_weight: torch.Tensor,
    quality_delta: torch.Tensor,
    beta: float = 0.5,
    margin_min: float = 0.05,
    margin_max: float = 0.20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Penalize failure of the degraded-adapter weight to increase.

    Returns the scalar loss and the detached dynamic margin for diagnostics.
    """
    delta = quality_delta.detach().to(dtype=clean_degraded_weight.dtype)
    margin = torch.clamp(beta * delta, min=margin_min, max=margin_max)
    loss = F.relu(margin - (degraded_degraded_weight - clean_degraded_weight))
    return loss.mean(), margin.detach()


def route_quality_order_hinge(
    degraded_weight: torch.Tensor,
    quality_score: torch.Tensor,
    margin: float = 0.05,
    min_quality_gap: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Supervise cross-tile route ordering within one sensor batch.

    ``quality_score`` is defined so that larger values indicate worse quality.
    For every non-tied pair, the worse tile is therefore required to receive
    a larger degraded-branch weight.  The score is detached because it is a
    target derived from the audited input-quality metadata; gradients flow
    only through ``degraded_weight``.

    The second return value is the number of ordered pairs used.  A zero-pair
    batch returns a differentiable zero so callers can aggregate safely.
    """
    if degraded_weight.ndim != 1 or quality_score.ndim != 1:
        raise ValueError("degraded_weight and quality_score must be one-dimensional")
    if degraded_weight.shape != quality_score.shape:
        raise ValueError("degraded_weight and quality_score must have the same shape")
    if degraded_weight.numel() < 2:
        return degraded_weight.sum() * 0.0, degraded_weight.new_zeros((), dtype=torch.long)

    score = quality_score.detach().to(dtype=degraded_weight.dtype)
    score_delta = score[:, None] - score[None, :]
    upper = torch.triu(
        torch.ones_like(score_delta, dtype=torch.bool),
        diagonal=1,
    )
    pair_mask = upper & (score_delta.abs() > float(min_quality_gap))
    if not torch.any(pair_mask):
        return degraded_weight.sum() * 0.0, degraded_weight.new_zeros((), dtype=torch.long)

    weight_delta = degraded_weight[:, None] - degraded_weight[None, :]
    signed_margin = torch.sign(score_delta[pair_mask]) * weight_delta[pair_mask]
    loss = F.relu(float(margin) - signed_margin).mean()
    return loss, pair_mask.sum().detach()


def ramp_weight(epoch_fraction: float, maximum: float, warmup_fraction: float = 0.2, ramp_fraction: float = 0.2) -> float:
    """Zero during warmup, linear during ramp, then fixed at maximum."""
    if epoch_fraction <= warmup_fraction:
        return 0.0
    ramp_end = warmup_fraction + ramp_fraction
    if epoch_fraction < ramp_end:
        return maximum * (epoch_fraction - warmup_fraction) / ramp_fraction
    return maximum
