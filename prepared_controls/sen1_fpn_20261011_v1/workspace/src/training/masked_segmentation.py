# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Valid-pixel-aware semantic segmentation losses and metrics.

The STURM-Flood loader replaces non-finite input values with zero only after
recording their locations in ``valid_mask``. Every loss and metric here uses
that mask, so replacement values and unknown target pixels cannot contribute
to optimization or validation scores.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_IGNORE_INDEX = 255


def _main_logits(
    outputs: torch.Tensor
    | tuple[torch.Tensor, ...]
    | list[torch.Tensor]
    | Mapping[str, Any],
) -> torch.Tensor:
    """Extract the main logits from YOLO26 inference or train-time outputs."""
    if isinstance(outputs, Mapping):
        if "logits" not in outputs:
            raise ValueError("structured model output must contain a 'logits' field")
        outputs = outputs["logits"]
    if isinstance(outputs, (tuple, list)):
        if not outputs:
            raise ValueError("semantic model returned an empty output sequence")
        outputs = outputs[0]
    if not isinstance(outputs, torch.Tensor) or outputs.ndim != 4:
        raise ValueError(f"expected [B,C,H,W] semantic logits, got {type(outputs).__name__}")
    return outputs


def _target_and_weights(
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target.ndim != 3:
        raise ValueError(f"target must have shape [B,H,W], got {tuple(target.shape)}")
    if valid_mask.ndim == 4 and valid_mask.shape[1] == 1:
        valid_mask = valid_mask[:, 0]
    if valid_mask.ndim != 3 or valid_mask.shape != target.shape:
        raise ValueError(
            f"valid_mask must match target [B,H,W], got target={tuple(target.shape)}, "
            f"valid_mask={tuple(valid_mask.shape)}"
        )
    target = target.to(dtype=torch.long)
    weights = valid_mask.to(dtype=torch.float32).clamp_min(0.0)
    weights = weights * target.ne(ignore_index).to(dtype=weights.dtype)
    return target, weights


def _resize_logits(logits: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    if logits.shape[-2:] == target_size:
        return logits
    return F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)


def _zero_loss(logits: torch.Tensor) -> torch.Tensor:
    # Keep an autograd connection when a batch is entirely invalid.
    return logits.sum() * 0.0


def masked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
) -> torch.Tensor:
    """Cross-entropy averaged only over valid target pixels."""
    if logits.ndim != 4:
        raise ValueError(f"logits must have shape [B,C,H,W], got {tuple(logits.shape)}")
    target, weights = _target_and_weights(target, valid_mask, ignore_index)
    logits = _resize_logits(logits, target.shape[-2:])
    if not torch.isfinite(logits).all():
        raise ValueError("non-finite semantic logits encountered")
    if weights.sum() <= 0:
        return _zero_loss(logits)
    per_pixel = F.cross_entropy(
        logits,
        target,
        weight=class_weights,
        ignore_index=ignore_index,
        reduction="none",
    )
    return (per_pixel * weights.to(dtype=per_pixel.dtype)).sum() / weights.sum().clamp_min(1.0)


def masked_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    num_classes: int = 2,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Multi-class soft Dice loss with invalid pixels excluded."""
    if logits.ndim != 4 or logits.shape[1] != num_classes:
        raise ValueError(
            f"logits must have shape [B,{num_classes},H,W], got {tuple(logits.shape)}"
        )
    target, weights = _target_and_weights(target, valid_mask, ignore_index)
    logits = _resize_logits(logits, target.shape[-2:])
    if not torch.isfinite(logits).all():
        raise ValueError("non-finite semantic logits encountered")
    if weights.sum() <= 0:
        return _zero_loss(logits)

    safe_target = target.clamp(min=0, max=num_classes - 1)
    one_hot = F.one_hot(safe_target, num_classes=num_classes).permute(0, 3, 1, 2).to(logits.dtype)
    valid = weights.to(dtype=logits.dtype).unsqueeze(1)
    probabilities = F.softmax(logits, dim=1)
    intersection = (probabilities * one_hot * valid).sum(dim=(0, 2, 3))
    denominator = (probabilities * valid).sum(dim=(0, 2, 3)) + (one_hot * valid).sum(dim=(0, 2, 3))
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


class MaskedSemanticLoss(nn.Module):
    """YOLO26-compatible segmentation loss with valid-pixel and aux masking."""

    def __init__(
        self,
        *,
        num_classes: int = 2,
        class_weights: torch.Tensor | None = None,
        dice_weight: float = 1.0,
        aux_weight: float = 0.4,
        ignore_index: int = DEFAULT_IGNORE_INDEX,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = float(dice_weight)
        self.aux_weight = float(aux_weight)
        self.ignore_index = int(ignore_index)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.detach().float())
        else:
            self.class_weights = None

    def forward(
        self,
        outputs: torch.Tensor
        | tuple[torch.Tensor, ...]
        | list[torch.Tensor]
        | Mapping[str, Any],
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if isinstance(outputs, Mapping):
            if "logits" not in outputs:
                raise ValueError("structured model output must contain a 'logits' field")
            outputs = outputs["logits"]
        if isinstance(outputs, (tuple, list)):
            if not outputs:
                raise ValueError("semantic model returned no logits")
            main_logits = outputs[0]
            aux_logits = outputs[1] if len(outputs) > 1 else None
        else:
            main_logits, aux_logits = outputs, None

        ce = masked_cross_entropy(
            main_logits,
            target,
            valid_mask,
            class_weights=self.class_weights,
            ignore_index=self.ignore_index,
        )
        dice = masked_dice_loss(
            main_logits,
            target,
            valid_mask,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )
        main_loss = ce + self.dice_weight * dice
        aux_loss = main_loss.new_zeros(())
        if aux_logits is not None:
            aux_ce = masked_cross_entropy(
                aux_logits,
                target,
                valid_mask,
                class_weights=self.class_weights,
                ignore_index=self.ignore_index,
            )
            aux_dice = masked_dice_loss(
                aux_logits,
                target,
                valid_mask,
                num_classes=self.num_classes,
                ignore_index=self.ignore_index,
            )
            aux_loss = aux_ce + self.dice_weight * aux_dice
        total = main_loss + self.aux_weight * aux_loss
        valid_pixels = valid_mask.to(dtype=torch.float32)
        if valid_pixels.ndim == 4 and valid_pixels.shape[1] == 1:
            valid_pixels = valid_pixels[:, 0]
        valid_pixels = valid_pixels * target.ne(self.ignore_index).to(valid_pixels.dtype)
        items = {
            "total_loss": total.detach(),
            "main_ce_loss": ce.detach(),
            "main_dice_loss": dice.detach(),
            "aux_loss": aux_loss.detach(),
            "valid_pixels": valid_pixels.sum().detach(),
        }
        return total, items


class MaskedSegmentationMeter:
    """Accumulate confusion statistics over valid pixels only."""

    def __init__(self, num_classes: int = 2, ignore_index: int = DEFAULT_IGNORE_INDEX) -> None:
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)

    @property
    def valid_pixel_count(self) -> int:
        return int(self.confusion.sum().item())

    def update(
        self,
        outputs: torch.Tensor
        | tuple[torch.Tensor, ...]
        | list[torch.Tensor]
        | Mapping[str, Any],
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        logits = _main_logits(outputs)
        if logits.shape[1] != self.num_classes:
            raise ValueError(f"expected {self.num_classes} classes, got {logits.shape[1]}")
        target, weights = _target_and_weights(target, valid_mask, self.ignore_index)
        logits = _resize_logits(logits, target.shape[-2:])
        if not torch.isfinite(logits).all():
            raise ValueError("non-finite semantic logits encountered")
        prediction = logits.argmax(dim=1)
        valid = weights > 0
        if not valid.any():
            return
        target_flat = target[valid].to(device="cpu", dtype=torch.long)
        prediction_flat = prediction[valid].to(device="cpu", dtype=torch.long)
        encoded = target_flat * self.num_classes + prediction_flat
        counts = torch.bincount(encoded, minlength=self.num_classes**2)
        self.confusion += counts.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, Any]:
        matrix = self.confusion.to(dtype=torch.float64)
        true_positive = matrix.diag()
        false_positive = matrix.sum(dim=0) - true_positive
        false_negative = matrix.sum(dim=1) - true_positive
        support = matrix.sum(dim=1)
        iou = true_positive / (true_positive + false_positive + false_negative).clamp_min(1.0)
        precision = true_positive / (true_positive + false_positive).clamp_min(1.0)
        recall = true_positive / (true_positive + false_negative).clamp_min(1.0)
        f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
        total = matrix.sum().clamp_min(1.0)
        result: dict[str, Any] = {
            "valid_pixel_count": self.valid_pixel_count,
            "pixel_accuracy": float(true_positive.sum() / total),
            "mean_iou": float(iou.mean()),
            "mean_f1": float(f1.mean()),
            "per_class_iou": [float(value) for value in iou],
            "per_class_precision": [float(value) for value in precision],
            "per_class_recall": [float(value) for value in recall],
            "per_class_f1": [float(value) for value in f1],
            "per_class_support": [int(value) for value in support],
            "confusion_matrix": [[int(value) for value in row] for row in self.confusion],
        }
        if self.num_classes > 1:
            result["flood_iou"] = float(iou[1])
            result["flood_precision"] = float(precision[1])
            result["flood_recall"] = float(recall[1])
            result["flood_f1"] = float(f1[1])
        return result


def masked_segmentation_metrics(
    outputs: torch.Tensor
    | tuple[torch.Tensor, ...]
    | list[torch.Tensor]
    | Mapping[str, Any],
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    num_classes: int = 2,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
) -> dict[str, Any]:
    """Compute one-batch metrics with the same mask semantics as the loss."""
    meter = MaskedSegmentationMeter(num_classes=num_classes, ignore_index=ignore_index)
    meter.update(outputs, target, valid_mask)
    return meter.compute()


def loss_from_batch(
    outputs: torch.Tensor
    | tuple[torch.Tensor, ...]
    | list[torch.Tensor]
    | Mapping[str, Any],
    batch: Mapping[str, Any],
    criterion: MaskedSemanticLoss,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Convenience bridge for the project batch dictionary."""
    return criterion(outputs, batch["mask"], batch["valid_mask"])
