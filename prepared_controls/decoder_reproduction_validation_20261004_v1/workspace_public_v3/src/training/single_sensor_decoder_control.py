# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Local preparation for separately trained, original-topology decoders.

No formal/test runner. Keep both registered sensor adapters to preserve original
construction, RNG consumption and state keys; the unused route gets no gradient.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.dataset.sturm_flood import grouped_sensor_collate
from src.evaluation.formal import is_better_checkpoint
from src.models.yolo26_semantic import load_matching_yolo26_weights
from src.models.yolo26_unet_decoder import Yolo26UNetDecoder


def validate_scope(split: str, seed: int, sensor: str) -> None:
    if split not in {"train", "val"}:
        raise ValueError("local control accepts train/val only")
    if seed in {2026, 2027, 2028} or not 10000 <= seed < 20000:
        raise ValueError("local control requires a non-formal diagnostic seed")
    if sensor not in {"S1", "S2"}:
        raise ValueError("sensor must be S1 or S2")


def make_single_sensor_loader(dataset, *, sensor: str, split: str, seed: int,
                              train: bool) -> DataLoader:
    validate_scope(split, seed, sensor)
    if train != (split == "train"):
        raise ValueError("training flag must agree with split")
    if not dataset.rows or any(r["sensor"] != sensor for r in dataset.rows):
        raise ValueError("loader must contain only the declared sensor")
    base = getattr(dataset, "base", dataset)
    if getattr(base, "split", None) != split:
        raise ValueError("dataset split must match declared train/val scope")
    # Original sampler's conditional distribution: uniform replacement within
    # one sensor. num_samples=N_sensor, batch16, no drop_last, same seed+1.
    sampler = None
    if train:
        sampler = WeightedRandomSampler(
            torch.full((len(dataset.rows),), 1.0 / len(dataset.rows), dtype=torch.double),
            num_samples=len(dataset.rows), replacement=True,
            generator=torch.Generator().manual_seed(seed + 1),
        )
    return DataLoader(dataset, batch_size=16, sampler=sampler, shuffle=False,
                      num_workers=0, pin_memory=torch.cuda.is_available(),
                      collate_fn=grouped_sensor_collate)


def build_original_decoder(weight_path):
    model = Yolo26UNetDecoder(common_channels=16, num_classes=2, use_input_skip=True)
    report = load_matching_yolo26_weights(model.core, weight_path)
    return model, report


class LocalSelectionState:
    """Original lexicographic ordering and patience10, local metadata only.

    A single-sensor loader makes overall F1 that sensor's F1. Its use in a
    formal experiment remains a pending protocol decision, not authorization.
    """
    def __init__(self):
        self.best = None
        self.stale = 0

    def observe(self, epoch: int, validation: dict) -> bool:
        candidate = dict(epoch=epoch,
                         clean_flood_f1=validation["clean"]["overall"]["flood_f1"],
                         degraded_flood_f1=validation["degraded"]["overall"]["flood_f1"])
        if is_better_checkpoint(candidate, self.best):
            self.best, self.stale = candidate, 0
        else:
            self.stale += 1
        return self.stale >= 10
