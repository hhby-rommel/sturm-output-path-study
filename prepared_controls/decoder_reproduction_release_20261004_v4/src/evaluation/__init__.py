# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Evaluation utilities for formal STURM-Flood experiments."""

from .formal import evaluate_clean_and_degraded, fixed_rng, is_better_checkpoint

__all__ = ["evaluate_clean_and_degraded", "fixed_rng", "is_better_checkpoint"]
