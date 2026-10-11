# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Official EfficientViT-B0 segmentation core adapted to the STURM protocol.

The Cityscapes checkpoint initializes the RGB core. Sensor-specific 2/9-to-3
adapters and the two-class classifier are newly initialized. The 2026 flood
distillation paper's teacher, pseudo-labels, and student weights are not used.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .optional_comparison import _GroupedSensorModel

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT = ROOT / "external" / "efficientvit-official"
OFFICIAL_COMMIT = "de7d7733cc0329f391b33f1f459271562ec27bd5"
WEIGHTS_SHA256 = "923d6fdd5e93640cc0c2f3f213764f34e80b477cd98a6b294d870ea6df5acc50"
SOURCE_LOCK = ROOT / "configs" / "efficientvit_b0_source_lock.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source() -> dict:
    """Validate the pinned source plus explicit Windows import patch, offline."""
    lock = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))
    if lock["commit"] != OFFICIAL_COMMIT:
        raise ValueError("EfficientViT source lock commit mismatch")
    for relative, expected in lock["python_files_sha256"].items():
        source = OFFICIAL_ROOT / relative
        # Normalize checkout line endings, not code content.
        if not source.is_file() or hashlib.sha256(
            source.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest() != expected:
            raise ValueError(f"EfficientViT source differs from audited lock: {relative}")
    actual = {p.relative_to(OFFICIAL_ROOT).as_posix()
              for p in (OFFICIAL_ROOT / "efficientvit").rglob("*.py")}
    expected = {p for p in lock["python_files_sha256"] if p.startswith("efficientvit/")}
    if actual != expected:
        raise ValueError("unexpected EfficientViT package Python files")
    return {"commit": lock["commit"], "source_lock_sha256": _sha256(SOURCE_LOCK),
            "verified_python_files": len(lock["python_files_sha256"]),
            "patch_sha256": lock["patch_sha256"]}


class EfficientViTB0STURM(_GroupedSensorModel):
    """One shared B0 core for S1/S2 with native sensor adapters."""

    architecture_name = "efficientvit_seg_b0_cityscapes_core_sturm_two_class"
    source_commit = OFFICIAL_COMMIT

    def __init__(self, *, weights_path: str | Path, num_classes: int = 2) -> None:
        super().__init__(common_channels=3)
        if num_classes != 2:
            raise ValueError("STURM comparison requires two classes")
        path = Path(weights_path)
        if not path.is_file() or _sha256(path) != WEIGHTS_SHA256:
            raise ValueError("official EfficientViT-B0 weight file is missing or has the wrong hash")
        if not (OFFICIAL_ROOT / "efficientvit" / "models" / "efficientvit" / "seg.py").is_file():
            raise FileNotFoundError("pinned EfficientViT source is missing")
        self.source_provenance = verify_source()
        if str(OFFICIAL_ROOT) not in sys.path:
            sys.path.insert(0, str(OFFICIAL_ROOT))
        from efficientvit.models.efficientvit.seg import efficientvit_seg_b0

        if Path(sys.modules[efficientvit_seg_b0.__module__].__file__).resolve() != (
            OFFICIAL_ROOT / "efficientvit/models/efficientvit/seg.py"
        ).resolve():
            raise ValueError("EfficientViT import resolved outside audited source")

        self.core = efficientvit_seg_b0("cityscapes")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint["state_dict"]
        self.core.load_state_dict(state, strict=True)
        self.pretrained_loaded_keys = len(state)
        classifier = self.core.head.output_ops[0].op_list[-1].conv
        if not isinstance(classifier, nn.Conv2d) or classifier.out_channels != 19:
            raise ValueError("official Cityscapes classifier structure changed")
        self.core.head.output_ops[0].op_list[-1].conv = nn.Conv2d(
            classifier.in_channels, num_classes, kernel_size=1, bias=True
        )
        self.pretraining = str(path.resolve())

    def forward_native(self, image: torch.Tensor, sensor: str | int | torch.Tensor) -> torch.Tensor:
        common = self.input_adapters(image, self._sensor_name(sensor))
        logits = self.core(common)
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
