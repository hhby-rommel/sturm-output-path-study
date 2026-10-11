# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Formal event-level STURM-Flood experiment runner.

The runner never starts a model matrix implicitly: one invocation owns one
model and one seed. Use ``--smoke`` before any formal invocation. In smoke
mode the validation subset also acts as a mock test, so real test events are
never opened.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.sturm_flood import SturmFloodDataset, grouped_sensor_collate  # noqa: E402
from src.evaluation.formal import evaluate_clean_and_degraded, is_better_checkpoint, seed_everything  # noqa: E402
from src.models.unet_semantic import IndependentNativeSturmUNet  # noqa: E402
from src.models.yolo11_semantic import SharedYolo11, load_matching_yolo11_weights  # noqa: E402
from src.models.yolo26_semantic import SharedYolo26, load_matching_yolo26_weights  # noqa: E402
from src.models.yolo26_unet_decoder import Yolo26UNetDecoder  # noqa: E402
from src.models.yolo26_main_only_control import SharedYolo26MainOnly  # noqa: E402
from src.models.optional_comparison import SegFormerB0Like, TorchvisionDeepLabV3MobileNetV3  # noqa: E402
from src.training.engine import train_one_epoch  # noqa: E402
from src.training.masked_segmentation import MaskedSemanticLoss  # noqa: E402
from src.training.optimization import (  # noqa: E402
    OptimizationSpec,
    build_optimizer,
    build_scheduler,
)


MODEL_IDS = (
    "yolo26n_native",
    "yolo26n_native_c16",
    "yolo26n_native_c16_main_only",
    "yolo26n_decoder",
    "yolo26n_decoder_dcl",
    "yolo11n_eca",
    "sturm_unet",
    "yolo26s_decoder",
    "deeplabv3_mobilenetv3",
    "segformer_b0_like",
)

OPTIONAL_COMPARISON_MODEL_IDS = (
    "deeplabv3_mobilenetv3",
    "segformer_b0_like",
)

YOLO_MAX_EPOCHS = 50
UNET_MAX_EPOCHS = 30
DEFAULT_PATIENCE = 10
STURM_UNET_DROPOUT = 0.30
STURM_UNET_BASE_CHANNELS = 64
DECODER_COMMON_CHANNELS = 16
DECODER_USE_INPUT_SKIP = True
DCL_SCHEDULE = {
    "consistency_weight_max": 0.1,
    "route_weight_max": 0.05,
    "warmup_fraction": 0.2,
    "ramp_fraction": 0.2,
}
OPTIMIZATION_SPEC_VERSIONS = (
    "optimizer_v1",
    "unet_optimizer_v2_lr1e4",
    "unet_optimizer_v2_lr3e4",
    "unet_optimizer_v2_lr1e3",
)
FORMAL_UNET_V2_VERSION = "unet_no_bn_optimizer_v2_v1"
EXPECTED_FORMAL_UNET_V2 = {
    "version": FORMAL_UNET_V2_VERSION,
    "model_id": "sturm_unet",
    "sensor_models": "independent_native",
    "base_channels": STURM_UNET_BASE_CHANNELS,
    "dropout": STURM_UNET_DROPOUT,
    "use_batchnorm": False,
    "input_normalization": "none_v1",
    "training_variant": "m1_degraded",
    "loss": "masked_ce_plus_dice_v1",
    "optimizer_spec_version": "unet_optimizer_v2_lr1e4",
    "max_epochs": UNET_MAX_EPOCHS,
    "early_stopping_patience": DEFAULT_PATIENCE,
    "checkpoint_selection": "pixel_pooled_degraded_f1_clean_tie_earlier_v1",
    "seeds": [2026, 2027, 2028],
}
EXPECTED_OPTIMIZATION_SPECS = {
    "optimizer_v1": {
        "optimizer": "adamw", "learning_rate": 3e-4, "weight_decay": 1e-4,
        "scheduler": "none", "warmup_fraction": 0.0, "minimum_lr_ratio": 1.0,
        "exclude_norm_and_bias_from_weight_decay": False, "track_parameter_update": False,
    },
    "unet_optimizer_v2_lr1e4": {
        "optimizer": "adamw", "learning_rate": 1e-4, "weight_decay": 1e-4,
        "scheduler": "warmup_cosine", "warmup_fraction": 0.05, "minimum_lr_ratio": 0.01,
        "exclude_norm_and_bias_from_weight_decay": True, "track_parameter_update": True,
    },
    "unet_optimizer_v2_lr3e4": {
        "optimizer": "adamw", "learning_rate": 3e-4, "weight_decay": 1e-4,
        "scheduler": "warmup_cosine", "warmup_fraction": 0.05, "minimum_lr_ratio": 0.01,
        "exclude_norm_and_bias_from_weight_decay": True, "track_parameter_update": True,
    },
    "unet_optimizer_v2_lr1e3": {
        "optimizer": "adamw", "learning_rate": 1e-3, "weight_decay": 1e-4,
        "scheduler": "warmup_cosine", "warmup_fraction": 0.05, "minimum_lr_ratio": 0.01,
        "exclude_norm_and_bias_from_weight_decay": True, "track_parameter_update": True,
    },
}


def load_formal_protocol_config(path: Path | None = None) -> dict[str, Any]:
    """Load protocol values that affect construction/training.

    Absolute data paths are intentionally not validated: the checked-in YAML
    is copied to a server checkout whose filesystem root legitimately differs.
    """
    config_path = path or (ROOT / "configs" / "project.yaml")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"invalid project config mapping: {config_path}")
    formal = config.get("formal_experiments")
    if not isinstance(formal, dict):
        raise ValueError("project config lacks formal_experiments mapping")
    expected = {
        "seeds": [2026, 2027, 2028],
        "early_stopping_patience": DEFAULT_PATIENCE,
        "fixed_degraded_tag": 1,
        "yolo_max_epochs": YOLO_MAX_EPOCHS,
        "unet_max_epochs": UNET_MAX_EPOCHS,
        "primary_model": "yolo26n_decoder",
    }
    for key, value in expected.items():
        if formal.get(key) != value:
            raise ValueError(f"formal_experiments.{key}={formal.get(key)!r}, expected {value!r}")
    if formal.get("freeze_batchnorm_running_stats") != {"yolo": True, "sturm_unet": False}:
        raise ValueError("project BatchNorm policy does not match the registered runner policy")
    if float(formal.get("sturm_unet_dropout", -1.0)) != STURM_UNET_DROPOUT:
        raise ValueError("project U-Net dropout does not match the registered runner value")
    if formal.get("sturm_unet_base_channels") != STURM_UNET_BASE_CHANNELS:
        raise ValueError("project U-Net base width does not match the registered runner value")
    decoder = formal.get("decoder")
    if (
        not isinstance(decoder, dict)
        or decoder.get("common_channels") != DECODER_COMMON_CHANNELS
        or decoder.get("use_input_skip") is not DECODER_USE_INPUT_SKIP
    ):
        raise ValueError("project decoder settings do not match common=16 and input skip")
    dcl = config.get("dcl")
    if not isinstance(dcl, dict):
        raise ValueError("project config lacks dcl mapping")
    for key, value in DCL_SCHEDULE.items():
        if float(dcl.get(key, -1.0)) != value:
            raise ValueError(f"dcl.{key}={dcl.get(key)!r}, expected {value!r}")
    optimization_specs = config.get("optimization_specs")
    if optimization_specs != EXPECTED_OPTIMIZATION_SPECS:
        raise ValueError("project optimization_specs do not exactly match the registered runner protocol")
    for version in OPTIMIZATION_SPEC_VERSIONS:
        OptimizationSpec.from_mapping(version, optimization_specs[version])
    if config.get("formal_unet_v2") != EXPECTED_FORMAL_UNET_V2:
        raise ValueError("project formal_unet_v2 does not exactly match the registered runner protocol")
    return config


def load_optimization_spec(protocol: dict[str, Any], version: str) -> OptimizationSpec:
    if version not in OPTIMIZATION_SPEC_VERSIONS:
        raise ValueError(f"unregistered optimization spec: {version}")
    return OptimizationSpec.from_mapping(version, protocol["optimization_specs"][version])


def freeze_batchnorm_for_model(model_id: str) -> bool:
    """Return the registered BN-running-statistics policy for a model."""
    if model_id not in MODEL_IDS:
        raise ValueError(f"unknown model id: {model_id}")
    # Only models that actually load pretrained YOLO checkpoints retain those
    # running statistics. From-scratch U-Net/optional controls must update BN.
    return model_id in {
        "yolo26n_native",
        "yolo26n_native_c16",
        "yolo26n_native_c16_main_only",
        "yolo26n_decoder",
        "yolo26n_decoder_dcl",
        "yolo11n_eca",
        "yolo26s_decoder",
    }


def resolve_patience(
    requested: int | None,
    *,
    epochs: int,
    validation_only: bool,
    formal_default: int = DEFAULT_PATIENCE,
) -> int:
    """Keep v1 frozen while giving scheduler diagnostics their full budget."""
    expected = epochs if validation_only else formal_default
    resolved = expected if requested is None else requested
    if resolved != expected:
        mode = "validation-only diagnostic" if validation_only else "formal optimizer-v1"
        raise ValueError(f"{mode} patience must remain {expected}, got {resolved}")
    return resolved


def refuse_existing_run_artifacts(run_dir: Path, *, resume: bool) -> None:
    """Prevent silent overwrites before any run artifact is written."""
    protected = (
        "best.pt",
        "last.pt",
        "history.csv",
        "validation_history.jsonl",
        "result.json",
        "test_evaluation.json",
        "validation_evaluation.json",
        "provenance.json",
    )
    existing = [name for name in protected if (run_dir / name).exists()]
    if (run_dir / "epoch_models").exists():
        existing.append("epoch_models")
    if not existing:
        return
    completed = {"result.json", "test_evaluation.json", "validation_evaluation.json"}.intersection(existing)
    if resume and not completed:
        return
    action = "resume" if resume else "start"
    raise RuntimeError(
        f"refusing to {action} into a run directory with protected artifacts: "
        f"{run_dir} ({', '.join(existing)})"
    )


class RowView(Dataset[dict[str, Any]]):
    def __init__(self, base: SturmFloodDataset, indices: list[int]) -> None:
        self.base = base
        self.indices = indices
        self.rows = [base.rows[index] for index in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.base[self.indices[index]]


def _smoke_view(dataset: SturmFloodDataset, per_sensor: int) -> RowView:
    selected: list[int] = []
    for sensor in ("S1", "S2"):
        candidates = [index for index, row in enumerate(dataset.rows) if row["sensor"] == sensor]
        if not candidates:
            raise RuntimeError(f"split has no {sensor} rows")
        selected.extend(candidates[:per_sensor])
    return RowView(dataset, selected)


def make_loader(dataset: Dataset, *, batch_size: int, train: bool, seed: int) -> DataLoader:
    rows = dataset.rows  # type: ignore[attr-defined]
    sampler = None
    if train:
        counts = {sensor: sum(row["sensor"] == sensor for row in rows) for sensor in ("S1", "S2")}
        if min(counts.values()) == 0:
            raise RuntimeError(f"training selection misses a sensor: {counts}")
        weights = torch.tensor([0.5 / counts[row["sensor"]] for row in rows], dtype=torch.double)
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(rows),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=grouped_sensor_collate,
    )


def build_model(model_id: str, device: torch.device, *, unet_normalization: str = "batchnorm") -> tuple[torch.nn.Module, dict[str, Any], str, int]:
    if model_id == "deeplabv3_mobilenetv3":
        model = TorchvisionDeepLabV3MobileNetV3(common_channels=3, num_classes=2).to(device)
        return model, {
            "pretrained": False,
            "architecture": model.architecture_name,
            "offline_initialization": True,
        }, "m1_degraded", YOLO_MAX_EPOCHS
    if model_id == "segformer_b0_like":
        model = SegFormerB0Like(common_channels=3, num_classes=2).to(device)
        return model, {
            "pretrained": False,
            "architecture": model.architecture_name,
            "offline_initialization": True,
        }, "m1_degraded", 50
    if model_id == "sturm_unet":
        model = IndependentNativeSturmUNet(
            base_channels=STURM_UNET_BASE_CHANNELS,
            dropout=STURM_UNET_DROPOUT,
            use_batchnorm=unet_normalization == "batchnorm",
        ).to(device)
        return model, {
            "pretrained": False,
            "dropout": STURM_UNET_DROPOUT,
            "normalization": unet_normalization,
            "conv2d_initialization": "kaiming_normal_fan_in_relu_zero_bias",
            "convtranspose2d_and_head_initialization": "xavier_uniform_zero_bias",
        }, "m1_degraded", UNET_MAX_EPOCHS
    if model_id == "yolo11n_eca":
        model = SharedYolo11(scale="n", common_channels=3, num_classes=2, use_attention=True).to(device)
        path = ROOT / "models" / "pretrained" / "yolo11n-seg.pt"
        return model, load_matching_yolo11_weights(model.core, path), "m1_degraded", YOLO_MAX_EPOCHS
    scale = "s" if model_id == "yolo26s_decoder" else "n"
    if model_id in {"yolo26n_native", "yolo26n_native_c16", "yolo26n_native_c16_main_only"}:
        common_channels = 3 if model_id == "yolo26n_native" else 16
        model_class = SharedYolo26MainOnly if model_id == "yolo26n_native_c16_main_only" else SharedYolo26
        model = model_class(scale=scale, common_channels=common_channels, num_classes=2).to(device)
    else:
        model = Yolo26UNetDecoder(
            scale=scale,
            common_channels=DECODER_COMMON_CHANNELS,
            num_classes=2,
            use_input_skip=DECODER_USE_INPUT_SKIP,
        ).to(device)
    path = ROOT / "models" / "pretrained" / f"yolo26{scale}-sem.pt"
    variant = "shared_dcl" if model_id == "yolo26n_decoder_dcl" else "m1_degraded"
    initialization = load_matching_yolo26_weights(model.core, path)
    if model_id == "yolo26n_native_c16_main_only":
        initialization = dict(initialization, control="native_main_only_no_high_resolution_decoder",
                              supervision="main_only", auxiliary_logits_used_in_loss=False,
                              historical_native_metrics_reusable=False)
    return model, initialization, variant, YOLO_MAX_EPOCHS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_hash() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def git_status() -> dict[str, Any]:
    """Record repository state so a result cannot be mistaken for a clean commit."""
    result = subprocess.run(["git", "status", "--porcelain", "--branch"], cwd=ROOT, capture_output=True, text=True)
    lines = result.stdout.splitlines()
    return {
        "returncode": result.returncode,
        "branch_line": lines[0] if lines else None,
        "dirty_paths": lines[1:] if len(lines) > 1 else [],
    }


def environment_snapshot() -> dict[str, Any]:
    pip_freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    try:
        conda_export = subprocess.run(["conda", "env", "export", "--no-builds"], capture_output=True, text=True)
        conda_text = conda_export.stdout if conda_export.returncode == 0 else None
        conda_warning = None if conda_export.returncode == 0 else conda_export.stderr.strip()
    except FileNotFoundError:
        conda_text = None
        conda_warning = "conda executable not found on PATH"
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "pip_freeze": pip_freeze.stdout.splitlines() if pip_freeze.returncode == 0 else None,
        "conda_env_export": conda_text,
        "snapshot_warnings": [
            message
            for message in (
                None if pip_freeze.returncode == 0 else pip_freeze.stderr.strip(),
                conda_warning,
            )
            if message
        ],
    }


class _NativeProfileWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, sensor: str) -> None:
        super().__init__()
        self.model = model
        self.sensor = sensor

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        outputs = self.model.forward_native(image, self.sensor)  # type: ignore[attr-defined]
        if isinstance(outputs, (tuple, list)):
            return outputs[0]
        if isinstance(outputs, dict):
            return outputs["logits"]
        return outputs


def profile_model(model: torch.nn.Module, device: torch.device, *, batch_size: int, smoke: bool) -> dict[str, Any]:
    model.eval()
    repetitions = 2 if smoke else 20
    warmup = 1 if smoke else 5
    result: dict[str, Any] = {"batch_size": batch_size, "repetitions": repetitions, "per_sensor": {}}
    try:
        from thop import profile as thop_profile
    except ImportError:
        thop_profile = None
    for sensor, channels in (("S1", 2), ("S2", 9)):
        wrapper = _NativeProfileWrapper(model, sensor).to(device)
        sample = torch.zeros(batch_size, channels, 128, 128, device=device)
        with torch.inference_mode():
            for _ in range(warmup):
                wrapper(sample)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            for _ in range(repetitions):
                wrapper(sample)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
        flops = None
        if thop_profile is not None:
            try:
                macs, _ = thop_profile(wrapper, inputs=(sample,), verbose=False)
                flops = 2.0 * float(macs)
            except Exception as exc:  # profiling must not invalidate trained results
                result.setdefault("warnings", []).append(f"{sensor} FLOPs unavailable: {exc}")
        result["per_sensor"][sensor] = {
            "channels": channels,
            "flops_per_batch": flops,
            "milliseconds_per_batch": 1000.0 * elapsed / repetitions,
            "milliseconds_per_tile": 1000.0 * elapsed / (repetitions * batch_size),
        }
    return result


def _capture_rng_state(train_loader: DataLoader | None = None) -> dict[str, Any]:
    """Capture all stochastic state needed to continue an epoch exactly.

    The sampler has its own ``torch.Generator``; restoring only the global
    torch/Python/NumPy states would silently replay a different tile order on
    resume.  Keep this state in the checkpoint rather than relying on a seed.
    """
    sampler_state = None
    if train_loader is not None:
        sampler = getattr(train_loader, "sampler", None)
        generator = getattr(sampler, "generator", None)
        if generator is not None:
            sampler_state = generator.get_state()
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "sampler": sampler_state,
    }


def _restore_rng_state(state: dict[str, Any], train_loader: DataLoader | None = None) -> None:
    """Restore the state captured by :func:`_capture_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([saved.cpu() for saved in state["cuda"]])
    sampler_state = state.get("sampler")
    if sampler_state is not None and train_loader is not None:
        sampler = getattr(train_loader, "sampler", None)
        generator = getattr(sampler, "generator", None)
        if generator is None:
            raise RuntimeError("checkpoint contains sampler RNG state but loader has no generator")
        generator.set_state(sampler_state.cpu())


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    *,
    epoch: int,
    seed: int,
    selection: dict[str, float],
    stale_epochs: int,
    history: list[dict[str, Any]],
    max_epochs: int,
    patience: int,
    optimizer_spec_version: str,
    train_loader: DataLoader | None = None,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "optimizer_spec_version": optimizer_spec_version,
            "epoch": epoch,
            "seed": seed,
            "selection": selection,
            "stale_epochs": stale_epochs,
            "history": history,
            "max_epochs": max_epochs,
            "patience": patience,
            "rng_state": _capture_rng_state(train_loader),
        },
        temporary,
    )
    temporary.replace(path)


def save_epoch_model_snapshot(run_dir: Path, model: torch.nn.Module, *, epoch: int, seed: int,
                              optimizer_spec_version: str) -> Path:
    """Keep an analysis-only model state for each validation-only U-Net epoch."""
    directory = run_dir / "epoch_models"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"epoch_{epoch:03d}.pt"
    if path.exists() or path.with_suffix(".pt.tmp").exists():
        raise FileExistsError(f"refusing to overwrite epoch snapshot: {path}")
    temporary = path.with_suffix(".pt.tmp")
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "seed": seed,
        "optimizer_spec_version": optimizer_spec_version,
        "snapshot_purpose": "validation_only_bn_timeline",
    }, temporary)
    temporary.replace(path)
    return path


def load_checkpoint(path: Path, model, optimizer, scheduler, device: torch.device) -> dict[str, Any]:
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler_state = state.get("scheduler_state_dict")
    if scheduler is None and scheduler_state is not None:
        raise RuntimeError("checkpoint has scheduler state but current optimization spec has none")
    if scheduler is not None:
        if scheduler_state is None:
            raise RuntimeError("checkpoint lacks scheduler state required by current optimization spec")
        scheduler.load_state_dict(scheduler_state)
    return state


def selection_from(metrics: dict[str, Any], epoch: int) -> dict[str, float]:
    return {
        "epoch": float(epoch),
        "clean_flood_f1": float(metrics["clean"]["overall"]["flood_f1"]),
        "degraded_flood_f1": float(metrics["degraded"]["overall"]["flood_f1"]),
    }


def dcl_weight(
    epoch: int,
    max_epochs: int,
    *,
    maximum: float = DCL_SCHEDULE["consistency_weight_max"],
    warmup_fraction: float = DCL_SCHEDULE["warmup_fraction"],
    ramp_fraction: float = DCL_SCHEDULE["ramp_fraction"],
) -> float:
    fraction = epoch / max(max_epochs, 1)
    if fraction <= warmup_fraction:
        return 0.0
    ramp_end = warmup_fraction + ramp_fraction
    if fraction < ramp_end:
        return maximum * (fraction - warmup_fraction) / ramp_fraction
    return maximum


def event_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    metric_names = ("flood_f1", "flood_iou", "flood_precision", "flood_recall")
    for view in ("clean", "degraded"):
        events = metrics[view]["per_event"]
        result[view] = {"event_count": len(events)}
        for metric in metric_names:
            values = [float(row[metric]) for row in events.values()]
            result[view][f"{metric}_mean"] = float(np.mean(values)) if values else 0.0
            result[view][f"{metric}_sample_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    # Preserve the full per-sensor metrics in the per-run report and make the
    # aggregate script able to compute the same cross-seed quantities.
    result["per_sensor"] = {
        view: {
            sensor: {metric: float(values[metric]) for metric in metric_names}
            for sensor, values in metrics[view]["per_sensor"].items()
        }
        for view in ("clean", "degraded")
    }
    return result


def write_history(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "epoch", "seconds", "train_loss", "learning_rate", "gradient_norm_mean",
        "gradient_norm_max", "gradient_clip_fraction", "parameter_update_l2",
        "relative_parameter_update_l2", "dcl_weight", "clean_f1", "degraded_f1",
        "s1_gradient_norm_mean", "s1_gradient_norm_max", "s1_gradient_above_clip_threshold_fraction",
        "s1_parameter_update_l2", "s1_relative_parameter_update_l2",
        "s2_gradient_norm_mean", "s2_gradient_norm_max", "s2_gradient_above_clip_threshold_fraction",
        "s2_parameter_update_l2", "s2_relative_parameter_update_l2",
        "clean_iou", "degraded_iou", "clean_precision", "degraded_precision",
        "clean_recall", "degraded_recall", "clean_valid_pixels", "degraded_valid_pixels",
        "clean_invalid_ratio", "degraded_invalid_ratio", "s1_clean_f1", "s1_degraded_f1",
        "s2_clean_f1", "s2_degraded_f1",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_IDS, required=True)
    parser.add_argument("--seed", type=int, choices=(2026, 2027, 2028), required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "formal")
    parser.add_argument(
        "--optimizer-spec-version",
        choices=OPTIMIZATION_SPEC_VERSIONS,
        default="optimizer_v1",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--save-epoch-models", action="store_true",
                        help="Save each U-Net epoch model for validation-only BN timeline audit")
    parser.add_argument("--diagnostic-stop-after-epoch", type=int,
                        help="Pause a validation-only run after this epoch while retaining the declared LR schedule")
    parser.add_argument("--unet-normalization", choices=("batchnorm", "none"), default="batchnorm",
                        help="U-Net normalization; default preserves formal-v1 protocol")
    parser.add_argument("--formal-unet-v2", action="store_true",
                        help="Run the registered independent no-BN U-Net-v2 formal protocol")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_formal_protocol_config()
    optimization_spec = load_optimization_spec(protocol, args.optimizer_spec_version)
    if args.optimizer_spec_version != "optimizer_v1" and args.model != "sturm_unet":
        raise ValueError("registered optimizer-v2 candidates are restricted to sturm_unet diagnostics")
    default_formal_dir = (ROOT / "results" / "formal").resolve()
    if args.optimizer_spec_version != "optimizer_v1" and args.output_dir.resolve() == default_formal_dir:
        raise ValueError("optimizer-v2 diagnostics require an explicit non-formal --output-dir")
    if args.formal_unet_v2:
        if (
            args.model != "sturm_unet"
            or args.optimizer_spec_version != EXPECTED_FORMAL_UNET_V2["optimizer_spec_version"]
            or args.unet_normalization != "none"
            or args.validation_only
            or args.save_epoch_models
            or args.diagnostic_stop_after_epoch is not None
        ):
            raise ValueError("--formal-unet-v2 requires registered U-Net-v2 options without diagnostic flags")
    elif args.validation_only != (args.optimizer_spec_version != "optimizer_v1"):
        raise ValueError("optimizer-v2 diagnostics require --validation-only; optimizer-v1 forbids it")
    if args.save_epoch_models and (args.model != "sturm_unet" or not args.validation_only):
        raise ValueError("--save-epoch-models requires validation-only sturm_unet diagnostics")
    if args.unet_normalization != "batchnorm" and (
        args.model != "sturm_unet" or not (args.validation_only or args.formal_unet_v2)
    ):
        raise ValueError("non-default U-Net normalization requires a registered U-Net-v2 path")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, initialization, train_variant, default_epochs = build_model(
        args.model, device, unet_normalization=args.unet_normalization,
    )
    epochs = 1 if args.smoke and args.epochs is None else (args.epochs or default_epochs)
    if epochs < 1 or epochs > default_epochs:
        raise ValueError(f"epochs must be in [1, {default_epochs}] for {args.model}")
    if args.formal_unet_v2 and (
        args.batch_size != int(protocol["formal_experiments"]["batch_size"])
        or (not args.smoke and epochs != EXPECTED_FORMAL_UNET_V2["max_epochs"])
    ):
        raise ValueError("formal U-Net-v2 requires batch size 16 and the registered 30-epoch budget")
    if args.diagnostic_stop_after_epoch is not None and (
        not args.validation_only or not 1 <= args.diagnostic_stop_after_epoch < epochs
    ):
        raise ValueError("--diagnostic-stop-after-epoch requires validation-only and 1 <= stop < epochs")
    args.patience = resolve_patience(
        args.patience,
        epochs=epochs,
        validation_only=args.validation_only,
        formal_default=int(protocol["formal_experiments"]["early_stopping_patience"]),
    )

    manifest = ROOT / "data" / "manifests" / "manifest.csv"
    split = ROOT / "data" / "splits" / "event_split_v1.csv"
    train_data: Dataset = SturmFloodDataset(manifest, split, "train")
    val_data: Dataset = SturmFloodDataset(manifest, split, "val")
    test_data: Dataset | None = None
    if args.smoke:
        train_data = _smoke_view(train_data, 4)  # type: ignore[arg-type]
        val_data = _smoke_view(val_data, 4)  # type: ignore[arg-type]
    elif not args.validation_only:
        test_data = SturmFloodDataset(manifest, split, "test")
    train_loader = make_loader(train_data, batch_size=args.batch_size, train=True, seed=args.seed + 1)
    val_loader = make_loader(val_data, batch_size=args.batch_size, train=False, seed=args.seed + 2)
    test_loader = make_loader(test_data, batch_size=args.batch_size, train=False, seed=args.seed + 3) if test_data else None

    if args.formal_unet_v2:
        run_dir = args.output_dir.resolve() / FORMAL_UNET_V2_VERSION / args.model / str(args.seed) / ("smoke" if args.smoke else "formal")
    elif args.optimizer_spec_version == "optimizer_v1":
        run_dir = args.output_dir.resolve() / args.model / str(args.seed) / ("smoke" if args.smoke else "formal")
    else:
        run_dir = (
            args.output_dir.resolve()
            / args.optimizer_spec_version
            / args.model
            / str(args.seed)
            / ("smoke" if args.smoke else "diagnostic")
        )
    refuse_existing_run_artifacts(run_dir, resume=args.resume)
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    result_path = run_dir / "result.json"
    test_path = run_dir / "test_evaluation.json"
    diagnostic_evaluation_path = run_dir / "validation_evaluation.json"
    shutil.copy2(ROOT / "configs" / "project.yaml", run_dir / "project.yaml")
    provenance_files = (
        manifest,
        split,
        ROOT / "configs" / "project.yaml",
        ROOT / "docs" / "server_protocol.md",
        ROOT / "docs" / "unet_v2_formal_protocol_20260926.md",
        ROOT / "scripts" / "server_main.py",
        ROOT / "src" / "models" / "unet_semantic.py",
        ROOT / "src" / "models" / "yolo26_unet_decoder.py",
        ROOT / "src" / "training" / "engine.py",
        ROOT / "src" / "training" / "optimization.py",
        ROOT / "src" / "evaluation" / "formal.py",
    )
    provenance = {
        "git_commit": git_hash(),
        "git_status": git_status(),
        "command": [sys.executable, *sys.argv],
        "environment": environment_snapshot(),
        "hashes": {str(path.relative_to(ROOT)): sha256(path) for path in provenance_files},
    }
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    criterion = MaskedSemanticLoss(num_classes=2, dice_weight=1.0, aux_weight=0.4).to(device)
    optimizer = build_optimizer(model, optimization_spec)
    scheduler = build_scheduler(
        optimizer,
        optimization_spec,
        total_steps=max(len(train_loader) * epochs, 1),
    )
    start_epoch = 0
    best: dict[str, float] | None = None
    history: list[dict[str, Any]] = []
    stale_epochs = 0
    if args.resume:
        if not last_path.exists():
            raise FileNotFoundError(f"resume requested but missing {last_path}")
        resumed = load_checkpoint(last_path, model, optimizer, scheduler, device)
        if resumed.get("optimizer_spec_version", "optimizer_v1") != optimization_spec.version:
            raise ValueError("resume optimizer specification does not match checkpoint")
        start_epoch = int(resumed["epoch"])
        if "rng_state" not in resumed:
            raise RuntimeError(
                "checkpoint lacks stochastic state; refusing a non-exact resume. "
                "Restart this run from its registered initialization."
            )
        saved_max_epochs = int(resumed.get("max_epochs", epochs))
        if saved_max_epochs != epochs:
            raise ValueError(
                f"resume max_epochs mismatch: checkpoint={saved_max_epochs}, requested={epochs}; "
                "keep the original epoch budget so the DCL schedule is unchanged"
            )
        saved_patience = int(resumed.get("patience", args.patience))
        if saved_patience != args.patience:
            raise ValueError(
                f"resume patience mismatch: checkpoint={saved_patience}, requested={args.patience}"
            )
        best = resumed.get("selection")
        history = list(resumed.get("history", []))
        stale_epochs = int(resumed.get("stale_epochs", 0))
        _restore_rng_state(resumed["rng_state"], train_loader)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(start_epoch + 1, epochs + 1):
        epoch_started = time.perf_counter()
        current_dcl_weight = (
            dcl_weight(
                epoch,
                epochs,
                maximum=float(DCL_SCHEDULE["consistency_weight_max"]),
                warmup_fraction=float(DCL_SCHEDULE["warmup_fraction"]),
                ramp_fraction=float(DCL_SCHEDULE["ramp_fraction"]),
            )
            if train_variant == "shared_dcl"
            else 0.0
        )
        train_stats = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device=device,
            grad_clip_norm=5.0,
            scheduler=scheduler,
            track_parameter_update=optimization_spec.track_parameter_update,
            sensor_gradient_diagnostics=args.formal_unet_v2,
            freeze_batchnorm=freeze_batchnorm_for_model(args.model),
            model_variant=train_variant,
            consistency_weight=current_dcl_weight,
        )
        validation = evaluate_clean_and_degraded(model, val_loader, criterion, device, seed=args.seed, tag=1)
        current = selection_from(validation, epoch)
        improved = is_better_checkpoint(current, best)
        if improved:
            best = current
            stale_epochs = 0
        else:
            stale_epochs += 1
        clean_overall = validation["clean"]["overall"]
        degraded_overall = validation["degraded"]["overall"]
        record = {
            "epoch": epoch,
            "seconds": time.perf_counter() - epoch_started,
            "train_loss": float(train_stats["loss"]),
            "learning_rate": float(train_stats["learning_rate"]),
            "gradient_norm_mean": float(train_stats["gradient_norm_mean"]),
            "gradient_norm_max": float(train_stats["gradient_norm_max"]),
            "gradient_clip_fraction": float(train_stats["gradient_clip_fraction"]),
            "parameter_update_l2": float(train_stats["parameter_update_l2"]),
            "relative_parameter_update_l2": float(train_stats["relative_parameter_update_l2"]),
            "dcl_weight": current_dcl_weight,
            "clean_f1": current["clean_flood_f1"],
            "degraded_f1": current["degraded_flood_f1"],
            "clean_iou": float(validation["clean"]["overall"]["flood_iou"]),
            "degraded_iou": float(validation["degraded"]["overall"]["flood_iou"]),
            "clean_precision": float(clean_overall["flood_precision"]),
            "degraded_precision": float(degraded_overall["flood_precision"]),
            "clean_recall": float(clean_overall["flood_recall"]),
            "degraded_recall": float(degraded_overall["flood_recall"]),
            "clean_valid_pixels": int(clean_overall["valid_pixel_count"]),
            "degraded_valid_pixels": int(degraded_overall["valid_pixel_count"]),
            "clean_invalid_ratio": float(validation["clean"]["invalid_pixel_ratio"]),
            "degraded_invalid_ratio": float(validation["degraded"]["invalid_pixel_ratio"]),
            "s1_clean_f1": float(validation["clean"]["per_sensor"]["S1"]["flood_f1"]),
            "s1_degraded_f1": float(validation["degraded"]["per_sensor"]["S1"]["flood_f1"]),
            "s2_clean_f1": float(validation["clean"]["per_sensor"]["S2"]["flood_f1"]),
            "s2_degraded_f1": float(validation["degraded"]["per_sensor"]["S2"]["flood_f1"]),
        }
        if args.formal_unet_v2:
            for sensor in ("S1", "S2"):
                prefix = sensor.lower()
                for metric in (
                    "gradient_norm_mean", "gradient_norm_max",
                    "gradient_above_clip_threshold_fraction", "parameter_update_l2",
                    "relative_parameter_update_l2",
                ):
                    record[f"{prefix}_{metric}"] = float(train_stats[f"sensor_{metric}"][sensor])
        history.append(record)
        if improved:
            # Save after appending this epoch so the best checkpoint is a
            # self-contained diagnostic artifact as well as a model snapshot.
            save_checkpoint(
                best_path, model, optimizer, scheduler, epoch=epoch, seed=args.seed, selection=current,
                stale_epochs=stale_epochs, history=history,
                max_epochs=epochs, patience=args.patience,
                optimizer_spec_version=optimization_spec.version, train_loader=train_loader,
            )
        save_checkpoint(
            last_path, model, optimizer, scheduler, epoch=epoch, seed=args.seed, selection=best or current,
            stale_epochs=stale_epochs, history=history,
            max_epochs=epochs, patience=args.patience,
            optimizer_spec_version=optimization_spec.version, train_loader=train_loader,
        )
        if args.save_epoch_models:
            save_epoch_model_snapshot(
                run_dir, model, epoch=epoch, seed=args.seed,
                optimizer_spec_version=optimization_spec.version,
            )
        write_history(run_dir / "history.csv", history)
        with (run_dir / "validation_history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"epoch": epoch, "metrics": validation}, ensure_ascii=False) + "\n")
        print(f"{args.model} seed={args.seed} epoch={epoch}/{epochs} clean_f1={record['clean_f1']:.4f} degraded_f1={record['degraded_f1']:.4f}", flush=True)
        if epoch == args.diagnostic_stop_after_epoch:
            print(f"diagnostic_paused_after_epoch={epoch} run_dir={run_dir}", flush=True)
            return
        if not args.validation_only and stale_epochs >= args.patience:
            break

    if not best_path.exists():
        raise RuntimeError("training ended without a best checkpoint")
    load_checkpoint(best_path, model, optimizer, scheduler, device)
    best_validation = evaluate_clean_and_degraded(model, val_loader, criterion, device, seed=args.seed, tag=1)
    if not args.validation_only and test_path.exists():
        raise RuntimeError(f"test evaluation already exists; refusing a second test access: {test_path}")
    if args.validation_only:
        final_evaluation = best_validation
    else:
        evaluation_loader = val_loader if args.smoke else test_loader
        assert evaluation_loader is not None
        final_evaluation = evaluate_clean_and_degraded(
            model, evaluation_loader, criterion, device, seed=args.seed, tag=1
        )
    test_payload = {
        "real_test_split_used": not args.smoke and not args.validation_only,
        "smoke_mock_test_uses_validation_subset": args.smoke,
        "diagnostic_validation_only": args.validation_only,
        "metrics": final_evaluation,
        "event_summary": event_summary(final_evaluation),
    }
    evaluation_path = diagnostic_evaluation_path if args.validation_only else test_path
    evaluation_path.write_text(json.dumps(test_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {
        "status": (
            "SMOKE_COMPLETED"
            if args.smoke
            else "DIAGNOSTIC_COMPLETED"
            if args.validation_only
            else "FORMAL_COMPLETED"
        ),
        "model_id": args.model,
        "formal_protocol_version": FORMAL_UNET_V2_VERSION if args.formal_unet_v2 else "formal_optimizer_v1" if not args.validation_only else None,
        "seed": args.seed,
        "initialization": initialization,
        "optimizer_spec_version": optimization_spec.version,
        "save_epoch_models": args.save_epoch_models,
        "optimization_spec": optimization_spec.as_dict(),
        "freeze_batchnorm_running_stats": freeze_batchnorm_for_model(args.model),
        "dcl_schedule": DCL_SCHEDULE if train_variant == "shared_dcl" else None,
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "best_selection": best,
        "best_validation": best_validation,
        "evaluation": test_payload,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
        "inference_profile": profile_model(model, device, batch_size=args.batch_size, smoke=args.smoke),
        "provenance": provenance,
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"result={result_path}")


if __name__ == "__main__":
    main()
