# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Full STURM-Flood audit and event-protocol builder.

The audit is deliberately conservative:

* metadata ``ems_code`` is the authoritative event id;
* Sentinel-1 and Sentinel-2 are scanned as independent tile streams;
* no SAR-optical pairs are constructed;
* the raw mask codes are preserved, while the training protocol follows the
  official upstream ``mask > 0`` binary preprocessing rule;
* non-finite input pixels are recorded and excluded through a valid-pixel mask.

One manifest row represents one image tile and its same-sensor mask.  This is
more useful for the downstream loader than emitting a second row for every
mask file and prevents accidental cross-sensor pairing.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio


EVENT_RE = re.compile(r"(EMSR\d+)", re.IGNORECASE)
SENSORS = (("S1", "Sentinel1"), ("S2", "Sentinel2"))
MANIFEST_FIELDS = [
    "sensor",
    "event_id",
    "filename_event_id",
    "event_id_match",
    "aoi_code",
    "event_type",
    "country",
    "floodmap_id",
    "floodmap_date",
    "acquisition_date",
    "tile_id",
    "image_path",
    "mask_path",
    "is_mask",
    "height",
    "width",
    "channels",
    "image_dtype",
    "mask_dtype",
    "image_crs",
    "image_nodata",
    "mask_crs",
    "mask_nodata",
    "image_nonfinite_values",
    "image_declared_nodata_pixels",
    "invalid_pixel_count",
    "valid_pixel_ratio",
    "mask_nonfinite_values",
    "mask_declared_nodata_pixels",
    "mask_shape_ok",
    "mask_finite",
    "raw_label_codes",
    "raw_positive_ratio",
]


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _event_from_filename(name: str) -> str:
    match = EVENT_RE.search(name)
    return match.group(1).upper() if match else "UNKNOWN"


def _metadata(root: Path, sensor: str) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    path = root / f"Sentinel{sensor[1]}_metadata.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_tile = {row["tile_id"]: row for row in rows}
    duplicate_tiles = len(by_tile) != len(rows)
    event_rows = defaultdict(list)
    for row in rows:
        event_rows[row["ems_code"].upper()].append(row)
    return by_tile, {
        "path": str(path),
        "rows": len(rows),
        "unique_tiles": len(by_tile),
        "duplicate_tiles": duplicate_tiles,
        "events": sorted(event_rows),
    }


def _codes(mask: np.ndarray, nodata: float | None) -> tuple[str, float, bool, int]:
    finite = np.isfinite(mask)
    declared_nodata = np.zeros(mask.shape, dtype=bool)
    if nodata is not None and np.isfinite(nodata):
        declared_nodata = finite & (mask == nodata)
        finite &= ~declared_nodata
    values, counts = np.unique(mask[finite], return_counts=True)
    code_text = ";".join(f"{int(value)}:{int(count)}" for value, count in zip(values, counts))
    positive_ratio = float(np.count_nonzero(mask[finite] > 0) / max(int(finite.sum()), 1))
    return code_text, positive_ratio, bool(np.isfinite(mask).all()), int(declared_nodata.sum())


def _scan_tile(
    sensor: str,
    root: Path,
    image_path: Path,
    mask_path: Path,
    meta: dict[str, str] | None,
) -> dict[str, Any]:
    with rasterio.open(image_path) as image_ds:
        image = image_ds.read()
        image_shape = (image_ds.height, image_ds.width)
        image_dtype = ",".join(image_ds.dtypes)
        image_crs = str(image_ds.crs) if image_ds.crs else ""
        image_nodata = image_ds.nodata

    with rasterio.open(mask_path) as mask_ds:
        mask = mask_ds.read(1)
        mask_shape = (mask_ds.height, mask_ds.width)
        mask_dtype = str(mask_ds.dtypes[0])
        mask_crs = str(mask_ds.crs) if mask_ds.crs else ""
        mask_nodata = mask_ds.nodata

    image_finite = np.isfinite(image)
    declared_nodata = np.zeros(image_finite.shape[1:], dtype=bool)
    if image_nodata is not None and np.isfinite(image_nodata):
        declared_nodata = image_finite.any(axis=0) & (image == image_nodata).any(axis=0)
    valid_pixels = image_finite.all(axis=0) & ~declared_nodata
    image_nonfinite_values = int((~image_finite).sum())
    image_declared_nodata_pixels = int(declared_nodata.sum())
    invalid_pixel_count = int((~valid_pixels).sum())
    code_text, positive_ratio, mask_finite, mask_declared_nodata_pixels = _codes(mask, mask_nodata)
    tile_id = image_path.name
    meta = meta or {}
    event_id = meta.get("ems_code", "").upper() or _event_from_filename(tile_id)
    filename_event_id = _event_from_filename(tile_id)

    return {
        "sensor": sensor,
        "event_id": event_id,
        "filename_event_id": filename_event_id,
        "event_id_match": str(event_id == filename_event_id),
        "aoi_code": meta.get("aoi_code", ""),
        "event_type": meta.get("event_type", ""),
        "country": meta.get("country", ""),
        "floodmap_id": meta.get("floodmap_id", ""),
        "floodmap_date": meta.get("floodmap_date", ""),
        "acquisition_date": meta.get("sentinel_date", meta.get("sentinel_timestamp", "")),
        "tile_id": tile_id,
        "image_path": _rel(image_path, root),
        "mask_path": _rel(mask_path, root),
        "is_mask": "False",
        "height": image_shape[0],
        "width": image_shape[1],
        "channels": image.shape[0],
        "image_dtype": image_dtype,
        "mask_dtype": mask_dtype,
        "image_crs": image_crs,
        "image_nodata": "" if image_nodata is None else image_nodata,
        "mask_crs": mask_crs,
        "mask_nodata": "" if mask_nodata is None else mask_nodata,
        "image_nonfinite_values": image_nonfinite_values,
        "image_declared_nodata_pixels": image_declared_nodata_pixels,
        "invalid_pixel_count": invalid_pixel_count,
        "valid_pixel_ratio": float(valid_pixels.mean()),
        "mask_nonfinite_values": int((~np.isfinite(mask)).sum()),
        "mask_declared_nodata_pixels": mask_declared_nodata_pixels,
        "mask_shape_ok": str(mask_shape == image_shape == (128, 128)),
        "mask_finite": str(mask_finite),
        "raw_label_codes": code_text,
        "raw_positive_ratio": positive_ratio,
    }


def scan_dataset(root: Path, workers: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    jobs: list[tuple[str, Path, Path, dict[str, str] | None]] = []
    metadata_info = {}
    for sensor, folder in SENSORS:
        image_dir = root / folder / sensor
        mask_dir = root / folder / "Floodmaps"
        metadata, info = _metadata(root, sensor)
        metadata_info[sensor] = info
        image_paths = sorted(image_dir.glob("*.tif"))
        image_names = {path.name for path in image_paths}
        mask_names = {path.name for path in mask_dir.glob("*.tif")}
        if image_names != mask_names:
            raise RuntimeError(
                f"{sensor} image/mask filename mismatch: "
                f"images_only={len(image_names - mask_names)}, "
                f"masks_only={len(mask_names - image_names)}"
            )
        if image_names != set(metadata):
            raise RuntimeError(
                f"{sensor} image/metadata filename mismatch: "
                f"filesystem_only={len(image_names - set(metadata))}, "
                f"metadata_only={len(set(metadata) - image_names)}"
            )
        jobs.extend(
            (sensor, root, image_path, mask_dir / image_path.name, metadata.get(image_path.name))
            for image_path in image_paths
        )

    records: list[dict[str, Any]] = []
    failures: list[str] = []
    total = len(jobs)
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_scan_tile, *job) for job in jobs]
        for future in as_completed(futures):
            completed += 1
            try:
                records.append(future.result())
            except Exception as exc:  # pragma: no cover - only exercised by corrupt input
                failures.append(f"{type(exc).__name__}: {exc}")
            if completed % 1000 == 0 or completed == total:
                print(f"scanned {completed}/{total}", flush=True)
    if failures:
        raise RuntimeError("tile scan failures:\n" + "\n".join(failures[:20]))
    records.sort(key=lambda row: (row["sensor"], row["tile_id"]))
    return records, metadata_info


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def event_statistics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row["event_id"]].append(row)
    output = []
    for event_id in sorted(grouped):
        rows = grouped[event_id]
        by_sensor = {sensor: [r for r in rows if r["sensor"] == sensor] for sensor in ("S1", "S2")}
        output.append(
            {
                "event_id": event_id,
                "s1_tiles": len(by_sensor["S1"]),
                "s2_tiles": len(by_sensor["S2"]),
                "total_tiles": len(rows),
                "s1_present": str(bool(by_sensor["S1"])),
                "s2_present": str(bool(by_sensor["S2"])),
                "s1_nan_files": sum(bool(int(r["image_nonfinite_values"])) for r in by_sensor["S1"]),
                "s2_nan_files": sum(bool(int(r["image_nonfinite_values"])) for r in by_sensor["S2"]),
                "s1_valid_pixel_mean": np.mean([float(r["valid_pixel_ratio"]) for r in by_sensor["S1"]]) if by_sensor["S1"] else "",
                "s2_valid_pixel_mean": np.mean([float(r["valid_pixel_ratio"]) for r in by_sensor["S2"]]) if by_sensor["S2"] else "",
                "s1_positive_ratio_mean": np.mean([float(r["raw_positive_ratio"]) for r in by_sensor["S1"]]) if by_sensor["S1"] else "",
                "s2_positive_ratio_mean": np.mean([float(r["raw_positive_ratio"]) for r in by_sensor["S2"]]) if by_sensor["S2"] else "",
                "aoi_count": len({r["aoi_code"] for r in rows}),
                "countries": ";".join(sorted({r["country"] for r in rows if r["country"]})),
                "event_types": ";".join(sorted({r["event_type"] for r in rows if r["event_type"]})),
                "floodmap_ids": ";".join(sorted({r["floodmap_id"] for r in rows if r["floodmap_id"]})),
                "filename_event_mismatch_count": sum(r["event_id_match"] != "True" for r in rows),
            }
        )
    return output


def assign_events(records: list[dict[str, Any]], seed: int, mode: str = "balanced") -> list[dict[str, str]]:
    """Assign whole events while optimizing tile and sensor balance.

    The event quotas are fixed at 36/12/12 for the 60-event dataset.  Since
    event sizes are highly unequal, the objective also targets 60/20/20 of
    tiles and S1/S2 counts.  The two sensitivity modes change the relative
    objective weights, producing auditable alternative partitions rather than
    merely changing an unused shuffle seed.
    """
    import random

    rng = random.Random(seed)
    event_sensor_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in records:
        event_sensor_counts[row["event_id"]][row["sensor"]] += 1
    events = list(event_sensor_counts)
    if len(events) < 3:
        raise ValueError(f"need at least 3 identifiable events, found {len(events)}")
    fractions = {"train": 0.60, "val": 0.20, "test": 0.20}
    n_train = int(round(len(events) * fractions["train"]))
    n_val = int(round(len(events) * fractions["val"]))
    slots = ["train"] * n_train + ["val"] * n_val + ["test"] * (len(events) - n_train - n_val)
    total_sensor = Counter()
    for event in events:
        total_sensor.update(event_sensor_counts[event])

    if mode == "balanced":
        tile_weight, sensor_weight, presence_weight = 2.0, 1.0, 0.25
    elif mode == "tile_priority":
        tile_weight, sensor_weight, presence_weight = 3.0, 0.5, 0.25
    elif mode == "sensor_priority":
        tile_weight, sensor_weight, presence_weight = 0.75, 3.0, 0.5
    else:
        raise ValueError(f"unknown split mode: {mode}")

    split_names = tuple(fractions)
    total_tiles = sum(total_sensor.values())
    total_events_by_sensor = Counter(
        sensor for event in event_sensor_counts for sensor in event_sensor_counts[event]
    )

    def score(assignments: dict[str, str]) -> float:
        tile_counts = Counter()
        sensor_counts = {split: Counter() for split in split_names}
        present_counts = {split: Counter() for split in split_names}
        for event, split in assignments.items():
            tile_counts[split] += sum(event_sensor_counts[event].values())
            sensor_counts[split].update(event_sensor_counts[event])
            for sensor in event_sensor_counts[event]:
                present_counts[split][sensor] += 1
        tile_error = sum(
            ((tile_counts[split] - total_tiles * fractions[split]) / max(total_tiles, 1)) ** 2
            for split in split_names
        )
        sensor_error = sum(
            ((sensor_counts[split][sensor] - total_sensor[sensor] * fractions[split]) / max(total_sensor[sensor], 1)) ** 2
            for split in split_names
            for sensor in ("S1", "S2")
        )
        presence_error = sum(
            ((present_counts[split][sensor] - total_events_by_sensor[sensor] * fractions[split]) / max(total_events_by_sensor[sensor], 1)) ** 2
            for split in split_names
            for sensor in ("S1", "S2")
        )
        return tile_weight * tile_error + sensor_weight * sensor_error + presence_weight * presence_error

    best_assignment: dict[str, str] | None = None
    best_score = float("inf")
    for _ in range(96):
        shuffled = events[:]
        rng.shuffle(shuffled)
        assignments = {event: split for event, split in zip(shuffled, slots)}
        current_score = score(assignments)
        stale = 0
        for _ in range(2500):
            left, right = rng.sample(split_names, 2)
            left_events = [event for event, split in assignments.items() if split == left]
            right_events = [event for event, split in assignments.items() if split == right]
            event_left = rng.choice(left_events)
            event_right = rng.choice(right_events)
            candidate = assignments.copy()
            candidate[event_left] = right
            candidate[event_right] = left
            candidate_score = score(candidate)
            if candidate_score < current_score - 1e-12:
                assignments = candidate
                current_score = candidate_score
                stale = 0
            else:
                stale += 1
                if stale >= 400:
                    break
        if current_score < best_score - 1e-12:
            best_assignment = assignments
            best_score = current_score
    if best_assignment is None:
        raise RuntimeError("failed to construct an event split")
    return [{"event_id": event, "split": best_assignment[event], "seed": str(seed)} for event in sorted(best_assignment)]


def split_summary(records: list[dict[str, Any]], assignments: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_event = {row["event_id"]: row["split"] for row in assignments}
    grouped = defaultdict(list)
    for row in records:
        grouped[by_event[row["event_id"]]].append(row)
    result = []
    for split in ("train", "val", "test"):
        rows = grouped[split]
        result.append(
            {
                "split": split,
                "events": len({r["event_id"] for r in rows}),
                "tiles": len(rows),
                "s1_tiles": sum(r["sensor"] == "S1" for r in rows),
                "s2_tiles": sum(r["sensor"] == "S2" for r in rows),
                "nan_files": sum(bool(int(r["image_nonfinite_values"])) for r in rows),
            }
        )
    return result


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_protocol_report(
    path: Path,
    root: Path,
    records: list[dict[str, Any]],
    metadata_info: dict[str, Any],
    event_rows: list[dict[str, Any]],
    split_sets: dict[str, tuple[list[dict[str, str]], list[dict[str, Any]]]],
) -> None:
    label_counts = Counter()
    sensor_label_counts = {"S1": Counter(), "S2": Counter()}
    for row in records:
        for token in row["raw_label_codes"].split(";"):
            if token:
                code, count = token.split(":", 1)
                label_counts[int(code)] += int(count)
                sensor_label_counts[row["sensor"]][int(code)] += int(count)
    nan_rows = [r for r in records if int(r["image_nonfinite_values"]) > 0]
    sensor_counts = Counter(r["sensor"] for r in records)
    shape_counts = Counter((r["sensor"], r["channels"], r["height"], r["width"], r["image_dtype"]) for r in records)
    event_sensor_sets = defaultdict(set)
    for row in records:
        event_sensor_sets[row["event_id"]].add(row["sensor"])
    s1_events = sum("S1" in sensors for sensors in event_sensor_sets.values())
    s2_events = sum("S2" in sensors for sensors in event_sensor_sets.values())
    both_events = sum(sensors == {"S1", "S2"} for sensors in event_sensor_sets.values())

    lines = [
        "# STURM-Flood 全量数据审计与锁定协议",
        "",
        f"审计时间（UTC）：{datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"数据根目录：`{root.resolve()}`",
        "",
        "## 1. 审计结论",
        "",
        f"- manifest 为每个传感器影像瓦片一行，并保留同传感器掩膜路径；未构造 SAR–光学配对。总影像瓦片数：{len(records)}。",
        f"- 传感器数量：S1={sensor_counts['S1']}，S2={sensor_counts['S2']}。事件数：{len(event_rows)}。",
        f"- 事件覆盖：S1={s1_events} 个事件，S2={s2_events} 个事件，两者同时覆盖={both_events} 个事件。",
        f"- 图像/掩膜尺寸、文件名配对和元数据文件名匹配均在扫描阶段强制校验。",
        f"- 含非有限值的输入影像瓦片：{len(nan_rows)}；另按 GeoTIFF 声明的 nodata 值排除无效像素。这些像素必须在输入、质量特征、损失和评价中由有效像素掩码排除。",
        "",
        "## 2. 原始标签与训练映射（锁定）",
        "",
        "官方 STURM-Flood 上游代码 `external/STURM-Flood/utils/utility.py` 使用 `np.where(mask_array > 0, 1, 0)` 生成二值水体掩膜；仓库没有给出原始 1–5 各码的独立语义名称。因此本项目不臆造逐码语义，锁定以下可复现操作映射：",
        "",
        """| 原始条件 | 训练标签 |
|---|---:|
| `raw_mask == 0` | 0（非水体） |
| `raw_mask > 0` | 1（水体） |
| `valid_pixel_mask == False` | ignore，不参与损失和评价 |""",
        "",
        "有效像素定义为 `np.isfinite(image).all(axis=0)` 且不命中图像 GeoTIFF 声明的有限 nodata 值；不使用洪水标签计算质量向量。原始标签码只保留在 manifest 中用于追溯，不把 4/5 等编码重新命名为未被官方确认的类别。",
        "",
        f"全量原始标签像素计数：`{dict(sorted(label_counts.items()))}`。",
        f"分传感器原始标签像素计数：S1=`{dict(sorted(sensor_label_counts['S1'].items()))}`；S2=`{dict(sorted(sensor_label_counts['S2'].items()))}`。",
        "",
        "## 3. 输入与事件审计",
        "",
        f"影像布局组合：`{dict((str(k), v) for k, v in shape_counts.items())}`。",
        "",
        "| 字段 | S1 | S2 |",
        "|---|---:|---:|",
        f"| 元数据行 | {metadata_info['S1']['rows']} | {metadata_info['S2']['rows']} |",
        f"| 元数据唯一 tile_id | {metadata_info['S1']['unique_tiles']} | {metadata_info['S2']['unique_tiles']} |",
        f"| 含非有限值的图像 | {sum(r['sensor']=='S1' and int(r['image_nonfinite_values'])>0 for r in records)} | {sum(r['sensor']=='S2' and int(r['image_nonfinite_values'])>0 for r in records)} |",
        f"| 声明 nodata 像素（按图像有效区规则） | {sum(int(r['image_declared_nodata_pixels']) for r in records if r['sensor']=='S1')} | {sum(int(r['image_declared_nodata_pixels']) for r in records if r['sensor']=='S2')} |",
        "",
        "事件 ID 以元数据 `ems_code` 为准，并与文件名中的 `EMSRxxx` 做一致性校验；S1、S2 仅共享事件级划分，不因同名 tile 自动组成配对样本。",
        "",
        "## 4. 事件级划分协议（锁定）",
        "",
        "主划分使用 seed=2026，事件不可拆分，事件配额为 train/val/test=36/12/12；在此约束下同时优化瓦片量约 60%/20%/20% 和 S1/S2 数量比例。敏感性划分固定使用 tile-priority（seed=2027）与 sensor-priority（seed=2028）目标，仅用于敏感性分析。",
        "",
    ]
    for name, (_, rows) in split_sets.items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| split | events | tiles | S1 | S2 | NaN files |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in rows:
            lines.append(f"| {row['split']} | {row['events']} | {row['tiles']} | {row['s1_tiles']} | {row['s2_tiles']} | {row['nan_files']} |")
        lines.append("")
    lines += [
        "## 5. 证据文件",
        "",
        "- `data/manifests/manifest.csv`：瓦片级清单与原始标签统计。",
        "- `data/manifests/event_statistics.csv`：事件级传感器覆盖与质量统计。",
        "- `data/splits/event_split_v1.csv`：主事件划分。",
        "- `data/splits/event_split_sensitivity_a.csv`、`event_split_sensitivity_b.csv`：固定敏感性划分。",
        "- 原始数据完整性依据：`docs/data_integrity_20260915.md`。",
        "",
        "## 6. 当前边界",
        "",
        "本报告锁定数据标签和事件划分协议，不代表 YOLO26 输入通道改造、训练数据加载器或正式实验已经通过验证。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--event-stats", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--sensitivity-a", type=Path, required=True)
    parser.add_argument("--sensitivity-b", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=max(2, min(8, os.cpu_count() or 4)))
    parser.add_argument("--reuse-manifest", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    if args.reuse_manifest:
        with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            records = list(csv.DictReader(handle))
        metadata_info = {sensor: _metadata(root, sensor)[1] for sensor, _ in SENSORS}
    else:
        records, metadata_info = scan_dataset(root, args.workers)
    write_csv(args.manifest, records, MANIFEST_FIELDS)
    event_rows = event_statistics(records)
    write_csv(args.event_stats, event_rows, list(event_rows[0]) if event_rows else ["event_id"])

    split_sets: dict[str, tuple[list[dict[str, str]], list[dict[str, Any]]]] = {}
    for name, seed, mode, output_path in (
        ("main_seed_2026", args.seed, "balanced", args.split),
        ("sensitivity_seed_2027", args.seed + 1, "tile_priority", args.sensitivity_a),
        ("sensitivity_seed_2028", args.seed + 2, "sensor_priority", args.sensitivity_b),
    ):
        assignments = assign_events(records, seed, mode=mode)
        write_csv(output_path, assignments, ["event_id", "split", "seed"])
        split_sets[name] = (assignments, split_summary(records, assignments))

    assignment_signatures = {
        tuple((row["event_id"], row["split"]) for row in assignments)
        for assignments, _ in split_sets.values()
    }
    if len(assignment_signatures) != len(split_sets):
        raise RuntimeError("sensitivity split assignments are not distinct")

    write_protocol_report(args.report, root, records, metadata_info, event_rows, split_sets)
    print(f"manifest={args.manifest} rows={len(records)}")
    print(f"event_stats={args.event_stats} events={len(event_rows)}")
    for name, (_, rows) in split_sets.items():
        print(name, rows)
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
