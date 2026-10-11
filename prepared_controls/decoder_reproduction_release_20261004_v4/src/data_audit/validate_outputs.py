# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Validate generated STURM-Flood audit and split artifacts without reading TIFFs."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    manifest = read_csv(ROOT / "data/manifests/manifest.csv")
    event_stats = read_csv(ROOT / "data/manifests/event_statistics.csv")
    event_ids = {row["event_id"] for row in event_stats}

    assert len(manifest) == 24277, len(manifest)
    assert len({(row["sensor"], row["tile_id"]) for row in manifest}) == len(manifest)
    assert not [row for row in manifest if row["event_id_match"] != "True"]
    assert not [row for row in manifest if row["mask_shape_ok"] != "True"]
    assert not [row for row in manifest if row["mask_finite"] != "True"]
    assert {int(code) for row in manifest for code in (item.split(":", 1)[0] for item in row["raw_label_codes"].split(";"))} == set(range(6))
    assert sum(int(row["image_nonfinite_values"]) > 0 for row in manifest) == 10
    assert sum(int(row["image_nonfinite_values"]) > 0 and row["sensor"] == "S2" for row in manifest) == 10
    assert len(event_stats) == 60
    assert sum(int(row["s1_tiles"]) > 0 for row in event_stats) == 47
    assert sum(int(row["s2_tiles"]) > 0 for row in event_stats) == 29
    assert sum(int(row["s1_tiles"]) > 0 and int(row["s2_tiles"]) > 0 for row in event_stats) == 16

    split_paths = [
        ROOT / "data/splits/event_split_v1.csv",
        ROOT / "data/splits/event_split_sensitivity_a.csv",
        ROOT / "data/splits/event_split_sensitivity_b.csv",
    ]
    signatures = []
    summaries = []
    for path in split_paths:
        rows = read_csv(path)
        assignments = {row["event_id"]: row["split"] for row in rows}
        assert len(rows) == 60
        assert set(assignments) == event_ids
        assert len(assignments) == len(rows)
        assert Counter(assignments.values()) == Counter({"train": 36, "val": 12, "test": 12})
        groups = {split: {event for event, assigned in assignments.items() if assigned == split} for split in ("train", "val", "test")}
        assert not (groups["train"] & groups["val"])
        assert not (groups["train"] & groups["test"])
        assert not (groups["val"] & groups["test"])
        assert groups["train"] | groups["val"] | groups["test"] == event_ids

        tile_rows = Counter()
        sensor_rows = {split: Counter() for split in groups}
        for row in manifest:
            split = assignments[row["event_id"]]
            tile_rows[split] += 1
            sensor_rows[split][row["sensor"]] += 1
        assert sum(tile_rows.values()) == len(manifest)
        summaries.append((path.name, tile_rows, sensor_rows))
        signatures.append(tuple(sorted(assignments.items())))

    assert len(set(signatures)) == 3
    config = yaml.safe_load((ROOT / "configs/project.yaml").read_text(encoding="utf-8"))
    assert config["manifest_path"].endswith("data/manifests/manifest.csv")
    assert config["label_protocol"]["positive_rule"] == "raw_mask_gt_zero"
    assert config["split_protocol"]["unit"] == "event"
    assert (ROOT / "docs/data_audit_20260915.md").exists()

    print("manifest_rows", len(manifest))
    print("event_count", len(event_stats), "event_coverage", {"S1": 47, "S2": 29, "both": 16})
    print("raw_label_codes", list(range(6)), "nonfinite_files", 10)
    for name, tile_rows, sensor_rows in summaries:
        print("  tiles", dict(tile_rows), "sensors", {split: dict(counts) for split, counts in sensor_rows.items()})


if __name__ == "__main__":
    main()
