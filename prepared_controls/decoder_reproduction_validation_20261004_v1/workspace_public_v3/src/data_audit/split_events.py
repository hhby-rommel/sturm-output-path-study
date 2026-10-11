# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Create deterministic event-level train/val/test assignments.

The splitter never separates tiles from the same event. It uses a greedy
stratification pass over event-level sensor counts and mean tile counts; the
final assignment is always inspected from the generated CSV before training.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def assign_events(rows: list[dict[str, str]], seed: int = 2026) -> list[dict[str, str]]:
    rng = random.Random(seed)
    event_stats = defaultdict(Counter)
    for row in rows:
        event = row.get("event_id", "UNKNOWN")
        if row.get("is_mask", "False").lower() == "true":
            continue
        sensor = row.get("sensor", "UNKNOWN")
        event_stats[event][sensor] += 1

    events = list(event_stats)
    if len(events) < 3:
        raise ValueError(f"need at least 3 identifiable events, found {len(events)}")
    rng.shuffle(events)
    events.sort(key=lambda event: sum(event_stats[event].values()), reverse=True)

    targets = {
        "train": 0.60 * len(events),
        "val": 0.20 * len(events),
        "test": 0.20 * len(events),
    }
    assignments: dict[str, str] = {}
    counts = Counter()
    sensor_counts = {split: Counter() for split in targets}

    for event in events:
        candidates = []
        for split in targets:
            overflow = max(0.0, counts[split] + 1 - targets[split])
            sensor_penalty = 0.0
            for sensor in ("S1", "S2"):
                total = sum(event_stats[item][sensor] for item in events)
                desired = total * targets[split] / len(events)
                sensor_penalty += abs(sensor_counts[split][sensor] + event_stats[event][sensor] - desired)
            candidates.append((overflow, sensor_penalty, counts[split], split))
        _, _, _, selected = min(candidates)
        assignments[event] = selected
        counts[selected] += 1
        sensor_counts[selected].update(event_stats[event])

    return [
        {"event_id": event, "split": assignments[event], "seed": str(seed)}
        for event in sorted(assignments)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    assignments = assign_events(rows, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["event_id", "split", "seed"])
        writer.writeheader()
        writer.writerows(assignments)
    print(f"wrote {len(assignments)} event assignments to {args.output}")


if __name__ == "__main__":
    main()
