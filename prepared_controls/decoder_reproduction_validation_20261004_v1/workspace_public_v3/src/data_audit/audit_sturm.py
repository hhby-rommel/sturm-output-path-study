# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Audit a downloaded STURM-Flood archive and build a manifest.

This script is intentionally conservative: it reports what is present and
never invents SAR-optical pairs. It can run before rasterio is installed; in
that case it still creates a path/event/sensor inventory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


EVENT_RE = re.compile(r"(EMSR\d+)", re.IGNORECASE)


def infer_sensor(path: Path) -> Optional[str]:
    text = "/".join(path.parts).lower()
    if "sentinel1" in text or "/s1/" in text or "_s1" in text:
        return "S1"
    if "sentinel2" in text or "/s2/" in text or "_s2" in text:
        return "S2"
    return None


def infer_event(path: Path) -> str:
    match = EVENT_RE.search(path.name)
    if match:
        return match.group(1).upper()
    match = EVENT_RE.search("_".join(path.parts))
    return match.group(1).upper() if match else "UNKNOWN"


def checksum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_raster_info(path: Path) -> Dict[str, Any]:
    try:
        import rasterio

        with rasterio.open(path) as dataset:
            return {
                "height": dataset.height,
                "width": dataset.width,
                "channels": dataset.count,
                "dtype": ",".join(dataset.dtypes),
                "nodata": dataset.nodata,
                "crs": str(dataset.crs) if dataset.crs else "",
            }
    except Exception as exc:
        return {"read_error": f"{type(exc).__name__}: {exc}"}


def is_mask_path(path: Path) -> bool:
    text = "/".join(path.parts).lower()
    return any(token in text for token in ("floodmap", "mask", "label", "groundtruth"))


def scan(root: Path) -> list[Dict[str, Any]]:
    suffixes = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        sensor = infer_sensor(path)
        records.append(
            {
                "path": str(path.resolve()),
                "relative_path": str(path.relative_to(root)),
                "filename": path.name,
                "stem": path.stem,
                "event_id": infer_event(path),
                "sensor": sensor or ("MASK" if is_mask_path(path) else "UNKNOWN"),
                "is_mask": is_mask_path(path),
                "bytes": path.stat().st_size,
                **read_raster_info(path),
            }
        )
    return records


def write_csv(records: Iterable[Dict[str, Any]], destination: Path) -> None:
    records = list(records)
    keys = sorted({key for record in records for key in record})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def summarize(records: list[Dict[str, Any]]) -> Dict[str, Any]:
    by_event = defaultdict(lambda: Counter())
    for record in records:
        by_event[record["event_id"]][record["sensor"]] += 1
    return {
        "record_count": len(records),
        "sensor_counts": dict(Counter(record["sensor"] for record in records)),
        "event_count": len(by_event),
        "events": {event: dict(counts) for event, counts in sorted(by_event.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--checksums", type=Path)
    args = parser.parse_args()

    if not args.root.exists():
        raise FileNotFoundError(args.root)
    records = scan(args.root)
    write_csv(records, args.manifest)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summarize(records), ensure_ascii=False, indent=2), encoding="utf-8")

    if args.checksums:
        checksum_rows = []
        for path in sorted(args.root.rglob("*")):
            if path.is_file():
                checksum_rows.append({"relative_path": str(path.relative_to(args.root)), "sha256": checksum(path)})
        write_csv(checksum_rows, args.checksums)

    print(json.dumps(summarize(records), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
