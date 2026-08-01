from __future__ import annotations

import json
import math
import os
import shutil
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .dataset import CropRecord


DISPLAY_RECORD_KEYS = (
    "method",
    "hex",
    "primary_rgb",
    "palette_hex",
    "weights",
    "confidence",
    "route",
    "observed",
    "pixels",
    "elapsed_ms",
    "palette_coverage_delta_e_ok",
    "jpeg60_delta_e_ok",
    "resize50_delta_e_ok",
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _minimal_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: record[key] for key in DISPLAY_RECORD_KEYS}


def _difference_score(records: Iterable[dict[str, Any]]) -> float:
    colors = [record["primary_rgb"] for record in records]
    total = 0.0
    for left in range(len(colors)):
        for right in range(left + 1, len(colors)):
            total += math.sqrt(
                sum((float(colors[left][channel]) - float(colors[right][channel])) ** 2 for channel in range(3))
            )
    return total


def _publish_asset(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == source.stat().st_size:
        return
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def export_browser_samples(
    crops: list[CropRecord],
    checkpoint_dir: str | Path,
    site_dir: str | Path,
    method_order: tuple[str, ...],
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Export a light 10k index plus one on-demand detail JSON per crop."""

    checkpoint_root = Path(checkpoint_dir)
    site_root = Path(site_dir)
    record_root = site_root / "sample-records"
    asset_root = site_root / "assets" / "crops"
    record_root.mkdir(parents=True, exist_ok=True)
    asset_root.mkdir(parents=True, exist_ok=True)

    python_methods = tuple(method for method in method_order if method != "color_thief_v3")
    color_payload = json.loads((checkpoint_root / "color_thief_v3.json").read_text(encoding="utf-8"))
    color_records = color_payload["records"]
    if len(color_records) != len(crops):
        raise RuntimeError("Color Thief browser export count does not match crop manifest")

    index_records: list[dict[str, Any]] = []
    with ExitStack() as stack:
        handles = {
            method: stack.enter_context((checkpoint_root / f"{method}.jsonl").open("r", encoding="utf-8"))
            for method in python_methods
        }
        for index, crop in enumerate(crops):
            full_records: dict[str, dict[str, Any]] = {}
            for method in python_methods:
                line = handles[method].readline()
                if not line:
                    raise RuntimeError(f"{method} checkpoint ended before crop {index}")
                full_records[method] = json.loads(line)
            full_records["color_thief_v3"] = color_records[index]
            if any(record.get("crop_id") != crop.crop_id for record in full_records.values()):
                raise RuntimeError(f"Checkpoint order mismatch at {crop.crop_id}")

            ordered_records = [full_records[method] for method in method_order]
            asset_path = f"assets/crops/{Path(crop.path).name}"
            record_path = f"sample-records/{index:05d}.json"
            group = {
                "crop_id": crop.crop_id,
                "batch": crop.batch,
                "variant": crop.variant,
                "source_file": crop.source_file,
                "asset_path": asset_path,
                "width": crop.width,
                "height": crop.height,
                "methods": {
                    method: _minimal_record(full_records[method]) for method in method_order
                },
            }
            _atomic_json(site_root / record_path, group)
            _publish_asset(Path(crop.path), asset_root / Path(crop.path).name)

            index_records.append(
                {
                    "crop_id": crop.crop_id,
                    "batch": crop.batch,
                    "variant": crop.variant,
                    "source_file": crop.source_file,
                    "asset_path": asset_path,
                    "record_path": record_path,
                    "route": full_records["adaptive_v1"]["route"],
                    "difference": round(_difference_score(ordered_records), 4),
                    "latency": round(max(float(record["elapsed_ms"]) for record in ordered_records), 4),
                }
            )
            if (index + 1) % 1_000 == 0:
                print(f"browser_export: {index + 1}/{len(crops)}", flush=True)

        for method, handle in handles.items():
            if handle.readline():
                raise RuntimeError(f"{method} checkpoint contains extra records")

    timestamp = generated_at or datetime.now(timezone.utc).isoformat()
    index_payload = {
        "schema_version": 1,
        "generated_at": timestamp,
        "count": len(index_records),
        "methods": list(method_order),
        "records": index_records,
    }
    _atomic_json(site_root / "sample-index.json", index_payload)
    return {
        "count": len(index_records),
        "index_path": "sample-index.json",
        "record_dir": "sample-records",
        "asset_dir": "assets/crops",
    }
