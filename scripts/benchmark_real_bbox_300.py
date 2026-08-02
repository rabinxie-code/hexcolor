#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hexbench.color import palette_coverage
from hexbench.images import open_image_srgb
from hexbench.roi_batch import process_image_with_boxes


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "real_bbox_300"
OUTPUT = ROOT / "data" / "real_bbox_300_benchmark.json"
METHODS = (
    "pixelero_observed_pruned",
    "hsv_pixelero_observed_pruned",
    "octree_observed_pruned",
)
LIQ_METHODS = (
    "pngquant_liq_speed4_observed_pruned",
    "pngquant_liq_speed6_observed_pruned",
    "pngquant_liq_speed10_observed_pruned",
)
RECOMMENDED_METHODS = METHODS + ("pngquant_liq_speed6_observed_pruned",)
LIQ6_METHODS = ("pngquant_liq_speed6_observed_pruned",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=SOURCE / "manifest.json")
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--coverage-samples", type=int, default=4096)
    parser.add_argument("--processes", type=int, default=64)
    parser.add_argument("--throughput-repeats", type=int, default=4)
    parser.add_argument("--throughput-trials", type=int, default=5)
    parser.add_argument(
        "--suite",
        choices=("recommended", "base", "libimagequant", "libimagequant6"),
        default="recommended",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def pct(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {name: round(float(value), 4) for name, value in (
        ("mean", np.mean(array)), ("p50", np.percentile(array, 50)), ("p95", np.percentile(array, 95))
    )}


def run_timed(item: tuple[str, list[list[int]], str, int, int]) -> dict[str, object]:
    path, boxes, method, samples, box_workers = item
    palettes, timing = process_image_with_boxes(
        path, [tuple(box) for box in boxes], method,
        max_samples=samples, min_cluster_ratio=0.01, box_workers=box_workers,
    )
    return {
        "total_ms": timing.total_ms,
        "decode_ms": timing.decode_ms,
        "sample_ms": timing.sample_ms,
        "palette_ms": timing.palette_ms,
        "palette_sizes": list(timing.palette_sizes),
    }


def run_coverage(item: tuple[str, list[list[int]], str, int, int]) -> list[float]:
    path, boxes, method, samples, coverage_samples = item
    palettes, _ = process_image_with_boxes(
        path, [tuple(box) for box in boxes], method,
        max_samples=samples, min_cluster_ratio=0.01, box_workers=1,
    )
    image, _ = open_image_srgb(path)
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    values: list[float] = []
    for result, (left, top, right, bottom) in zip(palettes, boxes):
        values.append(palette_coverage(
            rgb[top:bottom, left:right].reshape(-1, 3),
            np.asarray(result.palette, dtype=np.uint8),
            max_samples=coverage_samples,
        ))
    return values


def benchmark(records: list[dict[str, object]], base: Path, method: str, args: argparse.Namespace) -> dict[str, object]:
    paths_boxes = [(str(base / str(record["path"])), record["bboxes_xyxy"]) for record in records]
    latency_threads = 1 if method == "pixelero_observed_pruned" else 2
    latency = [run_timed((path, boxes, method, args.samples, latency_threads)) for path, boxes in paths_boxes]

    coverage_work = [(path, boxes, method, args.samples, args.coverage_samples) for path, boxes in paths_boxes]
    with ProcessPoolExecutor(max_workers=args.processes) as executor:
        coverage = [value for chunk in executor.map(run_coverage, coverage_work, chunksize=2) for value in chunk]

    throughput_work = [(path, boxes, method, args.samples, 1) for path, boxes in paths_boxes] * args.throughput_repeats
    with ProcessPoolExecutor(max_workers=args.processes) as executor:
        list(executor.map(run_timed, throughput_work[:args.processes], chunksize=1))
        walls: list[float] = []
        throughput_records: list[dict[str, object]] = []
        for _ in range(args.throughput_trials):
            started = time.perf_counter()
            trial_records = list(executor.map(
                run_timed, throughput_work,
                chunksize=max(1, len(throughput_work) // (args.processes * 4)),
            ))
            walls.append(time.perf_counter() - started)
            throughput_records.extend(trial_records)

    palette_sizes = [int(size) for record in latency for size in record["palette_sizes"]]
    measured_boxes = sum(len(boxes) for _, boxes in paths_boxes) * args.throughput_repeats
    image_rates = [len(throughput_work) / wall for wall in walls]
    box_rates = [measured_boxes / wall for wall in walls]
    return {
        "method": method,
        "single_process_latency_ms": {
            "total": pct([float(record["total_ms"]) for record in latency]),
            "decode": pct([float(record["decode_ms"]) for record in latency]),
            "sample": pct([float(record["sample_ms"]) for record in latency]),
            "palette_all_real_boxes": pct([float(record["palette_ms"]) for record in latency]),
            "box_threads": latency_threads,
        },
        "coverage_delta_e_ok_x100_lower_better": pct(coverage),
        "coverage_box_count": len(coverage),
        "palette_size": {
            "mean": round(statistics.fmean(palette_sizes), 4),
            "below_five_rate": round(sum(size < 5 for size in palette_sizes) / len(palette_sizes), 6),
        },
        "multiprocess": {
            "processes": args.processes,
            "trials": args.throughput_trials,
            "measured_images_per_trial": len(throughput_work),
            "measured_boxes_per_trial": measured_boxes,
            "wall_seconds": pct(walls),
            "images_s": pct(image_rates),
            "boxes_s": pct(box_rates),
            "mean_worker_total_ms": round(statistics.fmean(float(record["total_ms"]) for record in throughput_records), 4),
        },
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = manifest["records"]
    counts = [int(record["bbox_count"]) for record in records]
    methods = {
        "recommended": RECOMMENDED_METHODS,
        "base": METHODS,
        "libimagequant": LIQ_METHODS,
        "libimagequant6": LIQ6_METHODS,
    }[args.suite]
    results = [benchmark(records, args.manifest.parent, method, args) for method in methods]
    payload = {
        "dataset": manifest.get("remote", str(args.manifest)),
        "manifest": str(args.manifest),
        "images": len(records),
        "real_bbox_count": sum(counts),
        "bbox_count_per_image": {**pct([float(value) for value in counts]), "min": min(counts), "max": max(counts)},
        "bbox_source": manifest["bbox_source"],
        "bbox_source_format": manifest["bbox_source_format"],
        "extraction_samples_per_box": args.samples,
        "coverage_evaluation_samples_per_box": args.coverage_samples,
        "coverage_metric": "mean nearest-palette Oklab distance x 100; lower is better",
        "results": results,
    }
    output = args.output or (
        ROOT / "data" / "real_bbox_300_libimagequant_benchmark.json"
        if args.suite == "libimagequant" else OUTPUT
    )
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
