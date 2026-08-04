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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hexbench.color import palette_coverage, srgb_to_oklab
from hexbench.gpu_roi import (
    GpuBatchResult,
    gpu_kmeans_observed,
    gpu_liq_like_observed,
)
from hexbench.images import open_image_srgb
from hexbench.roi_batch import BoxPalette, libimagequant_observed_palette
from scripts.benchmark_gpu_real_bbox_300 import load_record_packed


ROOT = Path(__file__).resolve().parents[1]


def pct(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(np.mean(array)), 4),
        "p50": round(float(np.percentile(array, 50)), 4),
        "p95": round(float(np.percentile(array, 95)), 4),
    }


def cpu_reference(item: tuple[np.ndarray, int]) -> BoxPalette:
    pixels, palette_size = item
    return libimagequant_observed_palette(
        pixels, palette_size=palette_size, speed=6, min_cluster_ratio=0.01
    )


def directed_palette_distance(
    source: tuple[tuple[int, int, int], ...],
    source_weights: tuple[float, ...],
    target: tuple[tuple[int, int, int], ...],
) -> float:
    source_lab = srgb_to_oklab(np.asarray(source, dtype=np.uint8))
    target_lab = srgb_to_oklab(np.asarray(target, dtype=np.uint8))
    nearest = np.linalg.norm(
        source_lab[:, None, :] - target_lab[None, :, :], axis=2
    ).min(axis=1)
    weights = np.asarray(source_weights, dtype=np.float64)
    weights /= max(float(weights.sum()), 1e-12)
    return float(np.sum(nearest * weights) * 100.0)


def agreement(
    predicted: GpuBatchResult, references: list[BoxPalette]
) -> dict[str, dict[str, float]]:
    cpu_to_gpu = []
    symmetric = []
    for palette, weights, reference in zip(
        predicted.palettes, predicted.weights, references
    ):
        forward = directed_palette_distance(
            reference.palette, reference.weights, palette
        )
        backward = directed_palette_distance(
            palette, weights, reference.palette
        )
        cpu_to_gpu.append(forward)
        symmetric.append((forward + backward) * 0.5)
    return {
        "cpu_gt_to_gpu_weighted_delta_e_ok_x100": pct(cpu_to_gpu),
        "symmetric_weighted_delta_e_ok_x100": pct(symmetric),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/real_bbox_300/manifest.json",
    )
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--pipeline", choices=("oklab", "liq_like"), default="liq_like")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--image-limit", type=int)
    parser.add_argument("--coverage-samples", type=int, default=4096)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/real_bbox_300_gpu_cpu_gt_benchmark.json",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = manifest["records"]
    if args.image_limit is not None:
        records = records[: args.image_limit]
    work = [(args.manifest.parent, record, args.samples) for record in records]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        chunks = list(executor.map(load_record_packed, work, chunksize=2))
    padded = np.concatenate([chunk for chunk, _ in chunks])
    lengths = np.concatenate([item_lengths for _, item_lengths in chunks])
    sample_rows = [
        np.ascontiguousarray(row[: int(length)])
        for row, length in zip(padded, lengths)
    ]

    all_results: dict[int, dict[str, GpuBatchResult]] = {}
    timings: dict[int, dict[str, list[float]]] = {}
    references: dict[int, list[BoxPalette]] = {}
    cpu_reference_ms: dict[int, float] = {}
    for palette_size in (5, 16):
        all_results[palette_size] = {}
        timings[palette_size] = {}
        variants = ((0, "gpu2_liq_like"),) if args.pipeline == "liq_like" else (
            (1, "gpu_1_restart"),
            (2, "gpu_2_restarts"),
        )
        for restarts, name in variants:
            kwargs = {
                "iterations": args.iterations,
                "palette_size": palette_size,
            }
            run_palette = (
                gpu_liq_like_observed
                if args.pipeline == "liq_like"
                else gpu_kmeans_observed
            )
            if args.pipeline == "oklab":
                kwargs.update(
                    kmeans_restarts=restarts,
                    observed_correction="frequency_weighted",
                )
            run_palette(padded, lengths, **kwargs)
            torch.cuda.synchronize()
            elapsed = []
            result = None
            for _ in range(args.trials):
                before = time.perf_counter()
                result = run_palette(padded, lengths, **kwargs)
                torch.cuda.synchronize()
                elapsed.append((time.perf_counter() - before) * 1000.0)
            assert result is not None
            all_results[palette_size][name] = result
            timings[palette_size][name] = elapsed

        before = time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            references[palette_size] = list(
                executor.map(
                    cpu_reference,
                    ((pixels, palette_size) for pixels in sample_rows),
                    chunksize=8,
                )
            )
        cpu_reference_ms[palette_size] = (time.perf_counter() - before) * 1000.0

    coverage: dict[int, dict[str, list[float]]] = {
        size: {
            "cpu": [],
            **(
                {"gpu2_liq_like": []}
                if args.pipeline == "liq_like"
                else {"gpu_1_restart": [], "gpu_2_restarts": []}
            ),
        }
        for size in (5, 16)
    }
    cursor = 0
    for record in records:
        image, _ = open_image_srgb(args.manifest.parent / str(record["path"]))
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        for left, top, right, bottom in record["bboxes_xyxy"]:
            pixels = rgb[top:bottom, left:right].reshape(-1, 3)
            for palette_size in (5, 16):
                cpu_palette = np.asarray(
                    references[palette_size][cursor].palette, dtype=np.uint8
                )
                coverage[palette_size]["cpu"].append(
                    palette_coverage(
                        pixels, cpu_palette, max_samples=args.coverage_samples
                    )
                )
                names = (
                    ("gpu2_liq_like",)
                    if args.pipeline == "liq_like"
                    else ("gpu_1_restart", "gpu_2_restarts")
                )
                for name in names:
                    gpu_palette = np.asarray(
                        all_results[palette_size][name].palettes[cursor],
                        dtype=np.uint8,
                    )
                    coverage[palette_size][name].append(
                        palette_coverage(
                            pixels,
                            gpu_palette,
                            max_samples=args.coverage_samples,
                        )
                    )
            cursor += 1

    payload: dict[str, object] = {
        "gpu": torch.cuda.get_device_name(0),
        "images": len(records),
        "boxes": len(lengths),
        "samples_per_box": args.samples,
        "iterations": args.iterations,
        "pipeline": args.pipeline,
        "metric_note": (
            "CPU GT is libimagequant speed 6 plus observed-color correction and "
            "1% pruning. Delta-E metrics use Oklab x100; lower is better."
        ),
        "palette_sizes": {},
    }
    size_payload = payload["palette_sizes"]
    assert isinstance(size_payload, dict)
    for palette_size in (5, 16):
        variants = {}
        names = (
            ("gpu2_liq_like",)
            if args.pipeline == "liq_like"
            else ("gpu_1_restart", "gpu_2_restarts")
        )
        for name in names:
            result = all_results[palette_size][name]
            median_ms = float(np.median(timings[palette_size][name]))
            variants[name] = {
                "gpu_superbatch_ms": pct(timings[palette_size][name]),
                "gpu_amortized_ms_per_image_p50": round(
                    median_ms / len(records), 4
                ),
                "coverage_delta_e_ok_x100": pct(coverage[palette_size][name]),
                "palette_size_mean": round(
                    statistics.fmean(map(len, result.palettes)), 4
                ),
                **agreement(result, references[palette_size]),
            }
        size_payload[str(palette_size)] = {
            "cpu_reference_build_ms": round(cpu_reference_ms[palette_size], 4),
            "cpu_reference_coverage_delta_e_ok_x100": pct(
                coverage[palette_size]["cpu"]
            ),
            "cpu_reference_palette_size_mean": round(
                statistics.fmean(
                    len(item.palette) for item in references[palette_size]
                ),
                4,
            ),
            "variants": variants,
        }

    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
