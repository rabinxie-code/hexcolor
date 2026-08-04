#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hexbench.color import palette_coverage
from hexbench.gpu_roi import gpu_kmeans_observed, gpu_liq_like_observed
from hexbench.images import open_image_srgb
from hexbench.roi_batch import _stratified_box_pixels

ROOT = Path(__file__).resolve().parents[1]


def load_record(item):
    base, record, max_samples = item
    image, _ = open_image_srgb(base / str(record["path"]))
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    boxes = record["bboxes_xyxy"]
    return [_stratified_box_pixels(rgb, tuple(box), max_samples) for box in boxes], rgb, boxes


def load_record_packed(item):
    samples, _, _ = load_record(item)
    max_samples = item[2]
    padded = np.zeros((len(samples), max_samples, 3), dtype=np.uint8)
    lengths = np.asarray([len(sample) for sample in samples], dtype=np.int32)
    for index, sample in enumerate(samples):
        padded[index, :len(sample)] = sample
    return padded, lengths


def pct(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": round(float(values.mean()), 4), "p50": round(float(np.percentile(values, 50)), 4), "p95": round(float(np.percentile(values, 95)), 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/real_bbox_300/manifest.json")
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--coverage-samples", type=int, default=4096)
    parser.add_argument("--decode-workers", type=int, default=32)
    parser.add_argument("--process-workers", type=int, default=64)
    parser.add_argument("--prepare-trials", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--pipeline", choices=("oklab", "liq_like"), default="liq_like")
    parser.add_argument("--observed-correction", choices=("nearest", "frequency_weighted"), default="frequency_weighted")
    parser.add_argument("--palette-size", type=int, default=5)
    parser.add_argument("--kmeans-restarts", type=int, default=2)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "data/real_bbox_300_gpu_benchmark.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    records = manifest["records"]
    work = [(args.manifest.parent, record, args.samples) for record in records]
    preparation_times = []
    with ProcessPoolExecutor(max_workers=args.process_workers) as executor:
        list(executor.map(load_record_packed, work[:args.process_workers], chunksize=1))
        chunks = None
        for _ in range(args.prepare_trials):
            before = time.perf_counter()
            chunks = list(executor.map(load_record_packed, work, chunksize=2))
            padded = np.concatenate([chunk for chunk, _ in chunks])
            lengths = np.concatenate([chunk_lengths for _, chunk_lengths in chunks])
            preparation_times.append((time.perf_counter() - before) * 1000)
    assert chunks is not None
    box_count = len(lengths)
    palette_kwargs = {
        "iterations": args.iterations,
        "observed_correction": args.observed_correction,
        "palette_size": args.palette_size,
        "kmeans_restarts": args.kmeans_restarts,
    }
    run_palette = gpu_liq_like_observed if args.pipeline == "liq_like" else gpu_kmeans_observed
    if args.pipeline == "liq_like":
        palette_kwargs.pop("observed_correction")
        palette_kwargs.pop("kmeans_restarts")
    run_palette(padded, lengths, **palette_kwargs)
    torch.cuda.synchronize()
    times = []
    result = None
    for _ in range(args.trials):
        before = time.perf_counter()
        result = run_palette(padded, lengths, **palette_kwargs)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - before) * 1000)
    median_gpu = float(np.median(times))
    median_preparation = float(np.median(preparation_times))
    image_counts = np.asarray([int(record["bbox_count"]) for record in records])
    cumulative_boxes = np.cumsum(image_counts)
    batch_scaling = []
    for image_count in (1, 8, 32, 128):
        subset_boxes = int(cumulative_boxes[image_count - 1])
        run_palette(padded[:subset_boxes], lengths[:subset_boxes], **palette_kwargs)
        torch.cuda.synchronize()
        scaling_times = []
        for _ in range(10):
            before = time.perf_counter()
            run_palette(padded[:subset_boxes], lengths[:subset_boxes], **palette_kwargs)
            torch.cuda.synchronize()
            scaling_times.append((time.perf_counter() - before) * 1000)
        batch_p50 = float(np.median(scaling_times))
        batch_scaling.append({
            "images": image_count,
            "boxes": subset_boxes,
            "batch_p50_ms": round(batch_p50, 4),
            "amortized_ms_per_image": round(batch_p50 / image_count, 4),
        })

    with ThreadPoolExecutor(max_workers=args.decode_workers) as executor:
        loaded = list(executor.map(load_record, work))
    coverage = []
    observed_boxes = 0
    cursor = 0
    for _, rgb, boxes in loaded:
        for left, top, right, bottom in boxes:
            pixels = rgb[top:bottom, left:right].reshape(-1, 3)
            palette = np.asarray(result.palettes[cursor], dtype=np.uint8)
            coverage.append(palette_coverage(pixels, palette, max_samples=args.coverage_samples))
            observed = {tuple(color) for color in pixels.tolist()}
            observed_boxes += all(tuple(color) in observed for color in palette.tolist())
            cursor += 1
    sequential_ms = median_preparation + median_gpu
    overlapped_ms = max(median_preparation, median_gpu)
    payload = {
        "method": (
            f"gpu2_liq_like{args.palette_size}_observed_pruned"
            if args.pipeline == "liq_like"
            else f"triton_oklab_kmeans{args.palette_size}_r{args.kmeans_restarts}_{args.observed_correction}_observed_pruned"
        ), "gpu": torch.cuda.get_device_name(0),
        "pipeline": args.pipeline,
        "images": len(records), "boxes": box_count, "samples_per_box": args.samples, "iterations": args.iterations,
        "observed_correction": (
            "frequency_weighted" if args.pipeline == "liq_like" else args.observed_correction
        ),
        "palette_size": args.palette_size,
        "kmeans_restarts": None if args.pipeline == "liq_like" else args.kmeans_restarts,
        "persistent_process_feeder": {"workers": args.process_workers, "trials": args.prepare_trials, "superbatch_ms": pct(preparation_times)},
        "gpu_superbatch_ms": pct(times), "gpu_amortized_ms_per_image_p50": round(median_gpu / len(records), 4),
        "gpu_amortized_ms_per_box_p50": round(median_gpu / box_count, 4),
        "sequential_pipeline_amortized_ms_per_image": round(sequential_ms / len(records), 4),
        "sequential_pipeline_images_s": round(len(records) / (sequential_ms / 1000), 2),
        "overlapped_pipeline_amortized_ms_per_image": round(overlapped_ms / len(records), 4),
        "overlapped_pipeline_images_s": round(len(records) / (overlapped_ms / 1000), 2),
        "gpu_batch_scaling": batch_scaling + [{"images": len(records), "boxes": box_count, "batch_p50_ms": round(median_gpu, 4), "amortized_ms_per_image": round(median_gpu / len(records), 4)}],
        "coverage_delta_e_ok_x100_lower_better": pct(coverage), "observed_boxes": observed_boxes,
        "palette_size_mean": round(statistics.fmean(map(len, result.palettes)), 4),
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
