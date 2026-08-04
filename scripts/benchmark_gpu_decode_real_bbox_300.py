#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from nvidia import nvimgcodec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hexbench.color import palette_coverage
from hexbench.gpu_roi import (
    build_stratified_box_metadata,
    gpu_kmeans_observed_device,
    gpu_liq_like_observed_device,
    gpu_stratified_box_samples_from_metadata,
)
from hexbench.images import open_image_srgb


ROOT = Path(__file__).resolve().parents[1]


def pct(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(array.mean()), 4),
        "p50": round(float(np.percentile(array, 50)), 4),
        "p95": round(float(np.percentile(array, 95)), 4),
    }


def synchronize_ms(function):
    before = time.perf_counter()
    value = function()
    torch.cuda.synchronize()
    return value, (time.perf_counter() - before) * 1000.0


def decode_paths(decoder, paths):
    decoded = decoder.read(paths if len(paths) > 1 else paths[0])
    if len(paths) == 1:
        decoded = [decoded]
    return decoded


def run_pipeline(
    decoder, paths, gpu_metadata, iterations, observed_correction,
    palette_size, kmeans_restarts, pipeline,
):
    decoded = decode_paths(decoder, paths)
    images = torch.stack([torch.as_tensor(image) for image in decoded])
    samples, lengths = gpu_stratified_box_samples_from_metadata(images, gpu_metadata)
    if pipeline == "liq_like":
        result = gpu_liq_like_observed_device(
            samples, lengths, iterations=iterations, palette_size=palette_size
        )
    else:
        result = gpu_kmeans_observed_device(
            samples, lengths, iterations=iterations,
            observed_correction=observed_correction,
            palette_size=palette_size, kmeans_restarts=kmeans_restarts,
        )
    return result, images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/real_bbox_300/manifest.json")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--pipeline", choices=("oklab", "liq_like"), default="liq_like")
    parser.add_argument("--observed-correction", choices=("nearest", "frequency_weighted"), default="frequency_weighted")
    parser.add_argument("--palette-size", type=int, default=5)
    parser.add_argument("--kmeans-restarts", type=int, default=2)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--coverage-samples", type=int, default=4096)
    parser.add_argument("--cuda-streams", type=int, default=4)
    parser.add_argument("--image-limit", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "data/real_bbox_300_gpu_decode_benchmark.json")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = manifest["records"]
    if args.image_limit is not None:
        records = records[:args.image_limit]
    paths = [str(args.manifest.parent / str(record["path"])) for record in records]
    boxes = np.asarray([box for record in records for box in record["bboxes_xyxy"]], dtype=np.int32)
    image_indexes = np.asarray([
        image_index
        for image_index, record in enumerate(records)
        for _ in record["bboxes_xyxy"]
    ], dtype=np.int32)
    decoder = nvimgcodec.Decoder(options=f":num_cuda_streams={args.cuda_streams}")
    gpu_metadata = build_stratified_box_metadata(image_indexes, boxes)

    run_pipeline(
        decoder, paths, gpu_metadata, args.iterations, args.observed_correction,
        args.palette_size, args.kmeans_restarts, args.pipeline,
    )
    torch.cuda.synchronize()
    full_times: list[float] = []
    result = None
    images = None
    for _ in range(args.trials):
        before = time.perf_counter()
        (result, images) = run_pipeline(
            decoder, paths, gpu_metadata, args.iterations, args.observed_correction,
            args.palette_size, args.kmeans_restarts, args.pipeline,
        )
        torch.cuda.synchronize()
        full_times.append((time.perf_counter() - before) * 1000.0)
    assert result is not None and images is not None
    print("full-pipeline timing complete", flush=True)

    decoded, decode_ms = synchronize_ms(lambda: decode_paths(decoder, paths))
    images, stack_ms = synchronize_ms(lambda: torch.stack([torch.as_tensor(image) for image in decoded]))
    (samples, lengths), sample_ms = synchronize_ms(
        lambda: gpu_stratified_box_samples_from_metadata(images, gpu_metadata)
    )
    _, palette_ms = synchronize_ms(
        lambda: (
            gpu_liq_like_observed_device(
                samples, lengths, iterations=args.iterations,
                palette_size=args.palette_size,
            )
            if args.pipeline == "liq_like"
            else gpu_kmeans_observed_device(
                samples, lengths, iterations=args.iterations,
                observed_correction=args.observed_correction,
                palette_size=args.palette_size,
                kmeans_restarts=args.kmeans_restarts,
            )
        )
    )

    decoded_cpu = images.cpu().numpy()
    coverage = []
    exact_pixels = 0
    channel_absolute_error = 0
    channel_count = 0
    cursor = 0
    observed_pillow = 0
    for image_index, record in enumerate(records):
        pillow_image, _ = open_image_srgb(paths[image_index])
        pillow_rgb = np.asarray(pillow_image.convert("RGB"), dtype=np.uint8)
        difference = np.abs(decoded_cpu[image_index].astype(np.int16) - pillow_rgb.astype(np.int16))
        exact_pixels += int(np.all(difference == 0, axis=2).sum())
        channel_absolute_error += int(difference.sum())
        channel_count += int(difference.size)
        for left, top, right, bottom in record["bboxes_xyxy"]:
            pixels = pillow_rgb[top:bottom, left:right].reshape(-1, 3)
            palette = np.asarray(result.palettes[cursor], dtype=np.uint8)
            coverage.append(palette_coverage(pixels, palette, max_samples=args.coverage_samples))
            packed_pixels = (
                (pixels[:, 0].astype(np.uint32) << 16)
                | (pixels[:, 1].astype(np.uint32) << 8)
                | pixels[:, 2].astype(np.uint32)
            )
            packed_palette = (
                (palette[:, 0].astype(np.uint32) << 16)
                | (palette[:, 1].astype(np.uint32) << 8)
                | palette[:, 2].astype(np.uint32)
            )
            observed_pillow += bool(np.isin(packed_palette, packed_pixels).all())
            cursor += 1

    total_pixels = len(records) * int(records[0]["width"]) * int(records[0]["height"])
    payload = {
        "method": (
            f"nvimagecodec_webp_to_gpu_liq_like{args.palette_size}_observed_pruned"
            if args.pipeline == "liq_like"
            else f"nvimagecodec_webp_to_triton_oklab_kmeans{args.palette_size}_r{args.kmeans_restarts}_{args.observed_correction}_observed_pruned"
        ),
        "gpu": torch.cuda.get_device_name(0),
        "images": len(records),
        "boxes": len(boxes),
        "iterations": args.iterations,
        "pipeline": args.pipeline,
        "observed_correction": args.observed_correction,
        "palette_size": args.palette_size,
        "kmeans_restarts": (
            None if args.pipeline == "liq_like" else args.kmeans_restarts
        ),
        "cuda_streams": args.cuda_streams,
        "full_pipeline_superbatch_ms": pct(full_times),
        "full_pipeline_amortized_ms_per_image_p50": round(float(np.median(full_times)) / len(records), 4),
        "full_pipeline_images_s_p50": round(len(records) / (float(np.median(full_times)) / 1000.0), 2),
        "single_pass_stage_ms": {
            "decode_to_cuda": round(decode_ms, 4),
            "stack_cuda_images": round(stack_ms, 4),
            "bbox_sampling": round(sample_ms, 4),
            "palette_and_small_cpu_output": round(palette_ms, 4),
        },
        "coverage_delta_e_ok_x100_lower_better": pct(coverage),
        "observed_in_gpu_decode_boxes": len(boxes),
        "observed_in_pillow_decode_boxes": observed_pillow,
        "palette_size_mean": round(statistics.fmean(map(len, result.palettes)), 4),
        "decode_comparison_to_pillow": {
            "exact_pixel_rate": round(exact_pixels / total_pixels, 8),
            "mean_absolute_channel_error": round(channel_absolute_error / channel_count, 8),
        },
        "notes": "WebP is decoded by a CPU codec fallback inside nvImageCodec, then exposed as CUDA images; local files are warm-cache inputs.",
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
