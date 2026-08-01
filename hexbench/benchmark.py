from __future__ import annotations

import io
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .color import delta_e_ok, palette_coverage
from .browser_export import export_browser_samples
from .dataset import CropRecord, materialize_s3_crops, materialize_synthetic_cases
from .images import open_image_srgb
from .methods import METHODS
from .models import ExtractionResult


METHOD_LABELS = {
    "adaptive_v1": "Adaptive Hex v1",
    "tencent_hsv": "腾讯 HSV 直方图",
    "pixelero_rgb_hist": "Pixelero RGB 直方图",
    "octree": "Octree · native",
    "color_thief_v3": "Color Thief v3 · OKLCH",
    "pngquant_liq": "pngquant · libimagequant",
    "colorpipette_inspired": "ColorPipette-inspired",
}

BENCHMARK_METHOD_ORDER = (
    "adaptive_v1",
    "tencent_hsv",
    "pixelero_rgb_hist",
    "octree",
    "color_thief_v3",
    "pngquant_liq",
    "colorpipette_inspired",
)

TARGET_CROPS = 100_000_000_000
TARGET_DAYS = 3
TARGET_DAY_SCENARIOS = (3, 7, 15)
CAPACITY_SAFETY_FACTOR = 1.5
SOURCE_IMAGES = 5_000_000_000
AVERAGE_SOURCE_MPIX = 1.0
AVERAGE_COMPRESSED_MB = 0.5
TARGET_WORKER_GPIX_S = 0.5


def capacity_target(crops: int, days: float, safety_factor: float) -> tuple[float, float]:
    """Return required and provisioned crops/s for a completion window."""

    if crops <= 0:
        raise ValueError("crops must be positive")
    if days <= 0:
        raise ValueError("days must be positive")
    if safety_factor < 1:
        raise ValueError("safety_factor must be at least 1")
    required = crops / (days * 86400)
    return required, required * safety_factor


def source_pipeline_capacity(
    crops: int,
    source_images: int,
    days: float,
    average_mpix: float,
    average_compressed_mb: float,
    worker_gpix_s: float,
    safety_factor: float,
) -> dict[str, float | int]:
    """Capacity model when all regions are aggregated after decoding each source once."""

    if source_images <= 0 or average_mpix <= 0 or worker_gpix_s <= 0:
        raise ValueError("source_images, average_mpix and worker_gpix_s must be positive")
    required_crops_s, _ = capacity_target(crops, days, safety_factor)
    seconds = days * 86400
    images_s = source_images / seconds
    gpix_s = source_images * average_mpix / seconds / 1_000
    return {
        "source_images": source_images,
        "crops_per_source_image": crops / source_images,
        "average_source_mpix": average_mpix,
        "average_compressed_mb": average_compressed_mb,
        "images_s": images_s,
        "crops_s": required_crops_s,
        "source_gpix_s": gpix_s,
        "compressed_input_gb_s": images_s * average_compressed_mb / 1_000,
        "decoded_rgb_gb_s": gpix_s * 3,
        "worker_gpix_s": worker_gpix_s,
        "estimated_fused_workers_1_5x": math.ceil(gpix_s / worker_gpix_s * safety_factor),
    }


def build_capacity_scenario(days: int, crop_rates: dict[str, float]) -> dict[str, Any]:
    """Build one auditable 100B capacity scenario from measured crop rates."""

    required, provisioned = capacity_target(TARGET_CROPS, days, CAPACITY_SAFETY_FACTOR)
    decode_once = source_pipeline_capacity(
        TARGET_CROPS,
        SOURCE_IMAGES,
        days,
        AVERAGE_SOURCE_MPIX,
        AVERAGE_COMPRESSED_MB,
        TARGET_WORKER_GPIX_S,
        CAPACITY_SAFETY_FACTOR,
    )
    return {
        "days": days,
        "target_crops_s": round(required, 3),
        "provisioned_crops_s": round(provisioned, 3),
        "decode_once_model": decode_once,
        "cpu_equivalent_processes_1_5x": {
            method: math.ceil(provisioned / max(float(rate), 1e-12)) for method, rate in crop_rates.items()
        },
    }


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile)) if values else float("nan")


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _jpeg_variant(image: Image.Image) -> Image.Image:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=60, optimize=False)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        compressed.load()
        return compressed.convert("RGB")


def _resize_variant(image: Image.Image) -> Image.Image:
    width, height = image.size
    reduced = image.convert("RGB").resize((max(1, width // 2), max(1, height // 2)), Image.Resampling.BILINEAR)
    return reduced.resize((width, height), Image.Resampling.BILINEAR)


def _run_once(record: CropRecord, method_name: str) -> dict[str, Any]:
    method = METHODS[method_name]
    start = time.perf_counter_ns()
    image, metadata = open_image_srgb(record.path)
    decoded = time.perf_counter_ns()
    result = method(image, record.crop_id)
    finished = time.perf_counter_ns()

    base_pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    coverage = palette_coverage(base_pixels, np.asarray(result.palette, dtype=np.uint8))

    jpeg_image, _ = open_image_srgb(record.jpeg_path)
    resize_image, _ = open_image_srgb(record.resize_path)
    jpeg_result = method(jpeg_image, record.crop_id + ":jpeg60")
    resize_result = method(resize_image, record.crop_id + ":resize50")
    jpeg_delta = float(delta_e_ok(np.asarray(result.primary), np.asarray(jpeg_result.primary)))
    resize_delta = float(delta_e_ok(np.asarray(result.primary), np.asarray(resize_result.primary)))

    decode_ms = (decoded - start) / 1e6
    method_ms = (finished - decoded) / 1e6
    elapsed_ms = (finished - start) / 1e6
    pixels = image.width * image.height
    payload = result.to_dict()
    payload.update(
        {
            "crop_id": record.crop_id,
            "source_file": record.source_file,
            "batch": record.batch,
            "variant": record.variant,
            "asset_path": f"assets/crops/{Path(record.path).name}",
            "width": image.width,
            "height": image.height,
            "pixels": pixels,
            "decode_ms": round(decode_ms, 6),
            "method_ms": round(method_ms, 6),
            "elapsed_ms": round(elapsed_ms, 6),
            "method_mpix_s": round((pixels / 1e6) / max(method_ms / 1000.0, 1e-12), 6),
            "end_to_end_crops_s": round(1000.0 / max(elapsed_ms, 1e-12), 6),
            "palette_coverage_delta_e_ok": round(coverage, 6),
            "jpeg60_delta_e_ok": round(jpeg_delta, 6),
            "resize50_delta_e_ok": round(resize_delta, 6),
            "source_metadata": metadata,
        }
    )
    return payload


_WORKER_METHOD_NAME = ""


def _initialize_method_worker(method_name: str, warm_path: str) -> None:
    global _WORKER_METHOD_NAME
    _WORKER_METHOD_NAME = method_name
    image, _ = open_image_srgb(warm_path)
    METHODS[method_name](image, f"warmup:{method_name}")


def _run_worker(record: CropRecord) -> dict[str, Any]:
    return _run_once(record, _WORKER_METHOD_NAME)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # A killed process can leave only the final append incomplete.
            if line_number == len(lines):
                break
            raise
    return records


def _run_python_method_checkpointed(
    crops: list[CropRecord],
    method_name: str,
    workers: int,
    checkpoint_path: Path,
    resume: bool,
) -> list[dict[str, Any]]:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_jsonl(checkpoint_path) if resume else []
    expected_ids = [record.crop_id for record in crops]
    existing_ids = [str(record.get("crop_id")) for record in existing]
    if existing_ids != expected_ids[: len(existing_ids)]:
        raise RuntimeError(f"Checkpoint does not match this crop manifest: {checkpoint_path}")
    if len(existing) > len(crops):
        raise RuntimeError(f"Checkpoint contains more records than requested: {checkpoint_path}")

    remaining = crops[len(existing) :]
    if not remaining:
        print(f"{method_name}: resumed {len(existing)}/{len(crops)} records", flush=True)
        return existing

    mode = "a" if existing else "w"
    with checkpoint_path.open(mode, encoding="utf-8") as checkpoint:
        if workers == 1:
            warm_image, _ = open_image_srgb(crops[0].path)
            METHODS[method_name](warm_image, f"warmup:{method_name}")
            iterator = (_run_once(record, method_name) for record in remaining)
            executor = None
        else:
            executor = ProcessPoolExecutor(
                max_workers=workers,
                initializer=_initialize_method_worker,
                initargs=(method_name, crops[0].path),
            )
            iterator = executor.map(_run_worker, remaining, chunksize=max(1, len(remaining) // (workers * 40)))
        try:
            for index, record in enumerate(iterator, start=len(existing) + 1):
                existing.append(record)
                checkpoint.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                if index % 100 == 0:
                    checkpoint.flush()
                if index % 500 == 0 or index == len(crops):
                    print(f"{method_name}: {index}/{len(crops)}", flush=True)
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
    return existing


def _color_thief_result(raw: dict[str, Any], audit: dict[str, Any]) -> ExtractionResult:
    palette = tuple(tuple(int(channel) for channel in color) for color in raw.get("palette", []))
    if not palette:
        palette = ((0, 0, 0),)
    weights = tuple(float(value) for value in raw.get("weights", []))
    if len(weights) != len(palette) or sum(weights) <= 1e-12:
        weights = tuple(1.0 / len(palette) for _ in palette)
    return ExtractionResult(
        method="color_thief_v3",
        primary=palette[0],
        palette=palette,
        weights=weights,
        confidence=float(weights[0]),
        route="oklch_mmcq_q10",
        observed=False,
        diagnostics={
            "implementation": f"official colorthief {audit['version']} package",
            "runtime": audit["runtime"],
            "options": audit["options"],
            "quantizer": "MMCQ applied to scaled OKLCH coordinates",
            "output_semantics": "quantizer centroid; not guaranteed to occur in source pixels",
        },
    )


def _run_color_thief_subprocess(items: list[dict[str, str]]) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    runner = project_root / "scripts" / "benchmark_colorthief.mjs"
    with tempfile.TemporaryDirectory(prefix="hexbench-colorthief-") as directory:
        input_path = Path(directory) / "input.json"
        input_path.write_text(json.dumps({"items": items}), encoding="utf-8")
        completed = subprocess.run(
            ["node", str(runner), str(input_path)],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Color Thief v3 runner failed: {completed.stderr.strip()}")
        return json.loads(completed.stdout)


def _run_color_thief_v3(
    crops: list[CropRecord],
    synthetic_cases: list[dict[str, object]],
    workers: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    """Run the official Color Thief v3 package and normalize its records."""

    real_items = [
        {
            "id": record.crop_id,
            "kind": "real",
            "path": str(Path(record.path).resolve()),
            "jpeg_path": str(Path(record.jpeg_path).resolve()),
            "resize_path": str(Path(record.resize_path).resolve()),
        }
        for record in crops
    ]
    worker_count = max(1, min(workers, len(real_items)))
    chunk_size = math.ceil(len(real_items) / worker_count)
    chunks = [real_items[index : index + chunk_size] for index in range(0, len(real_items), chunk_size)]
    chunks[0].extend(
        {
            "id": str(case["case_id"]),
            "kind": "synthetic",
            "path": str(Path(str(case["path"])).resolve()),
        }
        for case in synthetic_cases
    )
    if worker_count == 1:
        raw_payloads = [_run_color_thief_subprocess(chunks[0])]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            raw_payloads = []
            for index, payload in enumerate(executor.map(_run_color_thief_subprocess, chunks), start=1):
                raw_payloads.append(payload)
                print(f"color_thief_v3: chunk {index}/{len(chunks)}", flush=True)

    raw_payload = raw_payloads[0]
    raw_records = [record for payload in raw_payloads for record in payload["records"]]

    audit = {
        "package": raw_payload["package"],
        "version": raw_payload["version"],
        "runtime": raw_payload["runtime"],
        "options": raw_payload["options"],
        "execution": "official Node package; Sharp decode and TypeScript MMCQ/OKLCH pipeline",
    }
    by_key = {(item["kind"], item["id"]): item for item in raw_records}
    records: list[dict[str, Any]] = []
    for record in crops:
        raw = by_key[("real", record.crop_id)]
        result = _color_thief_result(raw["base"], audit)
        jpeg_result = _color_thief_result(raw["jpeg"], audit)
        resize_result = _color_thief_result(raw["resize"], audit)
        image, metadata = open_image_srgb(record.path)
        base_pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
        coverage = palette_coverage(base_pixels, np.asarray(result.palette, dtype=np.uint8))
        jpeg_delta = float(delta_e_ok(np.asarray(result.primary), np.asarray(jpeg_result.primary)))
        resize_delta = float(delta_e_ok(np.asarray(result.primary), np.asarray(resize_result.primary)))
        decode_ms = float(raw["base"]["decode_ms"])
        method_ms = float(raw["base"]["method_ms"])
        elapsed_ms = float(raw["base"]["elapsed_ms"])
        pixels = image.width * image.height
        item = result.to_dict()
        item.update(
            {
                "crop_id": record.crop_id,
                "source_file": record.source_file,
                "batch": record.batch,
                "variant": record.variant,
                "asset_path": f"assets/crops/{Path(record.path).name}",
                "width": image.width,
                "height": image.height,
                "pixels": pixels,
                "decode_ms": round(decode_ms, 6),
                "method_ms": round(method_ms, 6),
                "elapsed_ms": round(elapsed_ms, 6),
                "method_mpix_s": round((pixels / 1e6) / max(method_ms / 1000.0, 1e-12), 6),
                "end_to_end_crops_s": round(1000.0 / max(elapsed_ms, 1e-12), 6),
                "palette_coverage_delta_e_ok": round(coverage, 6),
                "jpeg60_delta_e_ok": round(jpeg_delta, 6),
                "resize50_delta_e_ok": round(resize_delta, 6),
                "source_metadata": metadata,
            }
        )
        records.append(item)

    synthetic: dict[str, dict[str, Any]] = {}
    for case in synthetic_cases:
        raw = by_key[("synthetic", str(case["case_id"]))]
        item = _color_thief_result(raw["base"], audit).to_dict()
        expected_hex = case.get("expected_hex")
        if expected_hex:
            value = str(expected_hex).lstrip("#")
            expected_rgb = np.array(
                [int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)], dtype=np.uint8
            )
            item["expected_delta_e_ok"] = round(
                float(delta_e_ok(np.asarray(item["primary_rgb"]), expected_rgb)), 6
            )
        else:
            item["expected_delta_e_ok"] = None
        item["route_match"] = None
        synthetic[str(case["case_id"])] = item
    return records, synthetic, audit


def _summarize(method_name: str, records: list[dict[str, Any]], provisioned_crops_s: float) -> dict[str, Any]:
    elapsed = [float(record["elapsed_ms"]) for record in records]
    method_elapsed = [float(record["method_ms"]) for record in records]
    decode_elapsed = [float(record["decode_ms"]) for record in records]
    pixels = sum(int(record["pixels"]) for record in records)
    total_seconds = max(sum(elapsed) / 1000.0, 1e-12)
    crops_per_second = len(records) / total_seconds
    method_seconds = max(sum(method_elapsed) / 1000.0, 1e-12)
    method_mpix_s = (pixels / 1e6) / method_seconds
    worker_hours_1b = 1e9 / max(crops_per_second, 1e-12) / 3600.0
    processes = math.ceil(provisioned_crops_s / max(crops_per_second, 1e-12))
    route_counts = Counter(str(record["route"]) for record in records)
    jpeg_delta = [float(record["jpeg60_delta_e_ok"]) for record in records]
    resize_delta = [float(record["resize50_delta_e_ok"]) for record in records]
    coverage = [float(record["palette_coverage_delta_e_ok"]) for record in records]
    elapsed_mean = statistics.fmean(elapsed)
    elapsed_ci95 = 0.0 if len(elapsed) < 2 else 1.96 * statistics.stdev(elapsed) / math.sqrt(len(elapsed))
    verdict = {
        "adaptive_v1": "CONDITIONAL GO：fused batch kernel 后进入 1B pilot",
        "tencent_hsv": "不作默认；仅保留可解释质量基线",
        "pixelero_rgb_hist": "保留；两级直方图的快速 RGB 基线",
        "octree": "保留；原生快速量化基线",
        "color_thief_v3": "不作默认；默认 q10 与 centroid 语义",
        "pngquant_liq": "不作默认；质量向且有许可约束",
        "colorpipette_inspired": "不作默认；仅 palette 质量参考",
    }[method_name]
    risk = {
        "adaptive_v1": "高：3 天目标必须 fused batch kernel 与真实 shard 验证",
        "tencent_hsv": "高：容量、灰色、多峰和色相边界质量风险",
        "pixelero_rgb_hist": "中：RGB 轴切分与初始化会改变聚类边界",
        "octree": "中：RGB 树边界和叶均值不保证语义主色",
        "color_thief_v3": "中：每 10 像素采样、忽略近白与 JS/Sharp 运行栈",
        "pngquant_liq": "中高：全图 palette 优化成本与 GPL/商业许可选择",
        "colorpipette_inspired": "极高：分割开销与非观测色语义",
    }[method_name]
    return {
        "method": method_name,
        "label": METHOD_LABELS[method_name],
        "samples": len(records),
        "latency_ms": {
            "p50": round(_percentile(elapsed, 50), 4),
            "p95": round(_percentile(elapsed, 95), 4),
            "p99": round(_percentile(elapsed, 99), 4),
            "mean": round(elapsed_mean, 4),
            "mean_ci95": [round(elapsed_mean - elapsed_ci95, 4), round(elapsed_mean + elapsed_ci95, 4)],
            "decode_mean": round(statistics.fmean(decode_elapsed), 4),
            "method_mean": round(statistics.fmean(method_elapsed), 4),
        },
        "throughput": {
            "crops_s_single_process_e2e": round(crops_per_second, 3),
            "mpix_s_method_only": round(method_mpix_s, 3),
            "estimated_processes_target_1_5x": processes,
            "worker_hours_per_1b": round(worker_hours_1b, 2),
        },
        "quality_proxy": {
            "palette_coverage_mean_delta_e_ok": round(statistics.fmean(coverage), 4),
            "jpeg60_mean_delta_e_ok": round(statistics.fmean(jpeg_delta), 4),
            "jpeg60_p95_delta_e_ok": round(_percentile(jpeg_delta, 95), 4),
            "resize50_mean_delta_e_ok": round(statistics.fmean(resize_delta), 4),
            "resize50_p95_delta_e_ok": round(_percentile(resize_delta, 95), 4),
            "observed_output_rate": round(statistics.fmean(1.0 if record["observed"] else 0.0 for record in records), 4),
        },
        "route_counts": dict(sorted(route_counts.items())),
        "risk": risk,
        "verdict": verdict,
    }


def _synthetic_benchmark(
    cases: list[dict[str, object]],
    color_thief: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for case in cases:
        image, _ = open_image_srgb(str(case["path"]))
        expected_hex = case.get("expected_hex")
        expected_rgb = None
        if expected_hex:
            text = str(expected_hex).lstrip("#")
            expected_rgb = np.array([int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)], dtype=np.uint8)
        methods: dict[str, Any] = {}
        for method_name, method in METHODS.items():
            result = method(image, str(case["case_id"]))
            item = result.to_dict()
            item["expected_delta_e_ok"] = (
                round(float(delta_e_ok(np.asarray(result.primary), expected_rgb)), 6) if expected_rgb is not None else None
            )
            item["route_match"] = result.route == case.get("expected_route") if method_name == "adaptive_v1" else None
            methods[method_name] = item
        if color_thief is not None:
            methods["color_thief_v3"] = color_thief[str(case["case_id"])]
        output.append({**case, "methods": methods})
    return output


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _run_color_thief_checkpointed(
    crops: list[CropRecord],
    synthetic_cases: list[dict[str, object]],
    workers: int,
    checkpoint_path: Path,
    resume: bool,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    if resume and checkpoint_path.exists():
        cached = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        cached_ids = [str(record.get("crop_id")) for record in cached.get("records", [])]
        expected_ids = [record.crop_id for record in crops]
        if cached_ids == expected_ids:
            print(f"color_thief_v3: resumed {len(cached_ids)}/{len(crops)} records", flush=True)
            return cached["records"], cached["synthetic"], cached["audit"]
        raise RuntimeError(f"Checkpoint does not match this crop manifest: {checkpoint_path}")
    records, synthetic, audit = _run_color_thief_v3(crops, synthetic_cases, workers=workers)
    _write_json_atomic(checkpoint_path, {"records": records, "synthetic": synthetic, "audit": audit})
    return records, synthetic, audit


def _stratified_display_ids(crops: list[CropRecord], count: int) -> set[str]:
    count = max(1, min(count, len(crops)))
    indexes = np.rint(np.linspace(0, len(crops) - 1, count)).astype(np.int64)
    return {crops[int(index)].crop_id for index in indexes}


def run_benchmark(
    output_path: str | Path = "data/results/benchmark.json",
    site_data_path: str | Path = "site/data.js",
    source_dir: str | Path = "data/samples/s3_10k",
    source_manifest_path: str | Path = "data/sample_manifest_10k.json",
    crop_dir: str | Path = "data/benchmark_crops_10k",
    checkpoint_dir: str | Path = "data/results/checkpoints_10k",
    target_crops: int = 10_000,
    display_crops: int = 60,
    workers: int = 12,
    resume: bool = True,
) -> dict[str, Any]:
    if target_crops <= 0 or display_crops <= 0 or workers <= 0:
        raise ValueError("target_crops, display_crops and workers must be positive")

    source_manifest_file = Path(source_manifest_path)
    if not source_manifest_file.exists():
        raise RuntimeError(
            f"Missing {source_manifest_file}. Run scripts/download_s3_benchmark.py first."
        )
    crops = materialize_s3_crops(
        source_dir=source_dir,
        output_dir=crop_dir,
        source_manifest_path=source_manifest_file,
        target_crops=target_crops,
    )
    synthetic_cases = materialize_synthetic_cases()
    if len(crops) != target_crops:
        raise RuntimeError(
            f"Materialized {len(crops)} valid crops, but {target_crops} are required. "
            "Download additional source images or inspect corrupt files."
        )

    target_crops_s, provisioned_crops_s = capacity_target(
        TARGET_CROPS,
        TARGET_DAYS,
        CAPACITY_SAFETY_FACTOR,
    )
    checkpoint_root = Path(checkpoint_dir)
    display_ids = _stratified_display_ids(crops, display_crops)
    display_records: list[dict[str, Any]] = []
    summaries_by_method: dict[str, dict[str, Any]] = {}
    for method_name in METHODS:
        method_records = _run_python_method_checkpointed(
            crops,
            method_name,
            workers,
            checkpoint_root / f"{method_name}.jsonl",
            resume,
        )
        summaries_by_method[method_name] = _summarize(method_name, method_records, provisioned_crops_s)
        display_records.extend(record for record in method_records if record["crop_id"] in display_ids)

    color_thief_records, color_thief_synthetic, color_thief_audit = _run_color_thief_checkpointed(
        crops,
        synthetic_cases,
        workers,
        checkpoint_root / "color_thief_v3.json",
        resume,
    )
    summaries_by_method["color_thief_v3"] = _summarize(
        "color_thief_v3",
        color_thief_records,
        provisioned_crops_s,
    )
    display_records.extend(record for record in color_thief_records if record["crop_id"] in display_ids)
    summaries = [summaries_by_method[method_name] for method_name in BENCHMARK_METHOD_ORDER]

    decode_once_model = source_pipeline_capacity(
        TARGET_CROPS,
        SOURCE_IMAGES,
        TARGET_DAYS,
        AVERAGE_SOURCE_MPIX,
        AVERAGE_COMPRESSED_MB,
        TARGET_WORKER_GPIX_S,
        CAPACITY_SAFETY_FACTOR,
    )
    source_manifest = json.loads(source_manifest_file.read_text(encoding="utf-8"))
    source_public = {key: value for key, value in source_manifest.items() if key != "samples"}
    source_public["candidate_images"] = len(source_manifest.get("samples", []))
    source_public["benchmark_images"] = len({record.source_file for record in crops})
    source_public["benchmark_shards"] = len({record.batch for record in crops})
    original_audit_path = Path("data/results/colorpipette_original.json")
    original_audit = json.loads(original_audit_path.read_text(encoding="utf-8")) if original_audit_path.exists() else None
    original_projection = None
    if original_audit is not None:
        original_crops_s = 1.0 / max(float(original_audit["inference_seconds"]), 1e-12)
        original_projection = {
            "samples": 1,
            "crops_s_single_process_warm_model": round(original_crops_s, 6),
            "estimated_processes_target_1_5x": math.ceil(provisioned_crops_s / original_crops_s),
            "worker_hours_per_1b": round(1e9 / original_crops_s / 3600.0, 2),
            "comparison_warning": "One-image CPU audit; model load excluded from crops/s and not directly comparable to the 10,000-crop benchmark.",
        }
    crop_rates = {
        summary["method"]: float(summary["throughput"]["crops_s_single_process_e2e"]) for summary in summaries
    }
    if original_projection is not None:
        crop_rates["colorpipette_original"] = float(original_projection["crops_s_single_process_warm_model"])
    scale_scenarios = [build_capacity_scenario(days, crop_rates) for days in TARGET_DAY_SCENARIOS]
    payload: dict[str, Any] = {
        "schema_version": 4,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "source_images": len({record.source_file for record in crops}),
            "source_shards": len({record.batch for record in crops}),
            "real_crops": len(crops),
            "browsable_crops": len(crops),
            "embedded_visual_crops": len(display_ids),
            "method_observations": len(crops) * len(BENCHMARK_METHOD_ORDER),
            "crop_size": [320, 320],
            "synthetic_cases": len(synthetic_cases),
            "methods": len(BENCHMARK_METHOD_ORDER),
            "sampling_unit": "one deterministic crop from each distinct source image; top-left / center / bottom-right anchors balanced",
            "timing_scope": "per-crop decode + extraction; Python methods use Pillow, exact Color Thief uses its official Sharp loader; stability variants excluded",
            "execution": f"{workers} isolated worker processes; single-process-equivalent throughput is computed from summed per-crop elapsed time",
            "display_disclosure": f"All {len(crops):,} crops are searchable and viewable through an on-demand index; {len(display_ids)} records remain embedded for overview decoration.",
        },
        "hardware": {
            "cpu": _cpu_model(),
            "logical_cpus": os.cpu_count(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "gpu": "none visible in this Devbox",
        },
        "scale": {
            "crops": TARGET_CROPS,
            "days": TARGET_DAYS,
            "target_crops_s": round(target_crops_s, 3),
            "safety_factor": CAPACITY_SAFETY_FACTOR,
            "provisioned_crops_s": round(provisioned_crops_s, 3),
            "decode_once_model": decode_once_model,
            "scenarios": scale_scenarios,
            "projection_warning": "Three-day target. Single-process local extrapolation only; excludes S3 GET, shard packing and distributed batch efficiency.",
        },
        "source": source_public,
        "summaries": summaries,
        "records": display_records,
        "synthetic": _synthetic_benchmark(synthetic_cases, color_thief_synthetic),
        "external_audit": {
            "tencent": {
                "url": "https://cloud.tencent.com/developer/article/1119452",
                "published": "2018-05-10",
                "finding": "Only a five-step hue-quantization sketch is provided; bins, smoothing, achromatic handling and RGB recovery are unspecified.",
                "implementation_choice": "60 hue × 4 saturation × 4 value bins, circular 1-2-3-2-1 smoothing, four achromatic value buckets, observed-pixel recovery.",
            },
            "colorpipette": {
                "url": "https://github.com/Shuqi-67/ColorPipette",
                "finding": "Research palette generator, not crop HEX labeller: BASNet saliency + SpixelNet + Lab/LCh harmony and per-pixel Python aggregation.",
                "weights": {"basnet_mb": 333, "spixelnet_mb": 27},
                "environment": "README pins Python 3.6, PyTorch 1.10.0, CUDA 11 and Ubuntu 16.04; repository has no root license file.",
                "proxy_disclosure": "The benchmark uses a clearly named SLIC + deterministic contrast-saliency proxy; it is not claimed bit-identical to the original neural pipeline.",
                "original_cpu_audit": original_audit,
                "original_projection": original_projection,
            },
            "color_thief": {
                "url": "https://lokeshdhakar.com/projects/color-thief/",
                "repository": "https://github.com/lokesh/color-thief",
                "finding": "v3 defaults to quality=10 sampling and OKLCH coordinates, then applies its TypeScript MMCQ quantizer.",
                "license": "MIT",
                "exact_audit": color_thief_audit,
            },
            "pngquant": {
                "url": "https://github.com/kornelski/pngquant",
                "library_url": "https://pngquant.org/lib/",
                "finding": "pngquant is the PNG CLI; this benchmark calls its libimagequant palette/remap engine on decoded RGBA and excludes PNG encoding.",
                "implementation_choice": "imagequant-python 1.1.5 binding to libimagequant 2.15.1, top-3, quality 0..100, dithering disabled; not the current pngquant v3 CLI end-to-end path",
                "license": "libimagequant is GPL v3+ or commercial for non-GPL use",
            },
            "octree": {
                "url": "https://www.cg.tuwien.ac.at/research/publications/1988/purgathofer-1988-simple/",
                "finding": "The paper inserts colors into an octree and reduces deepest leaves to palette means with O(N) mapping.",
                "implementation_choice": "Pillow FASTOCTREE native adapter, top-3, no dithering; algorithm-family comparison, not line-for-line Pascal reproduction.",
            },
            "pixelero": {
                "url": "https://pixelero.wordpress.com/2014/11/12/just-saying-hi-or-color-quantization-using-histogram/",
                "finding": "The article clusters each RGB channel, forms at most 8×8×8 histogram bins, then clusters occupied bins to the target palette.",
                "implementation_choice": "8 weighted 1D clusters/channel, deterministic 3D-bin weighted k-means, top-3 output.",
            },
        },
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    output.write_text(serialized, encoding="utf-8")
    site_output = Path(site_data_path)
    site_output.parent.mkdir(parents=True, exist_ok=True)
    display_asset_root = site_output.parent / "assets" / "crops"
    display_asset_root.mkdir(parents=True, exist_ok=True)
    for crop in crops:
        if crop.crop_id in display_ids:
            destination = display_asset_root / Path(crop.path).name
            if Path(crop.path).resolve() != destination.resolve():
                shutil.copy2(crop.path, destination)
    browser_export = export_browser_samples(
        crops,
        checkpoint_dir=checkpoint_root,
        site_dir=site_output.parent,
        method_order=BENCHMARK_METHOD_ORDER,
        generated_at=payload["generated_at"],
    )
    payload["scope"]["browser_export"] = browser_export
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    output.write_text(serialized, encoding="utf-8")
    site_output.write_text("window.BENCHMARK_DATA = " + serialized + ";\n", encoding="utf-8")
    (site_output.parent / "benchmark.json").write_text(serialized, encoding="utf-8")
    return payload
