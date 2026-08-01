from __future__ import annotations

import hashlib
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from .color import as_rgb_tuple, pack_rgb24, srgb_to_oklab, unpack_rgb24
from .images import open_image_srgb
from .methods import (
    _hsv_weighted_bins,
    _pixelero_weighted_bins,
    _weighted_channel_centroids,
    _weighted_rgb_kmeans,
)


Box = tuple[int, int, int, int]

ROI_METHODS = {
    "pixelero": "pixelero_observed_pruned",
    "hsv_pixelero": "hsv_pixelero_observed_pruned",
    "octree": "octree_observed_pruned",
    "libimagequant": "pngquant_liq_speed6_observed_pruned",
}


@dataclass(frozen=True)
class BoxPalette:
    palette: tuple[tuple[int, int, int], ...]
    weights: tuple[float, ...]


@dataclass(frozen=True)
class BatchTiming:
    decode_ms: float
    sample_ms: float
    palette_ms: float
    total_ms: float
    palette_sizes: tuple[int, ...]


def deterministic_boxes(width: int, height: int, image_id: str, count: int = 20) -> list[Box]:
    """Generate repeatable object-like boxes spanning small to large areas."""

    digest = hashlib.blake2b(image_id.encode("utf-8"), digest_size=8).digest()
    rng = np.random.default_rng(int.from_bytes(digest, "little"))
    boxes: list[Box] = []
    for _ in range(count):
        area_fraction = float(np.exp(rng.uniform(np.log(0.015), np.log(0.32))))
        aspect = float(np.exp(rng.uniform(np.log(0.45), np.log(2.2))))
        box_width = int(round(math.sqrt(width * height * area_fraction * aspect)))
        box_height = int(round(math.sqrt(width * height * area_fraction / aspect)))
        box_width = int(np.clip(box_width, min(48, width), width))
        box_height = int(np.clip(box_height, min(48, height), height))
        left = int(rng.integers(0, max(1, width - box_width + 1)))
        top = int(rng.integers(0, max(1, height - box_height + 1)))
        boxes.append((left, top, left + box_width, top + box_height))
    return boxes


def _stratified_box_pixels(rgb: np.ndarray, box: Box, max_samples: int) -> np.ndarray:
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    pixel_count = width * height
    if pixel_count <= max_samples:
        return np.ascontiguousarray(rgb[top:bottom, left:right].reshape(-1, 3))
    columns = max(1, min(width, int(round(math.sqrt(max_samples * width / max(height, 1))))))
    rows = max(1, min(height, max_samples // columns))
    while rows * columns > max_samples:
        columns -= 1
    xs = np.linspace(left, right - 1, columns, dtype=np.int32)
    ys = np.linspace(top, bottom - 1, rows, dtype=np.int32)
    return np.ascontiguousarray(rgb[ys[:, None], xs[None, :]].reshape(-1, 3))


def _weighted_observed_kdtree(
    pixels: np.ndarray,
    centroids: np.ndarray,
    populations: np.ndarray,
    min_cluster_ratio: float,
) -> BoxPalette:
    total = max(float(np.sum(populations)), 1e-12)
    ratios = populations / total
    keep = np.flatnonzero(ratios >= min_cluster_ratio)
    if keep.size == 0:
        keep = np.asarray([int(np.argmax(populations))])
    targets = np.asarray(centroids[keep], dtype=np.float64)

    packed = pack_rgb24(pixels)
    unique, counts = np.unique(packed, return_counts=True)
    candidates = unpack_rgb24(unique)
    candidate_lab = srgb_to_oklab(candidates)
    tree = cKDTree(candidate_lab, compact_nodes=True, balanced_tree=True)
    target_lab = srgb_to_oklab(np.clip(np.rint(targets), 0, 255).astype(np.uint8))
    neighbor_count = min(64, len(candidates))
    distances, indexes = tree.query(target_lab, k=neighbor_count, workers=1)
    distances = np.atleast_2d(distances)
    indexes = np.atleast_2d(indexes)
    if len(keep) == 1:
        distances = distances.reshape(1, -1)
        indexes = indexes.reshape(1, -1)

    selected: set[int] = set()
    palette: list[tuple[int, int, int]] = []
    for row_distances, row_indexes in zip(distances, indexes):
        local_counts = counts[row_indexes].astype(np.float64)
        frequency = 0.25 + 0.75 * np.sqrt(local_counts / max(float(np.max(counts)), 1.0))
        scores = row_distances / frequency
        for order in np.argsort(scores):
            candidate_index = int(row_indexes[int(order)])
            if candidate_index not in selected:
                selected.add(candidate_index)
                palette.append(as_rgb_tuple(candidates[candidate_index]))
                break
    return BoxPalette(
        palette=tuple(palette),
        weights=tuple(float(value) for value in ratios[keep[: len(palette)]]),
    )


def _pixelero_bins_with_shared_centers(
    pixels: np.ndarray,
    weights: np.ndarray,
    channel_centers: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    indexes = [
        np.argmin(np.abs(pixels[:, channel, None] - channel_centers[channel][None, :]), axis=1)
        for channel in range(3)
    ]
    green_bins = len(channel_centers[1])
    blue_bins = len(channel_centers[2])
    keys = (indexes[0] * green_bins + indexes[1]) * blue_bins + indexes[2]
    bin_count = len(channel_centers[0]) * green_bins * blue_bins
    bin_weights = np.bincount(keys, weights=weights, minlength=bin_count).astype(np.float64)
    occupied = np.flatnonzero(bin_weights > 0)
    sums = np.stack(
        [np.bincount(keys, weights=weights * pixels[:, channel], minlength=bin_count) for channel in range(3)],
        axis=1,
    )
    return sums[occupied] / bin_weights[occupied, None], bin_weights[occupied]


def _libimagequant_indexed(
    pixels: np.ndarray,
    width: int,
    height: int,
    max_colors: int,
    speed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run libimagequant without PNG encoding/dithering."""
    try:
        import imagequant
    except ImportError as error:
        raise RuntimeError("libimagequant ROI methods require imagequant>=1.1.5") from error

    lib, ffi = imagequant.lib, imagequant.ffi
    rgba = np.concatenate(
        (pixels, np.full((len(pixels), 1), 255, dtype=np.uint8)), axis=1
    )
    raw = rgba.tobytes()
    attribute = lib.liq_attr_create()
    image = ffi.NULL
    result_pointer = ffi.new("liq_result**")
    remapped = ffi.NULL
    try:
        attribute.max_colors = max_colors
        if lib.liq_set_quality(attribute, 0, 100) != lib.LIQ_OK:
            raise RuntimeError("libimagequant rejected quality range")
        if lib.liq_set_speed(attribute, speed) != lib.LIQ_OK:
            raise RuntimeError(f"libimagequant rejected speed {speed}")
        image = lib.liq_image_create_rgba(attribute, raw, width, height, 0)
        if image == ffi.NULL:
            raise RuntimeError("libimagequant could not create image")
        code = lib.liq_image_quantize(image, attribute, result_pointer)
        if code != lib.LIQ_OK:
            raise RuntimeError(f"libimagequant quantization failed with code {code}")
        result = result_pointer[0]
        lib.liq_set_dithering_level(result, 0.0)
        pixel_count = width * height
        remapped = ffi.new("unsigned char[]", pixel_count)
        code = lib.liq_write_remapped_image(result, image, remapped, pixel_count)
        if code != lib.LIQ_OK:
            raise RuntimeError(f"libimagequant remapping failed with code {code}")
        indexed = np.frombuffer(ffi.buffer(remapped, pixel_count)[:], dtype=np.uint8).copy()
        palette = lib.liq_get_palette(result)
        colors = np.asarray(
            [[palette.entries[index].r, palette.entries[index].g, palette.entries[index].b]
             for index in range(palette.count)],
            dtype=np.float64,
        )
        return indexed.reshape(height, width), colors
    finally:
        if result_pointer[0] != ffi.NULL:
            lib.liq_result_destroy(result_pointer[0])
        if image != ffi.NULL:
            lib.liq_image_destroy(image)
        lib.liq_attr_destroy(attribute)
        if remapped != ffi.NULL:
            ffi.release(remapped)


def _libimagequant_bins(pixels: np.ndarray, speed: int) -> tuple[np.ndarray, np.ndarray]:
    """Run libimagequant on a sample and return palette mass."""

    indexed, colors = _libimagequant_indexed(pixels, len(pixels), 1, 5, speed)
    counts = np.bincount(indexed.reshape(-1), minlength=len(colors)).astype(np.float64)
    occupied = np.flatnonzero(counts > 0)
    return colors[occupied], counts[occupied]


def _box_palette(
    pixels: np.ndarray,
    method: str,
    min_cluster_ratio: float,
    shared_centers: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> BoxPalette:
    weights = np.ones(len(pixels), dtype=np.float64)
    if method == "pixelero_observed_pruned":
        bins, bin_weights = (
            _pixelero_bins_with_shared_centers(pixels, weights, shared_centers)
            if shared_centers is not None
            else _pixelero_weighted_bins(pixels, weights)
        )
    elif method == "hsv_pixelero_observed_pruned":
        bins, bin_weights = _hsv_weighted_bins(pixels, weights)
    elif method == "octree_observed_pruned":
        rgba = np.concatenate(
            (pixels, np.full((len(pixels), 1), 255, dtype=np.uint8)),
            axis=1,
        )
        pseudo_image = Image.fromarray(rgba.reshape(1, len(rgba), 4), mode="RGBA")
        quantized = pseudo_image.quantize(
            colors=5,
            method=Image.Quantize.FASTOCTREE,
            dither=Image.Dither.NONE,
        )
        raw_palette = np.asarray(quantized.getpalette("RGBA"), dtype=np.uint8).reshape(-1, 4)
        counts = np.zeros(len(raw_palette), dtype=np.float64)
        for population, index in quantized.getcolors(maxcolors=256) or []:
            counts[int(index)] = float(population)
        occupied = np.flatnonzero(counts > 0)
        bins, bin_weights = raw_palette[occupied, :3].astype(np.float64), counts[occupied]
    elif method.startswith("pngquant_liq_speed") and method.endswith("_observed_pruned"):
        speed_text = method.removeprefix("pngquant_liq_speed").removesuffix("_observed_pruned")
        speed = int(speed_text)
        if not 1 <= speed <= 10:
            raise ValueError(f"Invalid libimagequant speed: {speed}")
        bins, bin_weights = _libimagequant_bins(pixels, speed)
    else:
        raise ValueError(f"Unknown ROI method: {method}")
    if method == "octree_observed_pruned" or method.startswith("pngquant_liq_speed"):
        order = np.argsort(bin_weights)[::-1]
        centroids, populations = bins[order], bin_weights[order]
    else:
        centroids, populations = _weighted_rgb_kmeans(bins, bin_weights, 5)
    return _weighted_observed_kdtree(pixels, centroids, populations, min_cluster_ratio)


def process_image_boxes(
    path: str | Path,
    method: str,
    box_count: int = 20,
    max_samples: int = 4096,
    min_cluster_ratio: float = 0.01,
    box_workers: int = 1,
) -> tuple[list[BoxPalette], BatchTiming]:
    started = time.perf_counter_ns()
    image, _ = open_image_srgb(path)
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    decoded = time.perf_counter_ns()
    boxes = deterministic_boxes(image.width, image.height, Path(path).stem, box_count)
    return _process_decoded_boxes(
        rgb, boxes, method, max_samples, min_cluster_ratio, box_workers, started, decoded
    )


def process_image_with_boxes(
    path: str | Path,
    boxes: list[Box] | tuple[Box, ...],
    method: str,
    max_samples: int = 4096,
    min_cluster_ratio: float = 0.01,
    box_workers: int = 1,
) -> tuple[list[BoxPalette], BatchTiming]:
    """Extract palettes from caller-supplied pixel-space XYXY boxes."""

    started = time.perf_counter_ns()
    image, _ = open_image_srgb(path)
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    decoded = time.perf_counter_ns()
    checked: list[Box] = []
    for left, top, right, bottom in boxes:
        box = (
            max(0, min(int(left), image.width)),
            max(0, min(int(top), image.height)),
            max(0, min(int(right), image.width)),
            max(0, min(int(bottom), image.height)),
        )
        if box[2] > box[0] and box[3] > box[1]:
            checked.append(box)
    return _process_decoded_boxes(
        rgb, checked, method, max_samples, min_cluster_ratio, box_workers, started, decoded
    )


def _process_decoded_boxes(
    rgb: np.ndarray,
    boxes: list[Box],
    method: str,
    max_samples: int,
    min_cluster_ratio: float,
    box_workers: int,
    started: int,
    decoded: int,
) -> tuple[list[BoxPalette], BatchTiming]:
    samples = [_stratified_box_pixels(rgb, box, max_samples) for box in boxes]
    shared_centers = None
    if method == "pixelero_observed_pruned":
        pooled = np.concatenate(samples, axis=0)
        pooled_weights = np.ones(len(pooled), dtype=np.float64)
        shared_centers = tuple(
            _weighted_channel_centroids(pooled[:, channel], pooled_weights, 8)
            for channel in range(3)
        )
    sampled = time.perf_counter_ns()
    if box_workers <= 1:
        palettes = [_box_palette(pixels, method, min_cluster_ratio, shared_centers) for pixels in samples]
    else:
        with ThreadPoolExecutor(max_workers=box_workers) as executor:
            palettes = list(executor.map(
                lambda pixels: _box_palette(pixels, method, min_cluster_ratio, shared_centers),
                samples,
            ))
    finished = time.perf_counter_ns()
    return palettes, BatchTiming(
        decode_ms=(decoded - started) / 1e6,
        sample_ms=(sampled - decoded) / 1e6,
        palette_ms=(finished - sampled) / 1e6,
        total_ms=(finished - started) / 1e6,
        palette_sizes=tuple(len(item.palette) for item in palettes),
    )
