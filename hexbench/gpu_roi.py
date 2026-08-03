from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _stratified_sample_kernel(
    images,
    metadata,
    samples,
    image_height: tl.constexpr,
    image_width: tl.constexpr,
    sample_count: tl.constexpr,
    block: tl.constexpr,
):
    box = tl.program_id(0)
    offsets = tl.arange(0, block)
    meta = box * 7
    image_index = tl.load(metadata + meta)
    left = tl.load(metadata + meta + 1)
    top = tl.load(metadata + meta + 2)
    width = tl.load(metadata + meta + 3)
    height = tl.load(metadata + meta + 4)
    columns = tl.load(metadata + meta + 5)
    rows = tl.load(metadata + meta + 6)
    length = columns * rows
    valid = offsets < length
    row = offsets // columns
    column = offsets - row * columns
    x = left + column * (width - 1) // tl.maximum(columns - 1, 1)
    y = top + row * (height - 1) // tl.maximum(rows - 1, 1)
    source = ((image_index * image_height + y) * image_width + x) * 3
    target = (box * sample_count + offsets) * 3
    tl.store(samples + target, tl.load(images + source, mask=valid, other=0), mask=valid)
    tl.store(samples + target + 1, tl.load(images + source + 1, mask=valid, other=0), mask=valid)
    tl.store(samples + target + 2, tl.load(images + source + 2, mask=valid, other=0), mask=valid)


@triton.jit
def _rgb_to_oklab_kernel(rgb, lengths, lab, sample_count: tl.constexpr, block: tl.constexpr):
    box = tl.program_id(0)
    offsets = tl.arange(0, block)
    mask = offsets < tl.load(lengths + box)
    base = (box * sample_count + offsets) * 3
    red = tl.load(rgb + base, mask=mask, other=0).to(tl.float32) / 255.0
    green = tl.load(rgb + base + 1, mask=mask, other=0).to(tl.float32) / 255.0
    blue = tl.load(rgb + base + 2, mask=mask, other=0).to(tl.float32) / 255.0
    red = tl.where(red <= 0.04045, red / 12.92, tl.exp(tl.log(tl.maximum((red + 0.055) / 1.055, 1e-20)) * 2.4))
    green = tl.where(green <= 0.04045, green / 12.92, tl.exp(tl.log(tl.maximum((green + 0.055) / 1.055, 1e-20)) * 2.4))
    blue = tl.where(blue <= 0.04045, blue / 12.92, tl.exp(tl.log(tl.maximum((blue + 0.055) / 1.055, 1e-20)) * 2.4))
    light = 0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue
    medium = 0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue
    short = 0.0883024619 * red + 0.2817188374 * green + 0.6299787005 * blue
    light = tl.where(light > 0, tl.exp(tl.log(tl.maximum(light, 1e-20)) / 3.0), 0.0)
    medium = tl.where(medium > 0, tl.exp(tl.log(tl.maximum(medium, 1e-20)) / 3.0), 0.0)
    short = tl.where(short > 0, tl.exp(tl.log(tl.maximum(short, 1e-20)) / 3.0), 0.0)
    tl.store(lab + base, 0.2104542553 * light + 0.7936177850 * medium - 0.0040720468 * short, mask=mask)
    tl.store(lab + base + 1, 1.9779984951 * light - 2.4285922050 * medium + 0.4505937099 * short, mask=mask)
    tl.store(lab + base + 2, 0.0259040371 * light + 0.7827717662 * medium - 0.8086757660 * short, mask=mask)


@triton.jit
def _assign_kernel(lab, lengths, centroids, labels, sample_count: tl.constexpr, block: tl.constexpr):
    box = tl.program_id(0)
    offsets = tl.arange(0, block)
    mask = offsets < tl.load(lengths + box)
    base = (box * sample_count + offsets) * 3
    point_l = tl.load(lab + base, mask=mask, other=0.0)
    point_a = tl.load(lab + base + 1, mask=mask, other=0.0)
    point_b = tl.load(lab + base + 2, mask=mask, other=0.0)
    best_distance = tl.full((block,), float("inf"), tl.float32)
    best_cluster = tl.zeros((block,), tl.int32)
    center_base = box * 15
    for cluster in range(5):
        center_l = tl.load(centroids + center_base + cluster * 3)
        center_a = tl.load(centroids + center_base + cluster * 3 + 1)
        center_b = tl.load(centroids + center_base + cluster * 3 + 2)
        delta_l = point_l - center_l
        delta_a = point_a - center_a
        delta_b = point_b - center_b
        distance = delta_l * delta_l + delta_a * delta_a + delta_b * delta_b
        better = distance < best_distance
        best_distance = tl.where(better, distance, best_distance)
        best_cluster = tl.where(better, cluster, best_cluster)
    tl.store(labels + box * sample_count + offsets, best_cluster, mask=mask)


@triton.jit
def _reduce_kernel(lab, lengths, labels, old_centroids, new_centroids, counts, sample_count: tl.constexpr, block: tl.constexpr):
    box = tl.program_id(0)
    cluster = tl.program_id(1)
    offsets = tl.arange(0, block)
    valid = offsets < tl.load(lengths + box)
    assigned = tl.load(labels + box * sample_count + offsets, mask=valid, other=-1) == cluster
    mask = valid & assigned
    base = (box * sample_count + offsets) * 3
    count = tl.sum(mask.to(tl.float32), axis=0)
    sum_l = tl.sum(tl.load(lab + base, mask=mask, other=0.0), axis=0)
    sum_a = tl.sum(tl.load(lab + base + 1, mask=mask, other=0.0), axis=0)
    sum_b = tl.sum(tl.load(lab + base + 2, mask=mask, other=0.0), axis=0)
    center = (box * 5 + cluster) * 3
    denominator = tl.maximum(count, 1.0)
    tl.store(new_centroids + center, tl.where(count > 0, sum_l / denominator, tl.load(old_centroids + center)))
    tl.store(new_centroids + center + 1, tl.where(count > 0, sum_a / denominator, tl.load(old_centroids + center + 1)))
    tl.store(new_centroids + center + 2, tl.where(count > 0, sum_b / denominator, tl.load(old_centroids + center + 2)))
    tl.store(counts + box * 5 + cluster, count)


@triton.jit
def _observed_kernel(rgb, lab, lengths, centroids, palette, sample_count: tl.constexpr, block: tl.constexpr):
    box = tl.program_id(0)
    cluster = tl.program_id(1)
    offsets = tl.arange(0, block)
    valid = offsets < tl.load(lengths + box)
    base = (box * sample_count + offsets) * 3
    center = (box * 5 + cluster) * 3
    delta_l = tl.load(lab + base, mask=valid, other=0.0) - tl.load(centroids + center)
    delta_a = tl.load(lab + base + 1, mask=valid, other=0.0) - tl.load(centroids + center + 1)
    delta_b = tl.load(lab + base + 2, mask=valid, other=0.0) - tl.load(centroids + center + 2)
    distance = delta_l * delta_l + delta_a * delta_a + delta_b * delta_b
    winner = tl.argmin(tl.where(valid, distance, float("inf")), axis=0)
    source = (box * sample_count + winner) * 3
    tl.store(palette + center, tl.load(rgb + source))
    tl.store(palette + center + 1, tl.load(rgb + source + 1))
    tl.store(palette + center + 2, tl.load(rgb + source + 2))


@dataclass(frozen=True)
class GpuBatchResult:
    palettes: list[tuple[tuple[int, int, int], ...]]
    weights: list[tuple[float, ...]]


def gpu_stratified_box_samples(
    images: torch.Tensor,
    image_indexes: np.ndarray,
    boxes: np.ndarray,
    *,
    max_samples: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather deterministic stratified bbox samples directly from CUDA images."""
    gpu_metadata = build_stratified_box_metadata(
        image_indexes, boxes, max_samples=max_samples, device=images.device
    )
    return gpu_stratified_box_samples_from_metadata(
        images, gpu_metadata, max_samples=max_samples
    )


def build_stratified_box_metadata(
    image_indexes: np.ndarray,
    boxes: np.ndarray,
    *,
    max_samples: int = 2048,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Build the compact metadata tensor reused by GPU bbox sampling."""
    if max_samples & (max_samples - 1):
        raise ValueError("max_samples must be a power of two")
    boxes = np.asarray(boxes, dtype=np.int32)
    image_indexes = np.asarray(image_indexes, dtype=np.int32)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or len(boxes) != len(image_indexes):
        raise ValueError("boxes must be [box, xyxy] with one image index per box")
    metadata = np.empty((len(boxes), 7), dtype=np.int32)
    for index, (image_index, box) in enumerate(zip(image_indexes, boxes)):
        left, top, right, bottom = map(int, box)
        width, height = right - left, bottom - top
        if width * height <= max_samples:
            columns, rows = width, height
        else:
            columns = max(1, min(width, int(round(math.sqrt(max_samples * width / max(height, 1))))))
            rows = max(1, min(height, max_samples // columns))
            while rows * columns > max_samples:
                columns -= 1
        metadata[index] = image_index, left, top, width, height, columns, rows
    return torch.as_tensor(metadata, device=device)


def gpu_stratified_box_samples_from_metadata(
    images: torch.Tensor,
    gpu_metadata: torch.Tensor,
    *,
    max_samples: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather bbox samples using precomputed CUDA metadata."""
    if images.ndim != 4 or images.shape[3] != 3 or images.dtype != torch.uint8 or not images.is_cuda:
        raise ValueError("images must be CUDA uint8 [image, height, width, rgb]")
    if gpu_metadata.ndim != 2 or gpu_metadata.shape[1] != 7 or not gpu_metadata.is_cuda:
        raise ValueError("gpu_metadata must be CUDA int32 [box, 7]")
    lengths = gpu_metadata[:, 5] * gpu_metadata[:, 6]
    samples = torch.empty((len(gpu_metadata), max_samples, 3), dtype=torch.uint8, device=images.device)
    _stratified_sample_kernel[(len(gpu_metadata),)](
        images,
        gpu_metadata,
        samples,
        image_height=images.shape[1],
        image_width=images.shape[2],
        sample_count=max_samples,
        block=max_samples,
        num_warps=8,
    )
    return samples, lengths


def gpu_kmeans_observed_device(
    rgb: torch.Tensor,
    gpu_lengths: torch.Tensor,
    *,
    iterations: int = 8,
    min_cluster_ratio: float = 0.01,
) -> GpuBatchResult:
    """Run the palette pipeline from an existing CUDA sample tensor."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != torch.uint8 or not rgb.is_cuda:
        raise ValueError("rgb must be CUDA uint8 [box, sample, rgb]")
    if gpu_lengths.ndim != 1 or len(gpu_lengths) != len(rgb) or not gpu_lengths.is_cuda:
        raise ValueError("gpu_lengths must have one CUDA value per box")
    box_count, sample_count, _ = rgb.shape
    if sample_count > 65536 or sample_count & (sample_count - 1):
        raise ValueError("sample dimension must be a power of two no larger than 65536")
    lab = torch.empty((box_count, sample_count, 3), dtype=torch.float32, device="cuda")
    _rgb_to_oklab_kernel[(box_count,)](rgb, gpu_lengths, lab, sample_count=sample_count, block=sample_count, num_warps=8)
    positions = torch.arange(sample_count, device="cuda")[None, :]
    valid = positions < gpu_lengths[:, None]
    first = (lab * valid[:, :, None]).sum(dim=1) / gpu_lengths.to(torch.float32)[:, None]
    centroids = torch.empty((box_count, 5, 3), dtype=torch.float32, device="cuda")
    centroids[:, 0] = first
    closest = ((lab - first[:, None, :]) ** 2).sum(dim=2).masked_fill(~valid, -1.0)
    box_indexes = torch.arange(box_count, device="cuda")
    for cluster in range(1, 5):
        winner = torch.argmax(closest, dim=1)
        selected = lab[box_indexes, winner]
        centroids[:, cluster] = selected
        distance = ((lab - selected[:, None, :]) ** 2).sum(dim=2).masked_fill(~valid, -1.0)
        closest = torch.minimum(closest, distance)
    labels = torch.empty((box_count, sample_count), dtype=torch.int32, device="cuda")
    counts = torch.empty((box_count, 5), dtype=torch.float32, device="cuda")
    updated = torch.empty_like(centroids)
    for _ in range(iterations):
        _assign_kernel[(box_count,)](lab, gpu_lengths, centroids, labels, sample_count=sample_count, block=sample_count, num_warps=8)
        _reduce_kernel[(box_count, 5)](lab, gpu_lengths, labels, centroids, updated, counts, sample_count=sample_count, block=sample_count, num_warps=8)
        centroids, updated = updated, centroids
    _assign_kernel[(box_count,)](lab, gpu_lengths, centroids, labels, sample_count=sample_count, block=sample_count, num_warps=8)
    _reduce_kernel[(box_count, 5)](lab, gpu_lengths, labels, centroids, updated, counts, sample_count=sample_count, block=sample_count, num_warps=8)
    centroids = updated
    palette = torch.empty((box_count, 5, 3), dtype=torch.uint8, device="cuda")
    _observed_kernel[(box_count, 5)](rgb, lab, gpu_lengths, centroids, palette, sample_count=sample_count, block=sample_count, num_warps=8)
    order = torch.argsort(counts, dim=1, descending=True, stable=True)
    palette = torch.gather(palette, 1, order[:, :, None].expand(-1, -1, 3)).cpu().numpy()
    ratios = torch.gather(counts, 1, order).div(gpu_lengths[:, None]).cpu().numpy()
    output_palettes = []
    output_weights = []
    for colors, box_ratios in zip(palette, ratios):
        seen = set()
        selected_colors = []
        selected_weights = []
        for color, ratio in zip(colors, box_ratios):
            value = tuple(int(channel) for channel in color)
            if float(ratio) >= min_cluster_ratio and value not in seen:
                seen.add(value)
                selected_colors.append(value)
                selected_weights.append(float(ratio))
        if not selected_colors:
            selected_colors.append(tuple(int(channel) for channel in colors[0]))
            selected_weights.append(float(box_ratios[0]))
        output_palettes.append(tuple(selected_colors))
        output_weights.append(tuple(selected_weights))
    return GpuBatchResult(output_palettes, output_weights)


def gpu_kmeans_observed(
    samples: np.ndarray,
    lengths: np.ndarray,
    *,
    iterations: int = 8,
    min_cluster_ratio: float = 0.01,
) -> GpuBatchResult:
    """Copy CPU samples to CUDA and run deterministic observed Oklab k-means."""
    if samples.ndim != 3 or samples.shape[2] != 3 or samples.dtype != np.uint8:
        raise ValueError("samples must be uint8 [box, sample, rgb]")
    if len(lengths) != len(samples) or np.any(lengths <= 0) or np.any(lengths > samples.shape[1]):
        raise ValueError("invalid sample lengths")
    rgb = torch.as_tensor(samples, device="cuda")
    gpu_lengths = torch.as_tensor(lengths.astype(np.int32, copy=False), device="cuda")
    return gpu_kmeans_observed_device(
        rgb,
        gpu_lengths,
        iterations=iterations,
        min_cluster_ratio=min_cluster_ratio,
    )
