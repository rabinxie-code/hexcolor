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
def _assign_kernel(
    lab, lengths, centroids, labels,
    sample_count: tl.constexpr, cluster_count: tl.constexpr, block: tl.constexpr,
):
    box = tl.program_id(0)
    offsets = tl.arange(0, block)
    mask = offsets < tl.load(lengths + box)
    base = (box * sample_count + offsets) * 3
    point_l = tl.load(lab + base, mask=mask, other=0.0)
    point_a = tl.load(lab + base + 1, mask=mask, other=0.0)
    point_b = tl.load(lab + base + 2, mask=mask, other=0.0)
    best_distance = tl.full((block,), float("inf"), tl.float32)
    best_cluster = tl.zeros((block,), tl.int32)
    center_base = box * cluster_count * 3
    for cluster in range(cluster_count):
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
def _reduce_kernel(
    lab, lengths, labels, old_centroids, new_centroids, counts,
    sample_count: tl.constexpr, cluster_count: tl.constexpr, block: tl.constexpr,
):
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
    center = (box * cluster_count + cluster) * 3
    denominator = tl.maximum(count, 1.0)
    tl.store(new_centroids + center, tl.where(count > 0, sum_l / denominator, tl.load(old_centroids + center)))
    tl.store(new_centroids + center + 1, tl.where(count > 0, sum_a / denominator, tl.load(old_centroids + center + 1)))
    tl.store(new_centroids + center + 2, tl.where(count > 0, sum_b / denominator, tl.load(old_centroids + center + 2)))
    tl.store(counts + box * cluster_count + cluster, count)


@triton.jit
def _weighted_reduce_kernel(
    points, lengths, weights, labels, old_centroids, new_centroids, counts,
    sample_count: tl.constexpr, cluster_count: tl.constexpr, block: tl.constexpr,
):
    box = tl.program_id(0)
    cluster = tl.program_id(1)
    offsets = tl.arange(0, block)
    valid = offsets < tl.load(lengths + box)
    assigned = tl.load(
        labels + box * sample_count + offsets, mask=valid, other=-1
    ) == cluster
    mask = valid & assigned
    point_base = (box * sample_count + offsets) * 3
    point_weights = tl.load(
        weights + box * sample_count + offsets, mask=mask, other=0.0
    ).to(tl.float32)
    total = tl.sum(point_weights, axis=0)
    sum_x = tl.sum(
        tl.load(points + point_base, mask=mask, other=0.0) * point_weights,
        axis=0,
    )
    sum_y = tl.sum(
        tl.load(points + point_base + 1, mask=mask, other=0.0) * point_weights,
        axis=0,
    )
    sum_z = tl.sum(
        tl.load(points + point_base + 2, mask=mask, other=0.0) * point_weights,
        axis=0,
    )
    center = (box * cluster_count + cluster) * 3
    denominator = tl.maximum(total, 1.0)
    tl.store(
        new_centroids + center,
        tl.where(total > 0, sum_x / denominator, tl.load(old_centroids + center)),
    )
    tl.store(
        new_centroids + center + 1,
        tl.where(
            total > 0,
            sum_y / denominator,
            tl.load(old_centroids + center + 1),
        ),
    )
    tl.store(
        new_centroids + center + 2,
        tl.where(
            total > 0,
            sum_z / denominator,
            tl.load(old_centroids + center + 2),
        ),
    )
    tl.store(counts + box * cluster_count + cluster, total)


@triton.jit
def _observed_kernel(
    rgb, lab, lengths, centroids, palette,
    sample_count: tl.constexpr, cluster_count: tl.constexpr, block: tl.constexpr,
):
    box = tl.program_id(0)
    cluster = tl.program_id(1)
    offsets = tl.arange(0, block)
    valid = offsets < tl.load(lengths + box)
    base = (box * sample_count + offsets) * 3
    center = (box * cluster_count + cluster) * 3
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


def _unique_rgb_candidates(
    rgb: torch.Tensor, lengths: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return compact exact RGB colors, raw counts, and unique lengths."""
    box_count, sample_count, _ = rgb.shape
    positions = torch.arange(sample_count, device=rgb.device)[None, :]
    valid = positions < lengths[:, None]
    packed = (
        (rgb[:, :, 0].to(torch.int32) << 16)
        | (rgb[:, :, 1].to(torch.int32) << 8)
        | rgb[:, :, 2].to(torch.int32)
    ).masked_fill(~valid, 1 << 24)
    packed, _ = torch.sort(packed, dim=1)
    sorted_valid = packed < (1 << 24)
    starts = sorted_valid & torch.cat(
        (
            torch.ones((box_count, 1), dtype=torch.bool, device=rgb.device),
            packed[:, 1:] != packed[:, :-1],
        ),
        dim=1,
    )
    ranks = torch.cumsum(starts.to(torch.int64), dim=1).sub_(1).clamp_min_(0)
    candidate_lengths = starts.sum(dim=1)
    candidate_counts = torch.zeros(
        (box_count, sample_count), dtype=torch.int32, device=rgb.device
    )
    candidate_counts.scatter_add_(1, ranks, sorted_valid.to(torch.int32))
    candidates = torch.full(
        (box_count, sample_count), 1 << 24, dtype=torch.int32, device=rgb.device
    )
    candidates.scatter_reduce_(1, ranks, packed, reduce="amin", include_self=True)
    candidate_rgb = torch.stack(
        (
            (candidates >> 16) & 255,
            (candidates >> 8) & 255,
            candidates & 255,
        ),
        dim=2,
    ).to(torch.uint8)
    return candidate_rgb, candidate_counts, candidate_lengths


def _frequency_weighted_observed(
    rgb: torch.Tensor,
    lengths: torch.Tensor,
    centroids: torch.Tensor,
    *,
    nearest_candidates: int = 64,
) -> torch.Tensor:
    """Map centroids to frequent observed colors without leaving the GPU.

    Exact RGB values are deduplicated per bbox. For every centroid, only its
    nearest colors in Oklab are considered, then frequent colors receive the
    same soft preference used by the CPU pipeline. Previously selected colors
    are skipped so later clusters can refill instead of disappearing.
    """
    box_count, sample_count, _ = rgb.shape
    positions = torch.arange(sample_count, device=rgb.device)[None, :]
    candidate_rgb, candidate_counts, candidate_lengths = _unique_rgb_candidates(
        rgb, lengths
    )
    candidate_lab = torch.empty(
        (box_count, sample_count, 3), dtype=torch.float32, device=rgb.device
    )
    _rgb_to_oklab_kernel[(box_count,)](
        candidate_rgb,
        candidate_lengths,
        candidate_lab,
        sample_count=sample_count,
        block=sample_count,
        num_warps=8,
    )

    candidate_valid = positions < candidate_lengths[:, None]
    squared_distance = (
        (centroids[:, :, None, :] - candidate_lab[:, None, :, :]) ** 2
    ).sum(dim=3)
    squared_distance.masked_fill_(~candidate_valid[:, None, :], float("inf"))
    nearest_count = min(nearest_candidates, sample_count)
    nearest_distance, nearest_indexes = torch.topk(
        squared_distance, nearest_count, dim=2, largest=False, sorted=True
    )
    flat_indexes = nearest_indexes.reshape(box_count, -1)
    cluster_count = centroids.shape[1]
    nearest_frequency = torch.gather(candidate_counts, 1, flat_indexes).reshape(
        box_count, cluster_count, nearest_count
    )
    max_frequency = candidate_counts.amax(dim=1).to(torch.float32)
    frequency_factor = 0.25 + 0.75 * torch.sqrt(
        nearest_frequency.to(torch.float32) / max_frequency[:, None, None]
    )
    scores = torch.sqrt(nearest_distance) / frequency_factor

    chosen: list[torch.Tensor] = []
    box_indexes = torch.arange(box_count, device=rgb.device)
    for cluster in range(cluster_count):
        cluster_scores = scores[:, cluster]
        for earlier in chosen:
            cluster_scores = cluster_scores.masked_fill(
                nearest_indexes[:, cluster] == earlier[:, None], float("inf")
            )
        winner = torch.argmin(cluster_scores, dim=1)
        selected = nearest_indexes[box_indexes, cluster, winner]
        chosen.append(selected)
    chosen_indexes = torch.stack(chosen, dim=1)
    return torch.gather(
        candidate_rgb, 1, chosen_indexes[:, :, None].expand(-1, -1, 3)
    )


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


def _fit_kmeans_restart(
    lab: torch.Tensor,
    lengths: torch.Tensor,
    valid: torch.Tensor,
    *,
    cluster_count: int,
    iterations: int,
    restart: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit one deterministic Oklab k-means restart and return per-box SSE."""
    box_count, sample_count, _ = lab.shape
    box_indexes = torch.arange(box_count, device=lab.device)
    mean = (lab * valid[:, :, None]).sum(dim=1) / lengths.to(torch.float32)[:, None]
    if restart == 0:
        first = mean
    elif restart == 1:
        from_mean = ((lab - mean[:, None, :]) ** 2).sum(dim=2)
        first = lab[box_indexes, torch.argmax(from_mean.masked_fill(~valid, -1.0), dim=1)]
    else:
        seed_indexes = torch.remainder(
            box_indexes * 9973 + restart * 7919, lengths.to(torch.int64)
        )
        first = lab[box_indexes, seed_indexes]

    centroids = torch.empty(
        (box_count, cluster_count, 3), dtype=torch.float32, device=lab.device
    )
    centroids[:, 0] = first
    closest = ((lab - first[:, None, :]) ** 2).sum(dim=2).masked_fill(~valid, -1.0)
    for cluster in range(1, cluster_count):
        selected = lab[box_indexes, torch.argmax(closest, dim=1)]
        centroids[:, cluster] = selected
        distance = ((lab - selected[:, None, :]) ** 2).sum(dim=2).masked_fill(~valid, -1.0)
        closest = torch.minimum(closest, distance)

    labels = torch.empty(
        (box_count, sample_count), dtype=torch.int32, device=lab.device
    )
    counts = torch.empty(
        (box_count, cluster_count), dtype=torch.float32, device=lab.device
    )
    updated = torch.empty_like(centroids)
    kernel_kwargs = {
        "sample_count": sample_count,
        "cluster_count": cluster_count,
        "block": sample_count,
        "num_warps": 8,
    }
    for _ in range(iterations):
        _assign_kernel[(box_count,)](
            lab, lengths, centroids, labels, **kernel_kwargs
        )
        _reduce_kernel[(box_count, cluster_count)](
            lab, lengths, labels, centroids, updated, counts, **kernel_kwargs
        )
        centroids, updated = updated, centroids
    _assign_kernel[(box_count,)](
        lab, lengths, centroids, labels, **kernel_kwargs
    )
    _reduce_kernel[(box_count, cluster_count)](
        lab, lengths, labels, centroids, updated, counts, **kernel_kwargs
    )
    centroids = updated
    point_energy = ((lab * lab).sum(dim=2) * valid).sum(dim=1)
    cluster_energy = (
        counts * (centroids * centroids).sum(dim=2)
    ).sum(dim=1)
    loss = torch.clamp(point_energy - cluster_energy, min=0.0)
    return centroids, counts, loss


def _liq_median_cut(
    points: torch.Tensor,
    weights: torch.Tensor,
    lengths: torch.Tensor,
    palette_size: int,
    *,
    exact_split: bool = False,
    max_mse: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batched LIQ-inspired variance median-cut initialization."""
    box_count, sample_count, _ = points.shape
    positions = torch.arange(sample_count, device=points.device)[None, :]
    valid = positions < lengths[:, None]
    labels = torch.zeros(
        (box_count, sample_count), dtype=torch.int64, device=points.device
    )
    channel_importance = torch.tensor(
        (7.0, 9.0, 5.0), dtype=torch.float32, device=points.device
    ) / 16.0
    box_indexes = torch.arange(box_count, device=points.device)

    for new_cluster in range(1, palette_size):
        cluster_ids = torch.arange(new_cluster, device=points.device)
        membership = (
            labels[:, :, None] == cluster_ids[None, None, :]
        ) & valid[:, :, None]
        cluster_weights = weights[:, :, None] * membership
        totals = cluster_weights.sum(dim=1).clamp_min_(1e-12)
        means = torch.einsum("bnc,bnd->bcd", cluster_weights, points)
        means /= totals[:, :, None]
        squared_delta = (
            points[:, :, None, :] - means[:, None, :, :]
        ) ** 2
        if exact_split:
            squared_delta = torch.where(
                squared_delta < (1.0 / 256.0) ** 2,
                squared_delta * 0.25,
                squared_delta,
            )
        variance = torch.einsum(
            "bnc,bncd->bcd",
            cluster_weights,
            squared_delta,
        )
        variance /= totals[:, :, None]
        weighted_variance = variance * channel_importance[None, None, :]
        split_score = weighted_variance.max(dim=2).values
        split_score *= totals
        if max_mse is not None:
            box_distance = (
                (points[:, :, None, :] - means[:, None, :, :]) ** 2
            ).sum(dim=3)
            box_distance.masked_fill_(~membership, -1.0)
            box_max_error = box_distance.max(dim=1).values
            current_max_mse = max_mse * (
                1.0 + (new_cluster / palette_size) * 16.0
            )
            split_score *= torch.where(
                box_max_error > current_max_mse[:, None],
                box_max_error / current_max_mse[:, None].clamp_min(1e-20),
                1.0,
            )
        split_cluster = torch.argmax(split_score, dim=1)
        selected_variance = weighted_variance[
            box_indexes, split_cluster
        ]
        channel_order = torch.argsort(
            selected_variance, dim=1, descending=True, stable=True
        )
        channel = channel_order[:, 0]
        selected = (labels == split_cluster[:, None]) & valid
        selected_count = selected.sum(dim=1)
        ordered_channels = torch.gather(
            points, 2, channel[:, None, None].expand(-1, sample_count, 1)
        ).squeeze(2)
        if exact_split:
            second = torch.gather(
                points,
                2,
                channel_order[:, 1, None, None].expand(
                    -1, sample_count, 1
                ),
            ).squeeze(2)
            third = torch.gather(
                points,
                2,
                channel_order[:, 2, None, None].expand(
                    -1, sample_count, 1
                ),
            ).squeeze(2)
            high = torch.floor(ordered_channels * 65535.0).to(torch.int64)
            low = torch.floor(
                (third + second * 0.5 + 0.25) * 65535.0
            ).to(torch.int64)
            sort_value = (high << 16) | low
            order = torch.argsort(
                sort_value.masked_fill(~selected, -1),
                dim=1,
                descending=True,
                stable=True,
            )
        else:
            order = torch.argsort(
                ordered_channels.masked_fill(~selected, float("inf")),
                dim=1,
                stable=True,
            )
        median_position = torch.clamp((selected_count - 1) // 2, min=0)
        median_index = order[box_indexes, median_position]
        median = points[box_indexes, median_index]
        if exact_split:
            even = (selected_count % 2) == 0
            upper_index = order[
                box_indexes,
                torch.minimum(
                    median_position + 1,
                    torch.clamp(selected_count - 1, min=0),
                ),
            ]
            upper = points[box_indexes, upper_index]
            lower_weight = weights[box_indexes, median_index]
            upper_weight = weights[box_indexes, upper_index]
            pair_mean = (
                median * lower_weight[:, None]
                + upper * upper_weight[:, None]
            ) / (lower_weight + upper_weight).clamp_min(1e-12)[:, None]
            median = torch.where(even[:, None], pair_mean, median)
        distance = torch.sqrt(
            ((points - median[:, None, :]) ** 2).sum(dim=2)
        )
        color_weight = (
            distance * (torch.sqrt(1.0 + weights) - 1.0) * selected
        )
        ordered_weight = torch.gather(color_weight, 1, order)
        cumulative = torch.cumsum(ordered_weight, dim=1)
        half = ordered_weight.sum(dim=1, keepdim=True) * 0.5
        break_at = (cumulative <= half).sum(dim=1)
        if exact_split:
            # LIQ includes the item that crosses halfvar in the first box.
            break_at += 1
        break_at = torch.minimum(
            torch.maximum(break_at, torch.ones_like(break_at)),
            torch.maximum(selected_count - 1, torch.ones_like(selected_count)),
        )
        ordered_positions = torch.arange(
            sample_count, device=points.device
        )[None, :]
        new_member_ordered = (
            (ordered_positions >= break_at[:, None])
            & (ordered_positions < selected_count[:, None])
        )
        new_member = torch.zeros_like(selected)
        new_member.scatter_(1, order, new_member_ordered)
        labels = torch.where(new_member, new_cluster, labels)

    cluster_ids = torch.arange(palette_size, device=points.device)
    membership = (
        labels[:, :, None] == cluster_ids[None, None, :]
    ) & valid[:, :, None]
    cluster_weights = weights[:, :, None] * membership
    totals = cluster_weights.sum(dim=1).clamp_min_(1e-12)
    centroids = torch.einsum("bnc,bnd->bcd", cluster_weights, points)
    return centroids / totals[:, :, None]


def _fit_weighted_kmeans(
    points: torch.Tensor,
    lengths: torch.Tensor,
    weights: torch.Tensor,
    centroids: torch.Tensor,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    box_count, sample_count, _ = points.shape
    cluster_count = centroids.shape[1]
    labels = torch.empty(
        (box_count, sample_count), dtype=torch.int32, device=points.device
    )
    totals = torch.empty(
        (box_count, cluster_count), dtype=torch.float32, device=points.device
    )
    updated = torch.empty_like(centroids)
    kernel_kwargs = {
        "sample_count": sample_count,
        "cluster_count": cluster_count,
        "block": sample_count,
        "num_warps": 8,
    }
    for _ in range(iterations):
        _assign_kernel[(box_count,)](
            points, lengths, centroids, labels, **kernel_kwargs
        )
        _weighted_reduce_kernel[(box_count, cluster_count)](
            points,
            lengths,
            weights,
            labels,
            centroids,
            updated,
            totals,
            **kernel_kwargs,
        )
        centroids, updated = updated, centroids
    _assign_kernel[(box_count,)](
        points, lengths, centroids, labels, **kernel_kwargs
    )
    return centroids, labels


def _weighted_assignment_error(
    points: torch.Tensor,
    lengths: torch.Tensor,
    centroids: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    positions = torch.arange(points.shape[1], device=points.device)[None, :]
    valid = positions < lengths[:, None]
    distance = (
        (points[:, :, None, :] - centroids[:, None, :, :]) ** 2
    ).sum(dim=3)
    nearest = distance.min(dim=2).values
    return (nearest * weights * valid).sum(dim=1)


def _liq_feedback_trial(
    points: torch.Tensor,
    lengths: torch.Tensor,
    centroids: torch.Tensor,
    perceptual_weights: torch.Tensor,
    adjusted_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one LIQ feedback/K-means trial in source-code order."""
    box_count, sample_count, _ = points.shape
    cluster_count = centroids.shape[1]
    positions = torch.arange(points.shape[1], device=points.device)[None, :]
    valid = positions < lengths[:, None]
    distance = (
        (points[:, :, None, :] - centroids[:, None, :, :]) ** 2
    ).sum(dim=3)
    nearest_distance, labels64 = torch.min(distance, dim=2)
    assigned = torch.gather(
        centroids, 1, labels64[:, :, None].expand(-1, -1, 3)
    )
    reflected = points + points - assigned
    reflected_distance = (
        (reflected[:, :, None, :] - centroids[:, None, :, :]) ** 2
    ).sum(dim=3).min(dim=2).values
    new_weights = (perceptual_weights + adjusted_weights) * torch.sqrt(
        1.0 + reflected_distance
    )
    new_weights.masked_fill_(~valid, 0.0)

    labels = labels64.to(torch.int32)
    updated_centroids = torch.empty_like(centroids)
    totals = torch.empty(
        (box_count, cluster_count), dtype=torch.float32, device=points.device
    )
    _weighted_reduce_kernel[(box_count, cluster_count)](
        points,
        lengths,
        new_weights,
        labels,
        centroids,
        updated_centroids,
        totals,
        sample_count=sample_count,
        cluster_count=cluster_count,
        block=sample_count,
        num_warps=8,
    )
    error = (nearest_distance * perceptual_weights * valid).sum(dim=1)
    return updated_centroids, new_weights, error


def gpu_liq_like_observed_device(
    rgb: torch.Tensor,
    gpu_lengths: torch.Tensor,
    *,
    palette_size: int = 5,
    iterations: int = 8,
    min_cluster_ratio: float = 0.01,
    feedback: bool = False,
    exact_median_cut: bool = True,
    rounded_remap: bool = True,
) -> GpuBatchResult:
    """GPU 2.0: LIQ gamma/weights + median-cut + weighted k-means."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != torch.uint8 or not rgb.is_cuda:
        raise ValueError("rgb must be CUDA uint8 [box, sample, rgb]")
    if not 1 <= palette_size <= 32:
        raise ValueError("palette_size must be between 1 and 32")
    box_count, sample_count, _ = rgb.shape
    candidate_rgb, raw_counts, candidate_lengths = _unique_rgb_candidates(
        rgb, gpu_lengths
    )
    positions = torch.arange(sample_count, device=rgb.device)[None, :]
    valid = positions < candidate_lengths[:, None]
    # libimagequant 2.x uses internal_gamma/input_gamma = .5499/.45455.
    points = torch.pow(
        candidate_rgb.to(torch.float32) / 255.0,
        0.5499 / 0.45455,
    )
    points.masked_fill_(~valid[:, :, None], 0.0)
    # LIQ converts the default importance 255 to /128 and caps a color at
    # 10% of the image surface so a flat background cannot dominate.
    perceptual_weights = torch.minimum(
        raw_counts.to(torch.float32) * (255.0 / 128.0),
        gpu_lengths.to(torch.float32)[:, None] * 0.1,
    )
    perceptual_weights.masked_fill_(~valid, 0.0)
    weights = perceptual_weights
    centroids = _liq_median_cut(
        points,
        weights,
        candidate_lengths,
        palette_size,
        exact_split=exact_median_cut,
    )
    if feedback:
        first_centroids, weights, first_error = _liq_feedback_trial(
            points,
            candidate_lengths,
            centroids,
            perceptual_weights,
            weights,
        )
        second_initial = _liq_median_cut(
            points,
            weights,
            candidate_lengths,
            palette_size,
            exact_split=exact_median_cut,
            max_mse=torch.maximum(
                first_error
                / perceptual_weights.sum(dim=1).clamp_min(1e-12),
                torch.full_like(first_error, 45.0 / 65536.0),
            )
            * 1.2,
        )
        second_centroids, weights, second_error = _liq_feedback_trial(
            points,
            candidate_lengths,
            second_initial,
            perceptual_weights,
            weights,
        )
        better = second_error < first_error
        centroids = torch.where(
            better[:, None, None], second_centroids, first_centroids
        )
    centroids, labels = _fit_weighted_kmeans(
        points, candidate_lengths, weights, centroids, iterations
    )
    center_rgb = torch.clamp(
        torch.pow(torch.clamp(centroids, 0.0, 1.0), 0.45455 / 0.5499)
        * 256.0,
        0.0,
        255.0,
    ).to(torch.uint8)
    if rounded_remap:
        rounded_centroids = torch.pow(
            center_rgb.to(torch.float32) / 255.0,
            0.5499 / 0.45455,
        )
        _assign_kernel[(box_count,)](
            points,
            candidate_lengths,
            rounded_centroids,
            labels,
            sample_count=sample_count,
            cluster_count=palette_size,
            block=sample_count,
            num_warps=8,
        )
    raw_populations = torch.zeros(
        (box_count, palette_size), dtype=torch.float32, device=rgb.device
    )
    safe_labels = torch.where(valid, labels, 0).to(torch.int64)
    raw_populations.scatter_add_(
        1, safe_labels, raw_counts.to(torch.float32) * valid
    )
    order = torch.argsort(
        raw_populations, dim=1, descending=True, stable=True
    )
    raw_populations = torch.gather(raw_populations, 1, order)
    center_rgb = torch.gather(
        center_rgb, 1, order[:, :, None].expand(-1, -1, 3)
    )
    center_lab = torch.empty(
        (box_count, palette_size, 3),
        dtype=torch.float32,
        device=rgb.device,
    )
    center_lengths = torch.full(
        (box_count,), palette_size, dtype=torch.int32, device=rgb.device
    )
    center_block = triton.next_power_of_2(palette_size)
    _rgb_to_oklab_kernel[(box_count,)](
        center_rgb,
        center_lengths,
        center_lab,
        sample_count=palette_size,
        block=center_block,
        num_warps=4,
    )
    palette = _frequency_weighted_observed(
        rgb, gpu_lengths, center_lab
    ).cpu().numpy()
    ratios = raw_populations.div(gpu_lengths[:, None]).cpu().numpy()
    output_palettes = []
    output_weights = []
    for colors, box_ratios in zip(palette, ratios):
        selected_colors = []
        selected_weights = []
        seen = set()
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


def gpu_kmeans_observed_device(
    rgb: torch.Tensor,
    gpu_lengths: torch.Tensor,
    *,
    iterations: int = 8,
    min_cluster_ratio: float = 0.01,
    observed_correction: str = "frequency_weighted",
    palette_size: int = 5,
    kmeans_restarts: int = 2,
) -> GpuBatchResult:
    """Run the palette pipeline from an existing CUDA sample tensor."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != torch.uint8 or not rgb.is_cuda:
        raise ValueError("rgb must be CUDA uint8 [box, sample, rgb]")
    if gpu_lengths.ndim != 1 or len(gpu_lengths) != len(rgb) or not gpu_lengths.is_cuda:
        raise ValueError("gpu_lengths must have one CUDA value per box")
    if observed_correction not in {"nearest", "frequency_weighted"}:
        raise ValueError("observed_correction must be 'nearest' or 'frequency_weighted'")
    if not 1 <= palette_size <= 32:
        raise ValueError("palette_size must be between 1 and 32")
    if not 1 <= kmeans_restarts <= 8:
        raise ValueError("kmeans_restarts must be between 1 and 8")
    box_count, sample_count, _ = rgb.shape
    if sample_count > 65536 or sample_count & (sample_count - 1):
        raise ValueError("sample dimension must be a power of two no larger than 65536")
    lab = torch.empty((box_count, sample_count, 3), dtype=torch.float32, device="cuda")
    _rgb_to_oklab_kernel[(box_count,)](rgb, gpu_lengths, lab, sample_count=sample_count, block=sample_count, num_warps=8)
    positions = torch.arange(sample_count, device="cuda")[None, :]
    valid = positions < gpu_lengths[:, None]
    centroids = counts = best_loss = None
    for restart in range(kmeans_restarts):
        candidate_centroids, candidate_counts, candidate_loss = _fit_kmeans_restart(
            lab,
            gpu_lengths,
            valid,
            cluster_count=palette_size,
            iterations=iterations,
            restart=restart,
        )
        if best_loss is None:
            centroids, counts, best_loss = (
                candidate_centroids, candidate_counts, candidate_loss
            )
        else:
            better = candidate_loss < best_loss
            centroids = torch.where(
                better[:, None, None], candidate_centroids, centroids
            )
            counts = torch.where(better[:, None], candidate_counts, counts)
            best_loss = torch.where(better, candidate_loss, best_loss)
    assert centroids is not None and counts is not None
    order = torch.argsort(counts, dim=1, descending=True, stable=True)
    centroids = torch.gather(centroids, 1, order[:, :, None].expand(-1, -1, 3))
    if observed_correction == "nearest":
        palette = torch.empty((box_count, palette_size, 3), dtype=torch.uint8, device="cuda")
        _observed_kernel[(box_count, palette_size)](
            rgb, lab, gpu_lengths, centroids, palette,
            sample_count=sample_count, cluster_count=palette_size,
            block=sample_count, num_warps=8,
        )
    else:
        palette = _frequency_weighted_observed(rgb, gpu_lengths, centroids)
    palette = palette.cpu().numpy()
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
    observed_correction: str = "frequency_weighted",
    palette_size: int = 5,
    kmeans_restarts: int = 2,
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
        observed_correction=observed_correction,
        palette_size=palette_size,
        kmeans_restarts=kmeans_restarts,
    )


def gpu_liq_like_observed(
    samples: np.ndarray,
    lengths: np.ndarray,
    *,
    palette_size: int = 5,
    iterations: int = 8,
    min_cluster_ratio: float = 0.01,
    feedback: bool = False,
    exact_median_cut: bool = True,
    rounded_remap: bool = True,
) -> GpuBatchResult:
    """Copy CPU samples to CUDA and run the LIQ-inspired GPU 2.0 path."""
    if samples.ndim != 3 or samples.shape[2] != 3 or samples.dtype != np.uint8:
        raise ValueError("samples must be uint8 [box, sample, rgb]")
    if len(lengths) != len(samples) or np.any(lengths <= 0) or np.any(lengths > samples.shape[1]):
        raise ValueError("invalid sample lengths")
    return gpu_liq_like_observed_device(
        torch.as_tensor(samples, device="cuda"),
        torch.as_tensor(
            lengths.astype(np.int32, copy=False), device="cuda"
        ),
        palette_size=palette_size,
        iterations=iterations,
        min_cluster_ratio=min_cluster_ratio,
        feedback=feedback,
        exact_median_cut=exact_median_cut,
        rounded_remap=rounded_remap,
    )
