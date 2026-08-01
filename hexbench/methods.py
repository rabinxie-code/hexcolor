from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version

import numpy as np
from PIL import Image

from .color import as_rgb_tuple, oklab_to_srgb, pack_rgb24, q12_keys, rgb_to_hex, srgb_to_oklab, unpack_rgb24
from .images import image_to_pixels
from .models import ExtractionResult

DEFAULT_PALETTE_SIZE = 5


def _seed_from_id(image_id: str) -> int:
    digest = hashlib.blake2b(image_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def _sample_indexes(size: int, count: int, image_id: str) -> np.ndarray:
    if size <= count:
        return np.arange(size, dtype=np.int64)
    rng = np.random.default_rng(_seed_from_id(image_id))
    return np.sort(rng.choice(size, size=count, replace=False))


def _q12_histogram(pixels: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keys = q12_keys(pixels).astype(np.int32, copy=False)
    histogram = np.bincount(keys, weights=weights, minlength=4096).astype(np.float64, copy=False)
    return keys, histogram


def _top_indexes(histogram: np.ndarray, count: int) -> np.ndarray:
    nonzero = int(np.count_nonzero(histogram))
    if nonzero == 0:
        return np.array([], dtype=np.int64)
    count = min(count, nonzero)
    candidates = np.argpartition(histogram, -count)[-count:]
    return candidates[np.argsort(histogram[candidates])[::-1]]


def _observed_representative(pixels: np.ndarray, weights: np.ndarray | None = None) -> tuple[int, int, int]:
    values = np.asarray(pixels, dtype=np.uint8).reshape(-1, 3)
    if values.shape[0] == 0:
        return 0, 0, 0
    if weights is None:
        center = np.median(values, axis=0)
    else:
        local_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        total = float(np.sum(local_weights))
        center = np.average(values, axis=0, weights=local_weights) if total > 0 else np.median(values, axis=0)
    center_lab = srgb_to_oklab(center[None, :])[0]
    values_lab = srgb_to_oklab(values)
    index = int(np.argmin(np.sum((values_lab - center_lab) ** 2, axis=1)))
    return as_rgb_tuple(values[index])


def _exact_heavy_hitter(pixels: np.ndarray, weights: np.ndarray) -> tuple[int, int, int]:
    packed = pack_rgb24(pixels)
    unique, inverse = np.unique(packed, return_inverse=True)
    totals = np.bincount(inverse, weights=weights)
    winner = unique[int(np.argmax(totals))]
    red = int((winner >> 16) & 255)
    green = int((winner >> 8) & 255)
    blue = int(winner & 255)
    return red, green, blue


def _palette_from_q12(
    pixels: np.ndarray,
    weights: np.ndarray,
    keys: np.ndarray,
    histogram: np.ndarray,
    count: int = 3,
) -> tuple[tuple[tuple[int, int, int], ...], tuple[float, ...], tuple[int, ...]]:
    total = max(float(histogram.sum()), 1e-12)
    palette: list[tuple[int, int, int]] = []
    ratios: list[float] = []
    bins: list[int] = []
    for key in _top_indexes(histogram, max(count * 3, count)):
        mask = keys == int(key)
        representative = _observed_representative(pixels[mask], weights[mask])
        if palette:
            distance = float(np.min(np.linalg.norm(
                srgb_to_oklab(np.asarray(palette)) - srgb_to_oklab(np.asarray(representative)[None, :]), axis=1
            )))
            if distance < 0.018:
                continue
        palette.append(representative)
        ratios.append(float(histogram[key] / total))
        bins.append(int(key))
        if len(palette) == count:
            break
    return tuple(palette), tuple(ratios), tuple(bins)


def _gradient_score(sample_lab: np.ndarray, sample_flat_indexes: np.ndarray, width: int, height: int) -> tuple[float, np.ndarray]:
    if sample_lab.shape[0] < 8:
        return 0.0, np.array([1.0, 0.0])
    rows = sample_flat_indexes // width
    columns = sample_flat_indexes % width
    x = columns / max(width - 1, 1) - 0.5
    y = rows / max(height - 1, 1) - 0.5
    design = np.stack((np.ones_like(x), x, y), axis=1)
    coefficients, _, _, _ = np.linalg.lstsq(design, sample_lab, rcond=None)
    fitted = design @ coefficients
    residual = float(np.sum((sample_lab - fitted) ** 2))
    centered = sample_lab - np.mean(sample_lab, axis=0, keepdims=True)
    total = float(np.sum(centered**2))
    score = 0.0 if total <= 1e-12 else max(0.0, min(1.0, 1.0 - residual / total))
    spatial = np.linalg.norm(coefficients[1:, :], axis=1)
    if float(np.linalg.norm(spatial)) <= 1e-12:
        direction = np.array([1.0, 0.0])
    else:
        direction = spatial / np.linalg.norm(spatial)
    return score, direction


def _gradient_palette(
    sample_pixels: np.ndarray,
    sample_flat_indexes: np.ndarray,
    width: int,
    direction: np.ndarray,
) -> tuple[tuple[int, int, int], ...]:
    rows = sample_flat_indexes // width
    columns = sample_flat_indexes % width
    coordinates = np.stack((columns, rows), axis=1).astype(np.float64)
    coordinates -= np.mean(coordinates, axis=0, keepdims=True)
    projection = coordinates @ direction
    order = np.argsort(projection)
    sorted_pixels = sample_pixels[order]
    quantile_indexes = np.rint(np.linspace(0, len(sorted_pixels) - 1, 5)).astype(np.int64)
    palette: list[tuple[int, int, int]] = []
    for color in sorted_pixels[quantile_indexes]:
        candidate = as_rgb_tuple(color)
        if not palette or float(np.min(np.linalg.norm(
            srgb_to_oklab(np.asarray(palette)) - srgb_to_oklab(np.asarray(candidate)[None, :]), axis=1
        ))) >= 0.015:
            palette.append(candidate)
    return tuple(palette)


def adaptive_hex_v1(image: Image.Image, image_id: str = "") -> ExtractionResult:
    """CPU gold implementation of the Adaptive Hex v1 routing proposal."""

    data = image_to_pixels(image)
    pixels = data.pixels
    weights = data.weights.astype(np.float64, copy=False)
    flat_indexes = np.flatnonzero(data.valid)
    keys, histogram = _q12_histogram(pixels, weights)
    total_weight = max(float(histogram.sum()), 1e-12)
    top_bins = _top_indexes(histogram, 3)
    top1_ratio = float(histogram[top_bins[0]] / total_weight) if top_bins.size else 0.0

    local_indexes = _sample_indexes(len(pixels), 64, image_id or "adaptive")
    sample_pixels = pixels[local_indexes]
    sample_lab = srgb_to_oklab(sample_pixels)
    median_lab = np.median(sample_lab, axis=0)
    distances = np.linalg.norm(sample_lab - median_lab, axis=1)
    spread = float(np.median(distances)) if distances.size else 0.0
    rgb_std = float(np.sqrt(np.mean(np.var(sample_pixels.astype(np.float64) / 255.0, axis=0))))
    _, sample_exact_counts = np.unique(pack_rgb24(sample_pixels), return_counts=True)
    sample_exact_ratio = float(np.max(sample_exact_counts) / len(sample_pixels))
    gradient_r2, gradient_direction = _gradient_score(
        sample_lab,
        flat_indexes[local_indexes],
        data.rgb.shape[1],
        data.rgb.shape[0],
    )

    if gradient_r2 >= 0.84 and spread >= 0.035 and top1_ratio < 0.22:
        route = "gradient"
    elif top1_ratio >= 0.88 or (top1_ratio >= 0.55 and sample_exact_ratio >= 0.25) or rgb_std < 0.008:
        route = "flat"
    elif spread < 0.105 or top1_ratio >= 0.35:
        route = "mild"
    else:
        route = "texture"

    q_palette, q_weights, q_bins = _palette_from_q12(
        pixels, weights, keys, histogram, count=DEFAULT_PALETTE_SIZE
    )
    if route == "flat":
        winning = int(top_bins[0])
        mask = keys == winning
        primary = _exact_heavy_hitter(pixels[mask], weights[mask])
        palette = (primary,) + tuple(color for color in q_palette if color != primary)
        palette = palette[:DEFAULT_PALETTE_SIZE]
        output_weights = (top1_ratio,) + tuple(q_weights[index] for index, color in enumerate(q_palette) if color != primary)
        output_weights = output_weights[: len(palette)]
        confidence = 0.55 + 0.45 * top1_ratio
    elif route == "mild":
        primary = _observed_representative(sample_pixels)
        palette = (primary,) + tuple(color for color in q_palette if color != primary)
        palette = palette[:DEFAULT_PALETTE_SIZE]
        output_weights = (max(0.0, 1.0 - spread * 4.0),) + q_weights[: max(0, len(palette) - 1)]
        confidence = max(0.25, min(0.95, 1.0 - spread * 4.0))
    elif route == "gradient":
        stops = _gradient_palette(sample_pixels, flat_indexes[local_indexes], data.rgb.shape[1], gradient_direction)
        primary = stops[len(stops) // 2] if stops else _observed_representative(sample_pixels)
        palette = stops or (primary,)
        output_weights = tuple(1.0 / len(palette) for _ in palette)
        confidence = gradient_r2
    else:
        palette = q_palette or (_observed_representative(sample_pixels),)
        primary = palette[0]
        output_weights = q_weights or (1.0,)
        confidence = min(0.95, max(0.20, float(sum(output_weights))))

    diagnostics = {
        "top1_q12_ratio": top1_ratio,
        "sample_spread_oklab": spread,
        "sample_rgb_std": rgb_std,
        "sample_exact_ratio": sample_exact_ratio,
        "gradient_r2": gradient_r2,
        "q12_top_bins": list(q_bins),
        "sample_count": int(len(sample_pixels)),
        "valid_pixels": int(len(pixels)),
        "alpha_coverage": float(np.mean(data.alpha)),
        "source_mode": data.source_mode,
        "color_managed": data.color_managed,
        "output_semantics": "observed sRGB; gradient route returns observed stops",
    }
    return ExtractionResult(
        method="adaptive_v1",
        primary=primary,
        palette=tuple(palette),
        weights=tuple(float(value) for value in output_weights),
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        route=route,
        observed=True,
        diagnostics=diagnostics,
    )


def _hsv_arrays(pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = np.asarray(pixels, dtype=np.float64) / 255.0
    maximum = np.max(rgb, axis=1)
    minimum = np.min(rgb, axis=1)
    delta = maximum - minimum
    saturation = np.divide(delta, maximum, out=np.zeros_like(delta), where=maximum > 1e-12)
    hue = np.zeros_like(maximum)
    chromatic = delta > 1e-12
    red, green, blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    red_max = chromatic & (maximum == red)
    green_max = chromatic & (maximum == green) & ~red_max
    blue_max = chromatic & ~(red_max | green_max)
    hue[red_max] = np.mod((green[red_max] - blue[red_max]) / delta[red_max], 6.0)
    hue[green_max] = (blue[green_max] - red[green_max]) / delta[green_max] + 2.0
    hue[blue_max] = (red[blue_max] - green[blue_max]) / delta[blue_max] + 4.0
    hue = np.mod(hue / 6.0, 1.0)
    return hue, saturation, maximum


def tencent_hsv_histogram(image: Image.Image, image_id: str = "", hue_bins: int = 60) -> ExtractionResult:
    """Concrete, reproducible interpretation of the underspecified Tencent article."""

    del image_id
    data = image_to_pixels(image)
    pixels = data.pixels
    weights = data.weights.astype(np.float64, copy=False)
    hue, saturation, value = _hsv_arrays(pixels)
    hue_index = np.minimum((hue * hue_bins).astype(np.int32), hue_bins - 1)
    saturation_index = np.minimum((saturation * 4).astype(np.int32), 3)
    value_index = np.minimum((value * 4).astype(np.int32), 3)
    achromatic = saturation < 0.08
    chromatic_key = hue_index * 16 + saturation_index * 4 + value_index
    achromatic_key = hue_bins * 16 + value_index
    keys = np.where(achromatic, achromatic_key, chromatic_key).astype(np.int32)
    histogram = np.bincount(keys, weights=weights, minlength=hue_bins * 16 + 4).astype(np.float64)
    total = max(float(histogram.sum()), 1e-12)

    hue_histogram = np.bincount(hue_index[~achromatic], weights=weights[~achromatic], minlength=hue_bins)
    padded = np.concatenate((hue_histogram[-2:], hue_histogram, hue_histogram[:2]))
    smoothed = np.convolve(padded, np.array([1, 2, 3, 2, 1], dtype=np.float64), mode="valid")
    winning_hue = int(np.argmax(smoothed)) if np.any(smoothed) else 0
    achromatic_weight = float(np.sum(weights[achromatic]))

    if achromatic_weight / total >= 0.5:
        value_counts = histogram[hue_bins * 16 : hue_bins * 16 + 4]
        winning_value = int(np.argmax(value_counts))
        winning_key = hue_bins * 16 + winning_value
        route = "achromatic_histogram"
    else:
        local = histogram[winning_hue * 16 : (winning_hue + 1) * 16]
        winning_key = winning_hue * 16 + int(np.argmax(local))
        route = "hsv_histogram"

    selected = keys == winning_key
    if not np.any(selected):
        selected = hue_index == winning_hue
    primary = _observed_representative(pixels[selected], weights[selected])

    palette: list[tuple[int, int, int]] = [primary]
    ratios: list[float] = [float(np.sum(weights[selected]) / total)]
    for key in _top_indexes(histogram, 12):
        mask = keys == int(key)
        candidate = _observed_representative(pixels[mask], weights[mask])
        candidate_lab = srgb_to_oklab(np.asarray(candidate)[None, :])
        existing_lab = srgb_to_oklab(np.asarray(palette))
        if float(np.min(np.linalg.norm(existing_lab - candidate_lab, axis=1))) < 0.025:
            continue
        palette.append(candidate)
        ratios.append(float(histogram[key] / total))
        if len(palette) == DEFAULT_PALETTE_SIZE:
            break

    confidence = min(1.0, max(ratios[0], float(smoothed[winning_hue] / max(np.sum(smoothed), 1e-12))))
    diagnostics = {
        "article_status": "algorithm sketch only; this implementation fixes 60 H × 4 S × 4 V bins plus achromatic buckets",
        "hue_bins": hue_bins,
        "achromatic_ratio": achromatic_weight / total,
        "winning_hue_degrees": (winning_hue + 0.5) * 360.0 / hue_bins,
        "selected_ratio": ratios[0],
        "valid_pixels": int(len(pixels)),
        "output_semantics": "observed sRGB representative from winning HSV bucket",
    }
    return ExtractionResult(
        method="tencent_hsv",
        primary=primary,
        palette=tuple(palette),
        weights=tuple(ratios),
        confidence=confidence,
        route=route,
        observed=True,
        diagnostics=diagnostics,
    )


def _visible_palette_from_rgba(
    colors: np.ndarray,
    counts: np.ndarray,
    palette_size: int,
) -> tuple[tuple[tuple[int, int, int], ...], tuple[float, ...]]:
    """Rank a quantizer palette by visible pixel mass and drop transparent entries."""

    rgba = np.asarray(colors, dtype=np.uint8).reshape(-1, 4)
    populations = np.asarray(counts, dtype=np.float64).reshape(-1)
    visible_mass = populations * (rgba[:, 3].astype(np.float64) / 255.0)
    keep = np.flatnonzero(visible_mass > 1e-12)
    if keep.size == 0:
        keep = np.flatnonzero(populations > 0)
        visible_mass = populations
    if keep.size == 0:
        return ((0, 0, 0),), (1.0,)
    order = keep[np.argsort(visible_mass[keep])[::-1]][:palette_size]
    total = max(float(np.sum(visible_mass[order])), 1e-12)
    palette = tuple(as_rgb_tuple(rgba[index, :3]) for index in order)
    weights = tuple(float(visible_mass[index] / total) for index in order)
    return palette, weights


def pngquant_libimagequant(
    image: Image.Image, image_id: str = "", palette_size: int = DEFAULT_PALETTE_SIZE
) -> ExtractionResult:
    """Use pngquant's libimagequant engine on decoded RGBA pixels.

    The PNG encoder and dithering are deliberately excluded because this task
    needs palette labels, not a recompressed output file. Palette generation and
    remapping are the real libimagequant implementation exposed by imagequant.
    """

    del image_id
    try:
        import imagequant
    except ImportError as error:
        raise RuntimeError("pngquant_libimagequant requires imagequant>=1.1.5") from error

    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    height, width = rgba.shape[:2]
    indexed_bytes, raw_palette = imagequant.quantize_raw_rgba_bytes(
        rgba.tobytes(),
        width,
        height,
        dithering_level=0.0,
        max_colors=palette_size,
        min_quality=0,
        max_quality=100,
    )
    indexed = np.frombuffer(indexed_bytes, dtype=np.uint8)
    palette_rgba = np.asarray(raw_palette, dtype=np.uint8).reshape(-1, 4)
    counts = np.bincount(indexed, minlength=len(palette_rgba)).astype(np.float64)
    palette, weights = _visible_palette_from_rgba(palette_rgba, counts, palette_size)
    try:
        package_version = version("imagequant")
    except PackageNotFoundError:
        package_version = "unknown"
    raw_lib_version = int(imagequant.lib.liq_version())
    lib_version = f"{raw_lib_version // 10000}.{(raw_lib_version // 100) % 100}.{raw_lib_version % 100}"
    return ExtractionResult(
        method="pngquant_liq",
        primary=palette[0],
        palette=palette,
        weights=weights,
        confidence=weights[0],
        route="libimagequant_palette",
        observed=False,
        diagnostics={
            "implementation": f"imagequant {package_version} native binding to libimagequant {lib_version}",
            "libimagequant_version": lib_version,
            "png_encoder_included": False,
            "dithering_level": 0.0,
            "max_colors": palette_size,
            "quality": [0, 100],
            "valid_pixels": int(width * height),
            "output_semantics": "libimagequant palette centroid; not guaranteed to occur in source pixels",
        },
    )


def octree_quantization(
    image: Image.Image, image_id: str = "", palette_size: int = DEFAULT_PALETTE_SIZE
) -> ExtractionResult:
    """Native fast-octree adapter for the Gervautz–Purgathofer method family."""

    del image_id
    rgba = image.convert("RGBA")
    quantized = rgba.quantize(
        colors=palette_size,
        method=Image.Quantize.FASTOCTREE,
        dither=Image.Dither.NONE,
    )
    palette_rgba = np.asarray(quantized.getpalette("RGBA"), dtype=np.uint8).reshape(-1, 4)
    counts = np.zeros(len(palette_rgba), dtype=np.float64)
    for population, index in quantized.getcolors(maxcolors=256) or []:
        counts[int(index)] = float(population)
    palette, weights = _visible_palette_from_rgba(palette_rgba, counts, palette_size)
    return ExtractionResult(
        method="octree",
        primary=palette[0],
        palette=palette,
        weights=weights,
        confidence=weights[0],
        route="fast_octree_native",
        observed=False,
        diagnostics={
            "algorithm_lineage": "Gervautz–Purgathofer octree reduction",
            "implementation": "Pillow Image.Quantize.FASTOCTREE native adapter; not a line-for-line Pascal port",
            "max_colors": palette_size,
            "dithering": "none",
            "valid_pixels": int(image.width * image.height),
            "output_semantics": "octree leaf mean; not guaranteed to occur in source pixels",
        },
    )


def _weighted_channel_centroids(values: np.ndarray, weights: np.ndarray, cluster_count: int) -> np.ndarray:
    histogram = np.bincount(values.astype(np.int32), weights=weights, minlength=256).astype(np.float64)
    occupied = np.flatnonzero(histogram > 0)
    if occupied.size == 0:
        return np.array([0.0], dtype=np.float64)
    if occupied.size <= cluster_count:
        return occupied.astype(np.float64)

    # The Pixelero article initializes by equal divisions of 0..255.
    centroids = np.linspace(0.0, 255.0, cluster_count, dtype=np.float64)
    coordinates = np.arange(256, dtype=np.float64)
    for _ in range(32):
        assignments = np.argmin(np.abs(coordinates[:, None] - centroids[None, :]), axis=1)
        updated = centroids.copy()
        for index in range(cluster_count):
            mask = assignments == index
            mass = float(np.sum(histogram[mask]))
            if mass > 0:
                updated[index] = float(np.sum(coordinates[mask] * histogram[mask]) / mass)
        updated.sort()
        if np.max(np.abs(updated - centroids)) < 1e-7:
            centroids = updated
            break
        centroids = updated
    return centroids


def _weighted_rgb_kmeans(points: np.ndarray, weights: np.ndarray, cluster_count: int) -> tuple[np.ndarray, np.ndarray]:
    count = min(cluster_count, len(points))
    if count == 0:
        return np.zeros((1, 3), dtype=np.float64), np.ones(1, dtype=np.float64)
    if len(points) <= count:
        order = np.argsort(weights)[::-1]
        return points[order], weights[order]

    selected = [int(np.argmax(weights))]
    while len(selected) < count:
        distances = np.min(
            np.sum((points[:, None, :] - points[np.asarray(selected)][None, :, :]) ** 2, axis=2),
            axis=1,
        )
        distances[np.asarray(selected)] = -1.0
        selected.append(int(np.argmax(distances * np.maximum(weights, 1e-12))))
    centroids = points[np.asarray(selected)].copy()

    assignments = np.zeros(len(points), dtype=np.int32)
    for _ in range(32):
        distances = np.sum((points[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        updated_assignments = np.argmin(distances, axis=1).astype(np.int32)
        updated = centroids.copy()
        for index in range(count):
            mask = updated_assignments == index
            mass = float(np.sum(weights[mask]))
            if mass > 0:
                updated[index] = np.average(points[mask], axis=0, weights=weights[mask])
        assignments = updated_assignments
        if np.max(np.abs(updated - centroids)) < 1e-7:
            centroids = updated
            break
        centroids = updated

    populations = np.bincount(assignments, weights=weights, minlength=count).astype(np.float64)
    order = np.argsort(populations)[::-1]
    return centroids[order], populations[order]


def pixelero_rgb_histogram(
    image: Image.Image,
    image_id: str = "",
    channel_clusters: int = 8,
    palette_size: int = DEFAULT_PALETTE_SIZE,
) -> ExtractionResult:
    """Deterministic completion of Pixelero's two-stage RGB histogram sketch."""

    del image_id
    data = image_to_pixels(image)
    pixels = data.pixels
    weights = data.weights.astype(np.float64, copy=False)
    channel_centroids = [
        _weighted_channel_centroids(pixels[:, channel], weights, channel_clusters) for channel in range(3)
    ]
    channel_indexes = [
        np.argmin(np.abs(pixels[:, channel, None] - channel_centroids[channel][None, :]), axis=1)
        for channel in range(3)
    ]
    green_bins = len(channel_centroids[1])
    blue_bins = len(channel_centroids[2])
    keys = (channel_indexes[0] * green_bins + channel_indexes[1]) * blue_bins + channel_indexes[2]
    bin_count = len(channel_centroids[0]) * green_bins * blue_bins
    bin_weights = np.bincount(keys, weights=weights, minlength=bin_count).astype(np.float64)
    occupied = np.flatnonzero(bin_weights > 0)
    bin_rgb = np.stack(
        [np.bincount(keys, weights=weights * pixels[:, channel], minlength=bin_count) for channel in range(3)],
        axis=1,
    )
    bin_rgb = bin_rgb[occupied] / bin_weights[occupied, None]
    centroids, populations = _weighted_rgb_kmeans(bin_rgb, bin_weights[occupied], palette_size)
    palette_array = np.clip(np.rint(centroids), 0, 255).astype(np.uint8)
    palette = tuple(as_rgb_tuple(color) for color in palette_array)
    total = max(float(np.sum(populations)), 1e-12)
    output_weights = tuple(float(value / total) for value in populations)
    return ExtractionResult(
        method="pixelero_rgb_hist",
        primary=palette[0],
        palette=palette,
        weights=output_weights,
        confidence=output_weights[0],
        route="channel_histogram_kmeans",
        observed=False,
        diagnostics={
            "article_status": "prose + demo sketch; initialization and convergence fixed here for reproducibility",
            "channel_clusters": [int(len(item)) for item in channel_centroids],
            "maximum_3d_bins": int(bin_count),
            "occupied_3d_bins": int(len(occupied)),
            "final_clusters": len(palette),
            "valid_pixels": int(len(pixels)),
            "output_semantics": "weighted RGB centroid; not guaranteed to occur in source pixels",
        },
    )


def _hsv_weighted_bins(
    pixels: np.ndarray,
    weights: np.ndarray,
    hue_bins: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Return weighted RGB representatives of Tencent-style HSV buckets."""

    hue, saturation, value = _hsv_arrays(pixels)
    hue_index = np.minimum((hue * hue_bins).astype(np.int32), hue_bins - 1)
    saturation_index = np.minimum((saturation * 4).astype(np.int32), 3)
    value_index = np.minimum((value * 4).astype(np.int32), 3)
    achromatic = saturation < 0.08
    keys = np.where(
        achromatic,
        hue_bins * 16 + value_index,
        hue_index * 16 + saturation_index * 4 + value_index,
    ).astype(np.int32)
    bin_count = hue_bins * 16 + 4
    bin_weights = np.bincount(keys, weights=weights, minlength=bin_count).astype(np.float64)
    occupied = np.flatnonzero(bin_weights > 0)
    sums = np.stack(
        [np.bincount(keys, weights=weights * pixels[:, channel], minlength=bin_count) for channel in range(3)],
        axis=1,
    )
    return sums[occupied] / bin_weights[occupied, None], bin_weights[occupied]


def _pixelero_weighted_bins(
    pixels: np.ndarray,
    weights: np.ndarray,
    channel_clusters: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the occupied adaptive RGB bins from Pixelero's first stage."""

    channel_centroids = [
        _weighted_channel_centroids(pixels[:, channel], weights, channel_clusters) for channel in range(3)
    ]
    channel_indexes = [
        np.argmin(np.abs(pixels[:, channel, None] - channel_centroids[channel][None, :]), axis=1)
        for channel in range(3)
    ]
    green_bins = len(channel_centroids[1])
    blue_bins = len(channel_centroids[2])
    keys = (channel_indexes[0] * green_bins + channel_indexes[1]) * blue_bins + channel_indexes[2]
    bin_count = len(channel_centroids[0]) * green_bins * blue_bins
    bin_weights = np.bincount(keys, weights=weights, minlength=bin_count).astype(np.float64)
    occupied = np.flatnonzero(bin_weights > 0)
    sums = np.stack(
        [np.bincount(keys, weights=weights * pixels[:, channel], minlength=bin_count) for channel in range(3)],
        axis=1,
    )
    return sums[occupied] / bin_weights[occupied, None], bin_weights[occupied]


def _centroid_result(
    method: str,
    centroids: np.ndarray,
    populations: np.ndarray,
    route: str,
    diagnostics: dict[str, object],
) -> ExtractionResult:
    palette_array = np.clip(np.rint(centroids), 0, 255).astype(np.uint8)
    palette = tuple(as_rgb_tuple(color) for color in palette_array)
    total = max(float(np.sum(populations)), 1e-12)
    output_weights = tuple(float(value / total) for value in populations)
    return ExtractionResult(
        method=method,
        primary=palette[0],
        palette=palette,
        weights=output_weights,
        confidence=output_weights[0],
        route=route,
        observed=False,
        diagnostics={
            **diagnostics,
            "valid_bins": int(len(centroids)),
            "output_semantics": "weighted bin centroid; not guaranteed to occur in source pixels",
        },
    )


def hsv_bins_pixelero_kmeans(
    image: Image.Image,
    image_id: str = "",
    palette_size: int = DEFAULT_PALETTE_SIZE,
) -> ExtractionResult:
    """Tencent HSV bins followed by Pixelero's weighted RGB k-means."""

    del image_id
    data = image_to_pixels(image)
    weights = data.weights.astype(np.float64, copy=False)
    bins, bin_weights = _hsv_weighted_bins(data.pixels, weights)
    centroids, populations = _weighted_rgb_kmeans(bins, bin_weights, palette_size)
    return _centroid_result(
        "hsv_pixelero_kmeans",
        centroids,
        populations,
        "hsv_bins_rgb_kmeans",
        {"binning": "Tencent 60H × 4S × 4V + achromatic", "clustering": "Pixelero weighted RGB k-means"},
    )


def pixelero_bins_hsv_clustering(
    image: Image.Image,
    image_id: str = "",
    palette_size: int = DEFAULT_PALETTE_SIZE,
) -> ExtractionResult:
    """Pixelero adaptive RGB bins followed by Tencent-style HSV bucket merging."""

    del image_id
    data = image_to_pixels(image)
    weights = data.weights.astype(np.float64, copy=False)
    rgb_bins, rgb_weights = _pixelero_weighted_bins(data.pixels, weights)
    hsv_bins, hsv_weights = _hsv_weighted_bins(
        np.clip(np.rint(rgb_bins), 0, 255).astype(np.uint8),
        rgb_weights,
    )
    order = np.argsort(hsv_weights)[::-1][:palette_size]
    return _centroid_result(
        "pixelero_hsv_cluster",
        hsv_bins[order],
        hsv_weights[order],
        "pixelero_bins_hsv_merge",
        {"binning": "Pixelero adaptive per-channel RGB", "clustering": "Tencent 60H × 4S × 4V bucket merge"},
    )


def hsv_bins_octree_quantization(
    image: Image.Image,
    image_id: str = "",
    palette_size: int = DEFAULT_PALETTE_SIZE,
) -> ExtractionResult:
    """Tencent HSV bins converted to a weighted pseudo-image, then FASTOCTREE."""

    del image_id
    data = image_to_pixels(image)
    weights = data.weights.astype(np.float64, copy=False)
    bins, bin_weights = _hsv_weighted_bins(data.pixels, weights)
    sample_budget = 8192
    counts = np.maximum(1, np.rint(bin_weights / max(float(bin_weights.sum()), 1e-12) * sample_budget).astype(np.int32))
    samples = np.repeat(np.clip(np.rint(bins), 0, 255).astype(np.uint8), counts, axis=0)
    rgba = np.concatenate((samples, np.full((len(samples), 1), 255, dtype=np.uint8)), axis=1)
    pseudo_image = Image.fromarray(rgba.reshape(1, len(rgba), 4), mode="RGBA")
    quantized = pseudo_image.quantize(colors=palette_size, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
    palette_rgba = np.asarray(quantized.getpalette("RGBA"), dtype=np.uint8).reshape(-1, 4)
    populations = np.zeros(len(palette_rgba), dtype=np.float64)
    for population, index in quantized.getcolors(maxcolors=256) or []:
        populations[int(index)] = float(population)
    palette, output_weights = _visible_palette_from_rgba(palette_rgba, populations, palette_size)
    return ExtractionResult(
        method="hsv_octree",
        primary=palette[0],
        palette=palette,
        weights=output_weights,
        confidence=output_weights[0],
        route="hsv_bins_fast_octree",
        observed=False,
        diagnostics={
            "binning": "Tencent 60H × 4S × 4V + achromatic",
            "clustering": "Pillow FASTOCTREE over 8192 weighted HSV-bin representatives",
            "occupied_hsv_bins": int(len(bins)),
            "output_semantics": "octree leaf mean; not guaranteed to occur in source pixels",
        },
    )


def _map_palette_to_weighted_observed(
    image: Image.Image,
    result: ExtractionResult,
    method: str,
) -> ExtractionResult:
    """Map each palette entry to a nearby observed color, favouring frequent exact colors."""

    data = image_to_pixels(image)
    packed = pack_rgb24(data.pixels)
    unique, inverse = np.unique(packed, return_inverse=True)
    mass = np.bincount(inverse, weights=data.weights.astype(np.float64, copy=False))
    candidates = unpack_rgb24(unique)
    candidate_lab = srgb_to_oklab(candidates)
    frequency_factor = 0.25 + 0.75 * np.sqrt(mass / max(float(np.max(mass)), 1e-12))
    selected: set[int] = set()
    corrected: list[tuple[int, int, int]] = []
    corrections: list[float] = []
    for color in result.palette:
        target_lab = srgb_to_oklab(np.asarray(color, dtype=np.uint8)[None, :])[0]
        distances = np.linalg.norm(candidate_lab - target_lab, axis=1)
        scores = distances / frequency_factor
        if selected and len(selected) < len(scores):
            scores[np.fromiter(selected, dtype=np.int64)] = np.inf
        winner = int(np.argmin(scores))
        selected.add(winner)
        corrected.append(as_rgb_tuple(candidates[winner]))
        corrections.append(float(distances[winner] * 100.0))
    palette = tuple(corrected)
    return ExtractionResult(
        method=method,
        primary=palette[0],
        palette=palette,
        weights=result.weights,
        confidence=result.confidence,
        route=result.route + "_weighted_observed",
        observed=True,
        diagnostics={
            **result.diagnostics,
            "correction": "nearest observed Oklab color, distance divided by frequency factor",
            "source_method": result.method,
            "correction_delta_e_ok": corrections,
            "output_semantics": "all output colors occur exactly in source pixels",
        },
    )


def tencent_hsv_weighted_observed(image: Image.Image, image_id: str = "") -> ExtractionResult:
    return _map_palette_to_weighted_observed(image, tencent_hsv_histogram(image, image_id), "tencent_hsv_observed")


def pixelero_weighted_observed(image: Image.Image, image_id: str = "") -> ExtractionResult:
    return _map_palette_to_weighted_observed(image, pixelero_rgb_histogram(image, image_id), "pixelero_rgb_hist_observed")


def octree_weighted_observed(image: Image.Image, image_id: str = "") -> ExtractionResult:
    return _map_palette_to_weighted_observed(image, octree_quantization(image, image_id), "octree_observed")


def hsv_pixelero_weighted_observed(image: Image.Image, image_id: str = "") -> ExtractionResult:
    return _map_palette_to_weighted_observed(image, hsv_bins_pixelero_kmeans(image, image_id), "hsv_pixelero_kmeans_observed")


def pixelero_hsv_weighted_observed(image: Image.Image, image_id: str = "") -> ExtractionResult:
    return _map_palette_to_weighted_observed(image, pixelero_bins_hsv_clustering(image, image_id), "pixelero_hsv_cluster_observed")


def hsv_octree_weighted_observed(image: Image.Image, image_id: str = "") -> ExtractionResult:
    return _map_palette_to_weighted_observed(image, hsv_bins_octree_quantization(image, image_id), "hsv_octree_observed")


def _resize_like_colorpipette(image: Image.Image) -> Image.Image:
    width, height = image.size
    if min(width, height) <= 256:
        return image.convert("RGB")
    if height < width:
        new_height = 256
        new_width = int(round(width * 256 / height))
    else:
        new_width = 256
        new_height = int(round(height * 256 / width))
    new_width = max(16, int(math.ceil(new_width / 16.0) * 16))
    new_height = max(16, int(math.ceil(new_height / 16.0) * 16))
    return image.convert("RGB").resize((new_width, new_height), Image.Resampling.BICUBIC)


def _harmonize_lch(colors_lab: np.ndarray) -> np.ndarray:
    if colors_lab.shape[0] < 3:
        return colors_lab
    lightness = colors_lab[:, 0]
    chroma = np.linalg.norm(colors_lab[:, 1:], axis=1)
    points = np.stack((lightness, chroma), axis=1)
    center = np.mean(points, axis=0)
    _, _, right = np.linalg.svd(points - center, full_matrices=False)
    normal = right[-1]
    signed_distance = (points - center) @ normal
    # The original ColorPipette only moves colors lying more than ~15 units
    # from its fitted L/C line. Oklab is normalized, so 0.15 is analogous.
    excess = np.sign(signed_distance) * np.maximum(np.abs(signed_distance) - 0.15, 0.0)
    adjusted = points - excess[:, None] * normal[None, :]
    hues = np.arctan2(colors_lab[:, 2], colors_lab[:, 1])
    output = colors_lab.copy()
    output[:, 0] = np.clip(adjusted[:, 0], 0.0, 1.0)
    output[:, 1] = np.maximum(adjusted[:, 1], 0.0) * np.cos(hues)
    output[:, 2] = np.maximum(adjusted[:, 1], 0.0) * np.sin(hues)
    return output


def colorpipette_inspired(
    image: Image.Image, image_id: str = "", palette_size: int = DEFAULT_PALETTE_SIZE
) -> ExtractionResult:
    """Scalable proxy of ColorPipette's segmentation/saliency/harmony stages.

    This is intentionally named "inspired": the original repository requires a
    333 MB BASNet checkpoint, a 27 MB SpixelNet checkpoint and legacy native
    connectivity code. Here SLIC and deterministic contrast saliency stand in
    for those two neural models so the method can be benchmarked reproducibly.
    """

    del image_id
    try:
        from skimage.segmentation import slic
    except ImportError as error:
        raise RuntimeError("colorpipette_inspired requires scikit-image") from error

    resized = _resize_like_colorpipette(image)
    rgb = np.asarray(resized, dtype=np.uint8)
    rgb_float = rgb.astype(np.float64) / 255.0
    height, width = rgb.shape[:2]
    requested_segments = int(np.clip(height * width / 256.0, 24, 500))
    labels = slic(
        rgb_float,
        n_segments=requested_segments,
        compactness=12.0,
        sigma=0.8,
        start_label=0,
        channel_axis=-1,
    )
    segment_count = int(labels.max()) + 1
    flat_labels = labels.reshape(-1)
    flat_rgb = rgb.reshape(-1, 3)
    flat_lab = srgb_to_oklab(flat_rgb)

    global_center = np.median(flat_lab, axis=0)
    contrast = np.linalg.norm(flat_lab - global_center, axis=1).reshape(height, width)
    contrast /= max(float(np.percentile(contrast, 95)), 1e-12)
    yy, xx = np.mgrid[0:height, 0:width]
    center_prior = np.exp(-(((xx / max(width - 1, 1) - 0.5) ** 2 + (yy / max(height - 1, 1) - 0.5) ** 2) / 0.16))
    saliency = np.clip(0.78 * contrast + 0.22 * center_prior, 0.0, 1.0).reshape(-1)

    counts = np.bincount(flat_labels, minlength=segment_count).astype(np.float64)
    segment_saliency = np.bincount(flat_labels, weights=saliency, minlength=segment_count) / np.maximum(counts, 1.0)
    segment_rgb = np.stack(
        [np.bincount(flat_labels, weights=flat_rgb[:, channel], minlength=segment_count) for channel in range(3)],
        axis=1,
    ) / np.maximum(counts[:, None], 1.0)
    segment_lab = srgb_to_oklab(segment_rgb)
    size_ratio = counts / max(float(np.sum(counts)), 1.0)

    # Original ColorPipette derives background colors from low BASNet saliency.
    background_threshold = float(np.quantile(saliency, 0.22))
    background_pixels = flat_rgb[saliency <= background_threshold]
    if len(background_pixels) == 0:
        background_pixels = flat_rgb
    background_weights = np.ones(len(background_pixels), dtype=np.float64)
    bg_keys, bg_histogram = _q12_histogram(background_pixels, background_weights)
    background_palette, _, _ = _palette_from_q12(background_pixels, background_weights, bg_keys, bg_histogram, count=2)

    selected: list[int] = [int(np.argmax(segment_saliency + 0.15 * np.sqrt(size_ratio)))]
    while len(selected) < min(palette_size, segment_count):
        selected_lab = segment_lab[selected]
        distances = np.min(np.linalg.norm(segment_lab[:, None, :] - selected_lab[None, :, :], axis=2), axis=1)
        score = 0.62 * segment_saliency + 0.28 * np.clip(distances / 0.25, 0.0, 1.0) + 0.10 * np.sqrt(size_ratio)
        score[selected] = -np.inf
        candidate = int(np.argmax(score))
        if not np.isfinite(score[candidate]):
            break
        if float(distances[candidate]) < 0.018:
            break
        selected.append(candidate)

    selected_lab = _harmonize_lch(segment_lab[selected])
    palette_array = oklab_to_srgb(selected_lab)
    palette = tuple(as_rgb_tuple(color) for color in palette_array)
    selected_importance = segment_saliency[selected] * np.sqrt(size_ratio[selected])
    if float(np.sum(selected_importance)) <= 1e-12:
        selected_importance = np.ones(len(selected), dtype=np.float64)
    palette_weights = selected_importance / np.sum(selected_importance)
    primary = palette[0]
    confidence = float(np.clip(segment_saliency[selected[0]], 0.0, 1.0))
    diagnostics = {
        "implementation": "SLIC + deterministic contrast saliency + L/C harmony proxy",
        "original_requirements": "BASNet 333 MB + SpixelNet 27 MB + legacy connectivity extension",
        "resized_to": [width, height],
        "segments": segment_count,
        "background_palette": [rgb_to_hex(color) for color in background_palette],
        "valid_pixels": int(height * width),
        "output_semantics": "harmonized palette color; not guaranteed to occur in source pixels",
    }
    return ExtractionResult(
        method="colorpipette_inspired",
        primary=primary,
        palette=palette,
        weights=tuple(float(value) for value in palette_weights),
        confidence=confidence,
        route="saliency_superpixels_harmony",
        observed=False,
        diagnostics=diagnostics,
    )


METHODS: dict[str, Callable[[Image.Image, str], ExtractionResult]] = {
    "adaptive_v1": adaptive_hex_v1,
    "tencent_hsv": tencent_hsv_histogram,
    "pixelero_rgb_hist": pixelero_rgb_histogram,
    "octree": octree_quantization,
    "pngquant_liq": pngquant_libimagequant,
    "colorpipette_inspired": colorpipette_inspired,
    "hsv_pixelero_kmeans": hsv_bins_pixelero_kmeans,
    "pixelero_hsv_cluster": pixelero_bins_hsv_clustering,
    "hsv_octree": hsv_bins_octree_quantization,
    "tencent_hsv_observed": tencent_hsv_weighted_observed,
    "pixelero_rgb_hist_observed": pixelero_weighted_observed,
    "octree_observed": octree_weighted_observed,
    "hsv_pixelero_kmeans_observed": hsv_pixelero_weighted_observed,
    "pixelero_hsv_cluster_observed": pixelero_hsv_weighted_observed,
    "hsv_octree_observed": hsv_octree_weighted_observed,
}
