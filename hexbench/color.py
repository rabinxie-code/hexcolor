from __future__ import annotations

import numpy as np


def as_rgb_tuple(rgb: np.ndarray | tuple[int, int, int] | list[int]) -> tuple[int, int, int]:
    values = np.asarray(rgb).round().clip(0, 255).astype(np.uint8).reshape(3)
    return int(values[0]), int(values[1]), int(values[2])


def rgb_to_hex(rgb: np.ndarray | tuple[int, int, int] | list[int]) -> str:
    red, green, blue = as_rgb_tuple(rgb)
    return f"#{red:02X}{green:02X}{blue:02X}"


def rgb24(rgb: np.ndarray | tuple[int, int, int] | list[int]) -> int:
    red, green, blue = as_rgb_tuple(rgb)
    return (red << 16) | (green << 8) | blue


def pack_rgb24(pixels: np.ndarray) -> np.ndarray:
    values = np.asarray(pixels, dtype=np.uint32)
    return (values[..., 0] << 16) | (values[..., 1] << 8) | values[..., 2]


def unpack_rgb24(values: np.ndarray | int) -> np.ndarray:
    packed = np.asarray(values, dtype=np.uint32)
    return np.stack(((packed >> 16) & 255, (packed >> 8) & 255, packed & 255), axis=-1).astype(np.uint8)


def q12_keys(pixels: np.ndarray) -> np.ndarray:
    values = np.asarray(pixels, dtype=np.uint16)
    return ((values[..., 0] >> 4) << 8) | ((values[..., 1] >> 4) << 4) | (values[..., 2] >> 4)


def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """Convert uint8/float sRGB to Oklab, preserving all leading dimensions."""

    values = np.asarray(rgb, dtype=np.float64)
    if values.size == 0:
        return np.empty((*values.shape[:-1], 3), dtype=np.float64)
    if float(np.nanmax(values)) > 1.0:
        values = values / 255.0
    linear = np.where(
        values <= 0.04045,
        values / 12.92,
        np.power((values + 0.055) / 1.055, 2.4),
    )
    red, green, blue = np.moveaxis(linear, -1, 0)
    light = 0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue
    medium = 0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue
    short = 0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue
    light_root = np.cbrt(light)
    medium_root = np.cbrt(medium)
    short_root = np.cbrt(short)
    lab_l = 0.2104542553 * light_root + 0.7936177850 * medium_root - 0.0040720468 * short_root
    lab_a = 1.9779984951 * light_root - 2.4285922050 * medium_root + 0.4505937099 * short_root
    lab_b = 0.0259040371 * light_root + 0.7827717662 * medium_root - 0.8086757660 * short_root
    return np.stack((lab_l, lab_a, lab_b), axis=-1)


def oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    values = np.asarray(lab, dtype=np.float64)
    lab_l, lab_a, lab_b = np.moveaxis(values, -1, 0)
    light_root = lab_l + 0.3963377774 * lab_a + 0.2158037573 * lab_b
    medium_root = lab_l - 0.1055613458 * lab_a - 0.0638541728 * lab_b
    short_root = lab_l - 0.0894841775 * lab_a - 1.2914855480 * lab_b
    light = light_root**3
    medium = medium_root**3
    short = short_root**3
    red = 4.0767416621 * light - 3.3077115913 * medium + 0.2309699292 * short
    green = -1.2684380046 * light + 2.6097574011 * medium - 0.3413193965 * short
    blue = -0.0041960863 * light - 0.7034186147 * medium + 1.7076147010 * short
    linear = np.stack((red, green, blue), axis=-1).clip(0.0, 1.0)
    srgb = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )
    return np.rint(srgb.clip(0.0, 1.0) * 255.0).astype(np.uint8)


def delta_e_ok(rgb_a: np.ndarray, rgb_b: np.ndarray) -> np.ndarray:
    """Oklab Euclidean distance on a human-readable ~0-100 scale."""

    lab_a = srgb_to_oklab(np.asarray(rgb_a))
    lab_b = srgb_to_oklab(np.asarray(rgb_b))
    return np.linalg.norm(lab_a - lab_b, axis=-1) * 100.0


def palette_coverage(pixels: np.ndarray, palette: np.ndarray, max_samples: int = 4096) -> float:
    """Mean distance to the nearest palette color; lower is better."""

    source = np.asarray(pixels, dtype=np.uint8).reshape(-1, 3)
    colors = np.asarray(palette, dtype=np.uint8).reshape(-1, 3)
    if source.shape[0] == 0 or colors.shape[0] == 0:
        return float("nan")
    if source.shape[0] > max_samples:
        indexes = np.linspace(0, source.shape[0] - 1, max_samples, dtype=np.int64)
        source = source[indexes]
    source_lab = srgb_to_oklab(source)
    palette_lab = srgb_to_oklab(colors)
    distances = np.linalg.norm(source_lab[:, None, :] - palette_lab[None, :, :], axis=2)
    return float(np.mean(np.min(distances, axis=1)) * 100.0)
