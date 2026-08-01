from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps


@dataclass(frozen=True)
class PixelData:
    rgb: np.ndarray
    alpha: np.ndarray
    valid: np.ndarray
    source_mode: str
    color_managed: bool

    @property
    def pixels(self) -> np.ndarray:
        return self.rgb[self.valid]

    @property
    def weights(self) -> np.ndarray:
        return self.alpha[self.valid]


def open_image_srgb(path: str | Path) -> tuple[Image.Image, dict[str, object]]:
    with Image.open(path) as opened:
        opened.load()
        image = ImageOps.exif_transpose(opened)
        metadata = {
            "source_mode": image.mode,
            "source_size": list(image.size),
            "format": opened.format or "unknown",
            "has_icc": bool(opened.info.get("icc_profile")),
        }
        # Copy before the source file handle closes.
        return image.copy(), metadata


def image_to_pixels(image: Image.Image, alpha_threshold: int = 8) -> PixelData:
    source_mode = image.mode
    alpha = _alpha_channel(image)
    rgb_image, color_managed = _to_srgb(image)
    rgb = np.asarray(rgb_image, dtype=np.uint8)
    alpha_float = np.asarray(alpha, dtype=np.float32) / 255.0
    valid = alpha_float >= (alpha_threshold / 255.0)
    if not np.any(valid):
        valid = np.ones(alpha_float.shape, dtype=bool)
        alpha_float = np.ones(alpha_float.shape, dtype=np.float32)
    return PixelData(rgb=rgb, alpha=alpha_float, valid=valid, source_mode=source_mode, color_managed=color_managed)


def _alpha_channel(image: Image.Image) -> Image.Image:
    if "A" in image.getbands():
        return image.getchannel("A")
    return Image.new("L", image.size, 255)


def _to_srgb(image: Image.Image) -> tuple[Image.Image, bool]:
    icc_bytes = image.info.get("icc_profile")
    base = image.convert("RGB")
    if not icc_bytes:
        return base, False
    try:
        source_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
        target_profile = ImageCms.createProfile("sRGB")
        converted = ImageCms.profileToProfile(base, source_profile, target_profile, outputMode="RGB")
        return converted, True
    except (ImageCms.PyCMSError, OSError, ValueError):
        return base, False
