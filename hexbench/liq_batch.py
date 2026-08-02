from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def native_available() -> bool:
    try:
        from . import _liq_batch_native  # noqa: F401
    except ImportError:
        return False
    return True


def quantize_many(
    samples: list[np.ndarray], speed: int, workers: int
) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """Quantize many sampled boxes in one native call, or return None."""

    if os.environ.get("HEXBENCH_DISABLE_LIQ_BATCH") == "1":
        return None
    try:
        import imagequant
        from ._liq_batch_native import ffi, lib
    except ImportError:
        return None
    if not samples:
        return []
    lengths = np.asarray([len(sample) for sample in samples], dtype=np.uintp)
    offsets = np.empty(len(samples) + 1, dtype=np.uintp)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    packed = np.ascontiguousarray(np.concatenate(samples, axis=0), dtype=np.uint8)
    palettes = np.zeros((len(samples), 5, 3), dtype=np.uint8)
    counts = np.zeros((len(samples), 5), dtype=np.uint64)
    sizes = np.zeros(len(samples), dtype=np.uint8)
    library_path = Path(imagequant._libimagequant.__file__).as_posix().encode()
    status = lib.liq_quantize_many(
        library_path,
        ffi.from_buffer("uint8_t[]", packed),
        ffi.from_buffer("size_t[]", offsets),
        len(samples),
        speed,
        workers,
        ffi.from_buffer("uint8_t[]", palettes),
        ffi.from_buffer("uint64_t[]", counts),
        ffi.from_buffer("uint8_t[]", sizes),
    )
    if status != 0:
        raise RuntimeError(f"native libimagequant batch failed with code {status}")
    return [
        (palettes[index, : sizes[index]].astype(np.float64), counts[index, : sizes[index]].astype(np.float64))
        for index in range(len(samples))
    ]
