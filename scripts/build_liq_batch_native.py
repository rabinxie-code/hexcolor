#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from cffi import FFI


ROOT = Path(__file__).resolve().parents[1]
ffi = FFI()
ffi.cdef(
    """
    int liq_quantize_many(
        const char *library_path,
        const uint8_t *rgb,
        const size_t *offsets,
        size_t box_count,
        int speed,
        int workers,
        uint8_t *palettes,
        uint64_t *counts,
        uint8_t *sizes
    );
    """
)
ffi.set_source(
    "hexbench._liq_batch_native",
    '#include "liq_batch_native.c"',
    include_dirs=[str(ROOT / "hexbench")],
    libraries=["dl", "pthread"],
    extra_compile_args=["-O3", "-std=c11"],
)


if __name__ == "__main__":
    ffi.compile(tmpdir=str(ROOT), verbose=True)
