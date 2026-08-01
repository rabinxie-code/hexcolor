from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from .color import rgb24
from .images import open_image_srgb
from .methods import METHODS


METHOD_IDS = {
    "adaptive_v1": 1,
    "tencent_hsv": 2,
    "colorpipette_inspired": 3,
    "pixelero_rgb_hist": 4,
    "octree": 5,
    "pngquant_liq": 6,
}


def annotate_path(path: str | Path, method_name: str = "adaptive_v1", *, full: bool = False) -> dict[str, Any]:
    """Annotate one image and return a JSON-serializable record.

    The compact form mirrors the proposed production schema: RGB values are
    stored as integers, while ``hex`` remains as a convenience export field.
    """

    if method_name not in METHODS:
        raise ValueError(f"Unknown method {method_name!r}; choose one of {sorted(METHODS)}")

    source = Path(path)
    started = time.perf_counter_ns()
    image, metadata = open_image_srgb(source)
    decoded = time.perf_counter_ns()
    result = METHODS[method_name](image, source.as_posix())
    finished = time.perf_counter_ns()
    payload = result.to_dict()

    compact: dict[str, Any] = {
        "schema_version": 1,
        "id": source.as_posix(),
        "width": image.width,
        "height": image.height,
        "rgb24": payload["rgb24"],
        "hex": payload["hex"],
        "method_id": METHOD_IDS[method_name],
        "method": payload["method"],
        "route": payload["route"],
        "confidence": payload["confidence"],
        "palette_rgb24": [rgb24(tuple(color)) for color in payload["palette_rgb"]],
        "palette_weight_u16": [min(65535, max(0, round(float(weight) * 65535))) for weight in payload["weights"]],
        "observed": payload["observed"],
        "decode_ms": round((decoded - started) / 1e6, 6),
        "method_ms": round((finished - decoded) / 1e6, 6),
    }
    if full:
        compact["primary_rgb"] = payload["primary_rgb"]
        compact["palette_rgb"] = payload["palette_rgb"]
        compact["palette_hex"] = payload["palette_hex"]
        compact["weights"] = payload["weights"]
        compact["diagnostics"] = payload["diagnostics"]
        compact["source_metadata"] = metadata
    return compact


def _annotate_worker(arguments: tuple[str, str, bool]) -> dict[str, Any]:
    path, method_name, full = arguments
    return annotate_path(path, method_name, full=full)


def annotate_paths(
    paths: Iterable[str | Path],
    method_name: str = "adaptive_v1",
    *,
    workers: int = 1,
    full: bool = False,
) -> Iterator[dict[str, Any]]:
    """Stream deterministic annotations in input order.

    Multiprocessing is deliberately optional. At 100B scale this reference
    interface should be replaced by shard-local decode and a fused batch
    kernel, but it is useful for correctness checks and small pilots.
    """

    normalized = [Path(path).as_posix() for path in paths]
    if workers <= 1:
        for path in normalized:
            yield annotate_path(path, method_name, full=full)
        return

    arguments = ((path, method_name, full) for path in normalized)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(_annotate_worker, arguments, chunksize=8)
