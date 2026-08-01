"""Reference implementations and benchmarks for crop-level HEX extraction."""

from .batch import METHOD_IDS, annotate_path, annotate_paths
from .methods import METHODS, adaptive_hex_v1, colorpipette_inspired, tencent_hsv_histogram
from .models import ExtractionResult

__all__ = [
    "METHODS",
    "METHOD_IDS",
    "ExtractionResult",
    "adaptive_hex_v1",
    "annotate_path",
    "annotate_paths",
    "tencent_hsv_histogram",
    "colorpipette_inspired",
]
