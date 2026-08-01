#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hexbench.benchmark import BENCHMARK_METHOD_ORDER
from hexbench.browser_export import export_browser_samples
from hexbench.dataset import CropRecord


if __name__ == "__main__":
    crop_records = json.loads(Path("data/benchmark_crops_10k/manifest.json").read_text(encoding="utf-8"))
    crops = [CropRecord(**record) for record in crop_records]
    result = export_browser_samples(
        crops,
        checkpoint_dir="data/results/checkpoints_10k",
        site_dir="site",
        method_order=BENCHMARK_METHOD_ORDER,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
