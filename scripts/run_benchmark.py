#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hexbench.benchmark import run_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the HEX extraction benchmark")
    parser.add_argument("--target-crops", type=int, default=10_000)
    parser.add_argument("--display-crops", type=int, default=60)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--source-dir", default="data/samples/s3_10k")
    parser.add_argument("--manifest", default="data/sample_manifest_10k.json")
    parser.add_argument("--crop-dir", default="data/benchmark_crops_10k")
    parser.add_argument("--checkpoint-dir", default="data/results/checkpoints_10k")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_benchmark(
        source_dir=args.source_dir,
        source_manifest_path=args.manifest,
        crop_dir=args.crop_dir,
        checkpoint_dir=args.checkpoint_dir,
        target_crops=args.target_crops,
        display_crops=args.display_crops,
        workers=args.workers,
        resume=not args.no_resume,
    )
    print(json.dumps({"scope": result["scope"], "summaries": result["summaries"]}, ensure_ascii=False, indent=2))
