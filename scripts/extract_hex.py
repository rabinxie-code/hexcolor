#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hexbench.batch import annotate_paths
from hexbench.methods import METHODS


IMAGE_SUFFIXES = {".avif", ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def expand_inputs(inputs: Iterable[str]) -> list[Path]:
    images: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            images.extend(candidate for candidate in path.rglob("*") if candidate.suffix.lower() in IMAGE_SUFFIXES)
        elif path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
        else:
            raise ValueError(f"Unsupported image input: {path}")
    unique = {path.resolve(): path for path in images}
    return [unique[key] for key in sorted(unique, key=lambda item: item.as_posix())]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract compact crop-level RGB24/HEX annotations.")
    parser.add_argument("inputs", nargs="+", help="Image files or directories (directories are searched recursively).")
    parser.add_argument("--method", choices=sorted(METHODS), default="adaptive_v1")
    parser.add_argument("--workers", type=int, default=1, help="Local correctness/pilot workers; default: 1.")
    parser.add_argument("--full", action="store_true", help="Include palettes, source metadata and diagnostics.")
    parser.add_argument("--output", type=Path, help="Write JSONL here; stdout when omitted.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths = expand_inputs(args.inputs)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not paths:
        raise SystemExit("No supported images found.")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    output = args.output.open("w", encoding="utf-8") if args.output else sys.stdout
    try:
        for record in annotate_paths(paths, args.method, workers=args.workers, full=args.full):
            output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    finally:
        if args.output:
            output.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
