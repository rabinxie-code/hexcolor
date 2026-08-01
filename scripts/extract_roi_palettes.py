#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hexbench.color import rgb_to_hex
from hexbench.roi_batch import ROI_METHODS, process_image_with_boxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract top-five observed colors for multiple XYXY boxes per image."
    )
    parser.add_argument("manifest", type=Path, help="JSON manifest containing a records list.")
    parser.add_argument("--root", type=Path, help="Root for relative image paths; defaults to manifest directory.")
    parser.add_argument("--method", choices=sorted(ROI_METHODS), default="libimagequant")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--processes", type=int, default=1, help="Parallel images; use physical CPU cores in production.")
    parser.add_argument("--box-workers", type=int, default=1, help="Threads within an image; keep at 1 with many processes.")
    parser.add_argument("--max-samples", type=int, default=2048)
    parser.add_argument("--min-cluster-ratio", type=float, default=0.01)
    return parser.parse_args()


def run_record(item: tuple[str, list[list[int]], str, int, float, int]) -> dict[str, object]:
    path, boxes, method, max_samples, min_ratio, box_workers = item
    palettes, timing = process_image_with_boxes(
        path,
        [tuple(int(value) for value in box) for box in boxes],
        ROI_METHODS[method],
        max_samples=max_samples,
        min_cluster_ratio=min_ratio,
        box_workers=box_workers,
    )
    return {
        "path": path,
        "method": method,
        "boxes": [
            {
                "xyxy": box,
                "rgb": [list(color) for color in result.palette],
                "hex": [rgb_to_hex(color) for color in result.palette],
                "weights": list(result.weights),
            }
            for box, result in zip(boxes, palettes)
        ],
        "timing_ms": {
            "decode": timing.decode_ms,
            "sample": timing.sample_ms,
            "palette": timing.palette_ms,
            "total": timing.total_ms,
        },
    }


def main() -> None:
    args = parse_args()
    if args.processes < 1 or args.box_workers < 1:
        raise SystemExit("--processes and --box-workers must be at least 1")
    if args.max_samples < 1 or not 0.0 <= args.min_cluster_ratio < 1.0:
        raise SystemExit("invalid sampling or cluster ratio")
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = payload["records"] if isinstance(payload, dict) else payload
    root = (args.root or args.manifest.parent).resolve()
    work = [
        (
            str((root / str(record["path"])).resolve()),
            record["bboxes_xyxy"],
            args.method,
            args.max_samples,
            args.min_cluster_ratio,
            args.box_workers,
        )
        for record in records
    ]
    if args.processes == 1:
        results = map(run_record, work)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=args.processes)
        results = executor.map(run_record, work, chunksize=max(1, len(work) // (args.processes * 4)))
    try:
        with args.output.open("w", encoding="utf-8") as output:
            for result in results:
                output.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    finally:
        if executor is not None:
            executor.shutdown()


if __name__ == "__main__":
    main()
