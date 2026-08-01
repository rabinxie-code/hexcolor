#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one explicitly scoped ColorPipette original audit")
    parser.add_argument("--repo", default="/tmp/hex-colorpipette-source", help="Patched ColorPipette clone")
    parser.add_argument("--image", default="data/synthetic/mild_texture.png")
    parser.add_argument("--output", default="data/results/colorpipette_original.json")
    parser.add_argument("--palette-size", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    flask_root = repo / "src" / "flask"
    image_path = Path(args.image).resolve()
    output_path = Path(args.output).resolve()
    if not flask_root.is_dir():
        raise SystemExit(f"ColorPipette flask directory not found: {flask_root}")
    if not image_path.is_file():
        raise SystemExit(f"Audit image not found: {image_path}")

    # The upstream endpoint only uses src_img for names ending in png.
    audit_name = "hexbench_original_audit.png"
    audit_path = flask_root / "src_img" / audit_name
    with Image.open(image_path) as opened:
        opened.convert("RGB").save(audit_path, format="PNG")

    os.chdir(flask_root)
    sys.path.insert(0, str(flask_root))
    import torch

    torch.set_num_threads(min(16, os.cpu_count() or 1))
    try:
        import flask_cors  # noqa: F401
    except ImportError:
        flask_cors_stub = types.ModuleType("flask_cors")
        flask_cors_stub.CORS = lambda *unused_args, **unused_kwargs: None
        sys.modules["flask_cors"] = flask_cors_stub
    import app as upstream

    load_started = time.perf_counter()
    upstream.nets_init()
    load_seconds = time.perf_counter() - load_started

    request_path = f"/get/color_open?number={args.palette_size}&bcg_flag=false&img_name={audit_name}"
    def infer_once() -> tuple[dict[str, object], float]:
        inference_started = time.perf_counter()
        with upstream.app.test_request_context(request_path):
            response = upstream.generate_color_open()
            response_payload = response.get_json()
        return response_payload, time.perf_counter() - inference_started

    first_payload, first_inference_seconds = infer_once()
    response_payload, inference_seconds = infer_once()

    weights = {
        "basnet_bytes": (flask_root / "saliency" / "saved_models" / "basnet_best_train_gdi.pth").stat().st_size,
        "spixelnet_bytes": (flask_root / "spixel" / "pretrain_ckpt" / "SpixelNet_bsd_ckpt.tar").stat().st_size,
    }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "upstream": "https://github.com/Shuqi-67/ColorPipette",
        "scope": "one 320x320 S3 crop, CPU, cold model load + first inference + one warm repeat",
        "compatibility_changes": [
            "use scikit-image's current compiled label-connectivity routine",
            "avoid redundant torchvision ResNet download because BASNet checkpoint is complete",
            "set weights_only=False for legacy trusted upstream checkpoints",
            "stub unused Flask-CORS wiring in the isolated direct audit",
        ],
        "image": str(image_path),
        "model_load_seconds": round(load_seconds, 6),
        "first_inference_seconds": round(first_inference_seconds, 6),
        "inference_seconds": round(inference_seconds, 6),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 3),
        "weights": weights,
        "result": response_payload,
        "repeat_identical": first_payload == response_payload,
        "runtime_warnings": [
            "upstream NumPy uint8 overflow in app.py saliency averaging",
            "upstream NumPy uint8 underflow/overflow in har_colors.py Lab-to-LCh conversion",
        ],
        "production_projection_warning": "One-image CPU audit only; not used to claim distributed throughput.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
