from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class CropRecord:
    crop_id: str
    path: str
    source_file: str
    batch: str
    variant: str
    width: int
    height: int
    jpeg_path: str
    resize_path: str


def materialize_s3_crops(
    source_dir: str | Path = "data/samples/s3",
    output_dir: str | Path = "site/assets/crops",
    crop_size: int = 320,
    source_manifest_path: str | Path | None = None,
    target_crops: int | None = None,
) -> list[CropRecord]:
    source_root = Path(source_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[CropRecord] = []
    anchors = {
        "top_left": (0.08, 0.08),
        "center": (0.50, 0.50),
        "bottom_right": (0.92, 0.92),
    }
    if source_manifest_path is not None:
        manifest = json.loads(Path(source_manifest_path).read_text(encoding="utf-8"))
        source_items = [
            {
                "path": source_root / str(item["file"]),
                "batch": str(item.get("batch", "unknown")),
                "variants": [str(item["variant"])] if item.get("variant") else list(anchors),
            }
            for item in manifest.get("samples", [])
        ]
    else:
        source_items = [
            {
                "path": source_path,
                "batch": source_path.stem.split("_", 1)[0].removeprefix("b"),
                "variants": list(anchors),
            }
            for source_path in sorted(source_root.glob("*.jpg"))
        ]

    for source_item in source_items:
        source_path = Path(source_item["path"])
        batch = str(source_item["batch"])
        if not source_path.exists():
            continue
        source: Image.Image | None = None
        for variant in source_item["variants"]:
            if variant not in anchors:
                raise ValueError(f"Unknown crop variant {variant!r} for {source_path}")
            anchor_x, anchor_y = anchors[variant]
            crop_id = f"{source_path.stem}__{variant}"
            output_path = output_root / f"{crop_id}.jpg"
            jpeg_path = output_root / f"{crop_id}__jpeg60.jpg"
            resize_path = output_root / f"{crop_id}__resize50.png"

            if not output_path.exists() or not jpeg_path.exists() or not resize_path.exists():
                if source is None:
                    try:
                        with Image.open(source_path) as opened:
                            source = opened.convert("RGB")
                    except (OSError, ValueError):
                        break
                width, height = source.size
                side = max(64, int(round(min(width, height) * 0.48)))
                center_x = anchor_x * width
                center_y = anchor_y * height
                left = int(round(np.clip(center_x - side / 2, 0, width - side)))
                top = int(round(np.clip(center_y - side / 2, 0, height - side)))
                crop = source.crop((left, top, left + side, top + side))
                crop = crop.resize((crop_size, crop_size), Image.Resampling.LANCZOS)
                crop.save(output_path, format="JPEG", quality=92, optimize=True)

                # Stability variants are materialized once so every method and
                # decoder sees byte-identical inputs during the large run.
                with Image.open(output_path) as opened:
                    base = opened.convert("RGB")
                base.save(jpeg_path, format="JPEG", quality=60, optimize=False)
                reduced = base.resize(
                    (max(1, crop_size // 2), max(1, crop_size // 2)),
                    Image.Resampling.BILINEAR,
                )
                reduced.resize((crop_size, crop_size), Image.Resampling.BILINEAR).save(
                    resize_path,
                    format="PNG",
                )

            if not output_path.exists() or not jpeg_path.exists() or not resize_path.exists():
                continue
            records.append(
                CropRecord(
                    crop_id=crop_id,
                    path=str(output_path),
                    source_file=source_path.name,
                    batch=batch,
                    variant=variant,
                    width=crop_size,
                    height=crop_size,
                    jpeg_path=str(jpeg_path),
                    resize_path=str(resize_path),
                )
            )
            if len(records) % 500 == 0:
                print(f"materialize_crops: {len(records)}/{target_crops or '?'}", flush=True)
            if target_crops is not None and len(records) >= target_crops:
                break
        if target_crops is not None and len(records) >= target_crops:
            break
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps([asdict(record) for record in records], indent=2), encoding="utf-8")
    return records


def materialize_synthetic_cases(output_dir: str | Path = "data/synthetic", size: int = 256) -> list[dict[str, object]]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, object]] = []

    def save(name: str, image: Image.Image, expected_hex: str | None, expected_route: str | None) -> None:
        path = output_root / f"{name}.png"
        image.save(path)
        cases.append(
            {
                "case_id": name,
                "path": str(path),
                "expected_hex": expected_hex,
                "expected_route": expected_route,
            }
        )

    save("solid", Image.new("RGB", (size, size), "#D24A43"), "#D24A43", "flat")

    rng = np.random.default_rng(20260731)
    base = np.array([46, 126, 171], dtype=np.int16)
    mild = np.clip(base + rng.normal(0, 3.0, (size, size, 3)), 0, 255).astype(np.uint8)
    save("mild_texture", Image.fromarray(mild, "RGB"), "#2E7EAB", "mild")

    start = np.array([25, 71, 164], dtype=np.float64)
    end = np.array([241, 176, 55], dtype=np.float64)
    ramp = np.linspace(0.0, 1.0, size, dtype=np.float64)[None, :, None]
    gradient = np.broadcast_to(start * (1.0 - ramp) + end * ramp, (size, size, 3)).astype(np.uint8)
    save("linear_gradient", Image.fromarray(gradient, "RGB"), "#857B6D", "gradient")

    checker = np.empty((size, size, 3), dtype=np.uint8)
    colors = np.array([[24, 63, 92], [229, 111, 65], [238, 202, 104], [49, 132, 109]], dtype=np.uint8)
    tile = 16
    for row in range(size):
        for column in range(size):
            checker[row, column] = colors[((row // tile) + (column // tile)) % len(colors)]
    save("four_color_texture", Image.fromarray(checker, "RGB"), None, "texture")

    alpha = Image.new("RGBA", (size, size), (255, 0, 255, 0))
    draw = ImageDraw.Draw(alpha)
    draw.rectangle((32, 32, size - 33, size - 33), fill=(92, 211, 142, 255))
    save("transparent_fill", alpha, "#5CD38E", "flat")

    gray = np.linspace(28, 222, size, dtype=np.uint8)
    gray_gradient = np.broadcast_to(gray[None, :, None], (size, size, 3)).copy()
    save("gray_gradient", Image.fromarray(gray_gradient, "RGB"), "#7D7D7D", "gradient")

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    return cases
