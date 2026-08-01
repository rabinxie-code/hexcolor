from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from hexbench.batch import annotate_path, annotate_paths
from hexbench.benchmark import build_capacity_scenario, capacity_target, source_pipeline_capacity
from hexbench.color import delta_e_ok, rgb24, rgb_to_hex
from hexbench.dataset import materialize_s3_crops
from hexbench.methods import (
    adaptive_hex_v1,
    colorpipette_inspired,
    octree_quantization,
    pixelero_rgb_histogram,
    pngquant_libimagequant,
    tencent_hsv_histogram,
)


class ColorUtilityTests(unittest.TestCase):
    def test_rgb_encoding(self) -> None:
        self.assertEqual(rgb_to_hex((18, 52, 86)), "#123456")
        self.assertEqual(rgb24((18, 52, 86)), 0x123456)
        self.assertAlmostEqual(float(delta_e_ok(np.array([12, 34, 56]), np.array([12, 34, 56]))), 0.0)

    def test_three_day_capacity_target(self) -> None:
        required, provisioned = capacity_target(100_000_000_000, 3, 1.5)
        self.assertAlmostEqual(required, 385_802.4691358025)
        self.assertAlmostEqual(provisioned, 578_703.7037037037)

    def test_decode_once_capacity_model(self) -> None:
        model = source_pipeline_capacity(100_000_000_000, 5_000_000_000, 3, 1.0, 0.5, 0.5, 1.5)
        self.assertAlmostEqual(model["images_s"], 19_290.123456790123)
        self.assertAlmostEqual(model["source_gpix_s"], 19.290123456790123)
        self.assertAlmostEqual(model["compressed_input_gb_s"], 9.645061728395062)
        self.assertAlmostEqual(model["decoded_rgb_gb_s"], 57.87037037037037)
        self.assertEqual(model["estimated_fused_workers_1_5x"], 58)

    def test_three_seven_fifteen_day_scenarios(self) -> None:
        crop_rates = {"adaptive_v1": 105.089, "colorpipette_original": 1.913887}
        scenarios = [build_capacity_scenario(days, crop_rates) for days in (3, 7, 15)]
        self.assertEqual([scenario["days"] for scenario in scenarios], [3, 7, 15])
        self.assertEqual([scenario["decode_once_model"]["estimated_fused_workers_1_5x"] for scenario in scenarios], [58, 25, 12])
        self.assertEqual(scenarios[0]["cpu_equivalent_processes_1_5x"]["adaptive_v1"], 5507)
        self.assertEqual(scenarios[2]["cpu_equivalent_processes_1_5x"]["colorpipette_original"], 60475)


class MethodTests(unittest.TestCase):
    def test_manifest_sampling_uses_one_variant_per_source_and_exact_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "crops"
            source.mkdir()
            samples = []
            variants = ["top_left", "center", "bottom_right"]
            for index, variant in enumerate(variants):
                filename = f"b{index}_{1000 + index}.jpg"
                Image.new("RGB", (480, 360), (30 + index, 80, 140)).save(source / filename)
                samples.append({"file": filename, "batch": index, "variant": variant})
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"samples": samples}), encoding="utf-8")

            records = materialize_s3_crops(
                source_dir=source,
                output_dir=output,
                source_manifest_path=manifest,
                target_crops=2,
            )

            self.assertEqual(len(records), 2)
            self.assertEqual([record.variant for record in records], variants[:2])
            self.assertEqual(len({record.source_file for record in records}), 2)
            for record in records:
                self.assertTrue(Path(record.path).exists())
                self.assertTrue(Path(record.jpeg_path).exists())
                self.assertTrue(Path(record.resize_path).exists())

    def test_solid_color_is_exact(self) -> None:
        image = Image.new("RGB", (96, 96), (18, 52, 86))
        for method in (
            adaptive_hex_v1,
            tencent_hsv_histogram,
            pixelero_rgb_histogram,
            octree_quantization,
            pngquant_libimagequant,
            colorpipette_inspired,
        ):
            with self.subTest(method=method.__name__):
                result = method(image, "solid")
                self.assertEqual(result.primary, (18, 52, 86))

    def test_adaptive_routes_linear_gradient(self) -> None:
        size = 192
        start = np.array([20, 50, 170], dtype=np.float64)
        end = np.array([240, 180, 40], dtype=np.float64)
        ramp = np.linspace(0, 1, size)[None, :, None]
        pixels = np.broadcast_to(start * (1 - ramp) + end * ramp, (size, size, 3)).astype(np.uint8)
        result = adaptive_hex_v1(Image.fromarray(pixels), "gradient")
        self.assertEqual(result.route, "gradient")
        self.assertGreaterEqual(len(result.palette), 3)
        self.assertTrue(result.observed)

    def test_adaptive_ignores_fully_transparent_rgb(self) -> None:
        image = Image.new("RGBA", (128, 128), (255, 0, 255, 0))
        draw = ImageDraw.Draw(image)
        draw.rectangle((24, 24, 103, 103), fill=(92, 211, 142, 255))
        result = adaptive_hex_v1(image, "alpha")
        self.assertEqual(result.primary, (92, 211, 142))
        self.assertEqual(result.route, "flat")

    def test_tencent_has_achromatic_route(self) -> None:
        image = Image.new("RGB", (128, 128), (128, 128, 128))
        result = tencent_hsv_histogram(image, "gray")
        self.assertEqual(result.route, "achromatic_histogram")
        self.assertEqual(result.primary, (128, 128, 128))

    def test_native_quantizers_ignore_fully_transparent_rgb(self) -> None:
        image = Image.new("RGBA", (96, 96), (255, 0, 255, 0))
        draw = ImageDraw.Draw(image)
        draw.rectangle((12, 12, 83, 83), fill=(92, 211, 142, 255))
        for method in (octree_quantization, pngquant_libimagequant):
            with self.subTest(method=method.__name__):
                self.assertEqual(method(image, "alpha").primary, (92, 211, 142))

    def test_pixelero_histogram_is_deterministic(self) -> None:
        rng = np.random.default_rng(7)
        image = Image.fromarray(rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8))
        first = pixelero_rgb_histogram(image, "first")
        second = pixelero_rgb_histogram(image, "second")
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_adaptive_is_deterministic(self) -> None:
        rng = np.random.default_rng(42)
        pixels = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        first = adaptive_hex_v1(image, "same-id")
        second = adaptive_hex_v1(image, "same-id")
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_colorpipette_proxy_deduplicates_flat_palette(self) -> None:
        image = Image.new("RGB", (96, 96), (10, 90, 150))
        result = colorpipette_inspired(image, "flat")
        self.assertEqual(result.palette, ((10, 90, 150),))
        self.assertFalse(result.observed)

    def test_batch_annotation_compact_schema_and_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "a.png"
            second = Path(directory) / "b.png"
            Image.new("RGB", (16, 16), (26, 43, 60)).save(first)
            Image.new("RGB", (16, 16), (200, 10, 30)).save(second)

            record = annotate_path(first)
            self.assertEqual(record["hex"], "#1A2B3C")
            self.assertEqual(record["rgb24"], 0x1A2B3C)
            self.assertEqual(record["method_id"], 1)
            self.assertNotIn("diagnostics", record)

            records = list(annotate_paths([first, second]))
            self.assertEqual([Path(item["id"]).name for item in records], ["a.png", "b.png"])


if __name__ == "__main__":
    unittest.main()
