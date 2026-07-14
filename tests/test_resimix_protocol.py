"""Pure-file tests for frozen MoNuSeg-Lite protocol derivation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resimixpms.protocol import ProtocolError, derive_monuseg_lite_protocol  # noqa: E402
from resimixpms.manifests import (  # noqa: E402
    crop_boxes_for_shape,
    load_allowed_image_names,
    load_crop_plan,
    load_crop_records,
    validate_frozen_bundle,
)


class FrozenProtocolTest(unittest.TestCase):
    def _bundle(self, root: Path) -> Path:
        bundle = root / "source"
        bundle.mkdir()
        manifest = {
            "train_records": [{"image_name": "train_01.png"}, {"image_name": "train_02.png"}],
            "holdout_records": [{"image_name": f"holdout_{index}.png"} for index in range(6)],
            "train_crops": [
                {"image_name": "train_01.png", "x": 0, "y": 0, "width": 256, "height": 256},
                {"image_name": "train_02.png", "x": 0, "y": 0, "width": 256, "height": 256},
            ],
        }
        patches = {"evaluation_patches": [
            {"image_name": f"holdout_{index // 2}.png", "x": (index % 2) * 32, "y": 0, "width": 512, "height": 512}
            for index in range(12)
        ]}
        (bundle / "monuseg_lite_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (bundle / "monuseg_lite_patches.json").write_text(json.dumps(patches), encoding="utf-8")
        checksums = []
        for name in ("monuseg_lite_manifest.json", "monuseg_lite_patches.json"):
            checksums.append(f"{hashlib.sha256((bundle / name).read_bytes()).hexdigest()}  {name}")
        (bundle / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
        return bundle

    def test_derivation_uses_only_checked_bundle_selectors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._bundle(root)
            result = derive_monuseg_lite_protocol(
                source,
                {
                    "train_images": "monuseg_lite_manifest.json#/train_records",
                    "development_images": "monuseg_lite_manifest.json#/holdout_records",
                    "train_crops": "monuseg_lite_manifest.json#/train_crops",
                    "evaluation_patches": "monuseg_lite_patches.json#/evaluation_patches",
                },
                root / "derived",
            )
            validate_frozen_bundle(root / "derived" / "raw")
            self.assertEqual(len(load_allowed_image_names(result["train_manifest"])), 2)
            self.assertEqual(len(load_allowed_image_names(result["test_manifest"])), 6)
            self.assertEqual(len(load_crop_records(result["train_crop_manifest"])), 2)
            self.assertEqual(len(load_crop_records(result["eval_crop_manifest"])), 12)
            self.assertTrue(str(result["train_manifest"]).startswith(str(root / "derived")))

    def test_bad_count_or_external_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._bundle(root)
            selectors = {
                "train_images": "monuseg_lite_manifest.json#/train_records",
                "development_images": "monuseg_lite_manifest.json#/holdout_records",
                "train_crops": "monuseg_lite_manifest.json#/train_crops",
                "evaluation_patches": "outside.json#/evaluation_patches",
            }
            with self.assertRaises(ProtocolError):
                derive_monuseg_lite_protocol(source, selectors, root / "bad")

    def test_epoch_crop_indices_are_preserved_as_a_schedule(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "source"
            bundle.mkdir()
            names = ["train_a.tif", "train_b.tif"]
            manifest = {
                "train_files": names,
                "holdout_files": [f"holdout_{index}.tif" for index in range(6)],
                "max_train_crops_per_image": 4,
                "crop_indices": {
                    str(epoch): {
                        name: ([0, 1, 2, 3] if epoch == 0 else [3, 2, 1, 0]) for name in names
                    }
                    for epoch in range(10)
                },
            }
            patches = {
                "patches": [
                    {
                        "filename": f"holdout_{index // 2}.tif",
                        "x": (index % 2) * 256,
                        "y": 0,
                        "width": 256,
                        "height": 256,
                    }
                    for index in range(12)
                ]
            }
            for name, payload in (
                ("monuseg_lite_manifest.json", manifest),
                ("monuseg_lite_patches.json", patches),
            ):
                (bundle / name).write_text(json.dumps(payload), encoding="utf-8")
            checksums = [
                f"{hashlib.sha256((bundle / name).read_bytes()).hexdigest()}  {name}"
                for name in ("monuseg_lite_manifest.json", "monuseg_lite_patches.json")
            ]
            (bundle / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")

            result = derive_monuseg_lite_protocol(
                bundle,
                {
                    "train_images": "monuseg_lite_manifest.json#/train_files",
                    "development_images": "monuseg_lite_manifest.json#/holdout_files",
                    "train_crops": "monuseg_lite_manifest.json#/crop_indices",
                    "evaluation_patches": "monuseg_lite_patches.json#/patches",
                },
                root / "derived",
            )
            records, schedule = load_crop_plan(
                result["train_crop_manifest"],
                allowed_image_names=names,
                expected_crop_size=256,
                expected_overlap=92,
                expected_load="unclockwise",
                expected_epochs=10,
            )
            self.assertEqual(records, [])
            self.assertIsNotNone(schedule)
            self.assertEqual(schedule.indices_for("train_a.tif", epoch=0), (0, 1, 2, 3))
            self.assertEqual(schedule.indices_for("train_a", epoch=0), (0, 1, 2, 3))
            self.assertEqual(schedule.indices_for("train_a.tif", epoch=1), (3, 2, 1, 0))
            self.assertEqual(schedule.indices_for("train_a.tif", union=True), (0, 1, 2, 3))
            grid = crop_boxes_for_shape((512, 512), 256, 92, "unclockwise")
            self.assertEqual(
                schedule.select_boxes("train_a.tif", grid, epoch=1),
                [grid[3], grid[2], grid[1], grid[0]],
            )
            self.assertEqual(len(load_crop_records(result["eval_crop_manifest"])), 12)


if __name__ == "__main__":
    unittest.main()
