"""Small synthetic-array tests for the ResiMix frozen audit boundary."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_TEST_WORKTREE = Path(__file__).resolve().parents[1]

from resimixpms.donor import (  # noqa: E402
    DataIsolationError,
    TrainingSample,
    audit_training_samples,
    rgb_to_od,
    write_donor_bank,
)
from resimixpms.experiment import sha256_file  # noqa: E402
from resimixpms.runtime import ResiMixAugmentor  # noqa: E402
from resimixpms.transplant import CONTEXT_FEATURE_NAMES  # noqa: E402
from resimixpms.manifests import (  # noqa: E402
    ManifestPreflightError,
    load_allowed_image_names,
    load_crop_records,
    validate_frozen_bundle,
    validate_manifest_patient_isolation,
)


class FrozenManifestTests(unittest.TestCase):
    def _write_bundle(self, directory: Path) -> None:
        manifest = directory / "monuseg_lite_manifest.json"
        patches = directory / "monuseg_lite_patches.json"
        manifest.write_text(json.dumps({"images": [{"image": "train_a"}]}), encoding="utf-8")
        patches.write_text(
            json.dumps(
                {
                    "patches": [
                        {"name": "train_a", "x": 0, "y": 2, "width": 8, "height": 8},
                        {"image_name": "train_b", "x1": 4, "y1": 6, "x2": 12, "y2": 14},
                    ]
                }
            ),
            encoding="utf-8",
        )
        lines = []
        for filename in (manifest.name, patches.name):
            digest = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
            lines.append("{}  {}".format(digest, filename))
        (directory / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_frozen_bundle_and_flexible_manifest_loaders(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self._write_bundle(directory)
            bundle = validate_frozen_bundle(directory)
            self.assertEqual(set(bundle.file_sha256), {"monuseg_lite_manifest.json", "monuseg_lite_patches.json"})
            self.assertEqual(load_allowed_image_names(directory / "monuseg_lite_manifest.json"), ["train_a"])
            self.assertEqual(
                load_crop_records(directory / "monuseg_lite_patches.json"),
                [
                    {"image_name": "train_a", "x": 0, "y": 2, "width": 8, "height": 8},
                    {"image_name": "train_b", "x": 4, "y": 6, "width": 8, "height": 8},
                ],
            )
            (directory / "monuseg_lite_manifest.json").write_text("[]", encoding="utf-8")
            with self.assertRaises(ManifestPreflightError):
                validate_frozen_bundle(directory)

    def test_patient_isolation_fails_closed_before_image_io(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "tnbc.json"
            manifest.write_text(
                json.dumps({"records": [{"image_name": "p1_img"}, {"image_name": "p9_img"}]}),
                encoding="utf-8",
            )
            with self.assertRaises(ManifestPreflightError):
                validate_manifest_patient_isolation(
                    manifest,
                    allowed_patient_ids=range(1, 7),
                    forbidden_patient_ids={9, 10, 11},
                    name_to_patient={"p1_img": 1, "p9_img": 9},
                )
            manifest.write_text(json.dumps({"records": [{"image_name": "p1_img"}]}), encoding="utf-8")
            records = validate_manifest_patient_isolation(
                manifest,
                allowed_patient_ids=range(1, 7),
                forbidden_patient_ids={9, 10, 11},
                name_to_patient={"p1_img": 1},
            )
            self.assertEqual(records[0]["patient_id"], 1)


class DonorAuditTests(unittest.TestCase):
    @staticmethod
    def _synthetic_sample(split: str = "train") -> TrainingSample:
        instance_map = np.zeros((24, 24), dtype=np.int32)
        # Three interior hard instances and one boundary instance rejected by the donor filter.
        instance_map[5:8, 5:8] = 1
        instance_map[5:8, 12:15] = 2
        instance_map[12:15, 5:8] = 3
        instance_map[0:3, 12:15] = 4
        predictions = np.zeros((2, 24, 24), dtype=bool)
        predictions[0, 5:7, 12:14] = True  # 4/9: IoU-cliff donor.
        predictions[1, 12:15, 5:7] = True  # 6/9: matched low-quality donor.
        rgb = np.full((24, 24, 3), 128, dtype=np.uint8)
        return TrainingSample(
            source_id="tnbc_patient_1_train_a",
            dataset="tnbc",
            split=split,
            instance_map=instance_map,
            prediction_masks=predictions,
            coverage_map=np.zeros((24, 24), dtype=np.uint8),
            rgb=rgb,
            od=rgb_to_od(rgb),
            patient_id=1,
            source_metadata={"split": split},
        )

    def test_audit_classifies_filters_and_writes_training_only_bank(self) -> None:
        bank = audit_training_samples([self._synthetic_sample()])
        records = {record.instance_id: record for record in bank.audits}
        self.assertEqual(records[1].donor_class, "Missed")
        self.assertEqual(records[2].donor_class, "IoU-Cliff")
        self.assertEqual(records[3].donor_class, "Low-Quality Matched")
        self.assertTrue(records[1].eligible)
        self.assertTrue(records[2].eligible)
        self.assertTrue(records[3].eligible)
        self.assertFalse(records[4].eligible)
        self.assertIn("touches_image_boundary", records[4].rejection_reasons)
        self.assertEqual(len(bank.donors), 3)
        self.assertFalse(records[1].covered)
        self.assertGreater(records[1].hardness, records[3].hardness)
        self.assertTrue(records[1].mask.any())
        self.assertTrue(records[1].annulus_mask.any())
        self.assertTrue(np.isfinite(records[1].rgb_patch).all())
        self.assertTrue(np.isfinite(records[1].od_patch).all())

        with tempfile.TemporaryDirectory() as temporary:
            csv_path, summary_path = write_donor_bank(bank, temporary)
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertTrue(summary["training_only"])
            self.assertEqual(summary["donor_class_counts"]["Missed"], 1)
            self.assertTrue((Path(temporary) / rows[0]["payload_path"]).is_file())
            stats_path = Path(temporary) / "host_context_statistics.json"
            stats_path.write_text(json.dumps({
                "context_mean": [0.0] * len(CONTEXT_FEATURE_NAMES),
                "context_std": [1.0] * len(CONTEXT_FEATURE_NAMES),
                "natural_boundary_gradient_p95": 1.0e6,
                "legal_context_distance_p95": 1.0e6,
            }), encoding="utf-8")
            config_path = Path(temporary) / "resimix_config.json"
            config_path.write_text(json.dumps({
                "seed": 3407, "augmentation_probability": 0.5,
                "active_start_epoch": 2, "active_end_epoch": 9, "dataset": "tnbc",
                "donor_bank_manifest": str(csv_path),
                "donor_bank_manifest_sha256": sha256_file(csv_path),
                "donor_payload_dir": str(Path(temporary) / "donor_payloads"),
                "host_context_statistics": str(stats_path),
                "host_context_statistics_sha256": sha256_file(stats_path),
            }), encoding="utf-8")
            runtime = ResiMixAugmentor(config_path)
            self.assertEqual(sum(map(len, runtime.donors_by_category.values())), 3)

    def test_development_source_is_rejected_before_array_processing(self) -> None:
        with self.assertRaises(DataIsolationError):
            audit_training_samples([self._synthetic_sample(split="development")])


if __name__ == "__main__":
    unittest.main()
