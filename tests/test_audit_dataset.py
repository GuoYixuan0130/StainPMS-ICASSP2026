import unittest

import numpy as np

from tools.audit_dataset import (
    _analyze_label,
    _audit_split,
    _audit_split_isolation,
    _compare_raw_binary,
    _extract_manifest_entries,
    _normalize_entry,
)


class DatasetAuditTests(unittest.TestCase):
    def test_label_audit_and_watershed_comparison(self):
        prepared = np.zeros((8, 8), dtype=np.int32)
        prepared[2:6, 2:4] = 1
        prepared[2:6, 4:6] = 2
        raw = (prepared > 0).astype(np.uint8) * 255

        label_audit, labels = _analyze_label(prepared)
        self.assertEqual(label_audit["instance_count"], 2)
        self.assertTrue(label_audit["contiguous_positive_ids"])
        comparison = _compare_raw_binary(raw, labels)
        self.assertTrue(comparison["foreground_equal"])
        self.assertEqual(comparison["instance_count_delta"], 1)
        self.assertEqual(comparison["raw_components_split_by_preparation"], 1)

    def test_closed_tnbc_patient_is_rejected_before_file_access(self):
        config = {"dataset": "tnbc", "roots": {}}
        split = _audit_split(
            config,
            "train",
            {
                "allowed_patient_ids": [1, 2, 3, 4, 5, 6],
                "samples": [{"image": "09_1.png", "patient": 9}],
            },
        )
        self.assertEqual(split["status"], "protocol_violation")
        self.assertEqual(split["samples"], [])
        self.assertIn("closed", split["blockers"][0])

    def test_monuseg_official_test_path_is_rejected(self):
        config = {
            "dataset": "monuseg",
            "roots": {
                "image_root": "/data/monuseg/test/images",
                "label_root": "/data/monuseg/test/labels",
            },
        }
        split = _audit_split(
            config,
            "internal_pool",
            {"samples": ["TCGA-AA-0001-01Z.png"]},
        )
        self.assertEqual(split["status"], "protocol_violation")
        self.assertIn("official-test", split["blockers"][0])

    def test_safe_manifest_schema_is_supported(self):
        payload = {
            "records": [
                {"image_name": "01_1", "patient_number": "1"},
            ]
        }
        rows = _extract_manifest_entries(payload)
        entry = _normalize_entry(
            rows[0],
            dataset="tnbc",
            split_name="train",
            roots={
                "image_root": "/images",
                "image_extension": ".png",
                "label_root": "/labels",
            },
            metadata_map={},
        )
        self.assertEqual(entry["patient"], "1")
        self.assertTrue(entry["image_path"].replace("\\", "/").endswith("/images/01_1.png"))
        self.assertTrue(entry["label_path"].replace("\\", "/").endswith("/labels/01_1.mat"))

    def test_cross_split_patient_and_content_overlap_is_rejected(self):
        reports = {
            "train": {
                "samples": [
                    {
                        "sample_id": "a",
                        "patient": 1,
                        "case": None,
                        "image_sha256": "same",
                        "label_sha256": "left",
                    }
                ]
            },
            "development": {
                "samples": [
                    {
                        "sample_id": "b",
                        "patient": 1,
                        "case": None,
                        "image_sha256": "same",
                        "label_sha256": "right",
                    }
                ]
            },
        }
        isolation = _audit_split_isolation("tnbc", reports)
        self.assertEqual(isolation["status"], "protocol_violation")
        self.assertEqual(isolation["pairs"][0]["violations"]["patients"], ["1"])
        self.assertEqual(
            isolation["pairs"][0]["violations"]["image_sha256"], ["same"]
        )


if __name__ == "__main__":
    unittest.main()
