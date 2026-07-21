import unittest
from argparse import Namespace
from pathlib import Path

from tools.freeze_monuseg_train_manifest import build_manifest


class FreezeMonusegTrainManifestTests(unittest.TestCase):
    def test_train_tree_is_hashed_in_stable_order(self):
        root = Path(__file__).parent / "fixtures" / "monuseg_manifest"
        report = build_manifest(
            Namespace(
                image_root=str(root / "images"),
                label_root=str(root / "labels"),
                expected_count=2,
                protocol_id="test",
            )
        )
        self.assertEqual([row["sample_id"] for row in report["records"]], ["A", "B"])

    def test_test_path_is_rejected_before_enumeration(self):
        with self.assertRaises(ValueError):
            build_manifest(
                Namespace(
                    image_root="/sealed/test14/images",
                    label_root="/sealed/test14/labels",
                    expected_count=0,
                    protocol_id="test",
                )
            )


if __name__ == "__main__":
    unittest.main()
