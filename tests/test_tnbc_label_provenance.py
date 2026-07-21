import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat
from skimage import io

from tools.audit_tnbc_label_provenance import ProtocolViolation, audit


class TnbcLabelProvenanceTests(unittest.TestCase):
    def _args(self, root: Path, source: Path, raw_root: Path):
        return argparse.Namespace(
            source_manifest=[str(source)],
            image_root=str(root / "images"),
            prepared_label_root=str(root / "labels"),
            raw_root=[str(raw_root)],
            expected_count=1,
            watershed_min_distance=10,
            watershed_sigma=1.0,
        )

    def test_explicit_raw_path_compares_without_directory_walk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            (root / "labels").mkdir()
            (root / "raw" / "GT_01").mkdir(parents=True)
            (root / "images" / "01_1.png").write_bytes(b"image")
            prepared = np.zeros((8, 8), dtype=np.int32)
            prepared[2:6, 2:6] = 1
            savemat(root / "labels" / "01_1.mat", {"inst_map": prepared})
            raw = (prepared > 0).astype(np.uint8) * 255
            io.imsave(root / "raw" / "GT_01" / "1.png", raw, check_contrast=False)
            source = root / "source.json"
            source.write_text(json.dumps({"samples": ["01_1"]}), encoding="utf-8")
            report = audit(self._args(root, source, root / "raw"))
            row = report["samples"][0]
            self.assertEqual(report["status"], "complete")
            self.assertTrue(row["raw_vs_prepared"]["foreground_equal"])
            self.assertEqual(row["raw_vs_prepared"]["raw_binary_components_8"], 1)
            self.assertEqual(row["prepared"]["dtype"], "int32")

    def test_closed_patient_is_rejected_before_image_label_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "closed.json"
            source.write_text(json.dumps({"samples": ["09_1"]}), encoding="utf-8")
            with self.assertRaisesRegex(ProtocolViolation, "closed TNBC patient 9"):
                audit(self._args(root, source, root / "raw"))


if __name__ == "__main__":
    unittest.main()
