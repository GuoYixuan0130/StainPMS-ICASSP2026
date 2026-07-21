import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from run.dataset.manifest import load_dataset_manifest
from tools.freeze_tnbc_smoke_manifest import build_manifest


class TnbcSmokeManifestTests(unittest.TestCase):
    def _args(self, root: Path, source: Path, output: Path, allowed=(1,)):
        return argparse.Namespace(
            source_manifest=str(source),
            image_root=str(root / "images"),
            label_root=str(root / "labels"),
            output=str(output),
            allowed_patients=list(allowed),
            expected_count=1,
            protocol_id="synthetic_tnbc_smoke_v1",
        )

    def test_freezes_hash_verified_loader_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            (root / "labels").mkdir()
            (root / "images" / "01_1.png").write_bytes(b"image")
            (root / "labels" / "01_1.mat").write_bytes(b"label")
            source = root / "safe_source.json"
            source.write_text(json.dumps({"samples": ["01_1"]}), encoding="utf-8")
            output = root / "frozen.json"
            report = build_manifest(self._args(root, source, output))
            output.write_text(json.dumps(report), encoding="utf-8")
            manifest, records = load_dataset_manifest(
                output, expected_dataset="tnbc", verify_hashes=True
            )
            self.assertEqual(manifest["record_count"], 1)
            self.assertEqual(records[0]["sample_id"], "01_1")
            self.assertEqual(
                records[0]["image_sha256"], hashlib.sha256(b"image").hexdigest()
            )

    def test_closed_patient_rejected_before_file_existence_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "unsafe_source.json"
            source.write_text(json.dumps({"samples": ["09_1"]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "closed TNBC patient 9"):
                build_manifest(self._args(root, source, root / "unused.json", allowed=(1, 2, 3, 4, 5, 6)))


if __name__ == "__main__":
    unittest.main()
