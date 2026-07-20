import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from run.dataset.manifest import ManifestError, load_dataset_manifest


class ManifestLoaderTests(unittest.TestCase):
    def _fixture(self, root: Path):
        image = root / "sample.png"
        label = root / "sample.mat"
        image.write_bytes(b"image")
        label.write_bytes(b"label")
        payload = {
            "schema_version": 1,
            "dataset": "monuseg",
            "protocol_id": "synthetic_train_v1",
            "record_count": 1,
            "records": [
                {
                    "sample_id": "TCGA-AA-0001-01Z-00-DX1",
                    "image_path": image.name,
                    "image_sha256": hashlib.sha256(b"image").hexdigest(),
                    "label_path": label.name,
                    "label_sha256": hashlib.sha256(b"label").hexdigest(),
                }
            ],
        }
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        return manifest

    def test_ordered_manifest_with_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._fixture(Path(tmp))
            payload, records = load_dataset_manifest(
                manifest,
                expected_dataset="monuseg",
                verify_hashes=True,
            )
            self.assertEqual(payload["protocol_id"], "synthetic_train_v1")
            self.assertEqual(records[0]["manifest_index"], 0)
            self.assertEqual(records[0]["sample_id"], "TCGA-AA-0001-01Z-00-DX1")

    def test_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._fixture(root)
            (root / "sample.png").write_bytes(b"changed")
            with self.assertRaisesRegex(ManifestError, "image SHA256 mismatch"):
                load_dataset_manifest(
                    manifest,
                    expected_dataset="monuseg",
                    verify_hashes=True,
                )

    def test_identity_only_manifest_cannot_construct_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "test_identity.json"
            manifest.write_text(
                json.dumps(
                    {
                        "dataset": "monuseg",
                        "records": [{"sample_id": "TCGA-AA-0001-01Z-00-DX1"}],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ManifestError, "identity-only"):
                load_dataset_manifest(manifest, expected_dataset="monuseg")


if __name__ == "__main__":
    unittest.main()
