from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None, "requires the project PyTorch environment")
class PromptQProtocolTest(unittest.TestCase):
    def _manifest(self) -> dict:
        return {
            "dataset": "TNBC",
            "split_method": "fixed_patient_level",
            "train_image_root_relative": "train_12/images",
            "test_image_root_relative": "test/images",
            "router_train": ["01_1", "02_1", "03_1", "04_1", "05_1", "06_1"],
            "calibration": ["07_1", "07_2", "07_3", "08_1", "08_2", "08_3", "08_4"],
            "test": ["09_1"],
        }

    def _write_paths(self, root: Path, image_ids: list[str], image_dir: str, label_dir: str) -> None:
        images, labels = root / image_dir, root / label_dir
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        for image_id in image_ids:
            (images / f"{image_id}.png").touch()
            (labels / f"{image_id}.mat").touch()

    def test_direct_roles_never_resolve_closed_patients(self) -> None:
        from promptcredit.promptq.data import resolve_promptq_images

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "tnbc"
            manifest = self._manifest()
            self._write_paths(root, manifest["router_train"] + manifest["calibration"], "train_12/images", "train_12/labels")
            manifest_path = Path(temporary) / "split.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            train = resolve_promptq_images(root, manifest_path, "train")
            development = resolve_promptq_images(root, manifest_path, "development")
            observed = [item.image_id for item in train + development]
            self.assertFalse(any(image_id.startswith(("09_", "10_", "11_")) for image_id in observed))
            self.assertEqual(len(development), 7)

    def test_closed_patient_in_development_manifest_is_rejected(self) -> None:
        from promptcredit.promptq.data import resolve_promptq_images

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "tnbc"
            manifest = self._manifest()
            manifest["calibration"][-1] = "09_1"
            self._write_paths(root, manifest["router_train"] + manifest["calibration"], "train_12/images", "train_12/labels")
            manifest_path = Path(temporary) / "split.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                resolve_promptq_images(root, manifest_path, "development")


if __name__ == "__main__":
    unittest.main()
