"""Small end-to-end dataset/PMS integration test with no model inference."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import scipy.io as sio
from skimage import io


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resimixpms.coverage import StaticCoverageWriter  # noqa: E402
from resimixpms.experiment import sha256_file  # noqa: E402
from resimixpms.transplant import CONTEXT_FEATURE_NAMES, annulus_mask  # noqa: E402

try:  # The local lint environment intentionally has no CUDA/PyTorch stack.
    import torch  # noqa: F401
except ModuleNotFoundError:
    MONUSEG = None
else:
    # Other missing imports are a real AutoDL environment/configuration error,
    # not a reason to mark this mandatory integration test as skipped.
    from run.dataset.monuseg import MONUSEG  # noqa: E402


def disk(shape, center, radius):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius**2


@unittest.skipIf(MONUSEG is None, "requires the remote CA-SAM2 PyTorch environment")
class DatasetIntegrationTest(unittest.TestCase):
    def _config(self):
        criterion = SimpleNamespace(
            stain_top_k=20, stain_min_distance=12, stain_open_disk=2, stain_sigma=1.0,
            stain_baseline_dilate_radius=5, stain_merge_aware=False,
            stain_merge_min_distance=6, stain_merge_num_peaks=3,
            hed_alpha=1.0, hed_beta=0.0, hed_gamma=0.0,
            pms_loss_coef=0.5, pms_gt_match_radius=8, pms_baseline_prompts=False,
            pms_preserve_max_prompts=0,
        )
        transforms = SimpleNamespace(transform=[dict(type="Normalize")])
        return SimpleNamespace(data={"train": transforms, "test": transforms}, criterion=criterion)

    def _write_resimix_files(self, root: Path, train_manifest: Path, coverage_manifest: Path):
        payloads = root / "donor_payloads"
        payloads.mkdir()
        mask = disk((19, 19), (9, 9), 4)
        donor_rgb = np.full((19, 19, 3), (205, 180, 170), dtype=np.uint8)
        donor_rgb[mask] = (55, 70, 115)
        np.savez_compressed(payloads / "d0.npz", rgb=donor_rgb, mask=mask.astype(np.uint8),
                            annulus=annulus_mask(mask, width=4).astype(np.uint8), type_id=np.asarray(1))
        donor_csv = root / "donor_bank_manifest.csv"
        with donor_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("donor_id", "category", "payload_path", "type_id"))
            writer.writeheader()
            writer.writerow({"donor_id": "d0", "category": "Missed", "payload_path": "d0.npz", "type_id": "1"})
        stats = root / "host_context_statistics.json"
        stats.write_text(json.dumps({
            "context_mean": [0.0] * len(CONTEXT_FEATURE_NAMES),
            "context_std": [1.0] * len(CONTEXT_FEATURE_NAMES),
            "natural_boundary_gradient_p95": 1.0e6,
            "legal_context_distance_p95": 1.0e6,
        }), encoding="utf-8")
        config = root / "resimix_config.json"
        config.write_text(json.dumps({
            "seed": 3407, "augmentation_probability": 0.5,
            "active_start_epoch": 2, "active_end_epoch": 9,
            "donor_bank_manifest": str(donor_csv), "donor_payload_dir": str(payloads),
            "host_context_statistics": str(stats),
            "donor_bank_manifest_sha256": sha256_file(donor_csv),
            "host_context_statistics_sha256": sha256_file(stats),
            "dataset": "tnbc",
            "train_manifest": str(train_manifest),
            "train_manifest_sha256": sha256_file(train_manifest),
            "train_crop_manifest": "",
            "train_crop_manifest_sha256": "",
            "static_coverage_manifest": str(coverage_manifest),
            "static_coverage_manifest_sha256": sha256_file(coverage_manifest),
        }), encoding="utf-8")
        return config

    def test_manifest_static_coverage_and_synthetic_prompt_integration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            for split in ("train_12", "test"):
                (data_root / split / "images").mkdir(parents=True)
                (data_root / split / "labels").mkdir(parents=True)
            image = np.full((512, 512, 3), (170, 145, 130), dtype=np.uint8)
            inst = np.zeros((512, 512), dtype=np.int32)
            inst[disk(inst.shape, (256, 256), 7)] = 1
            io.imsave(data_root / "train_12" / "images" / "p1.png", image, check_contrast=False)
            sio.savemat(data_root / "train_12" / "labels" / "p1.mat", {"inst_map": inst})
            io.imsave(data_root / "test" / "images" / "p7.png", image[:256, :256], check_contrast=False)
            sio.savemat(data_root / "test" / "labels" / "p7.mat", {"inst_map": inst[:256, :256]})
            train_manifest = root / "train.json"
            test_manifest = root / "test.json"
            train_manifest.write_text(json.dumps({"records": [{"image_name": "p1.png", "patient_id": 1}]}), encoding="utf-8")
            test_manifest.write_text(json.dumps({"records": [{"image_name": "p7.png", "patient_id": 7}]}), encoding="utf-8")
            coverage_dir = root / "coverage"
            writer = StaticCoverageWriter.create(coverage_dir, {"p1": inst.shape}, provenance={"test": True})
            writer.write_full("p1", np.zeros_like(inst))
            writer.seal()
            config = self._write_resimix_files(root, train_manifest, coverage_dir / "coverage_manifest.json")
            args = SimpleNamespace(
                crop_size=256, overlap=0, use_pms=True, pms_self_bootstrap=False,
                pms_gt_match_radius=8, pms_baseline_prompts=False, pms_preserve_max_prompts=0,
                baseline_masks_dir=str(coverage_dir), coverage_manifest=str(coverage_dir / "coverage_manifest.json"),
                resimix_enabled=True, resimix_config=str(config), data_identity="tnbc",
                train_manifest=str(train_manifest), test_manifest=str(test_manifest),
                train_crop_manifest="", eval_crop_manifest="", allowed_patient_ids="",
                train_allowed_patient_ids="1,2,3,4,5,6", test_allowed_patient_ids="7,8",
                forbidden_patient_ids="9,10,11",
            )
            dataset = MONUSEG(args, self._config(), str(data_root), "sequence", mode="train")
            dataset.set_epoch(2)
            batch = dataset[0]
            events = dataset.consume_resimix_events()
            accepted = [event for event in events if event.get("status") == "accepted"]
            self.assertGreaterEqual(len(batch[0]), 1)
            self.assertTrue(events)
            for crop_index, masks in enumerate(batch[1]):
                coords = batch[11][crop_index].numpy()
                gt_masks = batch[13][crop_index].numpy()
                self.assertEqual(len(coords), len(gt_masks))
                for point, gt_mask in zip(coords.astype(int), gt_masks):
                    self.assertTrue(gt_mask[point[1], point[0]])
            for event in accepted:
                self.assertTrue(event.get("synthetic_prompt_added", False))
            # The synthetic prompt is marked by the PMS assembly only for an
            # accepted crop, proving no mutation is applied to the cache map.
            self.assertTrue(np.array_equal(dataset._baseline_cache["p1"], np.zeros_like(inst)))


if __name__ == "__main__":
    unittest.main()
