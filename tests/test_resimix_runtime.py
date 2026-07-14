"""Synthetic smoke tests for the dataset-facing ResiMix runtime."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resimixpms.runtime import ResiMixAugmentor, force_single_synthetic_medoid_prompt  # noqa: E402
from resimixpms.experiment import sha256_file  # noqa: E402
from resimixpms.transplant import CONTEXT_FEATURE_NAMES, annulus_mask  # noqa: E402


def disk(shape, center, radius):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius**2


class RuntimeSmokeTest(unittest.TestCase):
    def test_forced_synthetic_medoid_replaces_duplicate_synthetic_candidate(self):
        coords, instance_ids = force_single_synthetic_medoid_prompt(
            np.asarray([[30.0, 30.0], [0.0, 0.0], [12.0, 10.0]], dtype=np.float32),
            np.asarray([70, 5, 6], dtype=np.int32),
            np.asarray([10.0, 10.0], dtype=np.float32),
            70,
        )
        self.assertTrue(np.array_equal(instance_ids, np.asarray([5, 70], dtype=np.int32)))
        self.assertTrue(np.array_equal(coords, np.asarray([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)))

    def _runtime(self, directory: Path) -> ResiMixAugmentor:
        payload_dir = directory / "payloads"
        payload_dir.mkdir()
        donor_mask = disk((19, 19), (9, 9), 4)
        donor_rgb = np.full((19, 19, 3), (205, 180, 170), dtype=np.uint8)
        donor_rgb[donor_mask] = (55, 70, 115)
        np.savez_compressed(
            payload_dir / "d0.npz", rgb=donor_rgb, mask=donor_mask.astype(np.uint8),
            annulus=annulus_mask(donor_mask, width=4).astype(np.uint8), type_id=np.asarray(1),
        )
        donor_csv = directory / "donor_bank_manifest.csv"
        with donor_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("donor_id", "category", "payload_path", "type_id"))
            writer.writeheader()
            writer.writerow({"donor_id": "d0", "category": "Missed", "payload_path": "d0.npz", "type_id": "1"})
        stats = directory / "host_context_statistics.json"
        stats.write_text(json.dumps({
            "context_mean": [0.0] * len(CONTEXT_FEATURE_NAMES),
            "context_std": [1.0] * len(CONTEXT_FEATURE_NAMES),
            "natural_boundary_gradient_p95": 1.0e6,
            "legal_context_distance_p95": 1.0e6,
            "tissue_total_od_threshold": 0.15,
        }), encoding="utf-8")
        config = directory / "resimix_config.json"
        config.write_text(json.dumps({
            "seed": 3407, "augmentation_probability": 0.5,
            "active_start_epoch": 2, "active_end_epoch": 9,
            "donor_bank_manifest": str(donor_csv),
            "donor_payload_dir": str(payload_dir),
            "host_context_statistics": str(stats),
            "donor_bank_manifest_sha256": sha256_file(donor_csv),
            "host_context_statistics_sha256": sha256_file(stats),
        }), encoding="utf-8")
        return ResiMixAugmentor(config)

    def test_warmup_is_a_pixel_exact_noop_and_active_path_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = self._runtime(Path(temporary))
            image = np.full((96, 96, 3), (170, 145, 130), dtype=np.uint8)
            instances = np.zeros((96, 96), dtype=np.int32)
            instances[disk(instances.shape, (48, 48), 5)] = 1
            types = (instances > 0).astype(np.int16)
            coverage = np.zeros_like(instances)
            warmup = runtime.augment(image, instances, types, coverage, epoch=1, sample_key="warmup")
            self.assertTrue(np.array_equal(warmup.image, image))
            self.assertTrue(np.array_equal(warmup.instance_map, instances))
            self.assertTrue(np.array_equal(warmup.coverage_map, coverage))
            self.assertIsNone(warmup.synthetic_instance_id)

            accepted = None
            for index in range(80):
                attempt = runtime.augment(image, instances, types, coverage, epoch=2, sample_key=f"active-{index}")
                if attempt.synthetic_instance_id is not None:
                    accepted = attempt
                    break
            self.assertIsNotNone(accepted, "at least one deterministic active crop must be transplantable")
            self.assertTrue(np.array_equal(accepted.coverage_map, coverage))
            self.assertEqual(set(np.unique(accepted.instance_map)), {0, 1, accepted.synthetic_instance_id})
            self.assertEqual(int((accepted.instance_map == accepted.synthetic_instance_id).sum()) > 0, True)
            runtime.mark_prompt_added(accepted.event_index)
            events = runtime.consume_events()
            event = next(item for item in events if item.get("synthetic_instance_id") == accepted.synthetic_instance_id)
            self.assertTrue(event["synthetic_prompt_added"])


if __name__ == "__main__":
    unittest.main()
