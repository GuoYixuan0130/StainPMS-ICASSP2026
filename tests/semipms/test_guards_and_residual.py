import unittest
from pathlib import Path

import numpy as np

from semipms.guards import HiddenGTGuard, ImageRecord, deterministic_split, inspect_clean_initialization, patient_from_stem, sha256_file, validate_clean_checkpoint_name
from semipms.residual import (
    acceptance_features,
    frozen_accept,
    geometric_view,
    inverse_geometric_mask,
    inverse_stain_mask,
    propose_residual_points,
    transform_points_xy,
)


class SemiPMSGuardTest(unittest.TestCase):
    def test_closed_patient_guard(self):
        guard = HiddenGTGuard()
        record = ImageRecord(1, "01_1", "image", "label", "a")
        with self.assertRaises(PermissionError):
            guard.allow_unlabeled_label_read(record)
        guard.freeze_acceptance_rule()
        guard.allow_unlabeled_label_read(record)
        self.assertEqual(guard.hidden_gt_reads, 1)
        with self.assertRaises(PermissionError):
            guard.allow_unlabeled_label_read(ImageRecord(7, "07_1", "image", "label", "b"))

    def test_deterministic_six_plus_twenty_four_split(self):
        records = [
            ImageRecord(patient, f"{patient:02d}_{index}", "image", "label", f"{patient}-{index}")
            for patient in range(1, 7) for index in range(1, 6)
        ]
        first = deterministic_split(records)
        second = deterministic_split(list(reversed(records)))
        self.assertEqual([item.stem for item in first[0]], [item.stem for item in second[0]])
        self.assertEqual((len(first[0]), len(first[1])), (6, 24))

    def test_patient_parser_refuses_ambiguous_stems(self):
        self.assertEqual(patient_from_stem("06_3"), 6)
        with self.assertRaises(ValueError):
            patient_from_stem("TCGA_01")

    def test_clean_initialization_and_checksum_guards(self):
        with self.assertRaises(PermissionError):
            validate_clean_checkpoint_name(Path("tnbc_pms_best_e156.pth"))
        self.assertEqual(len(sha256_file(Path(__file__))), 64)


class SemiPMSResidualTest(unittest.TestCase):
    def test_inverse_geometric_mask_and_points(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        viewed = geometric_view(image)
        self.assertEqual(viewed.shape, image.shape)
        mask = np.zeros((20, 20), dtype=bool)
        mask[8:12, 8:12] = True
        shifted = np.zeros_like(mask)
        shifted[6:10, 11:15] = True
        self.assertTrue(np.array_equal(inverse_geometric_mask(shifted), mask))
        self.assertTrue(np.array_equal(inverse_stain_mask(mask), mask))
        np.testing.assert_equal(transform_points_xy(np.asarray([[8, 8]], dtype=np.float32)), [[11, 6]])

    def test_frozen_rule_and_residual_budget(self):
        residual = np.zeros((32, 32), dtype=np.float32)
        residual[5, 5], residual[20, 20], residual[10, 25] = 1.0, 0.8, 0.7
        self.assertGreaterEqual(len(propose_residual_points(residual, max_candidates=3)), 2)
        original = np.zeros((12, 12), dtype=bool); original[3:8, 3:8] = True
        features = acceptance_features(original, original, original, original.astype(float), np.zeros_like(original))
        rule = {"min_view_iou": .8, "max_centroid_displacement": 2, "min_area_stability": .8, "min_h_occupancy": .5, "min_boundary_stability": .8, "max_pseudo_conflict": .1}
        self.assertTrue(frozen_accept(features, rule))


if __name__ == "__main__":
    unittest.main()
