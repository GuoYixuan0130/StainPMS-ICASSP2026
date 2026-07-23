from __future__ import annotations

import unittest

import numpy as np

from stainpms.c2_component_audit import hard_exclusivity, score_calibration, selected_utility_labels
from stainpms.zero_training_oracle import annotate_pool_ious


class C2ComponentAuditTests(unittest.TestCase):
    def setUp(self):
        self.gt = np.zeros((12, 12), dtype=np.int32)
        self.gt[1:4, 1:4] = 1
        self.gt[1:4, 7:10] = 2

    def row(self, index, group, mask, score):
        return {
            "record_index": index,
            "prompt_group_id": group,
            "token": 0,
            "crop_index": 0,
            "bbox_xyxy": [0.0, 0.0, 12.0, 12.0],
            "mask": mask,
            "assembly_score": score,
            "quality": score,
            "edge_penalized": False,
            "gt_ious": {},
        }

    def test_unique_tp_duplicate_and_fp_labels_are_disjoint(self):
        first = self.gt == 1
        duplicate = self.gt == 1
        fp = np.zeros_like(first)
        rows = selected_utility_labels(
            annotate_pool_ious(
                [self.row(0, 10, first, 0.9), self.row(1, 11, duplicate, 0.7), self.row(2, 12, fp, 0.1)], self.gt
            ), self.gt
        )
        self.assertEqual([row["utility_label"] for row in rows], ["unique_tp", "duplicate", "unmatched_fp"])
        self.assertEqual(rows[0]["utility_target"], 1.0)
        calibration = score_calibration(rows)
        self.assertAlmostEqual(calibration["positive_fraction"], 1.0 / 3.0)
        self.assertGreater(calibration["auroc"], 0.9)

    def test_foreign_leakage_is_visible_in_hard_accounting(self):
        leaked = (self.gt == 1).copy()
        leaked[1:2, 7:9] = True
        other = self.gt == 2
        rows = selected_utility_labels(
            annotate_pool_ious([self.row(0, 10, leaked, 0.9), self.row(1, 11, other, 0.8)], self.gt), self.gt
        )
        audit = hard_exclusivity(rows, self.gt)
        self.assertIsNotNone(audit["hard_foreign_gt_fraction"]["mean"])
        self.assertGreater(audit["hard_foreign_gt_fraction"]["mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
