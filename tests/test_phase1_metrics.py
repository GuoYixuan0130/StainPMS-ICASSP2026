import unittest

import numpy as np

from stainpms.evaluator import evaluate_instance_pair
from stainpms.phase1_metrics import (
    attach_gt_error_classes,
    choose_edt_interior_points,
    final_instance_overlap_table,
    final_max_iou_by_gt,
    max_iou_with_final_prediction,
    strict_final_pairing,
    structural_errors,
    summarize_gt_rows,
)


class Phase1MetricTests(unittest.TestCase):
    def test_edt_point_is_inside_and_deterministic(self):
        inst = np.zeros((7, 7), dtype=np.int32)
        inst[1:6, 1:6] = 1
        inst[0:2, 5:7] = 2
        points = choose_edt_interior_points(inst)
        self.assertEqual(points[1], (3, 3))
        for instance_id, (x, y) in points.items():
            self.assertEqual(inst[y, x], instance_id)

    def test_error_classes_are_mutually_exclusive(self):
        rows = attach_gt_error_classes(
            [
                {"auto_point_count": 1, "auto_best_candidate_iou": 0.1, "auto_selected_candidate_iou": 0.1, "final_matched": True},
                {"auto_point_count": 0, "auto_best_candidate_iou": None, "auto_selected_candidate_iou": None, "final_matched": False},
                {"auto_point_count": 1, "auto_best_candidate_iou": 0.4, "auto_selected_candidate_iou": 0.2, "final_matched": False},
                {"auto_point_count": 1, "auto_best_candidate_iou": 0.8, "auto_selected_candidate_iou": 0.2, "final_matched": False},
                {"auto_point_count": 1, "auto_best_candidate_iou": 0.8, "auto_selected_candidate_iou": 0.7, "final_matched": False},
            ],
            0.5,
        )
        self.assertEqual(
            [row["error_class"] for row in rows],
            [
                "final_matched_tp",
                "point_miss",
                "candidate_generation_miss",
                "selection_ranking_miss",
                "assembly_nms_conflict_miss",
            ],
        )

    def test_conditional_and_e2e_ccr_have_distinct_denominators(self):
        rows = attach_gt_error_classes(
            [
                {"auto_point_count": 1, "auto_best_candidate_iou": 0.8, "auto_selected_candidate_iou": 0.4, "gt_point_best_candidate_iou": 0.9, "final_matched": False},
                {"auto_point_count": 0, "auto_best_candidate_iou": None, "auto_selected_candidate_iou": None, "gt_point_best_candidate_iou": 0.9, "final_matched": False},
            ],
            0.5,
        )
        summary = summarize_gt_rows(rows, thresholds=[0.5], match_iou=0.5)
        self.assertEqual(summary["ccr_auto_given_point"][0]["denominator"], 1)
        self.assertEqual(summary["ccr_auto_e2e"][0]["denominator"], 2)
        self.assertEqual(summary["ccr_auto_given_point"][0]["value"], 1.0)
        self.assertEqual(summary["ccr_auto_e2e"][0]["value"], 0.5)

    def test_structural_split_merge_and_boundary(self):
        gt = np.zeros((6, 6), dtype=np.int32)
        gt[1:5, 1:5] = 1
        pred = np.zeros_like(gt)
        pred[1:5, 1:2] = 1
        pred[1:5, 4:5] = 2
        report = structural_errors(gt, pred, 0.5)
        self.assertEqual(report["fn"], 1)
        self.assertEqual(report["fp"], 2)
        self.assertEqual(report["split_unmatched_gt_count"], 1)
        self.assertEqual(report["boundary_localization_unmatched_gt_count"], 1)

    def test_final_overlap_table_matches_reference_per_gt_scans(self):
        gt = np.zeros((8, 9), dtype=np.int32)
        gt[1:4, 1:4] = 1
        gt[4:7, 2:6] = 2
        gt[2:6, 6:8] = 3
        pred = np.zeros_like(gt)
        pred[1:4, 1:3] = 1
        pred[1:4, 3:4] = 2
        pred[4:7, 2:6] = 3
        pred[2:6, 6:8] = 4
        overlap = final_instance_overlap_table(gt, pred)
        fast = final_max_iou_by_gt(overlap)
        for gt_id in (1, 2, 3):
            expected = max_iou_with_final_prediction(gt == gt_id, pred)
            self.assertEqual(fast[gt_id][1], expected[1])
            self.assertAlmostEqual(fast[gt_id][0], expected[0], places=12)

        reference = structural_errors(gt, pred, 0.5)
        cached = structural_errors(
            gt,
            pred,
            0.5,
            pairing_info=strict_final_pairing(gt, pred, 0.5),
            overlap=overlap,
            best_iou_by_gt=fast,
        )
        for key in ("tp", "fp", "fn", "split_unmatched_gt_count", "merge_unmatched_pred_count", "boundary_localization_unmatched_gt_count"):
            self.assertEqual(cached[key], reference[key])

    def test_vectorized_strict_pairing_matches_frozen_evaluator(self):
        rng = np.random.default_rng(7)
        cases = []
        for _ in range(4):
            cases.append((rng.integers(0, 4, size=(9, 11), dtype=np.int32), rng.integers(0, 5, size=(9, 11), dtype=np.int32)))
        # Include an exact 0.5 IoU pair, which activates the evaluator's
        # inclusive maximum-cardinality branch.
        exact_gt = np.zeros((4, 6), dtype=np.int32)
        exact_pred = np.zeros_like(exact_gt)
        exact_gt[1:3, 1:3] = 1
        exact_pred[1:3, 1:5] = 1
        cases.append((exact_gt, exact_pred))
        for gt, pred in cases:
            expected = evaluate_instance_pair(gt, pred, mode="strict", match_iou=0.5)
            observed = strict_final_pairing(gt, pred, 0.5)["evaluator"]
            self.assertEqual(observed["pairing"], expected["pairing"])
            for metric in ("dq", "sq", "pq"):
                self.assertAlmostEqual(observed["metrics"][metric], expected["metrics"][metric], places=12)


if __name__ == "__main__":
    unittest.main()
