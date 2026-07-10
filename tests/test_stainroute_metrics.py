import numpy as np
import unittest

from stainroute.oracle import matched_iou_sum, pq_factorized
from tools.analyze_eval_artifacts import get_fast_pq


class StainRouteMetricTest(unittest.TestCase):
    def test_factorized_pq_matches_existing_implementation(self) -> None:
        # All retained pairs are strictly above 0.5, matching the legacy metric's
        # threshold convention exactly (up to its 1e-6 SQ stabilizer).
        gt = np.array(
            [
                [1, 1, 0, 2, 2, 0],
                [1, 1, 0, 2, 2, 0],
                [0, 0, 0, 0, 0, 0],
                [3, 3, 3, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        pred = np.array(
            [
                [9, 9, 0, 8, 8, 0],
                [9, 9, 0, 8, 0, 0],
                [0, 0, 0, 8, 0, 0],
                [7, 7, 7, 0, 6, 6],
            ],
            dtype=np.int32,
        )

        _, _, legacy_pq = get_fast_pq(gt, pred, match_iou=0.5)[0]
        self.assertAlmostEqual(matched_iou_sum(gt, pred), 2.6)
        self.assertAlmostEqual(pq_factorized(gt, pred), legacy_pq, delta=1.0e-6)

    def test_exact_half_iou_is_matched_by_stainroute_definition(self) -> None:
        gt = np.array([[1, 1, 1, 1]], dtype=np.int32)
        pred = np.array([[7, 7, 0, 0]], dtype=np.int32)

        self.assertAlmostEqual(matched_iou_sum(gt, pred), 0.5)
        self.assertAlmostEqual(pq_factorized(gt, pred), 0.5)

    def test_factorized_pq_handles_empty_and_shape_errors(self) -> None:
        empty = np.zeros((3, 3), dtype=np.int32)
        one_pred = empty.copy()
        one_pred[0, 0] = 1

        self.assertEqual(matched_iou_sum(empty, one_pred), 0.0)
        self.assertEqual(pq_factorized(empty, empty), 1.0)
        self.assertEqual(pq_factorized(empty, one_pred), 0.0)
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            pq_factorized(empty, np.zeros((2, 2), dtype=np.int32))
