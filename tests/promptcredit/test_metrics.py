from __future__ import annotations

import unittest

import numpy as np

from promptcredit.metrics import binary_iou, score_utility_summary, soft_iou


class PromptCreditMetricsTest(unittest.TestCase):
    def test_hard_and_soft_iou(self) -> None:
        target = np.asarray([[[1, 0], [0, 0]], [[1, 1], [0, 0]]], dtype=bool)
        hard = np.asarray([[[1, 0], [1, 0]], [[1, 1], [0, 0]]], dtype=bool)
        probability = np.asarray([[[1.0, 0.0], [0.5, 0.0]], [[1.0, 1.0], [0.0, 0.0]]])
        np.testing.assert_allclose(binary_iou(hard, target), [0.5, 1.0])
        np.testing.assert_allclose(soft_iou(probability, target), [1.0 / 1.5, 1.0])

    def test_score_utility_statistics(self) -> None:
        summary = score_utility_summary([0.9, 0.1, 0.8, 0.2], [0.9, 0.1, 0.8, 0.2])
        self.assertEqual(summary["auroc_iou_ge_0_5"], 1.0)
        self.assertEqual(summary["auprc_iou_ge_0_5"], 1.0)
        self.assertAlmostEqual(summary["spearman_point_score_vs_hard_iou"], 1.0)
        self.assertEqual(len(summary["reliability_diagram"]), 4)


if __name__ == "__main__":
    unittest.main()

