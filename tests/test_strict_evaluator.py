import tempfile
import unittest
from pathlib import Path

import numpy as np

from stainpms.evaluator import (
    aggregate_image_metrics,
    evaluate_instance_pair,
    write_evaluation_outputs,
)


class StrictEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.empty = np.zeros((8, 8), dtype=np.int32)
        self.one = self.empty.copy()
        self.one[2:6, 2:6] = 1

    def test_perfect_prediction(self):
        result = evaluate_instance_pair(self.one, self.one, mode="strict")
        self.assertTrue(result["included_in_macro"])
        self.assertFalse(result["no_match"])
        for name in ("dice1", "dice2", "aji", "aji_p", "dq", "sq", "pq"):
            self.assertAlmostEqual(result["metrics"][name], 1.0, places=5)

    def test_nonempty_maps_without_background_preserve_metrics(self):
        gt = np.full((8, 8), 5, dtype=np.int32)
        pred = np.full((8, 8), 17, dtype=np.int32)
        result = evaluate_instance_pair(gt, pred, mode="strict")
        self.assertTrue(result["metric_background_padding_applied"])
        self.assertFalse(result["gt_has_background_id_zero"])
        self.assertFalse(result["prediction_has_background_id_zero"])
        self.assertEqual(result["shape"], [8, 8])
        self.assertEqual(result["gt_instance_count"], 1)
        self.assertEqual(result["pred_instance_count"], 1)
        for name in ("dice1", "dice2", "aji", "aji_p", "dq", "sq", "pq"):
            self.assertAlmostEqual(result["metrics"][name], 1.0, places=5)

    def test_empty_prediction_with_nonempty_gt(self):
        result = evaluate_instance_pair(self.one, self.empty, mode="strict")
        self.assertTrue(result["included_in_macro"])
        self.assertTrue(result["no_match"])
        self.assertEqual(result["pairing"]["fn"], 1)
        self.assertTrue(all(value == 0.0 for value in result["metrics"].values()))

    def test_nonempty_prediction_with_empty_gt(self):
        result = evaluate_instance_pair(self.empty, self.one, mode="strict")
        self.assertTrue(result["included_in_macro"])
        self.assertEqual(result["pairing"]["fp"], 1)
        self.assertTrue(all(value == 0.0 for value in result["metrics"].values()))

    def test_both_empty_excluded_without_full_score(self):
        result = evaluate_instance_pair(self.empty, self.empty, mode="strict")
        self.assertFalse(result["included_in_macro"])
        self.assertTrue(result["both_empty"])
        self.assertTrue(all(value is None for value in result["metrics"].values()))

    def test_all_false_positives(self):
        result = evaluate_instance_pair(self.empty, self.one, mode="strict")
        self.assertEqual(result["pairing"], {
            "tp": 0,
            "fp": 1,
            "fn": 0,
            "paired_true": [],
            "paired_pred": [],
            "unpaired_true": [],
            "unpaired_pred": [1],
        })

    def test_all_false_negatives(self):
        result = evaluate_instance_pair(self.one, self.empty, mode="strict")
        self.assertEqual(result["pairing"]["tp"], 0)
        self.assertEqual(result["pairing"]["fn"], 1)
        self.assertEqual(result["pairing"]["fp"], 0)

    def test_nonempty_disjoint_prediction_is_all_fp_and_fn(self):
        pred = np.zeros_like(self.one)
        pred[0:2, 0:2] = 1
        result = evaluate_instance_pair(self.one, pred, mode="strict")
        self.assertTrue(result["no_match"])
        self.assertEqual(result["pairing"]["tp"], 0)
        self.assertEqual(result["pairing"]["fp"], 1)
        self.assertEqual(result["pairing"]["fn"], 1)
        self.assertTrue(all(value == 0.0 for value in result["metrics"].values()))

    def test_one_to_many_split(self):
        pred = np.zeros_like(self.one)
        pred[2:6, 2:4] = 1
        pred[2:6, 4:6] = 2
        result = evaluate_instance_pair(self.one, pred, mode="strict")
        self.assertEqual(result["pairing"]["tp"], 1)
        self.assertEqual(result["pairing"]["fp"], 1)
        self.assertEqual(result["pairing"]["fn"], 0)
        self.assertAlmostEqual(result["metrics"]["dq"], 2.0 / 3.0, places=6)
        self.assertAlmostEqual(result["metrics"]["sq"], 0.5, places=5)

    def test_many_to_one_merge(self):
        gt = np.zeros_like(self.one)
        gt[2:6, 2:4] = 1
        gt[2:6, 4:6] = 2
        result = evaluate_instance_pair(gt, self.one, mode="strict")
        self.assertEqual(result["pairing"]["tp"], 1)
        self.assertEqual(result["pairing"]["fp"], 0)
        self.assertEqual(result["pairing"]["fn"], 1)
        self.assertAlmostEqual(result["metrics"]["dq"], 2.0 / 3.0, places=6)

    def test_legacy_empty_behavior_is_preserved(self):
        result = evaluate_instance_pair(self.one, self.empty, mode="legacy_skip")
        self.assertFalse(result["included_in_macro"])
        self.assertEqual(result["skip_reason"], "legacy_skip_empty_gt_or_prediction")

    def test_macro_and_machine_readable_outputs(self):
        records = [
            evaluate_instance_pair(self.one, self.one, mode="strict", sample_id="perfect"),
            evaluate_instance_pair(self.one, self.empty, mode="strict", sample_id="fn"),
            evaluate_instance_pair(self.empty, self.empty, mode="strict", sample_id="empty"),
        ]
        summary = aggregate_image_metrics(records)
        self.assertEqual(summary["image_count"], 3)
        self.assertEqual(summary["included_image_count"], 2)
        self.assertAlmostEqual(summary["metrics_macro"]["aji"], 0.5, places=6)
        with tempfile.TemporaryDirectory() as tmp:
            payload = write_evaluation_outputs(records, tmp, context={"split": "synthetic"})
            self.assertEqual(payload["summary"]["both_empty_count"], 1)
            self.assertTrue((Path(tmp) / "metrics_per_image.json").is_file())
            self.assertTrue((Path(tmp) / "metrics_per_image.csv").is_file())
            self.assertTrue((Path(tmp) / "metrics_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
