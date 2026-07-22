import unittest

from stainpms.phase2a_tnbc_screen import assess_epoch5, build_epoch_record, metric_deltas


def strict_metrics(aji, pq):
    return {
        "included_in_macro": True,
        "metrics": {"dice1": 0.7, "dice2": 0.6, "aji": aji, "dq": 0.5, "sq": 0.5, "pq": pq},
    }


def summary(best7, best8, selected7=0.6, selected8=0.6):
    def group(best, selected):
        return {
            "gt_instance_count": 2,
            "candidate_iou": {
                "best_mean": best,
                "selected_standard_candidate_mean": selected,
                "selection_regret_mean": 0.02,
            },
            "ccr_auto_e2e": [{"threshold": 0.3, "value": 0.5}, {"threshold": 0.5, "value": best}, {"threshold": 0.7, "value": 0.2}],
        }
    return {"groups": {"patient:7": group(best7, selected7), "patient:8": group(best8, selected8)}}


def record(arm, epoch, aji7, aji8, pq7, pq8, best7, best8):
    images = [
        {"sample_id": "07_1", "patient": 7, "final_task_metrics": strict_metrics(aji7, pq7)},
        {"sample_id": "08_1", "patient": 8, "final_task_metrics": strict_metrics(aji8, pq8)},
    ]
    gt_rows = [
        {"patient": "7", "auto_selected_candidate_iou": "0.6"},
        {"patient": "7", "auto_selected_candidate_iou": "0.4"},
        {"patient": "8", "auto_selected_candidate_iou": "0.6"},
        {"patient": "8", "auto_selected_candidate_iou": "0.4"},
    ]
    return build_epoch_record(arm=arm, epoch=epoch, summary=summary(best7, best8), images=images, gt_rows=gt_rows, source_dir=None)


class TnbcScreenTests(unittest.TestCase):
    def test_epoch5_requires_all_owner_frozen_rules(self):
        c0 = record("c0", 5, 0.40, 0.50, 0.60, 0.60, 0.50, 0.50)
        c1 = record("c1", 5, 0.41, 0.51, 0.598, 0.600, 0.51, 0.50)
        result = assess_epoch5(c0, c1)
        self.assertEqual(result["decision"], "pass_all_promotion_rules")
        self.assertAlmostEqual(result["epoch5_c1_minus_c0"]["patient_macro"]["task_metrics_image_macro"]["aji"], 0.01)

    def test_epoch5_rejects_no_strict_macro_candidate_coverage_gain(self):
        c0 = record("c0", 5, 0.40, 0.50, 0.60, 0.60, 0.50, 0.50)
        c1 = record("c1", 5, 0.41, 0.51, 0.60, 0.60, 0.51, 0.49)
        result = assess_epoch5(c0, c1)
        self.assertEqual(result["decision"], "do_not_auto_promote")
        self.assertFalse(result["checks"]["best_ccr_patient_8_non_decrease"])
        self.assertFalse(result["checks"]["best_ccr_patient_macro_strict_increase"])

    def test_delta_is_patient_equal_not_gt_micro(self):
        c0 = record("c0", 5, 0.40, 0.50, 0.60, 0.60, 0.50, 0.50)
        c1 = record("c1", 5, 0.41, 0.50, 0.60, 0.60, 0.51, 0.50)
        delta = metric_deltas(c0, c1)
        self.assertAlmostEqual(delta["patient_macro"]["task_metrics_image_macro"]["aji"], 0.005)
        self.assertAlmostEqual(delta["patient_macro"]["mechanism"]["best_candidate_ccr_at_0_5"], 0.005)


if __name__ == "__main__":
    unittest.main()
