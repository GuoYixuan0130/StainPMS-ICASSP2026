from __future__ import annotations

import unittest

from tools.summarize_c2_ar_results import aggregate_comparison, arm_difference, normalized_arm, promotion_gate


def scope(offset: float) -> dict:
    def stage(pq: float) -> dict:
        return {
            "tp": 10 + offset,
            "fp": 2 - offset,
            "fn": 3 - offset,
            "dq": 0.7 + offset,
            "sq": 0.8 + offset,
            "pq": pq,
            "coverage_recall_at_0_5": 0.8 + offset,
            "raw_prediction_group_count": 12,
            "raw_prediction_mask_count": 12,
            "task_metrics_image_macro": {
                "dice1": 0.8 + offset,
                "dice2": 0.7 + offset,
                "aji": 0.6 + offset,
            },
        }

    return {
        "stages": {
            "native_final": stage(0.50 + offset),
            "final_pool_oracle": stage(0.60 + offset),
            "native_selected_pool_oracle": stage(0.70 + offset),
            "all_candidate_pool_oracle": stage(0.80 + offset),
        },
        "candidate_quality": {
            "all_candidate_coverage_recall_at_0_5": 0.9 + offset,
            "native_selected_coverage_recall_at_0_5": 0.8 + offset,
            "selection_regret_mean": 0.1 - offset,
            "all_candidate_best_iou_mean": 0.7 + offset,
            "all_candidate_best_iou_median": 0.75 + offset,
            "selected_candidate_iou_mean": 0.65 + offset,
            "selected_candidate_iou_median": 0.7 + offset,
        },
        "errors": {
            "generation_miss": 3 - offset,
            "selection_miss": 2 - offset,
            "assembly_loss": 1 + offset,
            "native_final_tp": 10 + offset,
            "native_final_false_positive_count": 2 - offset,
            "native_final_false_negative_count": 3 - offset,
        },
    }


class C2ARSummaryTests(unittest.TestCase):
    def test_six_input_report_applies_all_five_promotion_conditions(self):
        per_seed = []
        for seed in (2027, 1337):
            c0, c1, c2 = normalized_arm(scope(0.0)), normalized_arm(scope(0.01)), normalized_arm(scope(0.03))
            # Retain a positive native result but make both selected-to-final
            # and final-FP gaps smaller than C1.
            c2["stages"]["native_selected_pool_oracle"]["pq"] = 0.73
            c2["stages"]["final_pool_oracle"]["pq"] = 0.66
            c2["stages"]["native_final"]["pq"] = 0.59
            per_seed.append(
                {
                    "seed": seed,
                    "scopes": {
                        "patient_macro": {
                            "c2_ar_minus_c0": arm_difference(c2, c0),
                            "c2_ar_minus_c1": arm_difference(c2, c1),
                        }
                    },
                }
            )
        macro = [record["scopes"]["patient_macro"] for record in per_seed]
        aggregate = {
            comparison: aggregate_comparison([record[comparison] for record in macro])
            for comparison in ("c2_ar_minus_c0", "c2_ar_minus_c1")
        }
        result = promotion_gate(per_seed, aggregate)
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["conditions"]["both_seed_native_pq_positive_vs_c0"])


if __name__ == "__main__":
    unittest.main()
