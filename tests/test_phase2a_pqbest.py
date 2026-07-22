import json
import unittest
from pathlib import Path

from stainpms.phase2a_pqbest import choose_pq_best, paired_coverage_flips


def record(epoch, pq):
    return {
        "epoch": epoch,
        "patient_macro": {"task_metrics_image_macro": {"pq": pq}},
    }


class PqBestTests(unittest.TestCase):
    def test_second_seed_c0_c1_protocol_is_frozen_and_low_storage(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads(
            (root / "configs" / "phase2a" / "tnbc_c0_c1_second_seed_2027_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["protocol_id"], "tnbc_c0_c1_second_seed_2027_v1")
        self.assertEqual(config["optimization"]["seed"], 2027)
        self.assertEqual(config["optimization"]["planned_attempted_crop_batches"], 1350)
        self.assertEqual(set(config["arms"]), {"c0", "c1"})
        self.assertEqual(config["arms"]["c0"]["coverage_coefficient"], 0.0)
        self.assertEqual(config["arms"]["c0"]["quality_coefficient"], 0.0)
        self.assertEqual(config["arms"]["c1"]["coverage_coefficient"], 1.0)
        self.assertEqual(config["arms"]["c1"]["quality_coefficient"], 1.0)
        self.assertEqual(config["data"]["sealed_patients"], [9, 10, 11])
        self.assertTrue(config["retention"]["no_permanent_full_epoch_history"])
        self.assertEqual(config["retention"]["expected_two_arm_storage_gib"], "about 18 to 20 including diagnostics and logs")
        self.assertEqual(config["retention"]["minimum_free_storage_before_each_arm_gib"], 16)

    def test_third_seed_protocol_is_fixed_and_retains_all_three_seeds(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads(
            (root / "configs" / "phase2a" / "tnbc_c0_c1_third_seed_1337_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["protocol_id"], "tnbc_c0_c1_third_seed_1337_v1")
        self.assertEqual(config["optimization"]["seed"], 1337)
        self.assertEqual(config["reference"]["all_prespecified_seeds"], [3407, 2027, 1337])
        self.assertTrue(config["final_advancement_rule"]["all_conditions_required"])

    def test_selects_highest_pq_and_earlier_exact_tie(self):
        selected = choose_pq_best([record(1, 0.5), record(2, 0.7), record(3, 0.7)])
        self.assertEqual(selected["selected_epoch"], 2)
        self.assertEqual(selected["selected_patient_macro_pq"], 0.7)

    def test_paired_flips_count_empty_selected_candidate_as_failure(self):
        reference = [
            {"sample_id": "07_1", "gt_instance_id": "1", "patient": "7", "auto_best_candidate_iou": "0.8"},
            {"sample_id": "07_1", "gt_instance_id": "2", "patient": "7", "auto_best_candidate_iou": ""},
        ]
        candidate = [
            {"sample_id": "07_1", "gt_instance_id": "1", "patient": "7", "auto_best_candidate_iou": "0.4"},
            {"sample_id": "07_1", "gt_instance_id": "2", "patient": "7", "auto_best_candidate_iou": "0.9"},
        ]
        result = paired_coverage_flips(reference, candidate, field="auto_best_candidate_iou")
        self.assertEqual(result["denominator"], 2)
        self.assertEqual(result["reference_numerator"], 1)
        self.assertEqual(result["candidate_numerator"], 1)
        self.assertEqual(result["flips"]["success_to_failure"], 1)
        self.assertEqual(result["flips"]["failure_to_success"], 1)


if __name__ == "__main__":
    unittest.main()
