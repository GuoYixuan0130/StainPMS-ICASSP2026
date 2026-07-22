import unittest

from stainpms.phase2a_pqbest import choose_pq_best, paired_coverage_flips


def record(epoch, pq):
    return {
        "epoch": epoch,
        "patient_macro": {"task_metrics_image_macro": {"pq": pq}},
    }


class PqBestTests(unittest.TestCase):
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
