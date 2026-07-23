from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from tools.summarize_zero_training_oracle_diagnosis import c1_minus_c0, main, stage_gaps


def scope(offset: float):
    def stage(pq: float):
        return {
            "tp": 10 + offset,
            "fp": 2 + offset,
            "fn": 3 - offset,
            "dq": 0.7 + offset,
            "sq": 0.8 + offset,
            "pq": pq,
            "task_metrics_image_macro": {"dice1": 0.8 + offset, "dice2": 0.7 + offset, "aji": 0.6 + offset},
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
            "all_candidate_best_iou_q10": 0.5 + offset,
            "all_candidate_best_iou_q25": 0.6 + offset,
            "all_candidate_best_iou_q75": 0.8 + offset,
            "all_candidate_best_iou_q90": 0.9 + offset,
            "selected_candidate_iou_mean": 0.65 + offset,
            "selected_candidate_iou_median": 0.7 + offset,
            "selected_candidate_iou_q10": 0.45 + offset,
            "selected_candidate_iou_q25": 0.55 + offset,
            "selected_candidate_iou_q75": 0.75 + offset,
            "selected_candidate_iou_q90": 0.85 + offset,
        },
        "errors": {"generation_miss": 3 - offset, "selection_miss": 2 - offset},
    }


class ZeroTrainingSummaryTests(unittest.TestCase):
    def test_stage_gaps_follow_requested_upper_minus_lower_order(self):
        values = stage_gaps(scope(0.0))
        self.assertAlmostEqual(values["all_candidate_oracle_minus_native_selected_oracle"]["pq"], 0.1)
        self.assertAlmostEqual(values["native_selected_oracle_minus_final_pool_oracle"]["pq"], 0.1)
        self.assertAlmostEqual(values["final_pool_oracle_minus_native_final"]["pq"], 0.1)

    def test_c1_minus_c0_is_paired_for_stages_mechanism_and_errors(self):
        values = c1_minus_c0(scope(0.02), scope(0.0))
        self.assertAlmostEqual(values["stages"]["native_final"]["pq"], 0.02)
        self.assertAlmostEqual(values["mechanism"]["all_candidate_coverage_recall_at_0_5"], 0.02)
        self.assertAlmostEqual(values["mechanism"]["all_candidate_best_iou_mean"], 0.02)
        self.assertAlmostEqual(values["errors"]["generation_miss"], -0.02)

    def test_six_input_cli_writes_utf8_machine_and_human_outputs(self):
        workspace = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=workspace, prefix=".tmp_zero_training_summary_") as raw_directory:
            directory = Path(raw_directory)
            assignments = []
            for seed in (3407, 2027, 1337):
                for arm, offset in (("c0", 0.0), ("c1", 0.01)):
                    source = directory / f"{seed}_{arm}.json"
                    source.write_text(
                        json.dumps(
                            {
                                "status": "complete",
                                "seed": seed,
                                "arm": arm,
                                "reference_reproduction": {"status": "pass"},
                                "summary": {"patients": {"7": scope(offset), "8": scope(offset)}, "patient_macro": scope(offset)},
                            }
                        ),
                        encoding="utf-8",
                    )
                    assignments.extend(["--input", f"{seed}:{arm}={source}"])
            output = directory / "out"
            original_argv = sys.argv
            try:
                sys.argv = ["summarize", *assignments, "--output-dir", str(output)]
                self.assertEqual(main(), 0)
            finally:
                sys.argv = original_argv
            payload = json.loads((output / "zero_training_diagnosis.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "complete")
            self.assertIn("all_candidate_best_iou_mean", payload["three_seed_patient_macro"]["c1_minus_c0"]["mechanism"])
            self.assertIsNotNone(payload["three_seed_patient_macro"]["c1_minus_c0"]["stages"]["native_final"]["aji"]["mean"])
            self.assertTrue((output / "zero_training_diagnosis.csv").is_file())
            self.assertTrue((output / "zero_training_diagnosis.md").is_file())


if __name__ == "__main__":
    unittest.main()
