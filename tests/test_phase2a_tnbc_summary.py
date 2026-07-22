import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from summarize_phase2a_tnbc_warmstart_screen import diagnosis_record


def strict_metrics():
    return {
        "included_in_macro": True,
        "metrics": {"dice1": 0.7, "dice2": 0.6, "aji": 0.5, "dq": 0.4, "sq": 0.5, "pq": 0.2},
    }


class TnbcSummaryTests(unittest.TestCase):
    def test_diagnosis_record_reads_images_json_list(self):
        # The managed Windows sandbox denies writes to the system TEMP folder;
        # keep this short-lived fixture within the writable repository instead.
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            directory = Path(temporary)
            summary = {
                "groups": {
                    "patient:7": {
                        "gt_instance_count": 1,
                        "candidate_iou": {
                            "best_mean": 0.7,
                            "selected_standard_candidate_mean": 0.6,
                            "selection_regret_mean": 0.1,
                        },
                        "ccr_auto_e2e": [{"threshold": 0.5, "value": 0.6}],
                    },
                    "patient:8": {
                        "gt_instance_count": 1,
                        "candidate_iou": {
                            "best_mean": 0.8,
                            "selected_standard_candidate_mean": 0.7,
                            "selection_regret_mean": 0.1,
                        },
                        "ccr_auto_e2e": [{"threshold": 0.5, "value": 0.7}],
                    },
                }
            }
            (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (directory / "images.json").write_text(
                json.dumps(
                    [
                        {"sample_id": "07_1", "patient": 7, "final_task_metrics": strict_metrics()},
                        {"sample_id": "08_1", "patient": 8, "final_task_metrics": strict_metrics()},
                    ]
                ),
                encoding="utf-8",
            )
            with (directory / "gt_instances.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["patient", "auto_selected_candidate_iou"])
                writer.writeheader()
                writer.writerow({"patient": "7", "auto_selected_candidate_iou": "0.6"})
                writer.writerow({"patient": "8", "auto_selected_candidate_iou": "0.7"})

            record = diagnosis_record("c0", 1, directory)

        self.assertEqual(record["arm"], "c0")
        self.assertEqual(record["epoch"], 1)
        self.assertEqual(record["patients"]["7"]["image_count"], 1)


if __name__ == "__main__":
    unittest.main()
