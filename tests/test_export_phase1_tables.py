import csv
import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "export_phase1_tables", ROOT / "tools" / "export_phase1_tables.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def ratio(numerator, denominator):
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator,
    }


class ExportPhase1TablesTests(unittest.TestCase):
    def make_source(self, root: Path):
        block = {
            "gt_instance_count": 4,
            "auto_point_recall": ratio(2, 4),
            "ccr_gt_point": [ratio(4, 4) | {"threshold": t} for t in (0.3, 0.5, 0.7)],
            "ccr_auto_given_point": [ratio(2, 2) | {"threshold": t} for t in (0.3, 0.5, 0.7)],
            "ccr_auto_e2e": [ratio(2, 4) | {"threshold": t} for t in (0.3, 0.5, 0.7)],
            "candidate_iou": {
                "best_mean": 0.8,
                "selected_standard_candidate_mean": 0.7,
                "selection_regret_mean": 0.1,
                "qualified_candidate_count": 2,
                "qualified_but_not_final_count": 1,
            },
            "error_classes": {
                "final_matched_tp": 2,
                "point_miss": 1,
                "candidate_generation_miss": 1,
                "selection_ranking_miss": 0,
                "assembly_nms_conflict_miss": 0,
            },
            "image_count": 1,
            "auto_points": {
                "count": 3,
                "background_count": 1,
                "background_fraction": 1 / 3,
                "per_gt_count_mean": 0.5,
            },
            "final_structural_errors": {
                "tp": 2,
                "fp": 1,
                "fn": 2,
                "split_unmatched_gt_count": 0,
                "merge_unmatched_pred_count": 0,
                "boundary_localization_unmatched_gt_count": 1,
            },
        }
        summary = {
            "status": "complete",
            "dataset": "tnbc",
            "scope_label": "tnbc_p1_6",
            "metric_spec": {"sha256": "metric"},
            "checkpoint": {
                "checkpoint_sha256": "checkpoint",
                "classification": "historical_exploratory",
                "selection_history": "unknown",
                "training_manifest": "unknown",
            },
            "manifest": {"sha256": "manifest", "protocol_id": "protocol"},
            "repository": {"commit": "commit"},
            "overall": block,
            "groups": {"patient:1": block},
        }
        (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        rows = [
            {"sample_id": "01_1", "gt_instance_id": 1, "patient": 1, "auto_point_count": 1, "final_matched": True, "error_class": "final_matched_tp"},
            {"sample_id": "01_1", "gt_instance_id": 2, "patient": 1, "auto_point_count": 1, "final_matched": False, "error_class": "candidate_generation_miss"},
            {"sample_id": "01_1", "gt_instance_id": 3, "patient": 1, "auto_point_count": 0, "final_matched": True, "error_class": "final_matched_tp"},
            {"sample_id": "01_1", "gt_instance_id": 4, "patient": 1, "auto_point_count": 0, "final_matched": False, "error_class": "point_miss"},
        ]
        with (root / "gt_instances.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (root / "images.json").write_text(json.dumps([{"sample_id": "01_1"}]), encoding="utf-8")

    def test_exports_direct_per_gt_contingency(self):
        test_root = ROOT / ".tmp-test" / "phase1_export_test"
        source_dir = test_root / "source"
        output_dir = test_root / "out"
        source_dir.mkdir(parents=True, exist_ok=True)
        self.make_source(source_dir)
        report = MODULE.export_tables([MODULE.read_source(source_dir)], output_dir)
        self.assertEqual(report["status"], "complete")
        with (output_dir / "phase1_point_final_contingency.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)  # cohort plus patient
        for row in rows:
            self.assertEqual(row["with_own_point_final_matched"], "1")
            self.assertEqual(row["with_own_point_final_fn"], "1")
            self.assertEqual(row["without_own_point_final_matched"], "1")
            self.assertEqual(row["without_own_point_final_fn"], "1")

    def test_rejects_partition_disagreement(self):
        rows = [
            {"auto_point_count": 0, "final_matched": False, "error_class": "point_miss"}
        ]
        block = {
            "gt_instance_count": 1,
            "auto_point_recall": ratio(0, 1),
            "error_classes": {name: 0 for name in MODULE.ERROR_CLASSES},
        }
        with self.assertRaisesRegex(ValueError, "error counts"):
            MODULE.validate_partition(rows, block, "broken")


if __name__ == "__main__":
    unittest.main()
