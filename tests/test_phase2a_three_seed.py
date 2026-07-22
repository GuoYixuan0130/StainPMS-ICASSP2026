import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "phase2a_three_seed_summary", ROOT / "tools" / "summarize_phase2a_tnbc_three_seed.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def aggregate(aji_values, pq_values):
    return {
        "paired_c1_minus_c0": {
            "patient_macro": {
                "task_metrics_image_macro": {
                    "aji": {"mean": sum(aji_values) / 3, "values_by_seed": dict(zip(("3407", "2027", "1337"), aji_values))},
                    "pq": {"mean": sum(pq_values) / 3, "values_by_seed": dict(zip(("3407", "2027", "1337"), pq_values))},
                }
            }
        }
    }


class ThreeSeedAdvancementTests(unittest.TestCase):
    def test_pass_requires_all_three_frozen_conditions(self):
        decision = MODULE.advancement(aggregate([0.01, 0.02, -0.01], [0.01, -0.01, 0.02]))
        self.assertEqual(decision["status"], "pass_freeze_c1_and_request_owner_test_decision")

    def test_negative_mean_fails_even_when_two_seeds_are_positive(self):
        decision = MODULE.advancement(aggregate([0.01, 0.02, -0.01], [0.01, 0.001, -0.02]))
        self.assertEqual(decision["status"], "fail_stop_current_warmstart_route")


if __name__ == "__main__":
    unittest.main()
