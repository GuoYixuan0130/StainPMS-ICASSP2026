import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "phase2a_second_seed_summary", ROOT / "tools" / "summarize_phase2a_tnbc_second_seed.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def delta(aji, pq):
    return {"patient_macro": {"task_metrics_image_macro": {"aji": aji, "pq": pq}}}


class SecondSeedDecisionTests(unittest.TestCase):
    def test_both_positive_is_a_repeat_signal(self):
        self.assertEqual(MODULE.decision(delta(0.001, 0.002)), "second_seed_repeat_final_task_signal")

    def test_one_positive_is_mixed(self):
        self.assertEqual(MODULE.decision(delta(0.001, -0.002)), "mixed_or_unstable_signal")

    def test_neither_positive_is_not_reproducible(self):
        self.assertEqual(MODULE.decision(delta(0.0, -0.002)), "not_reproducible_under_current_route")


if __name__ == "__main__":
    unittest.main()
