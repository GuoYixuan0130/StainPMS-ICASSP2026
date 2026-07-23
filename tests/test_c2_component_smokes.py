from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from compare_c2_component_smokes import verify_arm  # noqa: E402


class C2ComponentSmokeTests(unittest.TestCase):
    @staticmethod
    def smoke(arm: str, exclusivity, utility):
        return {
            "status": "complete",
            "training_configuration": {"arm": arm},
            "losses": {
                "loss_c2_ar_exclusivity": exclusivity,
                "loss_c2_ar_utility": utility,
            },
        }

    def test_exclusivity_arm_accepts_absent_disabled_utility(self):
        report = verify_arm(
            self.smoke("c2_e", 0.05, None),
            "c2_e",
            exclusivity=0.25,
            utility=0.0,
        )
        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["inactive_term_zero_or_absent"])

    def test_utility_arm_accepts_absent_disabled_exclusivity(self):
        report = verify_arm(
            self.smoke("c2_u", None, 0.05),
            "c2_u",
            exclusivity=0.0,
            utility=0.25,
        )
        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["inactive_term_zero_or_absent"])


if __name__ == "__main__":
    unittest.main()
