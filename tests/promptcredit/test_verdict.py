from __future__ import annotations

import unittest

from promptcredit.audit.verdict import stage0_verdict


class PromptCreditVerdictTest(unittest.TestCase):
    def test_project_lead_truth_table(self) -> None:
        cases = [
            (True, False, True, "GO"),
            (False, True, True, "GO"),
            (True, True, True, "GO"),
            (True, True, False, "NO-GO"),
            (False, False, True, "NO-GO"),
        ]
        for assignment_gap, quality_gap, actionable_gradient, expected in cases:
            with self.subTest(assignment_gap=assignment_gap, quality_gap=quality_gap, actionable_gradient=actionable_gradient):
                self.assertEqual(
                    stage0_verdict(
                        assignment_gap=assignment_gap,
                        quality_gap=quality_gap,
                        actionable_gradient=actionable_gradient,
                        acceptable_cost=True,
                    ),
                    expected,
                )

    def test_unacceptable_cost_is_no_go(self) -> None:
        self.assertEqual(
            stage0_verdict(
                assignment_gap=False,
                quality_gap=True,
                actionable_gradient=True,
                acceptable_cost=False,
            ),
            "NO-GO",
        )


if __name__ == "__main__":
    unittest.main()

