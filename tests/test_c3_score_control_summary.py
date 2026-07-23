from __future__ import annotations

import unittest

from tools.audit_c3_score_control import c3_gate


def seed_record(seed: int, *, fp: float, duplicate: float, conflict: float, merge: float, full: float):
    values = {
        "native": 0.0,
        "fp_demotion_oracle": fp,
        "duplicate_order_oracle": duplicate,
        "conflict_order_oracle": conflict,
        "merge_risk_demotion_oracle": merge,
        "full_score_oracle": full,
    }
    return {
        "seed": seed,
        "patient_macro": {
            "deltas_vs_native_patient_macro": {name: {"pq": value} for name, value in values.items()},
            "retention_count_preservation": {
                name: {"all_images_preserved": True} for name in values
            },
        },
    }


class C3ScoreControlSummaryTests(unittest.TestCase):
    def test_conflict_direction_needs_same_largest_operator_in_both_seeds(self):
        gate = c3_gate([
            seed_record(2027, fp=0.001, duplicate=0.002, conflict=0.005, merge=0.0, full=0.01),
            seed_record(1337, fp=0.001, duplicate=0.002, conflict=0.004, merge=0.0, full=0.009),
        ])
        self.assertEqual(gate["status"], "one_direction_supported")
        self.assertEqual(gate["single_supported_operation"], "conflict_order_oracle")

    def test_conflicting_primary_operations_close_the_route(self):
        gate = c3_gate([
            seed_record(2027, fp=0.005, duplicate=0.002, conflict=0.004, merge=0.0, full=0.01),
            seed_record(1337, fp=0.002, duplicate=0.002, conflict=0.005, merge=0.0, full=0.01),
        ])
        self.assertEqual(gate["status"], "close_assembly_scoring_route")


if __name__ == "__main__":
    unittest.main()
