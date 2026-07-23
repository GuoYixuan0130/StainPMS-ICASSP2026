from __future__ import annotations

import unittest

from tools.summarize_c2_component_ablation import component_gate


def diff(*, pq=0.01, aji=0.01, dq=0.0, selected=0.01, assembly=-0.01, fp_penalty=-0.01, merge=-1.0):
    return {
        "stages": {
            "native_final": {"pq": pq, "aji": aji, "dq": dq},
            "native_selected_pool_oracle": {"pq": selected},
        },
        "errors": {"merge_overlap_fraction_gt_or_pred_gt_0": merge},
        "gaps": {
            "selected_to_final_assembly_gap": {"pq": assembly},
            "final_fp_penalty": {"pq": fp_penalty},
        },
    }


def seed_record(*, e=None, u=None):
    return {
        "scopes": {"patient_macro": {"c2_e_minus_c0": e or diff(), "c2_e_minus_c1": e or diff(), "c2_u_minus_c0": u or diff(), "c2_u_minus_c1": u or diff()}},
        "mechanisms": {
            "c1": {"exclusivity": {"hard_foreign_gt_fraction_mean": 0.2, "hard_overlap_positive_pair_fraction": 0.2}, "score": {"auroc": 0.6}},
            "c2_e": {"exclusivity": {"hard_foreign_gt_fraction_mean": 0.1, "hard_overlap_positive_pair_fraction": 0.1}, "score": {"auroc": 0.6}},
            "c2_u": {"exclusivity": {"hard_foreign_gt_fraction_mean": 0.2, "hard_overlap_positive_pair_fraction": 0.2}, "score": {"auroc": 0.7}},
        },
    }


class C2ComponentSummaryTests(unittest.TestCase):
    def test_exclusivity_gate_needs_assembly_and_native_preservation(self):
        self.assertEqual(component_gate([seed_record(), seed_record()], "c2_e")["status"], "mechanism_supported")
        failed = component_gate([seed_record(e=diff(assembly=0.01)), seed_record()], "c2_e")
        self.assertEqual(failed["status"], "not_supported")

    def test_utility_gate_needs_final_fp_penalty_reduction(self):
        self.assertEqual(component_gate([seed_record(), seed_record()], "c2_u")["status"], "mechanism_supported")
        failed = component_gate([seed_record(u=diff(fp_penalty=0.01)), seed_record()], "c2_u")
        self.assertEqual(failed["status"], "not_supported")


if __name__ == "__main__":
    unittest.main()
