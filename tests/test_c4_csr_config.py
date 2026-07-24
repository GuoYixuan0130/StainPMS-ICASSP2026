from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.run_c4_csr import invariance_passes, load_c3_reference


ROOT = Path(__file__).resolve().parents[1]


class C4CSRConfigTests(unittest.TestCase):
    def test_preregistered_c4_contract_is_fixed(self):
        payload = json.loads((ROOT / "configs" / "phase2a" / "tnbc_c4_csr_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["protocol_id"], "tnbc_c4_conflict_set_structured_ranking_v1")
        self.assertEqual(payload["scope"]["train_patients"], [1, 2, 3, 4, 5, 6])
        self.assertEqual(payload["scope"]["development_patients"], [7, 8])
        self.assertEqual(payload["scope"]["sealed_patients"], [9, 10, 11])
        self.assertEqual(payload["scope"]["seeds"], [2027, 1337])
        self.assertEqual(payload["training"]["epochs"], 20)
        self.assertEqual(payload["training"]["learning_rate"], 0.001)
        self.assertEqual(payload["ranker"]["width"], 64)
        self.assertLessEqual(payload["ranker"]["maximum_parameter_count"], 100000)
        self.assertFalse(payload["inference"]["uses_gt"])
        self.assertFalse(payload["inference"]["uses_evaluator_matching"])

    def test_reconstructed_joint_c3_reference_supplies_seed1337_gate_denominators(self):
        reused = {"path": "/tmp/seed2027_historical_c3.json", "sha256": "a" * 64}
        def row(seed, delta, top1, pairwise, source, identity=None):
            return {
                "seed": seed,
                "source_c1_oracle_directory": source,
                "historical_c3_reused_without_rerun": reused if seed == 2027 else None,
                "source_identity": identity,
                "patient_macro": {
                    "deltas_vs_native_patient_macro": {"conflict_order_oracle": {"pq": delta}},
                    "conflicts_both_patients": {
                        "unique_tp_native_top1": {"accuracy": top1},
                        "pairwise_ordering": {"all_negative": {"accuracy": pairwise}},
                    },
                },
            }
        payload = {
            "status": "complete",
            "lineage": {"seed2027": "historical verified C1 epoch-5 lineage", "seed1337": "reconstructed C1 seed-1337 lineage"},
            "historical_seed_reuse": {"2027": reused},
            "c3_gate": {"single_supported_operation": "conflict_order_oracle"},
            "per_seed": [
                row(2027, 0.013622816956216255, 0.4530791788856305, 0.5900652282990466, "/tmp/seed2027_c1"),
                row(1337, 0.012228712451025858, 0.4424778761061947, 0.5916206261510129, "/tmp/seed1337_c1_reconstructed", {"lineage": "reconstructed C1 seed-1337 lineage", "checkpoint_sha256": "b" * 64, "frozen_epoch5_manifest_sha256": "c" * 64}),
            ],
        }
        with patch("tools.run_c4_csr.read_json", return_value=payload):
            reference = load_c3_reference(Path("/frozen/c3.json"))
        self.assertAlmostEqual(reference["1337"]["conflict_order_oracle_delta_pq"], 0.012228712451025858)
        self.assertAlmostEqual(reference["1337"]["native_top1_accuracy"], 0.4424778761061947)
        self.assertAlmostEqual(reference["1337"]["native_pairwise_accuracy"], 0.5916206261510129)

    def test_invariance_gate_requires_false_for_gt_and_evaluator_use(self):
        invariance = {
            "selected_oracle_identical": True,
            "singleton_scores_unchanged": True,
            "candidate_pool_unchanged": True,
            "mask_unchanged": True,
            "keep_threshold_unchanged": True,
            "inference_uses_gt": False,
            "inference_uses_evaluator_matching": False,
        }
        self.assertTrue(invariance_passes(invariance))
        self.assertFalse(invariance_passes({**invariance, "inference_uses_gt": True}))
        self.assertFalse(invariance_passes({**invariance, "selected_oracle_identical": False}))


if __name__ == "__main__":
    unittest.main()
