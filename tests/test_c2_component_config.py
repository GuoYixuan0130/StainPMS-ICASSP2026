from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class C2ComponentConfigTests(unittest.TestCase):
    def test_exact_pre_registered_component_arms(self):
        payload = json.loads(
            (ROOT / "configs/phase2a/tnbc_c2_component_ablation_v1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["protocol_id"], "tnbc_c2_component_ablation_v1")
        self.assertEqual(payload["optimization"]["planned_attempted_crop_batches"], 1350)
        self.assertEqual(payload["optimization"]["epoch_checkpoint_retention"], "epoch5_full_state_only")
        self.assertEqual(payload["optimization"]["seeds"], [2027, 1337])
        self.assertEqual(payload["data"]["train_patients"], [1, 2, 3, 4, 5, 6])
        self.assertEqual(payload["data"]["development_patients"], [7, 8])
        self.assertEqual(payload["data"]["sealed_patients"], [9, 10, 11])
        arms = payload["arms"]
        self.assertEqual(arms["c2_e"]["coverage_coefficient"], 1.0)
        self.assertEqual(arms["c2_e"]["quality_coefficient"], 1.0)
        self.assertEqual(arms["c2_e"]["c2_ar"]["exclusivity_coefficient"], 0.25)
        self.assertEqual(arms["c2_e"]["c2_ar"]["utility_coefficient"], 0.0)
        self.assertEqual(arms["c2_u"]["c2_ar"]["exclusivity_coefficient"], 0.0)
        self.assertEqual(arms["c2_u"]["c2_ar"]["utility_coefficient"], 0.25)


if __name__ == "__main__":
    unittest.main()
