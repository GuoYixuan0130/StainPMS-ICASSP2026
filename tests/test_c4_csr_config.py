from __future__ import annotations

import json
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
