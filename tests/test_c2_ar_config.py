from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class C2ARConfigTests(unittest.TestCase):
    def test_frozen_two_seed_contract(self):
        payload = json.loads(
            (ROOT / "configs/phase2a/tnbc_c2_ar_two_seed_v1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["protocol_id"], "tnbc_c2_ar_two_seed_v1")
        self.assertEqual(payload["optimization"]["seeds"], [2027, 1337])
        self.assertEqual(payload["optimization"]["epochs"], 5)
        self.assertEqual(payload["optimization"]["planned_attempted_crop_batches"], 1350)
        self.assertEqual(payload["optimization"]["epoch_checkpoint_retention"], "all_full_states")
        self.assertEqual(payload["arms"]["c2_ar"]["coverage_coefficient"], 1.0)
        self.assertEqual(payload["arms"]["c2_ar"]["quality_coefficient"], 1.0)
        self.assertEqual(payload["arms"]["c2_ar"]["c2_ar"]["exclusivity_coefficient"], 0.25)
        self.assertEqual(payload["arms"]["c2_ar"]["c2_ar"]["utility_coefficient"], 0.25)
        self.assertEqual(payload["storage"]["minimum_free_gib"], 40.0)


if __name__ == "__main__":
    unittest.main()
