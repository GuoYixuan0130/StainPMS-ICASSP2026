from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class C3ScoreControlConfigTests(unittest.TestCase):
    def test_c3_configuration_locks_two_seed_read_only_scope(self):
        payload = json.loads(
            (ROOT / "configs/phase2a/tnbc_c3_score_control_audit_v1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["protocol_id"], "tnbc_c3_score_control_feasibility_audit_v1")
        self.assertEqual(payload["scope"]["allowed_patients"], [7, 8])
        self.assertEqual(payload["scope"]["allowed_seeds"], [2027, 1337])
        self.assertTrue(payload["scope"]["no_training"])
        self.assertIsNone(payload["fixed_native_assembly"]["score_keep_reject_threshold"])


if __name__ == "__main__":
    unittest.main()
