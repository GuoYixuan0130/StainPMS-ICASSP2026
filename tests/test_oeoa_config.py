from __future__ import annotations

import json
import unittest
from pathlib import Path

from stainpms.oeoa import ACTION_CLASSES, ROUTES


ROOT = Path(__file__).resolve().parents[1]


class OEOAConfigTests(unittest.TestCase):
    def test_preregistered_phase3a_contract_is_fixed(self):
        payload = json.loads((ROOT / "configs" / "phase3a" / "tnbc_oeoa_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["protocol_id"], "tnbc_phase3a_orthogonal_error_oracle_audit_v1")
        self.assertEqual(payload["scope"]["development_patients"], [7, 8])
        self.assertEqual(payload["scope"]["sealed_patients"], [9, 10, 11])
        self.assertEqual(payload["scope"]["seeds"], [2027, 1337])
        self.assertEqual(tuple(payload["atomic_actions"]), ACTION_CLASSES)
        self.assertEqual({name: tuple(actions) for name, actions in payload["routes"].items()}, ROUTES)
        self.assertEqual(payload["combinatorics"]["all_atomic_subsets"], 128)
        self.assertEqual(payload["target"]["c0_relative_gain"], 0.02)


if __name__ == "__main__":
    unittest.main()
