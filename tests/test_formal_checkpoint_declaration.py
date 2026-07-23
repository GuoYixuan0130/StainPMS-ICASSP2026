from __future__ import annotations

from pathlib import Path
import unittest

from stainpms.formal_checkpoint_declaration import enrich_declaration_from_state


def state() -> dict[str, object]:
    return {
        "phase": "2A-warmstart-c2-ar",
        "protocol": "tnbc_c2_ar_two_seed_v1",
        "dataset": "tnbc",
        "arm": "c2_ar",
        "epoch": 5,
    }


class FormalCheckpointDeclarationTests(unittest.TestCase):
    def test_missing_provenance_is_added_without_changing_hash(self):
        declaration = {
            "dataset": "tnbc",
            "checkpoint_sha256": "abc",
            "checkpoint_path": "/tmp/state.pth",
        }
        updated, changed = enrich_declaration_from_state(
            declaration, state(), checkpoint_path=Path("/tmp/state.pth"), checkpoint_sha256="abc"
        )
        self.assertEqual(set(changed), {"phase", "protocol", "arm", "epoch"})
        self.assertEqual(updated["arm"], "c2_ar")
        self.assertEqual(updated["epoch"], 5)

    def test_conflicting_provenance_is_rejected(self):
        declaration = {"dataset": "tnbc", "checkpoint_sha256": "abc", "arm": "c0"}
        with self.assertRaisesRegex(ValueError, "conflicts"):
            enrich_declaration_from_state(
                declaration, state(), checkpoint_path=Path("/tmp/state.pth"), checkpoint_sha256="abc"
            )


if __name__ == "__main__":
    unittest.main()
