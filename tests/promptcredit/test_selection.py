from __future__ import annotations

import copy
import unittest

from promptcredit.utils.selection import build_selection_payload, derive_selected_image_ids, validate_selection_payload


def _manifest() -> dict:
    return {
        "dataset": "TNBC",
        "content_sha256": "source-checksum",
        "router_train": ["01_1", "02_1", "03_1", "04_1", "05_1", "06_1"],
        "calibration": ["07_1", "08_1"],
        "test": ["09_1", "10_1", "11_1"],
    }


class PromptCreditSelectionTest(unittest.TestCase):
    def test_deterministic_rerun(self) -> None:
        manifest = _manifest()
        self.assertEqual(derive_selected_image_ids(manifest), derive_selected_image_ids(copy.deepcopy(manifest)))
        payload = build_selection_payload(manifest)
        self.assertEqual(validate_selection_payload(payload, manifest), payload["image_ids"])

    def test_no_calibration_or_test_path_access(self) -> None:
        manifest = _manifest()
        payload = build_selection_payload(manifest)
        payload["image_ids"] = ["07_1"] + payload["image_ids"][1:]
        # A changed membership also invalidates the checksum/frozen derivation.
        with self.assertRaises(ValueError):
            validate_selection_payload(payload, manifest)


if __name__ == "__main__":
    unittest.main()

