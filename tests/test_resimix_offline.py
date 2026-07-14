"""Offline initialization guard for the formal ResiMix protocol."""
from __future__ import annotations

from pathlib import Path
import runpy
import unittest


ROOT = Path(__file__).resolve().parents[1]


class OfflineInitializationTest(unittest.TestCase):
    def test_prompter_does_not_request_an_external_pretrained_backbone(self):
        config = runpy.run_path(str(ROOT / "args.py"))
        self.assertIs(config["prompter"]["backbone"]["pretrained"], False)


if __name__ == "__main__":
    unittest.main()
