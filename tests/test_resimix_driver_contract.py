"""Command-line safety checks for the formal ResiMix driver."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FormalDriverContractTest(unittest.TestCase):
    def test_artifact_root_is_explicitly_required(self):
        result = subprocess.run(
            [sys.executable, "tools/run_resimix_stage1.py", "--spec", "unused.json"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--artifact-root", result.stderr)


if __name__ == "__main__":
    unittest.main()
