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


class SmokeCLIContractTest(unittest.TestCase):
    def test_omitted_crop_manifests_remain_none(self):
        from tools.resimix_smoke import _parse_options

        options = _parse_options([
            "--dataset", "tnbc", "--data-path", "data", "--train-manifest", "train.json",
            "--test-manifest", "dev.json", "--coverage-manifest", "coverage.json",
            "--resimix-config", "resimix.json", "--output-dir", "out", "--overlap", "32",
            "--train-image-root", "images", "--train-label-root", "labels",
            "--test-image-root", "images", "--test-label-root", "labels",
        ])
        self.assertIsNone(options.train_crop_manifest)
        self.assertIsNone(options.eval_crop_manifest)


if __name__ == "__main__":
    unittest.main()
