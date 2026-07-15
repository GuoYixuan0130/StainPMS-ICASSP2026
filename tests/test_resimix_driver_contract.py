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

    def test_formal_training_keeps_metrics_not_full_model_copies(self):
        from tools.run_resimix_stage1 import _training_command

        data = {
            "data_path": "data", "checkpoint_path": "initialization.pth",
            "train_manifest": "train.json", "test_manifest": "development.json",
            "train_image_root": "train_images", "train_label_root": "train_labels",
            "test_image_root": "development_images", "test_label_root": "development_labels",
        }
        command = _training_command("tnbc", data, Path("coverage.json"), Path("run"), (0, 2, 4, 6, 8, 10))
        self.assertNotIn("--save_eval_checkpoints", command)


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


class RecoveryCLIContractTest(unittest.TestCase):
    def test_recovery_commands_do_not_train_tnbc_or_save_full_models(self):
        from tools.recover_resimix_stage1 import _commands

        artifact = Path("artifact")
        data = {
            "data_path": "data", "checkpoint_path": "initialization.pth",
            "train_manifest": "train.json", "test_manifest": "development.json",
            "train_image_root": "train_images", "train_label_root": "train_labels",
            "test_image_root": "development_images", "test_label_root": "development_labels",
            "train_crop_manifest": "crops.json", "eval_crop_manifest": "patches.json",
        }
        control, resimix = _commands(artifact, data, Path("coverage.json"), Path("resimix.json"))
        self.assertNotIn("tnbc", control)
        self.assertNotIn("tnbc", resimix)
        self.assertNotIn("--save_eval_checkpoints", control)
        self.assertNotIn("--save_eval_checkpoints", resimix)


if __name__ == "__main__":
    unittest.main()
