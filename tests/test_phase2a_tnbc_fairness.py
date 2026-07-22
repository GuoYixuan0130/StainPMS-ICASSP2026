import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from verify_phase2a_tnbc_fairness import stable_sha256, verify


def summary():
    epochs = []
    for epoch in range(1, 6):
        positions = [{"epoch_index": epoch - 1, "image_loader_index": 1, "crop_start_index": 2, "crop_end_index": 3, "global_crop_batch_index": epoch * 270 - 1, "index_sha256": "position"}]
        epochs.append(
            {
                "epoch": epoch,
                "attempted_crop_batches": 270,
                "effective_optimizer_updates": 269,
                "no_prompt_batch_count": 1,
                "no_prompt_batch_indices": positions,
                "no_prompt_batch_indices_sha256": stable_sha256(positions),
                "optimizer_updates": epoch * 269,
                "learning_rate_after_scheduler_step": 1e-5,
                "scheduler_state_after_step": {"last_epoch": epoch},
            }
        )
    return {
        "protocol": "tnbc_c0_c1_5epoch_exploratory_v1",
        "dataset": "tnbc",
        "screen_config": {"sha256": "config"},
        "data": {"manifest_sha256": "manifest", "coverage": {"sha256": "coverage"}},
        "determinism": {"seed": 3407},
        "planned_attempted_crop_batches": 1350,
        "runtime": {"crop_batches_seen": 1350},
        "epochs": epochs,
    }


class TnbcFairnessTests(unittest.TestCase):
    def test_equal_crop_paths_pass(self):
        result = verify(summary(), summary())
        self.assertEqual(result["status"], "pass")

    def test_no_prompt_position_mismatch_fails_closed(self):
        c0, c1 = summary(), summary()
        c1["epochs"][2]["no_prompt_batch_indices"][0]["crop_start_index"] = 7
        c1["epochs"][2]["no_prompt_batch_indices_sha256"] = stable_sha256(
            c1["epochs"][2]["no_prompt_batch_indices"]
        )
        result = verify(c0, c1)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["checks"][2]["status"], "fail")


if __name__ == "__main__":
    unittest.main()
