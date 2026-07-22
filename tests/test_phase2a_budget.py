import unittest

from stainpms.phase2a_budget import estimate_dataset_budget


def timing(profile, seconds, refresh=0.0):
    return {
        "status": "complete",
        "profile": profile,
        "data": {"manifest_sha256": "m", "protocol_id": "p"},
        "initialization": {"checkpoint_sha256": "c"},
        "timed": {
            "seconds_per_optimizer_update": seconds,
            "peak_memory_allocated_mib": 100,
        },
        "coverage_refresh": {"wall_seconds": refresh},
    }


class Phase2ABudgetTests(unittest.TestCase):
    def recipe(self):
        return {
            "optimization": {"epochs": 20, "crop_batch_size": 1},
            "stainpms": {"start_epoch": 5, "expected_refresh_count": 2},
            "timing": {"single_dataset_stop_gpu_hours": 1.0},
            "datasets": {
                "tnbc": {
                    "optimizer_updates_per_epoch": 10,
                    "planned_optimizer_updates": 200,
                    "checkpoint_count": 2,
                    "evaluation_seconds_per_checkpoint_proxy": 5,
                }
            },
        }

    def test_accounting_and_gate(self):
        report = estimate_dataset_budget(
            self.recipe(), "tnbc", timing("base", 1.0), timing("pms_active", 2.0, 10.0)
        )
        self.assertEqual(report["planned"]["base_optimizer_updates"], 50)
        self.assertEqual(report["planned"]["pms_active_optimizer_updates"], 150)
        self.assertEqual(report["estimated_total_seconds"], 380.0)
        self.assertEqual(report["status"], "gate_pass")

    def test_rejects_mismatched_manifest(self):
        active = timing("pms_active", 1.0)
        active["data"]["manifest_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "manifest_sha256"):
            estimate_dataset_budget(self.recipe(), "tnbc", timing("base", 1.0), active)


if __name__ == "__main__":
    unittest.main()
