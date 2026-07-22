import json
import unittest
from pathlib import Path

from tools.audit_warmstart_checkpoints import parse_named_path as parse_checkpoint
from tools.estimate_warmstart_feasibility_budget import estimate_c0_stages


ROOT = Path(__file__).resolve().parents[1]


class WarmStartFeasibilityTests(unittest.TestCase):
    def test_checkpoint_named_path_preserves_value_after_first_equals(self):
        name, path = parse_checkpoint("tnbc=checkpoints/model=name.pth")
        self.assertEqual(name, "tnbc")
        self.assertEqual(path.name, "model=name.pth")

    def test_budget_formula_counts_updates_and_one_initial_refresh(self):
        spec = {
            "train_images": 30,
            "screen_epochs": 5,
            "screen_planned_updates": 1350,
            "full_epochs": 10,
            "full_planned_updates": 2700,
        }
        timing = {
            "status": "complete",
            "profile": "pms_active",
            "data": {"record_count": 30},
            "timed": {"seconds_per_optimizer_update": 2.0},
            "coverage_refresh": {"wall_seconds": 60.0},
        }
        result = estimate_c0_stages(spec, timing)
        self.assertEqual(result["stages"]["screen"]["optimizer_updates"], 1350)
        self.assertEqual(result["stages"]["screen"]["C0_total_seconds"], 2760.0)
        self.assertEqual(result["stages"]["full"]["C0_total_seconds"], 5460.0)
        self.assertIsNone(result["stages"]["screen"]["C1_estimated_gpu_hours"])

    def test_frozen_proposal_has_exact_equal_budget_contract(self):
        proposal = json.loads(
            (ROOT / "configs/phase2a/warmstart_feasibility_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(proposal["datasets"]["tnbc"]["screen_planned_updates"], 1350)
        self.assertEqual(proposal["datasets"]["tnbc"]["full_planned_updates"], 2700)
        self.assertEqual(proposal["datasets"]["monuseg"]["screen_planned_updates"], 6660)
        self.assertEqual(proposal["datasets"]["monuseg"]["full_planned_updates"], 13320)
        self.assertFalse(proposal["timing_gate"]["old_200_epoch_gate_reused"])
        self.assertEqual(proposal["shared"]["seed"], 3407)
        self.assertEqual(proposal["shared"]["crop_batch_size"], 1)
        self.assertEqual(proposal["arms"]["C0"]["objective"], "unchanged StainPMS objective")
        self.assertFalse(proposal["arms"]["C0"]["candidate_auxiliary_loss"])
        self.assertTrue(proposal["arms"]["C1"]["candidate_auxiliary_loss"])

    def test_preflight_result_requires_weight_only_owner_decision(self):
        result = json.loads(
            (ROOT / "configs/phase2a/warmstart_preflight_result_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            result["status"],
            "owner_decision_required_before_implementation_or_training",
        )
        self.assertEqual(
            result["weight_warm_start_contract"]["load_checkpoint_fields"],
            ["model", "model1"],
        )
        self.assertIn(
            "texture_memory_bank_list",
            result["weight_warm_start_contract"]["do_not_load_checkpoint_fields"],
        )
        self.assertIsNone(result["equal_budget"]["tnbc"]["C1_gpu_hours"])
        self.assertIsNone(result["equal_budget"]["monuseg"]["C1_gpu_hours"])
        self.assertFalse(result["proposed_C1"]["new_parameters"])
        self.assertEqual(result["proposed_C1"]["prompt_group_weights"]["ordinary"], 1.0)
        self.assertEqual(
            result["proposed_C1"]["prompt_group_weights"]["PMS_residual"],
            "inherit pms_loss_coef * pms_residual_mask_weight",
        )


if __name__ == "__main__":
    unittest.main()
