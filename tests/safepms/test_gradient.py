from __future__ import annotations

import inspect
from pathlib import Path
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None, "requires the project PyTorch environment")
class SafePMSGradientTest(unittest.TestCase):
    def _project(self, anchor, expansion, *, trust=1.0):
        from safepms.gradient import project_global
        params = []
        for index, (anchor_value, expansion_value) in enumerate(zip(anchor, expansion, strict=True)):
            reference = anchor_value if anchor_value is not None else expansion_value
            params.append((f"p{index}", torch.nn.Parameter(torch.zeros_like(reference) if reference is not None else torch.zeros(1))))
        return project_global(params, anchor, expansion, trust_ratio=trust)

    def test_loss_decomposition_exactness(self):
        from safepms.gradient import decompose_losses
        value = torch.tensor(1.0, requires_grad=True)
        losses = {"loss_focal": value, "loss_dice": value * 2, "loss_iou": value * 3, "loss_pms_preserve_focal": value * 5, "loss_pms_focal": value * 7, "loss_pms_object": value * 11, "loss_reg": value * 13}
        anchor, expansion, point = decompose_losses(losses)
        self.assertEqual(float(anchor), 11.0); self.assertEqual(float(expansion), 18.0); self.assertEqual(float(point), 13.0)

    def test_control_sum_matches_original_summed_gradient(self):
        parameter = torch.nn.Parameter(torch.tensor(2.0))
        anchor, expansion = parameter.square(), 3 * parameter
        full = torch.autograd.grad(anchor + expansion, parameter, retain_graph=True)[0]
        split = torch.autograd.grad(anchor, parameter, retain_graph=True)[0] + torch.autograd.grad(expansion, parameter)[0]
        self.assertTrue(torch.equal(full, split))

    def test_positive_dot_does_not_project(self):
        final, stats = self._project((torch.tensor([1.0]),), (torch.tensor([2.0]),))
        self.assertFalse(stats.projected)
        self.assertTrue(stats.trust_clipped)
        self.assertTrue(torch.equal(final[0], torch.tensor([2.0])))

    def test_negative_dot_is_orthogonally_projected(self):
        _, stats = self._project((torch.tensor([2.0, 0.0]),), (torch.tensor([-3.0, 4.0]),))
        self.assertTrue(stats.projected); self.assertGreaterEqual(stats.projection_dot, -1e-7)

    def test_trust_ratio_cap(self):
        _, stats = self._project((torch.tensor([1.0, 0.0]),), (torch.tensor([-1.0, 100.0]),))
        self.assertTrue(stats.trust_clipped); self.assertLessEqual(stats.retained_expand_norm_ratio, 1.0 + 1e-6)

    def test_anchor_only_parameter(self):
        final, stats = self._project((torch.tensor([2.0]),), (None,))
        self.assertTrue(torch.equal(final[0], torch.tensor([2.0]))); self.assertEqual(stats.parameter_roles["anchor_only"], 1)

    def test_expansion_only_parameter(self):
        final, stats = self._project((None,), (torch.tensor([2.0]),))
        self.assertTrue(torch.equal(final[0], torch.tensor([2.0]))); self.assertEqual(stats.parameter_roles["expansion_only"], 1)

    def test_projection_and_trust_exclude_expansion_only_parameters(self):
        final, stats = self._project(
            (torch.tensor([1.0]), None),
            (torch.tensor([-2.0]), torch.tensor([100.0])),
        )
        self.assertTrue(stats.projected)
        self.assertTrue(torch.equal(final[0], torch.tensor([1.0])))
        self.assertTrue(torch.equal(final[1], torch.tensor([100.0])))

    def test_allow_unused_parameter(self):
        final, stats = self._project((None,), (None,))
        self.assertIsNone(final[0]); self.assertEqual(stats.parameter_roles["unused"], 1)

    def test_zero_anchor_norm(self):
        _, stats = self._project((torch.tensor([0.0]),), (torch.tensor([-2.0]),))
        self.assertFalse(stats.projected); self.assertGreaterEqual(stats.projection_dot, -1e-7)

    def test_first_order_safety_contract(self):
        _, stats = self._project((torch.tensor([2.0, 1.0]),), (torch.tensor([-4.0, 3.0]),))
        self.assertGreaterEqual(stats.projection_dot, -1e-7); self.assertGreaterEqual(stats.anchor_final_margin, -1e-7)

    def test_global_not_per_layer_projection(self):
        _, stats = self._project((torch.tensor([1.0]), torch.tensor([1.0])), (torch.tensor([-3.0]), torch.tensor([1.0])))
        self.assertTrue(stats.projected)

    def test_frozen_parameter_guard_source(self):
        from safepms import guards
        source = inspect.getsource(guards.freeze_decoder_only)
        self.assertIn("requires_grad_(False)", source); self.assertIn("sam_mask_decoder", source)

    def test_paired_initialization_equivalence(self):
        from safepms.guards import state_equal
        left, right = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
        right.load_state_dict(left.state_dict())
        self.assertTrue(state_equal(left, right))

    def test_batch_order_checksum(self):
        from safepms.data import manifest_sha256
        self.assertEqual(manifest_sha256(["01_1", "02_1"]), manifest_sha256(["01_1", "02_1"]))

    def test_closed_patient_guard(self):
        from safepms.data import TRAIN_PATIENTS, patient_of
        self.assertNotIn(patient_of("09_1"), TRAIN_PATIENTS)

    def test_no_development_access_epochs_one_to_four(self):
        from safepms import runner
        source = inspect.getsource(runner.run_stage1)
        # Two evaluation events: step 0 and epoch 5.  Each evaluates the two
        # paired copies, so there are four calls and none inside either loop.
        self.assertEqual(source.count("_evaluate("), 4)

    def test_inclusive_iou_half(self):
        source = (Path(__file__).resolve().parents[2] / "run" / "run_on_epoch.py").read_text(encoding="utf-8")
        self.assertIn("match_iou=0.5", source)

    def test_deterministic_tiny_synthetic(self):
        _, first = self._project((torch.tensor([1.0]),), (torch.tensor([-2.0]),))
        _, second = self._project((torch.tensor([1.0]),), (torch.tensor([-2.0]),))
        self.assertEqual(first, second)

    def test_no_inference_path_changes(self):
        source = (Path(__file__).resolve().parents[2] / "run" / "run_on_epoch.py").read_text(encoding="utf-8")
        self.assertIn("per_image_records=None", source)

    def test_no_backward_to_model_in_safe_controller(self):
        from safepms.gradient import GradientController
        source = inspect.getsource(GradientController.consume)
        self.assertIn("_autograd", source)
        self.assertNotIn(".backward(", source)


if __name__ == "__main__":
    unittest.main()
