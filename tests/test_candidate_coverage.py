import unittest

try:
    import torch
except ModuleNotFoundError:  # Local Windows audit environment may be CPU-only.
    torch = None


@unittest.skipIf(torch is None, "PyTorch is unavailable in the local audit environment")
class CandidateCoverageTests(unittest.TestCase):
    def setUp(self):
        from stainpms.candidate_coverage import (
            aggregate_candidate_prompt_groups,
            candidate_prompt_losses,
            compose_candidate_total_loss,
            stable_softmin,
        )

        self.aggregate_candidate_prompt_groups = aggregate_candidate_prompt_groups
        self.candidate_prompt_losses = candidate_prompt_losses
        self.compose_candidate_total_loss = compose_candidate_total_loss
        self.stable_softmin = stable_softmin

    def test_k1_equals_single_candidate_loss(self):
        values = torch.tensor([[0.37]], requires_grad=True)
        actual = self.stable_softmin(values, 0.1)
        torch.testing.assert_close(actual, values[:, 0])

    def test_equal_candidates_equal_common_loss(self):
        values = torch.full((3, 4), 0.42, requires_grad=True)
        actual = self.stable_softmin(values, 0.1)
        torch.testing.assert_close(actual, torch.full((3,), 0.42))

    def test_candidate_permutation_invariant(self):
        values = torch.tensor([[0.8, 0.2, 0.5, 0.4]])
        permuted = values[:, [2, 0, 3, 1]]
        torch.testing.assert_close(
            self.stable_softmin(values, 0.1),
            self.stable_softmin(permuted, 0.1),
        )

    def test_improving_best_candidate_reduces_coverage(self):
        before = torch.tensor([[0.8, 0.7, 0.6, 0.5]])
        after = before.clone()
        after[0, 3] = 0.1
        self.assertLess(
            self.stable_softmin(after, 0.1).item(),
            self.stable_softmin(before, 0.1).item(),
        )

    def test_tau_point_one_forward_and_gradient_are_finite(self):
        values = torch.tensor([[0.0, 1.0, 20.0, 100.0]], requires_grad=True)
        output = self.stable_softmin(values, 0.1).sum()
        output.backward()
        self.assertTrue(torch.isfinite(output))
        self.assertTrue(torch.isfinite(values.grad).all())

    def test_better_candidate_receives_larger_gradient_weight(self):
        values = torch.tensor([[0.1, 0.5, 0.8, 1.0]], requires_grad=True)
        self.stable_softmin(values, 0.1).sum().backward()
        self.assertGreater(values.grad[0, 0].item(), values.grad[0, 1].item())
        self.assertGreater(values.grad[0, 1].item(), values.grad[0, 2].item())
        self.assertLess(values.grad[0, 3].item(), 1e-3)

    def test_softmin_does_not_apply_equal_direct_fit_to_all_candidates(self):
        values = torch.tensor([[0.05, 0.5, 0.8, 1.2]], requires_grad=True)
        self.stable_softmin(values, 0.1).sum().backward()
        self.assertAlmostEqual(values.grad.sum().item(), 1.0, places=6)
        self.assertGreater(values.grad[0, 0].item(), 0.98)
        self.assertLess(values.grad[0, 2].item(), 1e-3)
        self.assertLess(values.grad[0, 3].item(), 1e-4)

    def test_hard_iou_target_is_fully_stop_gradient(self):
        logits = torch.randn(2, 4, 8, 8, requires_grad=True)
        quality = torch.randn(2, 4, requires_grad=True)
        gt = torch.zeros(2, 8, 8)
        gt[:, 2:6, 2:6] = 1
        result = self.candidate_prompt_losses(logits, quality, gt)
        self.assertFalse(result["hard_iou_target"].requires_grad)
        result["quality_per_prompt"].sum().backward()
        self.assertIsNone(logits.grad)
        self.assertIsNotNone(quality.grad)

    def test_empty_group_is_exact_zero_and_does_not_change_other_denominator(self):
        logits = torch.randn(2, 4, 8, 8, requires_grad=True)
        quality = torch.randn(2, 4, requires_grad=True)
        gt = torch.zeros(2, 8, 8)
        gt[:, 2:6, 2:6] = 1
        empty_logits = logits.new_empty((0, 4, 8, 8))
        empty_quality = quality.new_empty((0, 4))
        empty_gt = gt.new_empty((0, 8, 8))
        one_group = {
            "ordinary": {
                "candidate_logits": logits,
                "quality_predictions": quality,
                "gt_masks": gt,
                "alpha": 1.0,
            }
        }
        with_empty = {
            **one_group,
            "pms_residual": {
                "candidate_logits": empty_logits,
                "quality_predictions": empty_quality,
                "gt_masks": empty_gt,
                "alpha": 0.15,
            },
        }
        coverage_single, quality_single, _ = self.aggregate_candidate_prompt_groups(
            one_group
        )
        coverage_empty, quality_empty, audit = self.aggregate_candidate_prompt_groups(
            with_empty
        )
        torch.testing.assert_close(coverage_single, coverage_empty)
        torch.testing.assert_close(quality_single, quality_empty)
        self.assertEqual(audit["pms_residual"]["valid_prompt_count"], 0)
        self.assertEqual(audit["pms_residual"]["coverage_mean"], 0.0)
        self.assertEqual(audit["pms_residual"]["quality_mean"], 0.0)

    def test_disabling_audit_does_not_change_training_losses(self):
        logits = torch.randn(2, 4, 8, 8, requires_grad=True)
        quality = torch.randn(2, 4, requires_grad=True)
        gt = torch.zeros(2, 8, 8)
        gt[:, 1:6, 2:7] = 1
        groups = {
            "ordinary": {
                "candidate_logits": logits,
                "quality_predictions": quality,
                "gt_masks": gt,
                "alpha": 1.0,
            }
        }
        coverage_audit, quality_audit, audit = self.aggregate_candidate_prompt_groups(
            groups, collect_audit=True
        )
        coverage_fast, quality_fast, no_audit = self.aggregate_candidate_prompt_groups(
            groups, collect_audit=False
        )
        torch.testing.assert_close(coverage_audit, coverage_fast)
        torch.testing.assert_close(quality_audit, quality_fast)
        self.assertTrue(audit)
        self.assertEqual(no_audit, {})

    def test_zero_coefficients_exactly_degenerate_to_c0(self):
        parameter_c0 = torch.tensor(0.3, requires_grad=True)
        base_c0 = parameter_c0.square()
        base_c0.backward()
        gradient_c0 = parameter_c0.grad.detach().clone()

        parameter_zero = torch.tensor(0.3, requires_grad=True)
        base_zero = parameter_zero.square()
        coverage = (parameter_zero - 2.0).square()
        quality = (parameter_zero + 4.0).square()
        total = self.compose_candidate_total_loss(
            base_zero,
            coverage,
            quality,
            coverage_coefficient=0.0,
            quality_coefficient=0.0,
        )
        total.backward()
        torch.testing.assert_close(total, base_zero)
        torch.testing.assert_close(parameter_zero.grad, gradient_c0)

    def test_per_prompt_losses_match_existing_single_pair_losses(self):
        from pytorch_toolbelt.losses import BinaryFocalLoss, DiceLoss

        torch.manual_seed(7)
        logits = torch.randn(2, 4, 7, 9)
        quality = torch.randn(2, 4)
        gt = torch.zeros(2, 7, 9)
        gt[0, 1:5, 2:7] = 1
        gt[1, 2:7, 1:6] = 1
        result = self.candidate_prompt_losses(logits, quality, gt)
        dice = DiceLoss("binary")
        focal = BinaryFocalLoss()
        for i in range(2):
            for k in range(4):
                expected_dice = dice(logits[i : i + 1, k : k + 1], gt[i : i + 1])
                expected_focal = focal(
                    logits[i : i + 1, k : k + 1],
                    gt[i : i + 1].unsqueeze(1),
                )
                torch.testing.assert_close(
                    result["dice_per_candidate"][i, k], expected_dice
                )
                torch.testing.assert_close(
                    result["focal_per_candidate"][i, k], expected_focal
                )


if __name__ == "__main__":
    unittest.main()
