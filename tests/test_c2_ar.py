import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is unavailable in the local audit environment")
class C2ARTests(unittest.TestCase):
    def setUp(self):
        from stainpms.c2_ar import (
            c2_ar_losses,
            compose_c2_ar_total_loss,
            selected_mask_exclusivity_loss,
            unique_tp_utility_loss,
        )

        self.c2_ar_losses = c2_ar_losses
        self.compose_c2_ar_total_loss = compose_c2_ar_total_loss
        self.selected_mask_exclusivity_loss = selected_mask_exclusivity_loss
        self.unique_tp_utility_loss = unique_tp_utility_loss

    @staticmethod
    def two_instances(*, nearby=False):
        gt = torch.zeros(2, 12, 12)
        gt[0, 3:6, 2:5] = 1
        if nearby:
            gt[1, 3:6, 6:9] = 1
        else:
            gt[1, 8:11, 8:11] = 1
        return gt

    def test_no_nearby_nuclei_has_negligible_conflict(self):
        gt = self.two_instances(nearby=False)
        logits = torch.full_like(gt, -20.0, requires_grad=True)
        with torch.no_grad():
            logits[0, 3:6, 2:5] = 20.0
            logits[1, 8:11, 8:11] = 20.0
        loss, audit = self.selected_mask_exclusivity_loss(
            logits, gt, [2], neighbor_radius=1
        )
        self.assertEqual(audit["neighbor_pair_count"], 0)
        self.assertLess(audit["conflict"], 1.0e-8)
        self.assertLess(loss.item(), 1.0e-6)

    def test_cross_nucleus_leakage_increases_exclusivity_loss(self):
        gt = self.two_instances(nearby=True)
        clean = torch.full_like(gt, -12.0)
        clean[0, 3:6, 2:5] = 12.0
        clean[1, 3:6, 6:9] = 12.0
        leaked = clean.clone()
        leaked[0, 3:6, 6:9] = 12.0
        clean_loss, _ = self.selected_mask_exclusivity_loss(
            clean.requires_grad_(), gt, [2], neighbor_radius=2
        )
        leaked_loss, audit = self.selected_mask_exclusivity_loss(
            leaked.requires_grad_(), gt, [2], neighbor_radius=2
        )
        self.assertGreater(leaked_loss.item(), clean_loss.item())
        # One of the two selected masks entirely occupies the other nucleus,
        # while the other mask has no foreign leakage.  The group mean is
        # therefore exactly 0.5 rather than strictly larger.
        self.assertGreaterEqual(audit["foreign_leakage"], 0.5)
        self.assertGreater(audit["conflict"], 0.1)

    def test_utility_labels_count_unique_tp_duplicate_and_fp(self):
        gt = torch.zeros(3, 10, 10)
        gt[0, 1:4, 1:4] = 1
        gt[1, 1:4, 5:8] = 1
        gt[2, 6:9, 1:4] = 1
        logits = torch.full_like(gt, -12.0, requires_grad=True)
        with torch.no_grad():
            logits[0, 1:4, 1:4] = 12.0  # unique TP for GT 0
            logits[1, 1:4, 1:4] = 12.0  # duplicate against GT 0
            # third remains empty / unmatched FP
        quality = torch.tensor([0.2, 0.7, 0.8], requires_grad=True)
        loss, audit = self.unique_tp_utility_loss(logits, quality, gt, [3])
        self.assertEqual(audit["unique_tp_count"], 1)
        self.assertEqual(audit["duplicate_count"], 1)
        self.assertEqual(audit["unmatched_fp_count"], 1)
        self.assertAlmostEqual(audit["utility_target_mean"], 1.0 / 3.0, places=6)
        loss.backward()
        self.assertIsNone(logits.grad)
        self.assertIsNotNone(quality.grad)
        self.assertTrue(torch.isfinite(quality.grad).all())

    def test_merge_risk_lowers_positive_utility_target(self):
        gt = self.two_instances(nearby=True)
        logits = torch.full_like(gt, -12.0)
        logits[0, 3:6, 2:5] = 12.0
        # A partial foreign leak keeps the IoU with GT 0 strictly above the
        # evaluator's >0.5 match threshold, so it is a unique TP with a
        # reduced utility target rather than an unmatched merge.
        logits[0, 3:5, 6:8] = 12.0
        logits[1, 3:6, 6:9] = 12.0
        quality = torch.zeros(2, requires_grad=True)
        _, audit = self.unique_tp_utility_loss(
            logits.requires_grad_(), quality, gt, [2], merge_risk_overlap_fraction=0.1
        )
        self.assertGreaterEqual(audit["merge_risk_count"], 1)
        self.assertLess(audit["utility_target_mean"], 1.0)

    def test_zero_c2_coefficients_are_exact_c1_regression(self):
        parameter = torch.tensor(0.3, requires_grad=True)
        c1 = parameter.square()
        exclusivity = (parameter - 2.0).square()
        utility = (parameter + 3.0).square()
        total = self.compose_c2_ar_total_loss(
            c1,
            exclusivity,
            utility,
            exclusivity_coefficient=0.0,
            utility_coefficient=0.0,
        )
        total.backward()
        self.assertEqual(total.item(), c1.item())
        self.assertAlmostEqual(parameter.grad.item(), 0.6, places=6)

    def test_c2_combined_losses_are_finite_and_route_gradients(self):
        gt = self.two_instances(nearby=True)
        logits = torch.randn_like(gt, requires_grad=True)
        quality = torch.sigmoid(torch.randn(2, requires_grad=True))
        exclusivity, utility, _ = self.c2_ar_losses(logits, quality, gt, [2])
        total = exclusivity + utility
        total.backward()
        self.assertTrue(torch.isfinite(total))
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
