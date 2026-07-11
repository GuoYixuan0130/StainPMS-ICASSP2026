from __future__ import annotations

import inspect
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None, "requires the project PyTorch environment")
class NuSetPostmortemFusionTest(unittest.TestCase):
    def test_fixed_fusions_have_expected_shapes_and_equal_mean(self) -> None:
        from nuset.postmortem.fusion import FIXED_FUSIONS, fixed_fusions

        logits = torch.stack((torch.zeros(1, 2, 2), torch.ones(1, 2, 2), torch.full((1, 2, 2), 2.0), torch.full((1, 2, 2), 3.0)), dim=1)
        result = fixed_fusions(logits, size=4)
        self.assertEqual(tuple(result["equal_logit_mean"].shape), (1, 4, 4))
        self.assertTrue(torch.allclose(result["equal_logit_mean"], torch.full((1, 4, 4), 1.5)))
        self.assertEqual(set(result), set(FIXED_FUSIONS))

    def test_hard_majority_uses_token0_for_exact_ties(self) -> None:
        from nuset.postmortem.fusion import hard_majority_logits

        logits = torch.tensor([[[[1.0, -1.0]], [[1.0, 1.0]], [[-1.0, 1.0]], [[-1.0, -1.0]]]])
        result = hard_majority_logits(logits)
        self.assertTrue(torch.equal(result > 0, torch.tensor([[[True, False]]])))

    def test_convex_library_has_fixed_35_simplex_weights(self) -> None:
        from nuset.postmortem.fusion import simplex_weights

        weights = simplex_weights()
        self.assertEqual(tuple(weights.shape), (35, 4))
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(35)))

    def test_failure_mode_matches_preregistered_rules(self) -> None:
        from nuset.postmortem.metrics import failure_mode

        existing = {"top1_accuracy": .2, "mean_oracle_regret": .2, "selected_mask_mean_iou": .3}
        weak = {"top1_accuracy": .22, "mean_oracle_regret": .19, "selected_mask_mean_iou": .304}
        result = failure_mode(train_existing=existing, train_nurank=weak, development_existing=existing, development_nurank=weak, development_single=existing, development_pq_delta=.0)
        self.assertEqual(result["failure_mode"], "representation_or_objective_failure")

    def test_boundary_band_does_not_cover_deep_interior_or_exterior(self) -> None:
        from scipy.ndimage import distance_transform_edt

        truth = torch.zeros(15, 15, dtype=torch.bool).numpy()
        truth[3:12, 3:12] = True
        boundary = (truth & (distance_transform_edt(truth) <= 3)) | (~truth & (distance_transform_edt(~truth) <= 3))
        self.assertFalse(bool(boundary[7, 7]))
        self.assertFalse(bool(boundary[0, 0]))
        self.assertTrue(bool(boundary[3, 7]))

    def test_postmortem_source_has_no_optimizer_or_backward(self) -> None:
        from nuset.postmortem import runner

        source = inspect.getsource(runner)
        self.assertNotIn("optimizer.step", source)
        self.assertNotIn(".backward(", source)
        self.assertIn("TIME_CAP_SECONDS = 45 * 60", source)


if __name__ == "__main__":
    unittest.main()
