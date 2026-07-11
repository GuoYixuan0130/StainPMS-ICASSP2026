from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None, "requires the project PyTorch environment")
class PromptCreditMethodTest(unittest.TestCase):
    def test_legacy_and_gpu_nearest_indices_equivalent(self) -> None:
        from promptcredit.method.credit import gather_nearest_coordinates, legacy_nearest_indices

        coordinates = torch.tensor([[[0.0, 0.0], [4.0, 0.0], [9.0, 0.0]]], requires_grad=True)
        targets = [torch.tensor([[0.2, 0.0], [3.8, 0.0], [8.6, 0.0]])]
        legacy = legacy_nearest_indices(coordinates, targets)[0]
        gathered = gather_nearest_coordinates(coordinates, targets)
        self.assertTrue(torch.equal(legacy, gathered.source_indices[0].cpu()))
        self.assertTrue(torch.equal(gathered.coordinates[:, 0], coordinates[0, legacy]))

    def test_gathered_coordinates_retain_gradient(self) -> None:
        from promptcredit.method.credit import gather_nearest_coordinates

        coordinates = torch.tensor([[[0.0, 0.0], [5.0, 0.0]]], requires_grad=True)
        selection = gather_nearest_coordinates(coordinates, [torch.tensor([[4.9, 0.0]])])
        selection.coordinates.sum().backward()
        self.assertIsNotNone(coordinates.grad)
        self.assertGreater(float(coordinates.grad[0, 1].abs().sum()), 0.0)

    def test_directional_alpha_scales_only_gradient(self) -> None:
        from promptcredit.method.credit import directional_credit

        coordinate = torch.tensor([[[2.0, 0.0]]], requires_grad=True)
        directional_credit(coordinate, 0.0).square().sum().backward()
        self.assertEqual(float(coordinate.grad[0, 0, 0]), 0.0)
        coordinate.grad = None
        directional_credit(coordinate, 0.1).square().sum().backward()
        self.assertAlmostEqual(float(coordinate.grad[0, 0, 0]), 0.4, places=6)

    def test_quality_targets_and_duplicate_maximum(self) -> None:
        from promptcredit.method.quality import build_quality_targets, utility_target_from_hard_iou

        values = torch.tensor([0.0, 0.49, 0.50, 0.75, 1.0])
        expected = torch.tensor([0.0, 0.2327602, 0.25, 0.6931064, 0.9933072])
        torch.testing.assert_close(utility_target_from_hard_iou(values), expected)
        targets = build_quality_targets(
            torch.zeros(1, 4),
            [torch.tensor([1, 1, 2])],
            torch.tensor([0.49, 0.75, 1.0]),
        )
        self.assertEqual(targets.matched_count, 2)
        self.assertEqual(targets.duplicate_source_events, 1)
        self.assertAlmostEqual(float(targets.values[0, 1]), float(expected[3]), places=6)
        self.assertAlmostEqual(float(targets.values[0, 2]), float(expected[4]), places=6)
        self.assertEqual(float(targets.values[0, 0]), 0.0)

    def test_quality_focal_finite_and_score_modes(self) -> None:
        from promptcredit.method.quality import (
            QualityTargets,
            prompt_ranking_scores,
            quality_focal_loss,
            quality_focal_loss_with_audit,
        )

        logits = torch.tensor([[0.0, 1.0]], requires_grad=True)
        targets = QualityTargets(
            values=torch.tensor([[0.0, 0.8]]),
            matched_proposals=torch.tensor([[False, True]]),
            matched_count=1,
            duplicate_source_events=0,
        )
        loss = quality_focal_loss(logits, targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        _, audit = quality_focal_loss_with_audit(logits.detach(), targets)
        self.assertEqual(audit.proposal_total, 2)
        self.assertEqual(audit.matched_positive_count, 1)
        self.assertEqual(audit.unmatched_negative_count, 1)
        self.assertEqual(audit.normalization_denominator, 1)
        self.assertTrue(torch.isfinite(torch.tensor(audit.positive_loss_sum)))
        foreground = torch.tensor([0.2, 0.8])
        torch.testing.assert_close(prompt_ranking_scores(foreground, None, "objectness"), foreground)
        quality = torch.tensor([0.0, 0.0])
        torch.testing.assert_close(prompt_ranking_scores(foreground, quality, "quality"), torch.tensor([0.5, 0.5]))
        torch.testing.assert_close(prompt_ranking_scores(foreground, quality, "objectness_x_quality"), foreground * 0.5)

    def test_freeze_optimizer_checksum_and_legacy_checkpoint(self) -> None:
        from promptcredit.method.checkpoint import load_point_checkpoint_compat
        from promptcredit.method.freeze import (
            configure_promptcredit_v1_trainable,
            frozen_parameters_have_no_grad,
            module_state_sha256,
            optimizer_excludes_frozen,
        )

        class Point(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = torch.nn.Linear(2, 2)
                self.conv = torch.nn.Linear(2, 2)
                self.deform_layer = torch.nn.Linear(2, 2)
                self.reg_head = torch.nn.Linear(2, 2)
                self.cls_head = torch.nn.Linear(2, 2)
                self.quality_head = torch.nn.Linear(2, 1)

            def forward(self, value):
                return self.quality_head(self.cls_head(self.reg_head(self.deform_layer(self.conv(value)))))

        point, sam = Point(), torch.nn.Linear(2, 2)
        legacy_state = {key: value for key, value in point.state_dict().items() if not key.startswith("quality_head.")}
        compatibility = load_point_checkpoint_compat(point, legacy_state)
        self.assertEqual(compatibility["unexpected_keys"], [])
        self.assertEqual(load_point_checkpoint_compat(point, point.state_dict())["missing_keys"], [])
        manifest = configure_promptcredit_v1_trainable(point, sam)
        self.assertLess(manifest["quality_head_parameter_count"], 100_000)
        optimizer = torch.optim.AdamW([parameter for parameter in point.parameters() if parameter.requires_grad], lr=1e-4)
        self.assertTrue(optimizer_excludes_frozen(optimizer))
        before = module_state_sha256(sam)
        optimizer.zero_grad()
        point(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        self.assertTrue(frozen_parameters_have_no_grad(sam))
        self.assertEqual(before, module_state_sha256(sam))

    def test_quality_head_initialization_is_seed_3407_deterministic_and_neutral(self) -> None:
        from sam2_train.modeling.dpa_p2pnet import DPAP2PNet

        class DummyBackbone(torch.nn.Module):
            def forward(self, images):
                raise AssertionError("construction-only test")

        torch.manual_seed(1)
        first = DPAP2PNet(DummyBackbone(), num_levels=1, num_classes=1, hidden_dim=4, enable_quality_head=True)
        torch.manual_seed(999)
        second = DPAP2PNet(DummyBackbone(), num_levels=1, num_classes=1, hidden_dim=4, enable_quality_head=True)
        for first_parameter, second_parameter in zip(first.quality_head.parameters(), second.quality_head.parameters(), strict=True):
            torch.testing.assert_close(first_parameter, second_parameter)

        first.eval()
        logits = first.quality_head(torch.randn(1, 2, 3, 4))
        expected = torch.full_like(logits, float(torch.logit(torch.tensor(0.01))))
        torch.testing.assert_close(logits, expected)
        objectness = torch.tensor([0.2, 0.8, 0.5])
        quality_logits = logits.flatten()[:3]
        self.assertTrue(
            torch.equal(
                torch.argsort(objectness, descending=True, stable=True),
                torch.argsort(objectness * torch.sigmoid(quality_logits), descending=True, stable=True),
            )
        )

    def test_quality_head_does_not_change_legacy_coordinates_or_logits(self) -> None:
        from sam2_train.modeling.dpa_p2pnet import DPAP2PNet

        class DummyBackbone(torch.nn.Module):
            def forward(self, images):
                batch = images.shape[0]
                features = torch.ones(batch, 4, 4, 4, device=images.device)
                return [features], features

        torch.manual_seed(91)
        legacy = DPAP2PNet(DummyBackbone(), num_levels=1, num_classes=1, hidden_dim=4, dropout=0.0)
        torch.manual_seed(91)
        promptcredit = DPAP2PNet(
            DummyBackbone(), num_levels=1, num_classes=1, hidden_dim=4, dropout=0.0, enable_quality_head=True
        )
        legacy.eval()
        promptcredit.eval()
        image = torch.zeros(1, 3, 16, 16)
        legacy_output, *_ = legacy(image)
        promptcredit_output, *_ = promptcredit(image)
        torch.testing.assert_close(legacy_output["pred_coords"], promptcredit_output["pred_coords"])
        torch.testing.assert_close(legacy_output["pred_logits"], promptcredit_output["pred_logits"])

    def test_neutral_quality_keeps_point_nms_actions_identical(self) -> None:
        from promptcredit.smoke.runner import _prompt_action_source_ids

        output = {
            "pred_coords": torch.tensor([[[2.0, 2.0], [4.0, 2.0], [20.0, 20.0]]]),
            "pred_logits": torch.tensor([[[4.0, -4.0], [3.0, -3.0], [2.0, -2.0]]]),
            "pred_quality_logits": torch.full((1, 3), float(torch.logit(torch.tensor(0.01)))),
            "pred_masks": torch.ones(1, 1, 32, 32),
        }
        objectness_actions = _prompt_action_source_ids(
            output, mode="objectness", nms_radius=4.0, semantic_filtering=True
        )
        combined_actions = _prompt_action_source_ids(
            output, mode="objectness_x_quality", nms_radius=4.0, semantic_filtering=True
        )
        self.assertTrue(torch.equal(objectness_actions, combined_actions))


if __name__ == "__main__":
    unittest.main()
