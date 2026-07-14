"""Mechanical tests for the fixed SetPMS v1 formulation."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from sam2_train.modeling.stats_utils import get_fast_pq
from setpms import compute_setpms_loss, select_set_queries, unbalanced_sinkhorn_log


torch.manual_seed(3407)


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def _two_instance_fixture(*, include_fp: bool = False):
    gt_masks = torch.zeros(2, 8, 8)
    gt_masks[0, 1:3, 1:3] = 1.0
    gt_masks[1, 5:7, 5:7] = 1.0
    gt_points = torch.tensor([[1.5, 1.5], [5.5, 5.5]])

    mask_logits = torch.full((2 + int(include_fp), 8, 8), -12.0)
    mask_logits[0, 1:3, 1:3] = 12.0
    mask_logits[1, 5:7, 5:7] = 12.0
    coords = torch.tensor([[1.5, 1.5], [5.5, 5.5]])
    logits = torch.tensor([[8.0, -8.0], [8.0, -8.0]])
    if include_fp:
        mask_logits[2, 0:2, 6:8] = 12.0
        coords = torch.cat([coords, torch.tensor([[6.5, 0.5]])], dim=0)
        logits = torch.cat([logits, torch.tensor([[8.0, -8.0]])], dim=0)
    return mask_logits, coords, logits, gt_masks, gt_points


def test_joint_prediction_and_gt_permutation_is_invariant():
    values = _two_instance_fixture()
    reference = compute_setpms_loss(*values)
    pred_perm = torch.tensor([1, 0])
    gt_perm = torch.tensor([1, 0])
    permuted = compute_setpms_loss(
        values[0][pred_perm],
        values[1][pred_perm],
        values[2][pred_perm],
        values[3][gt_perm],
        values[4][gt_perm],
    )
    assert torch.allclose(reference.loss, permuted.loss, atol=1.0e-6)
    assert torch.allclose(reference.soft_pq, permuted.soft_pq, atol=1.0e-6)
    assert torch.allclose(reference.soft_aji, permuted.soft_aji, atol=1.0e-6)


def test_perfect_one_to_one_masks_approach_one_for_soft_aji_and_pq():
    result = compute_setpms_loss(*_two_instance_fixture())
    assert result.soft_aji.item() > 0.95
    assert result.soft_pq.item() > 0.95
    assert 0.0 <= result.soft_aji.item() <= 1.0
    assert 0.0 <= result.soft_pq.item() <= 1.0


def test_high_confidence_fp_reduces_soft_dq_and_aji():
    base = compute_setpms_loss(*_two_instance_fixture())
    with_fp = compute_setpms_loss(*_two_instance_fixture(include_fp=True))
    assert with_fp.soft_dq < base.soft_dq
    assert with_fp.soft_aji < base.soft_aji


def test_missing_prediction_reduces_soft_dq_and_aji():
    mask_logits, coords, logits, gt_masks, gt_points = _two_instance_fixture()
    full = compute_setpms_loss(mask_logits, coords, logits, gt_masks, gt_points)
    missing = compute_setpms_loss(
        mask_logits[:1], coords[:1], logits[:1], gt_masks, gt_points
    )
    assert missing.soft_dq < full.soft_dq
    assert missing.soft_aji < full.soft_aji


def test_duplicate_mask_increases_duplicate_penalty():
    mask_logits, coords, logits, gt_masks, gt_points = _two_instance_fixture()
    one = compute_setpms_loss(
        mask_logits[:1], coords[:1], logits[:1], gt_masks[:1], gt_points[:1]
    )
    duplicate = compute_setpms_loss(
        torch.cat([mask_logits[:1], mask_logits[:1]], dim=0),
        torch.cat([coords[:1], coords[:1]], dim=0),
        torch.cat([logits[:1], logits[:1]], dim=0),
        gt_masks[:1],
        gt_points[:1],
    )
    assert duplicate.duplicate_loss > one.duplicate_loss + 0.1


def test_iou_credit_increases_across_inclusive_half_threshold():
    gt_masks = torch.ones(1, 10, 10)
    gt_points = torch.tensor([[4.5, 4.5]])
    logits = torch.tensor([[8.0, -8.0]])
    coords = torch.tensor([[4.5, 4.5]])
    below = compute_setpms_loss(
        torch.full((1, 10, 10), _logit(0.49)), coords, logits, gt_masks, gt_points
    )
    above = compute_setpms_loss(
        torch.full((1, 10, 10), _logit(0.51)), coords, logits, gt_masks, gt_points
    )
    below_credit = (
        below.plan * torch.sigmoid((below.iou - 0.5) / 0.05)
    ).sum()
    above_credit = (
        above.plan * torch.sigmoid((above.iou - 0.5) / 0.05)
    ).sum()
    assert above.iou.item() > below.iou.item()
    assert above_credit > below_credit


def test_empty_inputs_and_zero_masks_are_finite():
    cases = [
        (
            torch.zeros(2, 4, 4),
            torch.zeros(2, 2),
            torch.zeros(2, 2),
            torch.empty(0, 4, 4),
            torch.empty(0, 2),
        ),
        (
            torch.empty(0, 4, 4),
            torch.empty(0, 2),
            torch.empty(0, 2),
            torch.ones(1, 4, 4),
            torch.tensor([[1.0, 1.0]]),
        ),
        (
            torch.empty(0, 4, 4),
            torch.empty(0, 2),
            torch.empty(0, 2),
            torch.empty(0, 4, 4),
            torch.empty(0, 2),
        ),
        (
            torch.full((1, 4, 4), -100.0),
            torch.tensor([[0.0, 0.0]]),
            torch.tensor([[0.0, 0.0]]),
            torch.zeros(1, 4, 4),
            torch.tensor([[0.0, 0.0]]),
        ),
    ]
    for values in cases:
        result = compute_setpms_loss(*values)
        for scalar in result.scalars().values():
            assert torch.isfinite(scalar).all()
        assert torch.isfinite(result.plan).all()


def test_log_domain_sinkhorn_is_finite_without_nan_or_inf():
    cost = torch.tensor([[0.0, 1.25, 0.9], [1.1, 0.1, 0.8], [0.7, 0.6, 0.2]])
    plan = unbalanced_sinkhorn_log(cost, torch.tensor([1.0, 0.4, 0.001]))
    assert torch.isfinite(plan).all()
    assert (plan >= 0).all()


def test_mask_objectness_and_coordinates_have_finite_nonzero_gradients():
    gt_masks = torch.zeros(1, 8, 8)
    gt_masks[0, 2:6, 2:6] = 1.0
    gt_points = torch.tensor([[3.5, 3.5]])
    mask_logits = torch.zeros(2, 8, 8, requires_grad=True)
    mask_logits.data[0, 2:6, 2:6] = 0.5
    pred_logits = torch.tensor([[1.0, -1.0], [1.5, -1.5]], requires_grad=True)
    pred_coords = torch.tensor([[3.0, 3.0], [7.0, 0.0]], requires_grad=True)
    result = compute_setpms_loss(mask_logits, pred_coords, pred_logits, gt_masks, gt_points)
    result.loss.backward()
    for tensor in (mask_logits, pred_logits, pred_coords):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum().item() > 0.0


def test_query_selection_is_deterministic_and_keeps_hungarian_queries_first():
    logits = torch.zeros(20, 2)
    logits[:, 0] = torch.arange(20, dtype=torch.float32)
    selected = select_set_queries(
        logits,
        torch.tensor([7, 3, 7]),
        gt_count=0,
        max_prompts=16,
    )
    assert selected.tolist()[:2] == [7, 3]
    assert len(selected) == 16
    assert len(set(selected.tolist())) == 16
    assert selected.tolist() == select_set_queries(
        logits, torch.tensor([7, 3, 7]), gt_count=0, max_prompts=16
    ).tolist()


def test_setpms_disabled_leaves_canonical_inference_source_untouched():
    """SetPMS is training-only; the inference implementation has no branch."""

    source = Path("run/run_on_epoch.py").read_text(encoding="utf-8").lower()
    source = source[source.index("def validation_on_epoch"):source.index("def _tta_average")]
    assert "setpms" not in source


def test_inclusive_iou_half_threshold_matches_canonical_metric():
    true = np.array([[1, 1], [0, 0]], dtype=np.int32)
    pred = np.array([[1, 0], [0, 0]], dtype=np.int32)
    (_, _, _), (paired_true, paired_pred, unpaired_true, unpaired_pred) = get_fast_pq(
        true, pred, match_iou=0.5
    )
    assert paired_true == [1]
    assert paired_pred == [1]
    assert unpaired_true == []
    assert unpaired_pred == []
