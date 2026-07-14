"""Differentiable SetPMS supervision primitives.

The transport plan is computed with log-domain unbalanced Sinkhorn and is
detached by default.  This is the specified v1 memory/stability choice: mask
IoU, point distance, and point-head objectness still receive gradients through
the downstream losses, while no gradients are propagated through Sinkhorn's
iterations themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F


_NUMERIC_EPS = 1.0e-8
_MASS_FLOOR = 1.0e-3


@dataclass
class SetPMSResult:
    """All SetPMS scalar terms plus the detached transport plan."""

    loss: torch.Tensor
    soft_pq: torch.Tensor
    soft_dq: torch.Tensor
    soft_sq: torch.Tensor
    soft_aji: torch.Tensor
    transport_cost: torch.Tensor
    duplicate_loss: torch.Tensor
    plan: torch.Tensor
    iou: torch.Tensor
    point_distance: torch.Tensor
    pred_mass: torch.Tensor

    def scalars(self) -> Mapping[str, torch.Tensor]:
        return {
            "soft_pq": self.soft_pq,
            "soft_dq": self.soft_dq,
            "soft_sq": self.soft_sq,
            "soft_aji": self.soft_aji,
            "transport_cost": self.transport_cost,
            "duplicate_loss": self.duplicate_loss,
            "set_loss": self.loss,
        }


def foreground_probability(pred_logits: torch.Tensor) -> torch.Tensor:
    """Return the canonical point-head nucleus probability.

    CA-SAM2/StainPMS uses class 0 for nucleus and its final class for
    no-object.  A one-logit head is also supported for unit tests.
    """

    if pred_logits.ndim == 0:
        raise ValueError("pred_logits must have a query dimension")
    if pred_logits.ndim == 1:
        return torch.sigmoid(pred_logits)
    if pred_logits.shape[-1] == 1:
        return torch.sigmoid(pred_logits[..., 0])
    return pred_logits.softmax(dim=-1)[..., 0]


def _stable_descending_indices(values: torch.Tensor) -> torch.Tensor:
    """Sort descending while retaining query-index order for exact ties."""

    try:
        return torch.argsort(values, descending=True, stable=True)
    except TypeError:  # pragma: no cover - legacy PyTorch compatibility
        # Query selection is intentionally non-differentiable.  The fallback
        # preserves the same deterministic (score, query-id) ordering.
        order = sorted(
            range(values.numel()),
            key=lambda index: (-float(values[index].detach().cpu()), index),
        )
        return torch.tensor(order, dtype=torch.long, device=values.device)


def select_set_queries(
    pred_logits: torch.Tensor,
    matched_query_indices: torch.Tensor | None,
    gt_count: int,
    *,
    max_prompts: int = 64,
    min_prompts: int = 16,
) -> torch.Tensor:
    """Deterministically select SetPMS automatic prompts for one crop.

    Hungarian-assigned queries are retained first in the matcher-provided
    order.  Remaining slots are filled by descending point-head nucleus
    probability, with query index as the stable tie breaker.  Duplicate or
    out-of-range matcher entries are ignored.  If there are fewer queries than
    the requested K, every available query is returned.
    """

    if pred_logits.ndim < 1:
        raise ValueError("pred_logits must be [Q, C] or [Q]")
    query_count = int(pred_logits.shape[0])
    if max_prompts <= 0 or query_count == 0:
        return torch.empty(0, dtype=torch.long, device=pred_logits.device)

    requested = min(int(max_prompts), max(int(min_prompts), 2 * int(gt_count)))
    requested = min(requested, query_count)
    selected: list[int] = []
    seen: set[int] = set()
    if matched_query_indices is not None:
        for raw_index in matched_query_indices.detach().flatten().cpu().tolist():
            index = int(raw_index)
            if 0 <= index < query_count and index not in seen:
                selected.append(index)
                seen.add(index)
                if len(selected) == requested:
                    break

    if len(selected) < requested:
        probabilities = foreground_probability(pred_logits)
        for raw_index in _stable_descending_indices(probabilities).detach().cpu().tolist():
            index = int(raw_index)
            if index not in seen:
                selected.append(index)
                seen.add(index)
                if len(selected) == requested:
                    break

    return torch.tensor(selected, dtype=torch.long, device=pred_logits.device)


def unbalanced_sinkhorn_log(
    cost: torch.Tensor,
    pred_mass: torch.Tensor,
    gt_mass: torch.Tensor | None = None,
    *,
    epsilon: float = 0.10,
    tau: float = 1.0,
    iterations: int = 20,
    numeric_eps: float = _NUMERIC_EPS,
) -> torch.Tensor:
    """Numerically stable log-domain unbalanced Sinkhorn transport plan."""

    if cost.ndim != 2:
        raise ValueError("cost must have shape [K, N]")
    prediction_count, gt_count = cost.shape
    if pred_mass.shape != (prediction_count,):
        raise ValueError("pred_mass must have shape [K]")
    if gt_mass is None:
        gt_mass = torch.ones(gt_count, dtype=cost.dtype, device=cost.device)
    if gt_mass.shape != (gt_count,):
        raise ValueError("gt_mass must have shape [N]")
    if epsilon <= 0 or tau <= 0 or iterations <= 0:
        raise ValueError("epsilon, tau, and iterations must be positive")
    if prediction_count == 0 or gt_count == 0:
        return cost.new_zeros((prediction_count, gt_count))

    # Sinkhorn is stable in float32 here because all costs are bounded by the
    # prescribed IoU/distance construction.  Keep the caller dtype so loss
    # terms remain compatible with AMP-disabled baseline training.
    log_kernel = -cost / float(epsilon)
    log_a = pred_mass.clamp_min(numeric_eps).log()
    log_b = gt_mass.clamp_min(numeric_eps).log()
    rho = float(tau) / (float(tau) + float(epsilon))
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)

    for _ in range(int(iterations)):
        log_u = rho * (log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1))
        log_v = rho * (log_b - torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0))

    log_plan = log_kernel + log_u.unsqueeze(1) + log_v.unsqueeze(0)
    # The bounded costs make these guards inactive in normal operation; they
    # convert pathological numerical input into a finite plan rather than NaN.
    log_plan = torch.nan_to_num(log_plan, nan=-80.0, neginf=-80.0, posinf=40.0)
    return log_plan.clamp(min=-80.0, max=40.0).exp()


def _as_mask_tensor(masks: torch.Tensor, name: str) -> torch.Tensor:
    if masks.ndim == 4:
        if masks.shape[1] != 1:
            raise ValueError(f"{name} with four dimensions must be [count, 1, H, W]")
        masks = masks[:, 0]
    if masks.ndim != 3:
        raise ValueError(f"{name} must have shape [count, H, W]")
    return masks


def _bounded(value: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def compute_setpms_loss(
    mask_logits: torch.Tensor,
    pred_coords: torch.Tensor,
    pred_logits: torch.Tensor,
    gt_masks: torch.Tensor,
    gt_points: torch.Tensor,
    *,
    crop_diagonal: torch.Tensor | float | None = None,
    detach_transport: bool = True,
    numeric_eps: float = _NUMERIC_EPS,
) -> SetPMSResult:
    """Compute the fixed v1 SetPMS loss for one automatic prompt-mask set."""

    mask_logits = _as_mask_tensor(mask_logits, "mask_logits")
    gt_masks = _as_mask_tensor(gt_masks, "gt_masks").to(
        device=mask_logits.device, dtype=mask_logits.dtype
    )
    prediction_count, height, width = mask_logits.shape
    gt_count = int(gt_masks.shape[0])
    if gt_masks.shape[-2:] != (height, width):
        raise ValueError("prediction and GT masks must share spatial dimensions")
    if pred_coords.shape != (prediction_count, 2):
        raise ValueError("pred_coords must have shape [K, 2]")
    if pred_logits.shape[0] != prediction_count:
        raise ValueError("pred_logits and mask_logits must share K")
    if gt_points.shape != (gt_count, 2):
        raise ValueError("gt_points must have shape [N, 2]")

    pred_coords = pred_coords.to(device=mask_logits.device, dtype=mask_logits.dtype)
    gt_points = gt_points.to(device=mask_logits.device, dtype=mask_logits.dtype)
    pred_mass = foreground_probability(pred_logits).to(mask_logits.dtype).clamp(
        min=_MASS_FLOOR, max=1.0
    )
    pred_soft_masks = torch.sigmoid(mask_logits)
    # Use the explicit spatial extent: ``reshape(0, -1)`` is ambiguous to
    # PyTorch when N=0, while [0, H*W] preserves the set-matrix contract.
    pred_flat = pred_soft_masks.reshape(prediction_count, height * width)
    gt_flat = gt_masks.reshape(gt_count, height * width)
    pred_area = pred_flat.sum(dim=1)
    gt_area = gt_flat.sum(dim=1)

    if crop_diagonal is None:
        diagonal = mask_logits.new_tensor(float((height * height + width * width) ** 0.5))
    else:
        diagonal = torch.as_tensor(crop_diagonal, dtype=mask_logits.dtype, device=mask_logits.device)
    diagonal = diagonal.clamp_min(numeric_eps)

    if prediction_count and gt_count:
        intersections = pred_flat @ gt_flat.transpose(0, 1)
        unions = (pred_area.unsqueeze(1) + gt_area.unsqueeze(0) - intersections).clamp_min(
            numeric_eps
        )
        iou = _bounded(intersections / unions)
        point_distance = (torch.cdist(pred_coords, gt_points, p=2) / diagonal).clamp(0.0, 1.0)
        cost = (1.0 - iou) + 0.25 * point_distance
        plan = unbalanced_sinkhorn_log(
            cost,
            pred_mass,
            torch.ones(gt_count, dtype=mask_logits.dtype, device=mask_logits.device),
            epsilon=0.10,
            tau=1.0,
            iterations=20,
            numeric_eps=numeric_eps,
        )
        if detach_transport:
            plan = plan.detach()
    else:
        intersections = mask_logits.new_zeros((prediction_count, gt_count))
        unions = mask_logits.new_zeros((prediction_count, gt_count))
        iou = mask_logits.new_zeros((prediction_count, gt_count))
        point_distance = mask_logits.new_zeros((prediction_count, gt_count))
        cost = mask_logits.new_zeros((prediction_count, gt_count))
        plan = mask_logits.new_zeros((prediction_count, gt_count))

    gate = torch.sigmoid((iou - 0.5) / 0.05)
    matching_quality = plan * gate
    gt_coverage = matching_quality.sum(dim=0).clamp(0.0, 1.0)
    matched_fraction = (
        matching_quality.sum(dim=1) / (pred_mass + numeric_eps)
    ).clamp(0.0, 1.0)

    soft_tp = gt_coverage.sum()
    soft_fn = (1.0 - gt_coverage).sum()
    soft_fp = (pred_mass * (1.0 - matched_fraction)).sum()
    soft_dq = _bounded(2.0 * soft_tp / (2.0 * soft_tp + soft_fp + soft_fn + numeric_eps))
    soft_sq = _bounded((matching_quality * iou).sum() / (matching_quality.sum() + numeric_eps))
    soft_pq = _bounded(soft_dq * soft_sq)

    aji_numerator = (plan * intersections).sum()
    aji_denominator = (
        (plan * unions).sum()
        + ((1.0 - matched_fraction) * pred_area).sum()
        + ((1.0 - gt_coverage) * gt_area).sum()
    )
    soft_aji = _bounded(aji_numerator / (aji_denominator + numeric_eps))

    if prediction_count > 1 and gt_count:
        pred_intersections = pred_flat @ pred_flat.transpose(0, 1)
        pred_unions = (
            pred_area.unsqueeze(1) + pred_area.unsqueeze(0) - pred_intersections
        ).clamp_min(numeric_eps)
        pred_pair_iou = _bounded(pred_intersections / pred_unions)
        affinity = plan @ plan.transpose(0, 1)
        off_diagonal = ~torch.eye(prediction_count, dtype=torch.bool, device=mask_logits.device)
        duplicate_loss = (affinity * pred_pair_iou * off_diagonal).sum() / (
            (affinity * off_diagonal).sum() + numeric_eps
        )
        duplicate_loss = _bounded(duplicate_loss)
    else:
        duplicate_loss = mask_logits.sum() * 0.0

    transport_cost = (plan * cost).sum() / (plan.sum() + numeric_eps)
    transport_cost = torch.nan_to_num(transport_cost, nan=0.0, posinf=0.0, neginf=0.0)

    if prediction_count == 0 and gt_count == 0:
        # An entirely empty synthetic input is neutral, not a false failure.
        soft_dq = mask_logits.new_tensor(1.0)
        soft_sq = mask_logits.new_tensor(1.0)
        soft_pq = mask_logits.new_tensor(1.0)
        soft_aji = mask_logits.new_tensor(1.0)

    loss = (
        0.5 * (1.0 - soft_pq)
        + 0.5 * (1.0 - soft_aji)
        + 0.1 * transport_cost
        + 0.1 * duplicate_loss
    )
    loss = torch.nan_to_num(loss, nan=0.0, posinf=1.0e6, neginf=0.0)
    return SetPMSResult(
        loss=loss,
        soft_pq=soft_pq,
        soft_dq=soft_dq,
        soft_sq=soft_sq,
        soft_aji=soft_aji,
        transport_cost=transport_cost,
        duplicate_loss=duplicate_loss,
        plan=plan,
        iou=iou,
        point_distance=point_distance,
        pred_mass=pred_mass,
    )
