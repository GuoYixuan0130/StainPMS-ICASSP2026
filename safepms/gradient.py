"""Loss partitioning and global anchor-constrained gradient composition."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Iterable

import torch


ANCHOR_KEYS = (
    "loss_focal", "loss_dice", "loss_iou",
    "loss_pms_preserve_focal", "loss_pms_preserve_dice", "loss_pms_preserve_iou",
)
EXPANSION_KEYS = ("loss_pms_focal", "loss_pms_dice", "loss_pms_iou", "loss_pms_object")
POINT_KEYS = ("loss_reg", "loss_cls", "loss_mask")
EPS = 1e-12


@dataclass(frozen=True)
class GradientSnapshot:
    dot: float
    norm_anchor: float
    norm_expand: float
    shared_norm_anchor: float
    shared_norm_expand: float
    cosine: float | None
    conflict: bool
    retained_expand_norm_ratio: float
    projected: bool
    trust_clipped: bool
    finite: bool
    projection_dot: float
    anchor_final_margin: float
    layerwise: dict[str, dict[str, Any]]
    parameter_roles: dict[str, int]
    other_decoder_dependency: bool


def decompose_losses(loss_dict: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the exact preregistered loss partition without changing weights."""
    if not loss_dict:
        raise ValueError("SafePMS requires a nonempty loss dictionary")
    reference = next(iter(loss_dict.values()))
    zero = reference * 0.0
    anchor = sum((loss_dict.get(name, zero) for name in ANCHOR_KEYS), zero)
    expansion = sum((loss_dict.get(name, zero) for name in EXPANSION_KEYS), zero)
    point = sum((loss_dict.get(name, zero) for name in POINT_KEYS), zero)
    return anchor, expansion, point


def _autograd(loss: torch.Tensor, parameters: list[torch.nn.Parameter], *, retain_graph: bool) -> tuple[torch.Tensor | None, ...]:
    if not loss.requires_grad:
        return tuple(None for _ in parameters)
    return torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)


def _sum_dot(left: Iterable[torch.Tensor | None], right: Iterable[torch.Tensor | None]) -> torch.Tensor:
    result = None
    for first, second in zip(left, right, strict=True):
        if first is not None and second is not None:
            value = (first * second).sum()
            result = value if result is None else result + value
    if result is None:
        # Scalar allocation is tiny and avoids any flatten/materialization.
        return torch.zeros((), dtype=torch.float64)
    return result


def _sum_square(values: Iterable[torch.Tensor | None]) -> torch.Tensor:
    result = None
    for value in values:
        if value is not None:
            term = (value * value).sum()
            result = term if result is None else result + term
    return torch.zeros((), dtype=torch.float64) if result is None else result


def _finite(values: Iterable[torch.Tensor | None]) -> bool:
    return all(value is None or bool(torch.isfinite(value).all()) for value in values)


def _nonzero(values: Iterable[torch.Tensor | None]) -> bool:
    return any(value is not None and bool(torch.count_nonzero(value).item()) for value in values)


def layer_name(parameter_name: str) -> str:
    name = parameter_name.lower()
    if "iou_prediction_head" in name or "iou_head" in name:
        return "iou_head"
    if "object_score" in name or "objectness" in name:
        return "object_score_head"
    if "output_hypernetworks" in name or "hypernetwork" in name:
        return "hypernetwork_heads"
    if "output_upscaling" in name or "upscaling" in name:
        return "upscaling_layers"
    if "mask_tokens" in name:
        return "mask_tokens"
    if "transformer" in name:
        return "transformer"
    return "other_decoder"


def _role(anchor: torch.Tensor | None, expansion: torch.Tensor | None) -> str:
    if anchor is None and expansion is None:
        return "unused"
    if anchor is None:
        return "expansion_only"
    if expansion is None:
        return "anchor_only"
    return "shared"


def _layerwise(named_parameters, anchor, expansion, safe_expansion) -> dict[str, dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for (name, _), ga, ge, gs in zip(named_parameters, anchor, expansion, safe_expansion, strict=True):
        bucket = buckets.setdefault(layer_name(name), {"dot": 0.0, "norm_anchor_sq": 0.0, "norm_expand_sq": 0.0, "shared": 0, "anchor_only": 0, "expansion_only": 0, "unused": 0})
        role = _role(ga, ge)
        bucket[role] += 1
        if ga is not None:
            bucket["norm_anchor_sq"] += float((ga * ga).sum().detach().cpu())
        if ge is not None:
            bucket["norm_expand_sq"] += float((ge * ge).sum().detach().cpu())
        if ga is not None and gs is not None:
            bucket["dot"] += float((ga * gs).sum().detach().cpu())
    for value in buckets.values():
        denom = (value["norm_anchor_sq"] * value["norm_expand_sq"]) ** 0.5
        value["cosine"] = value["dot"] / denom if denom else None
        value["norm_anchor"] = value.pop("norm_anchor_sq") ** 0.5
        value["norm_expand"] = value.pop("norm_expand_sq") ** 0.5
    for bucket in ("mask_tokens", "transformer", "upscaling_layers", "hypernetwork_heads", "iou_head", "object_score_head"):
        buckets.setdefault(bucket, {"dot": 0.0, "norm_anchor": 0.0, "norm_expand": 0.0, "cosine": None, "shared": 0, "anchor_only": 0, "expansion_only": 0, "unused": 0})
    return buckets


def project_global(named_parameters, anchor, expansion, *, trust_ratio: float = 1.0) -> tuple[list[torch.Tensor | None], GradientSnapshot]:
    """Project only shared expansion gradients, with one decoder-global scalar.

    The projection/trust calculation deliberately excludes anchor-only and
    expansion-only parameters.  Those parameters retain the exact gradient
    prescribed by the protocol; only parameters influenced by both losses are
    part of the shared-decoder geometry.
    """
    named = list(named_parameters)
    ga, ge = tuple(anchor), tuple(expansion)
    finite = _finite(ga) and _finite(ge)
    shared_anchor = tuple(first if first is not None and second is not None else None for first, second in zip(ga, ge, strict=True))
    shared_expand = tuple(second if first is not None and second is not None else None for first, second in zip(ga, ge, strict=True))
    dot_tensor = _sum_dot(shared_anchor, shared_expand)
    shared_na_sq_tensor, shared_ne_sq_tensor = _sum_square(shared_anchor), _sum_square(shared_expand)
    full_na_sq_tensor, full_ne_sq_tensor = _sum_square(ga), _sum_square(ge)
    dot, shared_na_sq, shared_ne_sq, full_na_sq, full_ne_sq = (float(item.detach().cpu()) for item in (dot_tensor, shared_na_sq_tensor, shared_ne_sq_tensor, full_na_sq_tensor, full_ne_sq_tensor))
    shared_norm_anchor, shared_norm_expand = shared_na_sq ** 0.5, shared_ne_sq ** 0.5
    norm_anchor, norm_expand = full_na_sq ** 0.5, full_ne_sq ** 0.5
    conflict = dot < 0.0
    alpha = -dot / (shared_na_sq + EPS) if conflict and shared_na_sq > 0 else 0.0
    safe = []
    for first, second in zip(ga, ge, strict=True):
        if second is None:
            safe.append(None)
        elif first is None:
            # Expansion-only parameters are not projected or trust-clipped.
            safe.append(second)
        else:
            safe.append(second + alpha * first)
    safe_shared = tuple(value if first is not None and second is not None else None for value, first, second in zip(safe, ga, ge, strict=True))
    safe_norm_sq = float(_sum_square(safe_shared).detach().cpu())
    safe_norm = safe_norm_sq ** 0.5
    clipped = safe_norm > trust_ratio * shared_norm_anchor and shared_norm_anchor > 0
    scale = trust_ratio * shared_norm_anchor / (safe_norm + EPS) if clipped else 1.0
    if clipped:
        safe = [value * scale if first is not None and second is not None else value for value, first, second in zip(safe, ga, ge, strict=True)]
    projected_dot = float(_sum_dot(ga, safe).detach().cpu())
    final = [first + second if first is not None and second is not None else first if second is None else second for first, second in zip(ga, safe, strict=True)]
    final_dot = float(_sum_dot(ga, final).detach().cpu())
    roles = {role: sum(_role(first, second) == role for first, second in zip(ga, ge, strict=True)) for role in ("anchor_only", "expansion_only", "shared", "unused")}
    layers = _layerwise(named, ga, ge, safe)
    layers["global_decoder"] = {
        "dot": dot,
        "norm_anchor": norm_anchor,
        "norm_expand": norm_expand,
        "cosine": dot / (norm_anchor * norm_expand) if norm_anchor and norm_expand else None,
        **roles,
    }
    snapshot = GradientSnapshot(
        dot=dot, norm_anchor=norm_anchor, norm_expand=norm_expand,
        shared_norm_anchor=shared_norm_anchor, shared_norm_expand=shared_norm_expand,
        cosine=dot / (norm_anchor * norm_expand) if norm_anchor and norm_expand else None,
        conflict=conflict,
        retained_expand_norm_ratio=(float(_sum_square(safe).detach().cpu()) ** 0.5 / norm_expand) if norm_expand else 1.0,
        projected=conflict and shared_na_sq > 0,
        trust_clipped=clipped,
        finite=finite and _finite(safe),
        projection_dot=projected_dot,
        anchor_final_margin=final_dot - full_na_sq,
        layerwise=layers,
        parameter_roles=roles,
        other_decoder_dependency=False,
    )
    return final, snapshot


class GradientController:
    """Optional train-loop hook for Stage 0 collection or Stage 1 SafePMS."""

    def __init__(self, named_parameters, *, mode: str, patient_order: list[str] | None = None, target_valid: int | None = None, deadline_monotonic: float | None = None):
        self.named_parameters = list(named_parameters)
        self.trainable_params = [parameter for _, parameter in self.named_parameters]
        self.mode = mode
        self.patient_order, self.target_valid = patient_order or [], target_valid
        self.deadline_monotonic = deadline_monotonic
        self.current_patient = None
        self._outer_accepted = False
        self.should_stop = False
        self.records: list[dict[str, Any]] = []
        self.invalid_batches = 0
        self.nonfinite_batches = 0
        self.other_decoder_dependency_batches = 0
        self.time_cap_exceeded = False
        self.step_count = 0

    def begin_outer_batch(self, index: int) -> None:
        if self.deadline_monotonic is not None and time.monotonic() >= self.deadline_monotonic:
            self.time_cap_exceeded = True
            self.should_stop = True
        image_id = self.patient_order[index] if index < len(self.patient_order) else None
        try:
            self.current_patient = int(str(image_id).split("_", 1)[0]) if image_id is not None else None
        except ValueError:
            self.current_patient = None
        self._outer_accepted = False

    def consume(self, loss_dict: dict[str, torch.Tensor]) -> dict[str, bool]:
        if self.mode == "audit" and self._outer_accepted:
            return {"skip": True}
        anchor, expansion, point = decompose_losses(loss_dict)
        other = _autograd(point, self.trainable_params, retain_graph=True)
        other_dependency = _nonzero(other)
        ga = _autograd(anchor, self.trainable_params, retain_graph=True)
        ge = _autograd(expansion, self.trainable_params, retain_graph=False)
        valid = _finite(ga) and _finite(ge) and _nonzero(ga) and _nonzero(ge) and not other_dependency
        final, snapshot = project_global(self.named_parameters, ga, ge)
        snapshot = GradientSnapshot(**{**snapshot.__dict__, "other_decoder_dependency": other_dependency})
        record = {
            "patient": self.current_patient,
            "valid": valid,
            "anchor_loss": float(anchor.detach().cpu()),
            "expansion_loss": float(expansion.detach().cpu()),
            "point_loss": float(point.detach().cpu()),
            **snapshot.__dict__,
        }
        if not snapshot.finite:
            self.nonfinite_batches += 1
        if other_dependency:
            self.other_decoder_dependency_batches += 1
        if self.mode == "audit":
            if valid:
                self._outer_accepted = True
                self.records.append(record)
                if self.target_valid is not None and len(self.records) >= self.target_valid:
                    self.should_stop = True
                    return {"stop": True}
            else:
                self.invalid_batches += 1
            return {"skip": True}
        if not snapshot.finite or other_dependency:
            self.invalid_batches += 1
            self.records.append(record)
            return {"skip": True}
        for parameter, value in zip(self.trainable_params, final, strict=True):
            parameter.grad = value
        self.records.append(record)
        self.step_count += 1
        return {"optimizer_step": True}
