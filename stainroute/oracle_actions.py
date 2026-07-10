"""GT-side action utility labels and joint budget oracle search.

This module is deliberately oracle-only. Its utilities must not be imported by
candidate generation, cached decoding, or any future pre-decode router.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Iterable, Sequence

import numpy as np

from stainroute.actions.schema import ActionCandidate
from stainroute.metrics import PQEvaluation, evaluate_pq


def _aji(gt: np.ndarray, pred: np.ndarray) -> float:
    """AJI matching used only as a recorded oracle utility field."""

    gt_ids = [int(value) for value in np.unique(gt) if int(value) != 0]
    pred_ids = [int(value) for value in np.unique(pred) if int(value) != 0]
    if not gt_ids and not pred_ids:
        return 1.0
    if not gt_ids or not pred_ids:
        return 0.0
    paired_pred: set[int] = set()
    intersection_sum = 0.0
    union_sum = 0.0
    for gt_id in gt_ids:
        gt_mask = gt == gt_id
        best_pred = 0
        best_iou = 0.0
        for pred_id in pred_ids:
            pred_mask = pred == pred_id
            intersection = float((gt_mask & pred_mask).sum())
            union = float((gt_mask | pred_mask).sum())
            iou = intersection / union if union else 0.0
            if iou > best_iou:
                best_pred, best_iou = pred_id, iou
        if best_pred:
            pred_mask = pred == best_pred
            intersection_sum += float((gt_mask & pred_mask).sum())
            union_sum += float((gt_mask | pred_mask).sum())
            paired_pred.add(best_pred)
        else:
            union_sum += float(gt_mask.sum())
    for pred_id in pred_ids:
        if pred_id not in paired_pred:
            union_sum += float((pred == pred_id).sum())
    return float(intersection_sum / union_sum) if union_sum else 0.0


@dataclass(frozen=True)
class ActionUtility:
    delta_matched_iou_sum: float
    delta_tp: int
    delta_fp: int
    delta_fn: int
    delta_dq: float
    delta_sq: float
    delta_pq: float
    delta_aji: float
    positive_utility_label: bool

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "delta_matched_iou_sum": self.delta_matched_iou_sum,
            "delta_tp": self.delta_tp,
            "delta_fp": self.delta_fp,
            "delta_fn": self.delta_fn,
            "delta_dq": self.delta_dq,
            "delta_sq": self.delta_sq,
            "delta_pq": self.delta_pq,
            "delta_aji": self.delta_aji,
            "positive_utility_label": self.positive_utility_label,
        }


def compute_action_utility(gt: np.ndarray, base_prediction: np.ndarray, action_prediction: np.ndarray) -> ActionUtility:
    """Compute action labels by fully re-evaluating the whole instance map."""

    base = evaluate_pq(gt, base_prediction)
    after = evaluate_pq(gt, action_prediction)
    return ActionUtility(
        delta_matched_iou_sum=after.matched_iou_sum - base.matched_iou_sum,
        delta_tp=after.tp - base.tp,
        delta_fp=after.fp - base.fp,
        delta_fn=after.fn - base.fn,
        delta_dq=after.dq - base.dq,
        delta_sq=after.sq - base.sq,
        delta_pq=after.pq - base.pq,
        delta_aji=_aji(gt, action_prediction) - _aji(gt, base_prediction),
        positive_utility_label=bool(after.pq > base.pq),
    )


@dataclass(frozen=True)
class OracleSubsetResult:
    action_ids: tuple[str, ...]
    cost: int
    evaluation: PQEvaluation


def _is_feasible(ids: Iterable[str], conflict_graph: dict[str, set[str]]) -> bool:
    selected = set(ids)
    return all(not (set(conflict_graph.get(action_id, set())) & selected) for action_id in selected)


def _better(candidate: OracleSubsetResult, current: OracleSubsetResult | None) -> bool:
    if current is None:
        return True
    if candidate.evaluation.pq != current.evaluation.pq:
        return candidate.evaluation.pq > current.evaluation.pq
    if candidate.cost != current.cost:
        return candidate.cost < current.cost
    return candidate.action_ids < current.action_ids


def exact_joint_oracle(
    actions: Sequence[ActionCandidate],
    *,
    budget: int,
    conflict_graph: dict[str, set[str]],
    evaluate_subset: Callable[[tuple[str, ...]], PQEvaluation],
) -> OracleSubsetResult:
    """Exhaustively evaluate all feasible subsets under decoder-cost budget."""

    ordered = tuple(sorted(actions, key=lambda action: action.action_id))
    best: OracleSubsetResult | None = None

    def visit(start: int, selected: tuple[ActionCandidate, ...], cost: int) -> None:
        nonlocal best
        ids = tuple(action.action_id for action in selected)
        result = OracleSubsetResult(ids, cost, evaluate_subset(ids))
        if _better(result, best):
            best = result
        for index in range(start, len(ordered)):
            action = ordered[index]
            next_cost = cost + action.action_cost
            if next_cost > budget:
                continue
            next_ids = ids + (action.action_id,)
            if not _is_feasible(next_ids, conflict_graph):
                continue
            visit(index + 1, selected + (action,), next_cost)

    visit(0, (), 0)
    assert best is not None
    return best


def beam_joint_oracle(
    actions: Sequence[ActionCandidate],
    *,
    budget: int,
    conflict_graph: dict[str, set[str]],
    evaluate_subset: Callable[[tuple[str, ...]], PQEvaluation],
    beam_width: int,
) -> OracleSubsetResult:
    """Deterministic GT-oracle beam search, validated against exact on small cases."""

    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    action_by_id = {action.action_id: action for action in actions}
    ordered_ids = tuple(sorted(action_by_id))
    states: dict[tuple[str, ...], OracleSubsetResult] = {
        (): OracleSubsetResult((), 0, evaluate_subset(()))
    }
    best = states[()]
    while states:
        expansions: dict[tuple[str, ...], OracleSubsetResult] = {}
        for state in states.values():
            selected = set(state.action_ids)
            for action_id in ordered_ids:
                if action_id in selected:
                    continue
                action = action_by_id[action_id]
                cost = state.cost + action.action_cost
                if cost > budget:
                    continue
                next_ids = tuple(sorted((*state.action_ids, action_id)))
                if not _is_feasible(next_ids, conflict_graph):
                    continue
                result = OracleSubsetResult(next_ids, cost, evaluate_subset(next_ids))
                previous = expansions.get(next_ids)
                if previous is None or _better(result, previous):
                    expansions[next_ids] = result
                if _better(result, best):
                    best = result
        ranked = sorted(
            expansions.values(),
            key=lambda item: (-item.evaluation.pq, item.cost, item.action_ids),
        )
        states = {item.action_ids: item for item in ranked[:beam_width]}
    return best


def normalized_oracle_recovery(base_pq: float, oracle_pq: float, perfect_pq: float = 1.0) -> float | None:
    denominator = perfect_pq - base_pq
    if denominator <= 0:
        return None
    return float((oracle_pq - base_pq) / denominator)
