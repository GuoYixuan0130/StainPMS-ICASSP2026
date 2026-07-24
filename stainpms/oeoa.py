"""Pure GT-only accounting for the Phase 3A OEOA audit.

This module consumes already-exported compact C0/C1 artifacts.  It never
constructs a model, data loader, trainer, optimiser, or checkpoint writer.
Every intervention here is an oracle replacement on an already assembled C1
final instance map, so the values are diagnostic upper bounds only.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from stainpms.evaluator import evaluate_instance_pair
from stainpms.phase1_metrics import (
    final_instance_overlap_table,
    final_max_iou_by_gt,
    instance_ids,
    mask_iou,
)
from stainpms.zero_training_oracle import (
    ORACLE_MATCH_IOU,
    annotate_pool_ious,
    maximum_cardinality_max_iou_matching,
    pool_gt_maxima,
)


ACTION_CLASSES = (
    "tp_boundary",
    "subthreshold_1to1",
    "merge",
    "split_or_duplicate",
    "complex_topology",
    "pure_fn",
    "pure_fp",
)

ROUTES: dict[str, tuple[str, ...]] = {
    "mask_boundary": ("tp_boundary",),
    "local_mask_rescue": ("subthreshold_1to1",),
    "topology": ("merge", "split_or_duplicate", "complex_topology"),
    "coverage": ("pure_fn",),
    "precision": ("pure_fp",),
    "mask_quality_total": ("tp_boundary", "subthreshold_1to1"),
    "detection_total": ("pure_fn", "pure_fp"),
}

METRIC_FIELDS = ("dice1", "dice2", "aji", "dq", "sq", "pq")


class _UnionFind:
    def __init__(self, count: int):
        self.parent = list(range(count))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def map_sha256(value: np.ndarray) -> str:
    """Stable SHA256 for an integer label map, including its shape."""

    array = np.ascontiguousarray(np.asarray(value, dtype=np.int32))
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def map_metrics(gt_map: np.ndarray, pred_map: np.ndarray, *, sample_id: str | None = None) -> dict[str, Any]:
    """Use exactly the repository's formal strict evaluator."""

    evaluation = evaluate_instance_pair(
        np.asarray(gt_map, dtype=np.int32),
        np.asarray(pred_map, dtype=np.int32),
        mode="strict",
        match_iou=ORACLE_MATCH_IOU,
        sample_id=sample_id,
    )
    metrics = evaluation["metrics"]
    if any(metrics.get(name) is None for name in METRIC_FIELDS):
        raise ValueError(f"Phase 3A requires non-empty strict metrics for {sample_id!r}")
    return evaluation


def compact_metrics(evaluation: Mapping[str, Any]) -> dict[str, float]:
    metrics = evaluation["metrics"]
    return {name: float(metrics[name]) for name in METRIC_FIELDS}


def average_metrics(rows: Iterable[Mapping[str, float]]) -> dict[str, float]:
    values = list(rows)
    if not values:
        raise ValueError("cannot average an empty metric collection")
    return {
        field: float(np.mean([float(row[field]) for row in values]))
        for field in METRIC_FIELDS
    }


def metric_delta(upper: Mapping[str, float], lower: Mapping[str, float]) -> dict[str, float]:
    return {field: float(upper[field]) - float(lower[field]) for field in METRIC_FIELDS}


def relabel_contiguously(inst_map: np.ndarray) -> np.ndarray:
    """Return an equivalent map with deterministic contiguous foreground IDs."""

    source = np.asarray(inst_map, dtype=np.int32)
    output = np.zeros_like(source)
    for new_id, old_id in enumerate(instance_ids(source), start=1):
        output[source == int(old_id)] = int(new_id)
    return output


def assert_label_map(value: np.ndarray, *, label: str) -> None:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{label} must be 2-D, got {array.shape}")
    if not np.issubdtype(array.dtype, np.integer) or (array < 0).any():
        raise ValueError(f"{label} must contain non-negative integer labels")
    # A label map has exactly one label per pixel.  This explicit check guards
    # against a malformed object-array input being silently coerced elsewhere.
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")


def _component_category(pred_count: int, gt_count: int, *, is_standard_tp: bool) -> str:
    if pred_count == 1 and gt_count == 1:
        return "tp_boundary" if is_standard_tp else "subthreshold_1to1"
    if pred_count == 1 and gt_count >= 2:
        return "merge"
    if pred_count >= 2 and gt_count == 1:
        return "split_or_duplicate"
    if pred_count >= 2 and gt_count >= 2:
        return "complex_topology"
    if pred_count == 0 and gt_count >= 1:
        return "pure_fn"
    if pred_count >= 1 and gt_count == 0:
        return "pure_fp"
    raise ValueError(f"unsupported overlap component P={pred_count}, G={gt_count}")


def build_overlap_components(
    gt_map: np.ndarray,
    pred_map: np.ndarray,
    *,
    sample_id: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the prescribed final-instance bipartite overlap components.

    An edge exists whenever a final prediction and GT instance share at least
    one pixel.  Isolated prediction and GT nodes remain components, rather than
    being discarded.  The standard evaluator's pairing determines the unique
    ``tp_boundary`` versus ``subthreshold_1to1`` split.
    """

    gt = np.asarray(gt_map, dtype=np.int32)
    pred = np.asarray(pred_map, dtype=np.int32)
    assert_label_map(gt, label="gt_map")
    assert_label_map(pred, label="pred_map")
    if gt.shape != pred.shape:
        raise ValueError(f"GT/pred shape mismatch: {gt.shape} vs {pred.shape}")

    overlap = final_instance_overlap_table(gt, pred)
    gt_ids = [int(value) for value in overlap["gt_ids"]]
    pred_ids = [int(value) for value in overlap["pred_ids"]]
    gt_index = {value: index for index, value in enumerate(gt_ids)}
    pred_index = {value: len(gt_ids) + index for index, value in enumerate(pred_ids)}
    union_find = _UnionFind(len(gt_ids) + len(pred_ids))
    intersections = np.asarray(overlap["intersections"], dtype=np.int64)

    edge_count = 0
    for gt_id in gt_ids:
        for pred_id in pred_ids:
            if int(intersections[gt_id, pred_id]) > 0:
                union_find.union(gt_index[gt_id], pred_index[pred_id])
                edge_count += 1

    grouped: dict[int, dict[str, list[int]]] = {}
    for gt_id in gt_ids:
        grouped.setdefault(union_find.find(gt_index[gt_id]), {"gt_ids": [], "pred_ids": []})["gt_ids"].append(gt_id)
    for pred_id in pred_ids:
        grouped.setdefault(union_find.find(pred_index[pred_id]), {"gt_ids": [], "pred_ids": []})["pred_ids"].append(pred_id)

    standard = map_metrics(gt, pred, sample_id=sample_id)
    pairing = standard.get("pairing") or {}
    paired = {
        (int(gt_id), int(pred_id))
        for gt_id, pred_id in zip(pairing.get("paired_true", []), pairing.get("paired_pred", []), strict=True)
    }

    def sort_key(item: dict[str, list[int]]) -> tuple[int, int, int, int]:
        pred_values = sorted(item["pred_ids"])
        gt_values = sorted(item["gt_ids"])
        return (
            0 if pred_values else 1,
            pred_values[0] if pred_values else 2**31 - 1,
            0 if gt_values else 1,
            gt_values[0] if gt_values else 2**31 - 1,
        )

    components: list[dict[str, Any]] = []
    for component_id, item in enumerate(sorted(grouped.values(), key=sort_key)):
        current_pred_ids = sorted(int(value) for value in item["pred_ids"])
        current_gt_ids = sorted(int(value) for value in item["gt_ids"])
        is_tp = bool(
            len(current_pred_ids) == 1
            and len(current_gt_ids) == 1
            and (current_gt_ids[0], current_pred_ids[0]) in paired
        )
        category = _component_category(len(current_pred_ids), len(current_gt_ids), is_standard_tp=is_tp)
        overlap_pixels = int(
            sum(
                int(intersections[gt_id, pred_id])
                for gt_id in current_gt_ids
                for pred_id in current_pred_ids
            )
        )
        component: dict[str, Any] = {
            "component_id": int(component_id),
            "category": category,
            "pred_ids": current_pred_ids,
            "gt_ids": current_gt_ids,
            "pred_count": int(len(current_pred_ids)),
            "gt_count": int(len(current_gt_ids)),
            "pred_area": int(sum(int((pred == value).sum()) for value in current_pred_ids)),
            "gt_area": int(sum(int((gt == value).sum()) for value in current_gt_ids)),
            "overlap_pixel_count": overlap_pixels,
            "standard_pq_tp": is_tp,
        }
        if len(current_pred_ids) == 1 and len(current_gt_ids) == 1:
            component["one_to_one_iou"] = float(
                mask_iou(pred == current_pred_ids[0], gt == current_gt_ids[0])
            )
        else:
            component["one_to_one_iou"] = None
        components.append(component)

    if sum(int(component["pred_count"]) for component in components) != len(pred_ids):
        raise RuntimeError("overlap graph lost prediction nodes")
    if sum(int(component["gt_count"]) for component in components) != len(gt_ids):
        raise RuntimeError("overlap graph lost GT nodes")
    if any(component["category"] not in ACTION_CLASSES for component in components):
        raise RuntimeError("overlap graph emitted an unknown action category")
    return components, {
        "evaluation": standard,
        "edge_count": int(edge_count),
        "component_count": int(len(components)),
        "category_counts": {
            name: int(sum(component["category"] == name for component in components))
            for name in ACTION_CLASSES
        },
    }


def apply_component_oracle(
    gt_map: np.ndarray,
    pred_map: np.ndarray,
    components: Sequence[Mapping[str, Any]],
    enabled_actions: Iterable[str],
) -> np.ndarray:
    """Apply exactly the requested component classes and reindex safely.

    Inactive components retain their native final prediction masks.  Active
    components replace all of their prediction masks with all component GT
    masks.  The all-action oracle returns the original GT map byte-for-byte,
    which makes the required pixel-level all-action check unambiguous.
    """

    enabled = frozenset(str(value) for value in enabled_actions)
    unknown = enabled.difference(ACTION_CLASSES)
    if unknown:
        raise ValueError(f"unknown OEOA action(s): {sorted(unknown)}")
    gt = np.asarray(gt_map, dtype=np.int32)
    pred = np.asarray(pred_map, dtype=np.int32)
    if not enabled:
        return pred.copy()
    if enabled == frozenset(ACTION_CLASSES):
        return gt.copy()

    output = np.zeros_like(pred)
    next_id = 1
    observed_pred: set[int] = set()
    observed_gt: set[int] = set()
    for component in components:
        category = str(component["category"])
        if category not in ACTION_CLASSES:
            raise ValueError(f"bad component category {category!r}")
        if category in enabled:
            source, ids = gt, [int(value) for value in component["gt_ids"]]
            observed_gt.update(ids)
        else:
            source, ids = pred, [int(value) for value in component["pred_ids"]]
            observed_pred.update(ids)
        for instance_id in ids:
            mask = source == instance_id
            if not mask.any():
                raise RuntimeError(f"component refers to absent instance {instance_id}")
            if bool((output[mask] != 0).any()):
                raise RuntimeError("component oracle would create overlapping output instances")
            output[mask] = int(next_id)
            next_id += 1

    component_pred_ids = {int(value) for component in components for value in component["pred_ids"]}
    component_gt_ids = {int(value) for component in components for value in component["gt_ids"]}
    if component_pred_ids != set(instance_ids(pred)) or component_gt_ids != set(instance_ids(gt)):
        raise RuntimeError("component oracle inputs do not cover complete native/GT maps")
    assert_label_map(output, label="oracle output")
    return output


def action_mask(actions: Iterable[str]) -> int:
    enabled = {str(value) for value in actions}
    unknown = enabled.difference(ACTION_CLASSES)
    if unknown:
        raise ValueError(f"unknown OEOA action(s): {sorted(unknown)}")
    return sum(1 << index for index, name in enumerate(ACTION_CLASSES) if name in enabled)


def actions_for_mask(mask: int) -> tuple[str, ...]:
    if mask < 0 or mask >= 2 ** len(ACTION_CLASSES):
        raise ValueError(f"invalid OEOA action mask {mask}")
    return tuple(name for index, name in enumerate(ACTION_CLASSES) if int(mask) & (1 << index))


def all_action_masks() -> tuple[int, ...]:
    return tuple(range(2 ** len(ACTION_CLASSES)))


def shapley_contributions(values: Mapping[int, float]) -> dict[str, float]:
    """Exact seven-action Shapley values over the complete 128-set table."""

    expected = set(all_action_masks())
    if set(int(key) for key in values) != expected:
        missing = sorted(expected.difference(values))
        extra = sorted(set(values).difference(expected))
        raise ValueError(f"Shapley requires all 128 subsets; missing={missing}, extra={extra}")
    n = len(ACTION_CLASSES)
    denominator = float(math.factorial(n))
    result: dict[str, float] = {}
    for index, action in enumerate(ACTION_CLASSES):
        contribution = 0.0
        bit = 1 << index
        for subset in expected:
            if subset & bit:
                continue
            size = int(subset).bit_count()
            weight = float(math.factorial(size) * math.factorial(n - size - 1)) / denominator
            contribution += weight * (float(values[subset | bit]) - float(values[subset]))
        result[action] = float(contribution)
    return result


def pairwise_interactions(values: Mapping[int, float]) -> dict[tuple[str, str], float]:
    """Return Δ(A+B)-Δ(A)-Δ(B) for every unordered atomic action pair."""

    result: dict[tuple[str, str], float] = {}
    empty = float(values[0])
    for left_index, right_index in combinations(range(len(ACTION_CLASSES)), 2):
        left_bit, right_bit = 1 << left_index, 1 << right_index
        value = float(values[left_bit | right_bit]) - float(values[left_bit]) - float(values[right_bit]) + empty
        result[(ACTION_CLASSES[left_index], ACTION_CLASSES[right_index])] = value
    return result


def candidate_pool_ceiling(records: Iterable[Mapping[str, Any]], gt_map: np.ndarray) -> dict[str, Any]:
    """Prompt-group-constrained max-cardinality, max-IoU candidate-pool ceiling."""

    gt = np.asarray(gt_map, dtype=np.int32)
    rows = annotate_pool_ious([dict(row) for row in records], gt)
    matching = maximum_cardinality_max_iou_matching(rows, instance_ids(gt), threshold=ORACLE_MATCH_IOU)
    matched = list(matching["matched"])
    ious = [float(row["iou"]) for row in matched]
    tp = int(matching["tp"])
    fn = int(len(instance_ids(gt)) - tp)
    dq = float(tp / (tp + 0.5 * fn)) if tp or fn else 0.0
    sq = float(np.mean(ious)) if ious else 0.0
    return {
        "matching_unit": "native_prompt_group; maximum IoU>0.5 cardinality, then maximum total IoU",
        "raw_candidate_mask_count": int(len(rows)),
        "raw_prompt_group_count": int(matching["raw_group_count"]),
        "gt_instance_count": int(len(instance_ids(gt))),
        "maximum_attainable_tp": tp,
        "remaining_fn": fn,
        "mean_matched_iou": sq if ious else None,
        "dq_ceiling": dq,
        "sq_ceiling": sq,
        "candidate_set_pq_ceiling": float(dq * sq),
        "matched_iou_sum": float(sum(ious)),
        "matched": matched,
        "covered_gt_count": int(matching["covered_gt_count"]),
        "one_to_one_conflict_gt_count": int(matching["one_to_one_conflict_gt_count"]),
    }


def localize_final_fns(
    *,
    gt_map: np.ndarray,
    final_map: np.ndarray,
    selected_records: Iterable[Mapping[str, Any]],
    all_records: Iterable[Mapping[str, Any]],
    sample_id: str | None = None,
) -> list[dict[str, Any]]:
    """Localize each native-final PQ FN against final/selected/all pools."""

    gt = np.asarray(gt_map, dtype=np.int32)
    final = np.asarray(final_map, dtype=np.int32)
    evaluation = map_metrics(gt, final, sample_id=sample_id)
    pairing = evaluation.get("pairing") or {}
    fn_gt_ids = [int(value) for value in pairing.get("unpaired_true", [])]
    final_maxima = final_max_iou_by_gt(final_instance_overlap_table(gt, final))
    selected = annotate_pool_ious([dict(row) for row in selected_records], gt)
    all_candidates = annotate_pool_ious([dict(row) for row in all_records], gt)
    selected_maxima = pool_gt_maxima(selected, instance_ids(gt))
    all_maxima = pool_gt_maxima(all_candidates, instance_ids(gt))

    output: list[dict[str, Any]] = []
    for gt_id in fn_gt_ids:
        final_iou, final_pred_id = final_maxima.get(gt_id, (0.0, None))
        selected_row = selected_maxima.get(gt_id)
        all_row = all_maxima.get(gt_id)
        selected_iou = float(selected_row["iou"]) if selected_row is not None else 0.0
        all_iou = float(all_row["iou"]) if all_row is not None else 0.0
        if selected_iou > ORACLE_MATCH_IOU:
            category = "assembly_or_keep_miss"
        elif all_iou > ORACLE_MATCH_IOU:
            category = "selection_miss"
        elif all_iou > 0.0:
            category = "candidate_mask_near_miss"
        else:
            category = "generation_miss"
        if 0.0 < all_iou < 0.1:
            near_bin = "(0,0.1)"
        elif 0.1 <= all_iou < 0.3:
            near_bin = "[0.1,0.3)"
        elif 0.3 <= all_iou <= ORACLE_MATCH_IOU:
            near_bin = "[0.3,0.5]"
        else:
            near_bin = None
        output.append(
            {
                "gt_instance_id": int(gt_id),
                "gt_area": int((gt == gt_id).sum()),
                "native_final_max_iou": float(final_iou),
                "native_final_best_prediction_id": final_pred_id,
                "selected_pool_max_iou": selected_iou,
                "selected_pool_best_record_index": int(selected_row["record"]["record_index"]) if selected_row is not None else None,
                "selected_pool_best_prompt_group_id": int(selected_row["record"]["prompt_group_id"]) if selected_row is not None else None,
                "all_candidate_pool_max_iou": all_iou,
                "all_candidate_pool_best_record_index": int(all_row["record"]["record_index"]) if all_row is not None else None,
                "all_candidate_pool_best_prompt_group_id": int(all_row["record"]["prompt_group_id"]) if all_row is not None else None,
                "fn_localization": category,
                "candidate_mask_near_miss_bin": near_bin,
            }
        )
    return output


def summarize_localizations(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Counts/proportions/area for the mutually exclusive FN localization rows."""

    values = list(rows)
    total = len(values)
    output: list[dict[str, Any]] = []
    categories = (
        "assembly_or_keep_miss",
        "selection_miss",
        "candidate_mask_near_miss",
        "generation_miss",
    )
    for category in categories:
        selected = [row for row in values if row.get("fn_localization") == category]
        output.append(
            {
                "fn_localization": category,
                "fn_count": int(len(selected)),
                "total_fn_count": int(total),
                "fn_proportion": float(len(selected) / total) if total else None,
                "total_gt_area": int(sum(int(row["gt_area"]) for row in selected)),
                "mean_gt_area": float(np.mean([float(row["gt_area"]) for row in selected])) if selected else None,
            }
        )
    return output
