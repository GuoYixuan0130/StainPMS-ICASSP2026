"""Read-only C3 score-control feasibility audit helpers.

Every operation in this module holds the selected-mask pool, masks, boxes,
prompt groups, NMS IoU, and native assembly implementation fixed.  The only
intervention is a GT-only replacement or permutation of the existing assembly
score.  Results are therefore upper bounds, never deployable performance.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from stainpms.c2_component_audit import selected_utility_labels
from stainpms.phase1_metrics import instance_ids
from stainpms.zero_training_oracle import ORACLE_MATCH_IOU, native_final_stage


OPERATIONS = (
    "native",
    "fp_demotion_oracle",
    "duplicate_order_oracle",
    "conflict_order_oracle",
    "merge_risk_demotion_oracle",
    "full_score_oracle",
)


def _quantiles(values: Iterable[float | int]) -> dict[str, float | int | None]:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    if not array.size:
        return {key: None for key in ("count", "mean", "q10", "q25", "median", "q75", "q90")}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "q10": float(np.quantile(array, 0.10)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
        "q90": float(np.quantile(array, 0.90)),
    }


def _box_iou(left: Iterable[float], right: Iterable[float]) -> float:
    lx1, ly1, lx2, ly2 = (float(value) for value in left)
    rx1, ry1, rx2, ry2 = (float(value) for value in right)
    width = max(0.0, min(lx2, rx2) - max(lx1, rx1))
    height = max(0.0, min(ly2, ry2) - max(ly1, ry1))
    intersection = width * height
    union = max(0.0, (lx2 - lx1) * (ly2 - ly1) + (rx2 - rx1) * (ry2 - ry1) - intersection)
    return float(intersection / union) if union else 0.0


class _UnionFind:
    def __init__(self, count: int):
        self.parent = list(range(count))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def conflict_components(records: list[dict[str, Any]], *, nms_iou: float) -> dict[str, Any]:
    """Build connected components from the three actual assembly conflicts.

    The native routine resolves duplicate prompt-group appearances before NMS,
    applies one cross-category NMS, then paints masks in score order.  Thus a
    component joins records sharing a prompt group, box-NMS overlap, or a
    non-empty mask overlap that can affect paint order.
    """

    union_find = _UnionFind(len(records))
    reasons: dict[tuple[int, int], set[str]] = defaultdict(set)
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            row_left, row_right = records[left], records[right]
            edge_reasons: set[str] = set()
            if int(row_left["prompt_group_id"]) == int(row_right["prompt_group_id"]):
                edge_reasons.add("prompt_group")
            if _box_iou(row_left["bbox_xyxy"], row_right["bbox_xyxy"]) > float(nms_iou):
                edge_reasons.add("nms_box_iou")
            if bool((np.asarray(row_left["mask"], dtype=bool) & np.asarray(row_right["mask"], dtype=bool)).any()):
                edge_reasons.add("paint_mask_overlap")
            if edge_reasons:
                union_find.union(left, right)
                reasons[(left, right)] = edge_reasons
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        grouped[union_find.find(index)].append(index)
    components = [sorted(indices) for _, indices in sorted(grouped.items(), key=lambda item: min(item[1]))]
    component_for_index = {
        index: component_id for component_id, indices in enumerate(components) for index in indices
    }
    reason_counts = {name: 0 for name in ("prompt_group", "nms_box_iou", "paint_mask_overlap")}
    for values in reasons.values():
        for name in values:
            reason_counts[name] += 1
    return {
        "components": components,
        "component_for_index": component_for_index,
        "edges": [
            {"left": int(left), "right": int(right), "reasons": sorted(values)}
            for (left, right), values in sorted(reasons.items())
        ],
        "edge_reason_counts": reason_counts,
        "edge_count": len(reasons),
    }


def _eligible_gt_ids(record: dict[str, Any], *, threshold: float) -> set[int]:
    return {
        int(gt_id)
        for gt_id, iou in record.get("gt_ious", {}).items()
        if float(iou) > float(threshold)
    }


def duplicate_competition_pairs(
    records: list[dict[str, Any]],
    component_for_index: dict[int, int],
    *,
    threshold: float = ORACLE_MATCH_IOU,
) -> list[tuple[int, int]]:
    """Return (unique-TP, duplicate) pairs that really share a conflict set."""

    unique_by_gt: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        if row["utility_label"] == "unique_tp" and row.get("matched_gt_instance_id") is not None:
            unique_by_gt[int(row["matched_gt_instance_id"])].append(index)
    pairs: list[tuple[int, int]] = []
    for duplicate_index, row in enumerate(records):
        if row["utility_label"] != "duplicate":
            continue
        for gt_id in _eligible_gt_ids(row, threshold=threshold):
            for unique_index in unique_by_gt.get(gt_id, []):
                if component_for_index[unique_index] == component_for_index[duplicate_index]:
                    pairs.append((unique_index, duplicate_index))
    return sorted(set(pairs))


def _demote_scores(scores: list[float], targets: Iterable[int]) -> list[float]:
    output = [float(value) for value in scores]
    target_indices = sorted(set(int(value) for value in targets), key=lambda index: (output[index], index))
    if not target_indices:
        return output
    minimum, maximum = min(output), max(output)
    span = max(1.0, maximum - minimum + 1.0)
    base = minimum - span
    step = 0.5 / max(1, len(target_indices))
    for rank, index in enumerate(target_indices):
        output[index] = base + rank * step
    return output


def _permute_component_scores(
    native_scores: list[float],
    components: Iterable[list[int]],
    priority,
    *,
    apply_component,
) -> list[float]:
    """Permute, never create/delete, score values inside eligible components."""

    output = [float(value) for value in native_scores]
    for component in components:
        if not apply_component(component):
            continue
        sorted_values = sorted(native_scores[index] for index in component)
        ranked_indices = sorted(component, key=lambda index: (priority(index), native_scores[index], index))
        for score, index in zip(sorted_values, ranked_indices, strict=True):
            output[index] = float(score)
    return output


def _permutation_with_fixed_final_count(
    records: list[dict[str, Any]],
    native_scores: list[float],
    components: Iterable[list[int]],
    priority,
    *,
    apply_component,
    gt_map: np.ndarray,
    nms_iou: float,
    native_final_instance_count: int,
) -> tuple[list[float], dict[str, int]]:
    """Apply only score permutations that preserve the native final count.

    The C3 conflict-order intervention is not allowed to obtain a gain by
    silently changing the number of retained output instances.  Components
    are considered in stable order.  A proposed within-component permutation
    is retained only if running the unchanged native assembler still emits
    exactly the native final-instance count.  No mask, candidate or score
    value is created or deleted.
    """

    current = [float(value) for value in native_scores]
    accepted = rejected = eligible = 0
    for component in components:
        if not apply_component(component):
            continue
        eligible += 1
        candidate = _permute_component_scores(
            current,
            [component],
            priority,
            apply_component=lambda _: True,
        )
        if candidate == current:
            accepted += 1
            continue
        stage, _ = _assemble_stage(records, candidate, gt_map, nms_iou=nms_iou)
        if int(stage["final_instance_count"]) == int(native_final_instance_count):
            current = candidate
            accepted += 1
        else:
            rejected += 1
    return current, {
        "eligible_component_count": int(eligible),
        "accepted_component_count": int(accepted),
        "rejected_for_final_count_change": int(rejected),
    }


def _assemble_stage(
    records: list[dict[str, Any]],
    scores: list[float],
    gt_map: np.ndarray,
    *,
    nms_iou: float,
) -> tuple[dict[str, Any], np.ndarray]:
    # Deliberately import only when a real audit runs: this module remains
    # unit-testable without PyTorch/CUDA.
    from run.run_on_epoch import _assemble_instance_map

    final_map = _assemble_instance_map(
        [row["bbox_xyxy"] for row in records],
        scores,
        [row["mask"] for row in records],
        [int(row["prompt_group_id"]) for row in records],
        gt_map.shape,
        float(nms_iou),
    )
    stage = native_final_stage(gt_map, final_map, threshold=ORACLE_MATCH_IOU)
    # The assembly protocol has no fixed score threshold.  We therefore audit
    # the actual final instance count explicitly, rather than silently assuming
    # that a score-only ordering intervention retained the same number of maps.
    stage["final_instance_count"] = int(len(instance_ids(final_map)))
    return stage, final_map


def _merge_records_causing_tp_loss(
    records: list[dict[str, Any]],
    native_scores: list[float],
    gt_map: np.ndarray,
    *,
    nms_iou: float,
    native_tp: int,
) -> list[int]:
    """Identify merge-risk masks whose removal raises native final TP.

    Removal is used solely for this detached causal label.  The reported
    intervention keeps the complete candidate pool and only demotes score.
    """

    harmful: list[int] = []
    for index, row in enumerate(records):
        if not bool(row.get("merge_risk", False)):
            continue
        retained_records = [candidate for position, candidate in enumerate(records) if position != index]
        retained_scores = [score for position, score in enumerate(native_scores) if position != index]
        stage, _ = _assemble_stage(retained_records, retained_scores, gt_map, nms_iou=nms_iou)
        if int(stage["tp"]) > int(native_tp):
            harmful.append(index)
    return harmful


def _stage_compact(stage: dict[str, Any]) -> dict[str, Any]:
    return {key: stage[key] for key in ("tp", "fp", "fn", "dq", "sq", "pq", "final_instance_count")}


def _delta(stage: dict[str, Any], native: dict[str, Any]) -> dict[str, float | int]:
    output: dict[str, float | int] = {}
    for key in ("tp", "fp", "fn", "dq", "sq", "pq"):
        output[key] = float(stage[key]) - float(native[key])
    return output


def summarize_conflicts(records: list[dict[str, Any]], graph: dict[str, Any]) -> dict[str, Any]:
    components = graph["components"]
    component_for_index = graph["component_for_index"]
    component_sizes = [len(component) for component in components]
    singleton_fp = sum(
        row["utility_label"] == "unmatched_fp" and len(components[component_for_index[index]]) == 1
        for index, row in enumerate(records)
    )
    conflicting_fp = sum(
        row["utility_label"] == "unmatched_fp" and len(components[component_for_index[index]]) > 1
        for index, row in enumerate(records)
    )
    top1_total = top1_correct = 0
    top1_ties = 0
    pair_counts = {name: 0 for name in ("all_negative", "unmatched_fp", "duplicate")}
    pair_correct = {name: 0 for name in pair_counts}
    pair_ties = {name: 0 for name in pair_counts}
    margins = {name: [] for name in pair_counts}
    for component in components:
        if len(component) < 2:
            continue
        unique = [index for index in component if records[index]["utility_label"] == "unique_tp"]
        negatives = [index for index in component if records[index]["utility_label"] in {"unmatched_fp", "duplicate"}]
        if unique and negatives:
            top1_total += 1
            top_score = max(float(records[index]["assembly_score"]) for index in component)
            top1_ties += int(sum(
                float(records[index]["assembly_score"]) == top_score for index in component
            ) > 1)
            top1_correct += int(any(
                records[index]["utility_label"] == "unique_tp"
                and float(records[index]["assembly_score"]) == top_score
                for index in component
            ))
        for unique_index in unique:
            for negative_index in negatives:
                label = str(records[negative_index]["utility_label"])
                margin = float(records[unique_index]["assembly_score"]) - float(records[negative_index]["assembly_score"])
                for key in ("all_negative", label):
                    pair_counts[key] += 1
                    pair_correct[key] += int(margin > 0.0)
                    pair_ties[key] += int(margin == 0.0)
                    margins[key].append(margin)
    return {
        "component_count": len(components),
        "non_singleton_component_count": sum(len(component) > 1 for component in components),
        "component_sizes": component_sizes,
        "component_size": _quantiles(component_sizes),
        "edge_count": int(graph["edge_count"]),
        "edge_reason_counts": dict(graph["edge_reason_counts"]),
        "singleton_unmatched_fp_count": int(singleton_fp),
        "conflicting_unmatched_fp_count": int(conflicting_fp),
        "duplicate_count": sum(row["utility_label"] == "duplicate" for row in records),
        "merge_risk_count": sum(bool(row.get("merge_risk", False)) for row in records),
        "unique_tp_native_top1": {
            "numerator": int(top1_correct),
            "denominator": int(top1_total),
            "tie_component_count": int(top1_ties),
            "accuracy": float(top1_correct / top1_total) if top1_total else None,
        },
        "pairwise_ordering": {
            key: {
                "correct": int(pair_correct[key]),
                "count": int(pair_counts[key]),
                "ties": int(pair_ties[key]),
                "accuracy": float(pair_correct[key] / pair_counts[key]) if pair_counts[key] else None,
                "positive_minus_negative_margin": _quantiles(margins[key]),
                "margin_values": margins[key],
            }
            for key in pair_counts
        },
    }


def audit_image(
    selected_records: list[dict[str, Any]],
    gt_map: np.ndarray,
    *,
    nms_iou: float,
    merge_risk_overlap_fraction: float = 0.1,
    return_maps: bool = False,
) -> dict[str, Any]:
    """Run all fixed C3 score-only GT-oracle interventions for one image."""

    records = selected_utility_labels(
        selected_records,
        gt_map,
        match_iou=ORACLE_MATCH_IOU,
        merge_risk_overlap_fraction=merge_risk_overlap_fraction,
    )
    native_scores = [float(row["assembly_score"]) for row in records]
    graph = conflict_components(records, nms_iou=nms_iou)
    native, native_map = _assemble_stage(records, native_scores, gt_map, nms_iou=nms_iou)

    fp_indices = [index for index, row in enumerate(records) if row["utility_label"] == "unmatched_fp"]
    fp_scores = _demote_scores(native_scores, fp_indices)

    duplicate_pairs = duplicate_competition_pairs(records, graph["component_for_index"])
    duplicate_indices = {duplicate for _, duplicate in duplicate_pairs}
    unique_indices = {unique for unique, _ in duplicate_pairs}
    duplicate_scores, duplicate_permutation = _permutation_with_fixed_final_count(
        records,
        native_scores,
        graph["components"],
        lambda index: 0 if index in duplicate_indices else (2 if index in unique_indices else 1),
        apply_component=lambda component: any(index in duplicate_indices for index in component),
        gt_map=gt_map,
        nms_iou=nms_iou,
        native_final_instance_count=int(native["final_instance_count"]),
    )

    conflict_scores, conflict_permutation = _permutation_with_fixed_final_count(
        records,
        native_scores,
        graph["components"],
        lambda index: (float(records[index]["utility_target"]), float(native_scores[index]), index),
        apply_component=lambda component: len(component) > 1,
        gt_map=gt_map,
        nms_iou=nms_iou,
        native_final_instance_count=int(native["final_instance_count"]),
    )

    harmful_merge_indices = _merge_records_causing_tp_loss(
        records,
        native_scores,
        gt_map,
        nms_iou=nms_iou,
        native_tp=int(native["tp"]),
    )
    merge_scores = _demote_scores(native_scores, harmful_merge_indices)
    full_scores = [
        float(row["utility_target"]) * (0.3 if bool(row.get("edge_penalized", False)) else 1.0)
        for row in records
    ]

    operation_scores = {
        "native": native_scores,
        "fp_demotion_oracle": fp_scores,
        "duplicate_order_oracle": duplicate_scores,
        "conflict_order_oracle": conflict_scores,
        "merge_risk_demotion_oracle": merge_scores,
        "full_score_oracle": full_scores,
    }
    stages: dict[str, dict[str, Any]] = {"native": _stage_compact(native)}
    maps: dict[str, np.ndarray] = {"native": native_map}
    for name, scores in operation_scores.items():
        if name == "native":
            continue
        stage, final_map = _assemble_stage(records, scores, gt_map, nms_iou=nms_iou)
        stages[name] = _stage_compact(stage)
        maps[name] = final_map
    deltas = {name: _delta(stage, stages["native"]) for name, stage in stages.items()}
    return {
        "stages": stages,
        "deltas_vs_native": deltas,
        "intervention_semantics": {
            "fixed_mask_candidate_pool_nms_and_assembly": True,
            "native_score_threshold": None,
            "fp_demotion_note": "native assembly has no keep/reject threshold; unmatched FPs are score-demoted below all non-FPs rather than removed",
            "duplicate_order_note": "only score values are permuted inside components containing an actual duplicate/unique-TP conflict",
            "conflict_order_note": "oracle utility permutes the unchanged native score multiset inside each non-singleton assembly component",
            "merge_risk_note": "harmful merge-risk is identified by detached leave-one-out TP gain; the reported intervention demotes score without removing candidates",
            "full_score_note": "GT-only utility replacement times the unchanged edge penalty; upper bound only",
        },
        "targets": {
            "unmatched_fp_score_demoted_count": len(fp_indices),
            "duplicate_competition_pair_count": len(duplicate_pairs),
            "duplicate_score_reordered_count": len(duplicate_indices) + len(unique_indices),
            "conflict_component_reordered_count": sum(len(component) > 1 for component in graph["components"]),
            "harmful_merge_risk_score_demoted_count": len(harmful_merge_indices),
            "duplicate_permutation": duplicate_permutation,
            "conflict_permutation": conflict_permutation,
        },
        "retention_count_preserved": {
            name: bool(stage["final_instance_count"] == stages["native"]["final_instance_count"])
            for name, stage in stages.items()
        },
        "conflicts": summarize_conflicts(records, graph),
        "records": records,
        **({"maps": maps} if return_maps else {}),
    }
