"""Read-only component-attribution accounting for C2-E/C2-U.

This module consumes the compact artifacts emitted by
``run_zero_training_oracle_diagnosis.py``.  It deliberately contains no model
or optimizer code: all labels are detached GT-only accounting labels and the
oracle-score intervention is an upper bound, never a deployable metric.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from stainpms.phase1_metrics import instance_ids
from stainpms.zero_training_oracle import (
    ORACLE_MATCH_IOU,
    decode_binary_rle,
    maximum_cardinality_max_iou_matching,
    native_final_stage,
)


def _quantiles(values: Iterable[float]) -> dict[str, float | None]:
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


def deserialize_gt(artifact: dict[str, Any]) -> np.ndarray:
    shape = tuple(int(value) for value in artifact["image_shape"])
    result = np.zeros(shape, dtype=np.int32)
    for row in artifact["gt_instances"]:
        result[decode_binary_rle(row["mask_rle"])] = int(row["gt_instance_id"])
    return result


def deserialize_selected(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in artifact["native_selected_before_assembly"]:
        item = dict(row)
        item["mask"] = decode_binary_rle(item.pop("mask_rle"))
        item["record_index"] = int(item["record_index"])
        item["prompt_group_id"] = int(item["prompt_group_id"])
        item["edge_penalized"] = bool(item.get("edge_penalized", False))
        item["quality"] = float(item["quality"])
        item["assembly_score"] = float(item["assembly_score"])
        output.append(item)
    return output


def _foreign_fraction(mask: np.ndarray, gt_map: np.ndarray, own_gt_id: int) -> float:
    foreign = (gt_map > 0) & (gt_map != int(own_gt_id))
    denominator = int(foreign.sum())
    return float((np.asarray(mask, dtype=bool) & foreign).sum() / denominator) if denominator else 0.0


def selected_utility_labels(
    selected_records: list[dict[str, Any]],
    gt_map: np.ndarray,
    *,
    match_iou: float = ORACLE_MATCH_IOU,
    merge_risk_overlap_fraction: float = 0.1,
) -> list[dict[str, Any]]:
    """Attach detached unique-TP/FP/duplicate/merge-risk utility labels.

    Matching is the frozen maximum-cardinality then maximum-IoU group matching
    from the oracle audit.  It is intentionally deterministic and not used to
    alter any native output.
    """

    matching = maximum_cardinality_max_iou_matching(
        selected_records, instance_ids(gt_map), threshold=match_iou
    )
    matched_by_record = {int(row["record_index"]): int(row["gt_instance_id"]) for row in matching["matched"]}
    matched_gt = set(matched_by_record.values())
    rows: list[dict[str, Any]] = []
    for record in selected_records:
        item = dict(record)
        index = int(item["record_index"])
        ious = {int(key): float(value) for key, value in item.get("gt_ious", {}).items()}
        if index in matched_by_record:
            gt_id = matched_by_record[index]
            foreign_fraction = _foreign_fraction(item["mask"], gt_map, gt_id)
            utility_target = max(0.0, 1.0 - foreign_fraction)
            label = "unique_tp"
            merge_risk = foreign_fraction > float(merge_risk_overlap_fraction)
        else:
            eligible = {gt_id for gt_id, value in ious.items() if value > float(match_iou)}
            gt_id = None
            foreign_fraction = None
            utility_target = 0.0
            label = "duplicate" if eligible & matched_gt else "unmatched_fp"
            merge_risk = False
        item.update(
            {
                "utility_label": label,
                "utility_target": float(utility_target),
                "matched_gt_instance_id": gt_id,
                "foreign_gt_fraction": foreign_fraction,
                "merge_risk": bool(merge_risk),
            }
        )
        rows.append(item)
    return rows


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < scores.size:
        stop = start + 1
        while stop < scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return float((ranks[labels.astype(bool)].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(labels.sum())
    if not positives:
        return None
    order = np.argsort(-scores, kind="mergesort")
    ranked = labels[order].astype(np.float64)
    precision = np.cumsum(ranked) / np.arange(1, ranked.size + 1)
    return float((precision * ranked).sum() / positives)


def score_calibration(rows: list[dict[str, Any]], *, bins: int = 10) -> dict[str, Any]:
    scores = np.asarray([float(row["assembly_score"]) for row in rows], dtype=np.float64)
    targets = np.asarray([float(row["utility_target"]) for row in rows], dtype=np.float64)
    labels = targets > 0.0
    if not scores.size:
        return {"valid_prediction_count": 0, "auroc": None, "auprc": None, "brier": None, "ece": None}
    calibration = np.clip(scores, 0.0, 1.0)
    ece = 0.0
    for lower, upper in zip(np.linspace(0.0, 1.0, bins, endpoint=False), np.linspace(1.0 / bins, 1.0, bins), strict=True):
        selected = (calibration >= lower) & (calibration < upper if upper < 1.0 else calibration <= upper)
        if selected.any():
            ece += float(selected.mean() * abs(calibration[selected].mean() - labels[selected].mean()))
    by_label = {
        label: _quantiles(float(row["assembly_score"]) for row in rows if row["utility_label"] == label)
        for label in ("unique_tp", "unmatched_fp", "duplicate")
    }
    merge_scores = _quantiles(
        float(row["assembly_score"]) for row in rows if bool(row.get("merge_risk", False))
    )
    positive_scores = scores[labels]
    score_position = {}
    for label in ("unmatched_fp", "duplicate"):
        label_scores = np.asarray(
            [float(row["assembly_score"]) for row in rows if row["utility_label"] == label], dtype=np.float64
        )
        score_position[label] = {
            "count": int(label_scores.size),
            "below_minimum_unique_tp_score_count": (
                int((label_scores < positive_scores.min()).sum()) if label_scores.size and positive_scores.size else None
            ),
            "below_median_unique_tp_score_count": (
                int((label_scores < np.median(positive_scores)).sum()) if label_scores.size and positive_scores.size else None
            ),
        }
    # Native assembly has no score cutoff: score affects within-prompt choice
    # and NMS ordering only.  The explicit null prevents falsely claiming a
    # threshold intervention where the deployed code has none.
    return {
        "valid_prediction_count": int(scores.size),
        "positive_unique_tp_count": int(labels.sum()),
        "negative_count": int((~labels).sum()),
        "positive_fraction": float(labels.mean()),
        "auroc": _binary_auc(labels, scores),
        "auprc": _average_precision(labels, scores),
        "brier": float(np.mean((calibration - targets) ** 2)),
        "ece": float(ece),
        "assembly_score_threshold": None,
        "threshold_note": "native assembly has no fixed score threshold; it ranks/suppresses by score",
        "scores_by_utility_label": by_label,
        "scores_for_merge_risk": merge_scores,
        "negative_score_position_relative_to_unique_tp": score_position,
    }


def hard_exclusivity(rows: list[dict[str, Any]], gt_map: np.ndarray) -> dict[str, Any]:
    foreign: list[float] = []
    soft_foreign: list[float] = []
    soft_overlap: list[float] = []
    pairs = 0
    positive_overlap: list[float] = []
    for row in rows:
        gt_id = row.get("matched_gt_instance_id")
        if gt_id is not None:
            foreign.append(_foreign_fraction(row["mask"], gt_map, int(gt_id)))
        if row.get("soft_foreign_gt_probability") is not None:
            soft_foreign.append(float(row["soft_foreign_gt_probability"]))
        if row.get("soft_selected_overlap") is not None:
            soft_overlap.append(float(row["soft_selected_overlap"]))
    for left_index, left in enumerate(rows):
        left_box = [float(value) for value in left["bbox_xyxy"]]
        left_gt = left.get("matched_gt_instance_id")
        for right in rows[left_index + 1 :]:
            if int(left["prompt_group_id"]) == int(right["prompt_group_id"]):
                continue
            right_gt = right.get("matched_gt_instance_id")
            if left_gt is None or right_gt is None or int(left_gt) == int(right_gt):
                continue
            right_box = [float(value) for value in right["bbox_xyxy"]]
            if left_box[2] <= right_box[0] or right_box[2] <= left_box[0] or left_box[3] <= right_box[1] or right_box[3] <= left_box[1]:
                continue
            pairs += 1
            intersection = int((left["mask"] & right["mask"]).sum())
            if intersection:
                union = int((left["mask"] | right["mask"]).sum())
                positive_overlap.append(float(intersection / union) if union else 0.0)
    return {
        "hard_foreign_gt_fraction": _quantiles(foreign),
        "soft_foreign_gt_probability": _quantiles(soft_foreign),
        "soft_selected_overlap": _quantiles(soft_overlap),
        "hard_overlap_candidate_pair_count": pairs,
        "hard_overlap_positive_pair_count": len(positive_overlap),
        "hard_overlap_positive_pair_fraction": float(len(positive_overlap) / pairs) if pairs else 0.0,
        "hard_overlap_iou_among_positive_pairs": _quantiles(positive_overlap),
    }


def oracle_score_intervention(
    rows: list[dict[str, Any]],
    gt_map: np.ndarray,
    *,
    instance_nms_iou: float,
) -> dict[str, Any]:
    """Run the frozen assembly using detached unique-TP utilities as scores."""

    # Imported locally: the runtime routine is deliberately reused rather than
    # reimplementing NMS/paint order in the reporting layer.
    from run.run_on_epoch import _assemble_instance_map

    scores = [
        float(row["utility_target"]) * (0.3 if bool(row.get("edge_penalized", False)) else 1.0)
        for row in rows
    ]
    pred = _assemble_instance_map(
        [row["bbox_xyxy"] for row in rows],
        scores,
        [row["mask"] for row in rows],
        [int(row["prompt_group_id"]) for row in rows],
        gt_map.shape,
        float(instance_nms_iou),
    )
    native = native_final_stage(gt_map, pred, threshold=ORACLE_MATCH_IOU)
    return {
        "kind": "gt_only_oracle_score_intervention_not_model_performance",
        "score_replacement": "detached unique-TP utility times the unchanged edge penalty",
        "tp": native["tp"],
        "fp": native["fp"],
        "fn": native["fn"],
        "dq": native["dq"],
        "sq": native["sq"],
        "pq": native["pq"],
    }


def image_component_audit(artifact: dict[str, Any], *, instance_nms_iou: float = 0.5) -> dict[str, Any]:
    gt_map = deserialize_gt(artifact)
    rows = selected_utility_labels(deserialize_selected(artifact), gt_map)
    counts = {label: sum(row["utility_label"] == label for row in rows) for label in ("unique_tp", "unmatched_fp", "duplicate")}
    counts["merge_risk"] = sum(bool(row["merge_risk"]) for row in rows)
    counts["valid_prediction_count"] = len(rows)
    return {
        "sample_id": str(artifact["sample_id"]),
        "patient": int(artifact["patient"]),
        "utility_labels": {
            **counts,
            "utility_effective_sample_fraction": float(len(rows) / len(rows)) if rows else 0.0,
            "class_fractions": {key: float(value / len(rows)) if rows else 0.0 for key, value in counts.items() if key != "valid_prediction_count"},
        },
        "score_calibration": score_calibration(rows),
        "exclusivity": hard_exclusivity(rows, gt_map),
        "oracle_score_intervention": oracle_score_intervention(rows, gt_map, instance_nms_iou=instance_nms_iou),
    }
