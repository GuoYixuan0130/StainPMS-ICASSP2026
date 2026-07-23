"""Pure accounting for the Phase-2A TNBC zero-training oracle diagnosis.

The functions here deliberately have no model, dataset, or CUDA dependency.
The GPU runner provides per-prompt masks and native final maps; this module
then computes the four owner-specified stages using fixed IoU > 0.5 matching.

Oracle stages are *pool* upper bounds.  Their DQ/SQ/PQ values are derived from
an ideal one-to-one matching after unmatched predictions are removed; they are
not deployable instance maps and must never be reported as native performance.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from stainpms.evaluator import evaluate_instance_pair
from stainpms.phase1_metrics import final_instance_overlap_table, instance_ids


ORACLE_MATCH_IOU = 0.5
ERROR_SENSITIVITY_FRACTIONS = (0.0, 0.1)


def encode_binary_rle(mask: np.ndarray) -> dict[str, Any]:
    """Encode a 2-D binary mask as deterministic uncompressed COCO-style RLE."""

    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise ValueError(f"RLE requires a 2-D mask, got {array.shape}")
    flat = np.asfortranarray(array).reshape(-1, order="F")
    changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1 if flat.size else np.empty(0, dtype=int)
    boundaries = np.concatenate(([0], changes, [flat.size]))
    counts = (boundaries[1:] - boundaries[:-1]).astype(int).tolist()
    if flat.size and bool(flat[0]):
        counts = [0, *counts]
    return {"size": [int(array.shape[0]), int(array.shape[1])], "counts": counts}


def decode_binary_rle(rle: dict[str, Any]) -> np.ndarray:
    """Decode :func:`encode_binary_rle` output without external dependencies."""

    h, w = [int(value) for value in rle["size"]]
    total = h * w
    flat = np.zeros(total, dtype=bool)
    offset = 0
    foreground = False
    for count_raw in rle["counts"]:
        count = int(count_raw)
        if count < 0 or offset + count > total:
            raise ValueError("invalid binary RLE count")
        if foreground:
            flat[offset : offset + count] = True
        offset += count
        foreground = not foreground
    if offset != total:
        raise ValueError("binary RLE does not cover its declared image size")
    return flat.reshape((h, w), order="F")


def _candidate_key(record: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(record.get("prompt_group_id", -1)),
        int(record.get("token", -1)),
        int(record.get("crop_index", -1)),
        int(record.get("record_index", -1)),
    )


def _mask_gt_ious(mask: np.ndarray, gt_map: np.ndarray) -> dict[int, float]:
    """IoU against every GT touched by ``mask`` in one label lookup."""

    binary = np.asarray(mask, dtype=bool)
    gt = np.asarray(gt_map, dtype=np.int32)
    if binary.shape != gt.shape:
        raise ValueError(f"mask/GT shape mismatch: {binary.shape} vs {gt.shape}")
    mask_area = int(binary.sum())
    if mask_area == 0:
        return {}
    labels, intersections = np.unique(gt[binary], return_counts=True)
    gt_areas = np.bincount(gt.reshape(-1), minlength=int(gt.max()) + 1)
    output: dict[int, float] = {}
    for raw_id, raw_intersection in zip(labels, intersections, strict=True):
        gt_id = int(raw_id)
        if gt_id == 0:
            continue
        intersection = int(raw_intersection)
        union = mask_area + int(gt_areas[gt_id]) - intersection
        output[gt_id] = float(intersection / union) if union else 0.0
    return output


def annotate_pool_ious(records: Iterable[dict[str, Any]], gt_map: np.ndarray) -> list[dict[str, Any]]:
    """Attach sparse GT-IoU tables to candidate records, preserving input order."""

    output: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        row["gt_ious"] = {str(key): value for key, value in _mask_gt_ious(row["mask"], gt_map).items()}
        output.append(row)
    return output


def pool_gt_maxima(records: Iterable[dict[str, Any]], gt_ids: Iterable[int]) -> dict[int, dict[str, Any] | None]:
    """Return the best mask record for each GT, with deterministic tie breaking."""

    maxima: dict[int, dict[str, Any] | None] = {int(gt_id): None for gt_id in gt_ids}
    for record in records:
        for raw_gt_id, raw_iou in record.get("gt_ious", {}).items():
            gt_id = int(raw_gt_id)
            value = float(raw_iou)
            current = maxima.get(gt_id)
            if current is None or value > float(current["iou"]) or (
                value == float(current["iou"]) and _candidate_key(record) < _candidate_key(current["record"])
            ):
                maxima[gt_id] = {"iou": value, "record": record}
    return maxima


def _group_edges(records: Iterable[dict[str, Any]], gt_ids: Iterable[int]) -> tuple[list[int], dict[tuple[int, int], dict[str, Any]]]:
    """Maximize within every prompt group before one-to-one oracle matching."""

    groups = sorted({int(record["prompt_group_id"]) for record in records})
    allowed_gt = {int(value) for value in gt_ids}
    edges: dict[tuple[int, int], dict[str, Any]] = {}
    for record in records:
        group = int(record["prompt_group_id"])
        for raw_gt_id, raw_iou in record.get("gt_ious", {}).items():
            gt_id = int(raw_gt_id)
            if gt_id not in allowed_gt:
                continue
            value = float(raw_iou)
            key = (group, gt_id)
            current = edges.get(key)
            if current is None or value > float(current["iou"]) or (
                value == float(current["iou"]) and _candidate_key(record) < _candidate_key(current["record"])
            ):
                edges[key] = {"iou": value, "record": record}
    return groups, edges


def maximum_cardinality_max_iou_matching(
    records: Iterable[dict[str, Any]],
    gt_ids: Iterable[int],
    *,
    threshold: float = ORACLE_MATCH_IOU,
) -> dict[str, Any]:
    """One-to-one prompt-group/GT matching: max cardinality, then total IoU.

    A group contains all crop appearances of one automatic point.  For the
    all-candidate pool it also contains its four native mask tokens.  The
    ``cardinality_bonus`` makes one additional eligible pair worth more than
    any possible sum-IoU difference, which implements the specified lexicopic
    objective exactly for finite matrices.
    """

    gt_list = sorted({int(value) for value in gt_ids})
    records_list = list(records)
    groups, edges = _group_edges(records_list, gt_list)
    if not groups or not gt_list:
        return {
            "groups": groups,
            "eligible_gt_ids": [],
            "matched": [],
            "tp": 0,
            "raw_group_count": len(groups),
            "raw_mask_count": len(records_list),
            "covered_gt_count": 0,
            "one_to_one_conflict_gt_count": 0,
            "multi_gt_group_count": 0,
        }
    values = np.zeros((len(groups), len(gt_list)), dtype=np.float64)
    eligible = np.zeros_like(values, dtype=bool)
    group_to_index = {value: index for index, value in enumerate(groups)}
    gt_to_index = {value: index for index, value in enumerate(gt_list)}
    for (group, gt_id), payload in edges.items():
        value = float(payload["iou"])
        if value > float(threshold):
            values[group_to_index[group], gt_to_index[gt_id]] = value
            eligible[group_to_index[group], gt_to_index[gt_id]] = True
    cardinality_bonus = float(min(values.shape) + 1)
    weights = np.where(eligible, cardinality_bonus + values, 0.0)
    rows, cols = linear_sum_assignment(-weights)
    selected = [(int(row), int(col)) for row, col in zip(rows, cols, strict=True) if eligible[row, col]]
    matched: list[dict[str, Any]] = []
    for row, col in selected:
        group = groups[row]
        gt_id = gt_list[col]
        payload = edges[(group, gt_id)]
        matched.append(
            {
                "prompt_group_id": group,
                "gt_instance_id": gt_id,
                "iou": float(payload["iou"]),
                "record_index": int(payload["record"]["record_index"]),
                "token": int(payload["record"].get("token", -1)),
                "crop_index": int(payload["record"].get("crop_index", -1)),
            }
        )
    covered_gt = {gt_list[index] for index in np.flatnonzero(eligible.any(axis=0))}
    group_eligible_counts = eligible.sum(axis=1)
    return {
        "groups": groups,
        "eligible_gt_ids": sorted(covered_gt),
        "matched": matched,
        "tp": len(matched),
        "raw_group_count": len(groups),
        "raw_mask_count": len(records_list),
        "covered_gt_count": len(covered_gt),
        "one_to_one_conflict_gt_count": int(len(covered_gt) - len(matched)),
        "multi_gt_group_count": int(np.count_nonzero(group_eligible_counts >= 2)),
    }


def oracle_pool_stage(records: Iterable[dict[str, Any]], gt_map: np.ndarray, *, threshold: float = ORACLE_MATCH_IOU) -> dict[str, Any]:
    """Score a selected/all-candidate pool after ideal unmatched-FP removal."""

    gt_ids = instance_ids(gt_map)
    matching = maximum_cardinality_max_iou_matching(records, gt_ids, threshold=threshold)
    tp = int(matching["tp"])
    fn = int(len(gt_ids) - tp)
    raw_groups = int(matching["raw_group_count"])
    raw_masks = int(matching["raw_mask_count"])
    ious = [float(pair["iou"]) for pair in matching["matched"]]
    dq = float(tp / (tp + 0.5 * fn)) if tp or fn else 0.0
    sq = float(np.mean(ious)) if ious else 0.0
    return {
        "kind": "pool_oracle_after_unmatched_fp_removal",
        "threshold_rule": f"IoU > {float(threshold):.1f}; max cardinality then max total IoU",
        "raw_prediction_group_count": raw_groups,
        "raw_prediction_mask_count": raw_masks,
        "oracle_filtered_prediction_count": tp,
        "raw_unmatched_group_count": raw_groups - tp,
        "tp": tp,
        "fp": 0,
        "fn": fn,
        "dq": dq,
        "sq": sq,
        "pq": dq * sq,
        "matched_iou_sum": float(sum(ious)),
        "matched": matching["matched"],
        "covered_gt_count": int(matching["covered_gt_count"]),
        "coverage_recall_at_0_5": float(matching["covered_gt_count"] / len(gt_ids)) if gt_ids else None,
        "one_to_one_conflict_gt_count": int(matching["one_to_one_conflict_gt_count"]),
        "multi_gt_group_count": int(matching["multi_gt_group_count"]),
    }


def filter_final_map_by_pairs(final_map: np.ndarray, matched_pred_ids: Iterable[int]) -> np.ndarray:
    """Remove unmatched final instances without changing any retained mask."""

    prediction = np.asarray(final_map, dtype=np.int32)
    keep = {int(value) for value in matched_pred_ids}
    output = np.zeros_like(prediction)
    for pred_id in keep:
        output[prediction == pred_id] = pred_id
    return output


def final_pool_oracle_stage(gt_map: np.ndarray, final_map: np.ndarray, *, threshold: float = ORACLE_MATCH_IOU) -> dict[str, Any]:
    """Filter only unmatched native-final masks and evaluate the retained map."""

    gt = np.asarray(gt_map, dtype=np.int32)
    pred = np.asarray(final_map, dtype=np.int32)
    records: list[dict[str, Any]] = []
    for index, pred_id in enumerate(instance_ids(pred)):
        records.append(
            {
                "record_index": index,
                "prompt_group_id": int(pred_id),
                "token": -1,
                "crop_index": -1,
                "mask": pred == pred_id,
            }
        )
    annotated = annotate_pool_ious(records, gt)
    oracle = oracle_pool_stage(annotated, gt, threshold=threshold)
    retained = filter_final_map_by_pairs(pred, [pair["prompt_group_id"] for pair in oracle["matched"]])
    evaluator = evaluate_instance_pair(gt, retained, mode="strict", match_iou=threshold)
    oracle["strict_metrics_after_filtering"] = evaluator["metrics"]
    oracle["strict_pairing_after_filtering"] = evaluator["pairing"]
    oracle["retained_final_map"] = retained
    return oracle


def native_final_stage(gt_map: np.ndarray, final_map: np.ndarray, *, threshold: float = ORACLE_MATCH_IOU) -> dict[str, Any]:
    """Strict native-final metrics, unmodified by any oracle operation."""

    evaluation = evaluate_instance_pair(gt_map, final_map, mode="strict", match_iou=threshold)
    pairing = evaluation["pairing"] or {}
    return {
        "kind": "native_final",
        "tp": int(pairing.get("tp", 0)),
        "fp": int(pairing.get("fp", 0)),
        "fn": int(pairing.get("fn", 0)),
        "dq": evaluation["metrics"]["dq"],
        "sq": evaluation["metrics"]["sq"],
        "pq": evaluation["metrics"]["pq"],
        "strict_metrics": evaluation["metrics"],
        "strict_pairing": pairing,
    }


def _quantiles(values: Iterable[float]) -> dict[str, float | None]:
    numeric = np.asarray([float(value) for value in values], dtype=np.float64)
    if numeric.size == 0:
        return {key: None for key in ("mean", "median", "q10", "q25", "q75", "q90")}
    return {
        "mean": float(numeric.mean()),
        "median": float(np.quantile(numeric, 0.50)),
        "q10": float(np.quantile(numeric, 0.10)),
        "q25": float(np.quantile(numeric, 0.25)),
        "q75": float(np.quantile(numeric, 0.75)),
        "q90": float(np.quantile(numeric, 0.90)),
    }


def _native_structural_errors(gt_map: np.ndarray, final_map: np.ndarray, native: dict[str, Any], *, fractions: Iterable[float] = ERROR_SENSITIVITY_FRACTIONS) -> dict[str, Any]:
    """Transparent duplicate/split/merge counting with fraction sensitivity."""

    overlap = final_instance_overlap_table(gt_map, final_map)
    table = np.asarray(overlap["intersections"], dtype=np.int64)
    gt_ids = [int(value) for value in overlap["gt_ids"]]
    pred_ids = [int(value) for value in overlap["pred_ids"]]
    gt_areas = np.asarray(overlap["gt_areas"], dtype=np.float64)
    pred_areas = np.asarray(overlap["pred_areas"], dtype=np.float64)
    pairing = native["strict_pairing"] or {}
    paired_gt = {int(value) for value in pairing.get("paired_true", [])}
    paired_pred = {int(value) for value in pairing.get("paired_pred", [])}
    paired_gt_by_pred = {int(pred): int(gt) for gt, pred in zip(pairing.get("paired_true", []), pairing.get("paired_pred", []), strict=True)}
    unmatched_gt = [value for value in gt_ids if value not in paired_gt]
    unmatched_pred = [value for value in pred_ids if value not in paired_pred]

    duplicate = 0
    for pred_id in unmatched_pred:
        if not gt_ids:
            continue
        overlaps = table[np.asarray(gt_ids, dtype=int), pred_id]
        positive = [gt_ids[index] for index, value in enumerate(overlaps) if int(value) > 0]
        if any(gt_id in paired_gt for gt_id in positive):
            duplicate += 1

    sensitivity: dict[str, dict[str, int]] = {}
    for fraction in fractions:
        split = 0
        merge = 0
        for gt_id in unmatched_gt:
            contributors = sum(
                int(table[gt_id, pred_id]) / gt_areas[gt_id] > float(fraction)
                for pred_id in pred_ids
                if gt_areas[gt_id] > 0
            )
            split += int(contributors >= 2)
        for pred_id in unmatched_pred:
            contributors = sum(
                int(table[gt_id, pred_id]) / pred_areas[pred_id] > float(fraction)
                for gt_id in gt_ids
                if pred_areas[pred_id] > 0
            )
            merge += int(contributors >= 2)
        sensitivity[f"overlap_fraction_gt_or_pred_gt_{fraction:g}"] = {"split": split, "merge": merge}
    return {
        "definition": {
            "duplicate": "unmatched final prediction overlaps at least one GT already strictly paired to another final prediction",
            "split": "unmatched GT overlaps at least two final predictions; threshold sensitivity records each overlap as a GT-area fraction",
            "merge": "unmatched final prediction overlaps at least two GT instances; threshold sensitivity records each overlap as a prediction-area fraction",
        },
        "duplicate_unmatched_prediction_count": duplicate,
        "unmatched_gt_count": len(unmatched_gt),
        "unmatched_prediction_count": len(unmatched_pred),
        "sensitivity": sensitivity,
        "paired_gt_by_pred": paired_gt_by_pred,
    }


def error_partition(
    *,
    gt_map: np.ndarray,
    all_candidate_records: Iterable[dict[str, Any]],
    selected_records: Iterable[dict[str, Any]],
    native_final: dict[str, Any],
    final_map: np.ndarray,
    threshold: float = ORACLE_MATCH_IOU,
) -> dict[str, Any]:
    """Per-GT loss decomposition and supplementary native-final errors."""

    gt_ids = instance_ids(gt_map)
    all_max = pool_gt_maxima(all_candidate_records, gt_ids)
    selected_max = pool_gt_maxima(selected_records, gt_ids)
    pairing = native_final["strict_pairing"] or {}
    paired_gt = {int(value) for value in pairing.get("paired_true", [])}
    generation = []
    selection = []
    assembly = []
    final_tp = []
    per_gt: list[dict[str, Any]] = []
    for gt_id in gt_ids:
        all_value = float(all_max[gt_id]["iou"]) if all_max[gt_id] is not None else 0.0
        selected_value = float(selected_max[gt_id]["iou"]) if selected_max[gt_id] is not None else 0.0
        if all_value <= threshold:
            label = "generation_miss"
            generation.append(gt_id)
        elif selected_value <= threshold:
            label = "selection_miss"
            selection.append(gt_id)
        elif gt_id not in paired_gt:
            label = "assembly_loss"
            assembly.append(gt_id)
        else:
            label = "native_final_tp"
            final_tp.append(gt_id)
        per_gt.append(
            {
                "gt_instance_id": gt_id,
                "all_candidate_best_iou": all_value,
                "selected_pool_best_iou": selected_value,
                "native_final_matched": gt_id in paired_gt,
                "error_class": label,
            }
        )
    all_oracle = oracle_pool_stage(all_candidate_records, gt_map, threshold=threshold)
    selected_oracle = oracle_pool_stage(selected_records, gt_map, threshold=threshold)
    return {
        "per_gt": per_gt,
        "counts": {
            "generation_miss": len(generation),
            "selection_miss": len(selection),
            "assembly_loss": len(assembly),
            "native_final_tp": len(final_tp),
            "native_final_false_positive_count": int(native_final.get("fp", 0)),
            "native_final_false_negative_count": int(native_final.get("fn", 0)),
            "all_candidate_one_to_one_conflict_gt_count": all_oracle["one_to_one_conflict_gt_count"],
            "selected_one_to_one_conflict_gt_count": selected_oracle["one_to_one_conflict_gt_count"],
        },
        "all_candidate_coverage": _quantiles(float(row["all_candidate_best_iou"]) for row in per_gt),
        "selected_candidate_coverage": _quantiles(float(row["selected_pool_best_iou"]) for row in per_gt),
        "native_final_structural_errors": _native_structural_errors(gt_map, final_map, native_final),
    }


def stage_gap(upper: dict[str, Any], lower: dict[str, Any]) -> dict[str, float | int | None]:
    """Upper-minus-lower accounting; accepts native or oracle stage records."""

    output: dict[str, float | int | None] = {}
    for key in ("tp", "fp", "fn", "dq", "sq", "pq"):
        upper_value = upper.get(key)
        lower_value = lower.get(key)
        output[key] = None if upper_value is None or lower_value is None else float(upper_value) - float(lower_value)
    return output


def summarize_numeric(values: Iterable[float | None]) -> dict[str, float | int | None]:
    numeric = np.asarray([float(value) for value in values if value is not None], dtype=np.float64)
    if numeric.size == 0:
        return {"count": 0, "mean": None, "std_sample": None, "positive_count": 0}
    return {
        "count": int(numeric.size),
        "mean": float(numeric.mean()),
        "std_sample": float(numeric.std(ddof=1)) if numeric.size > 1 else 0.0,
        "positive_count": int(np.count_nonzero(numeric > 0.0)),
    }
