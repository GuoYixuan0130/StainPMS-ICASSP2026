"""Metric and stability primitives with an inclusive IoU >= 0.5 contract."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, spearmanr

from .protocol import PQ_IOU_THRESHOLD


def relabel_instances(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32)
    out = np.zeros(labels.shape, dtype=np.int32)
    for new_id, old_id in enumerate(np.unique(labels)[1:], start=1):
        out[labels == old_id] = new_id
    return out


def pairwise_overlap(true: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    true = relabel_instances(true)
    pred = relabel_instances(pred)
    true_ids = np.unique(true)[1:]
    pred_ids = np.unique(pred)[1:]
    intersection = np.zeros((len(true_ids), len(pred_ids)), dtype=np.float64)
    if not len(true_ids) or not len(pred_ids):
        return true_ids, pred_ids, intersection, intersection.copy()
    pred_lookup = {int(pid): idx for idx, pid in enumerate(pred_ids)}
    for ti, true_id in enumerate(true_ids):
        overlapping = pred[true == true_id]
        for pred_id in np.unique(overlapping):
            if pred_id:
                intersection[ti, pred_lookup[int(pred_id)]] = float((overlapping == pred_id).sum())
    true_area = np.asarray([(true == iid).sum() for iid in true_ids], dtype=np.float64)
    pred_area = np.asarray([(pred == iid).sum() for iid in pred_ids], dtype=np.float64)
    union = true_area[:, None] + pred_area[None, :] - intersection
    return true_ids, pred_ids, intersection, union


def iou_matrix(true: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    true_ids, pred_ids, inter, union = pairwise_overlap(true, pred)
    return true_ids, pred_ids, inter, union, np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def inclusive_iou_pairs(true: np.ndarray, pred: np.ndarray, threshold: float = PQ_IOU_THRESHOLD) -> tuple[list[int], list[int], list[int], list[int], np.ndarray]:
    true_ids, pred_ids, _, _, iou = iou_matrix(true, pred)
    if not len(true_ids) or not len(pred_ids):
        return [], [], true_ids.astype(int).tolist(), pred_ids.astype(int).tolist(), iou
    eligible = iou >= threshold
    # Maximising cardinality before IoU handles exact 0.5 edges deterministically.
    bonus = float(min(iou.shape) + 1)
    row, col = linear_sum_assignment(-np.where(eligible, bonus + iou, 0.0))
    keep = eligible[row, col]
    row, col = row[keep], col[keep]
    paired_true = true_ids[row].astype(int).tolist()
    paired_pred = pred_ids[col].astype(int).tolist()
    paired_true_set, paired_pred_set = set(paired_true), set(paired_pred)
    return (
        paired_true,
        paired_pred,
        [int(item) for item in true_ids if int(item) not in paired_true_set],
        [int(item) for item in pred_ids if int(item) not in paired_pred_set],
        iou,
    )


def pq_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    paired_true, paired_pred, unpaired_true, unpaired_pred, iou = inclusive_iou_pairs(true, pred)
    true_ids, pred_ids, _, _, _ = iou_matrix(true, pred)
    true_lookup = {int(value): index for index, value in enumerate(true_ids)}
    pred_lookup = {int(value): index for index, value in enumerate(pred_ids)}
    paired_iou = [iou[true_lookup[t], pred_lookup[p]] for t, p in zip(paired_true, paired_pred)]
    tp, fp, fn = len(paired_true), len(unpaired_pred), len(unpaired_true)
    denom = tp + 0.5 * fp + 0.5 * fn
    dq = float(tp / denom) if denom else 1.0
    sq = float(np.mean(paired_iou)) if tp else 0.0
    return {"dq": dq, "sq": sq, "pq": dq * sq, "tp": tp, "fp": fp, "fn": fn}


def aji(true: np.ndarray, pred: np.ndarray) -> float:
    true_ids, pred_ids, inter, union, iou = iou_matrix(true, pred)
    if not len(true_ids) and not len(pred_ids):
        return 1.0
    if not len(true_ids) or not len(pred_ids):
        return 0.0
    selected_pred = np.argmax(iou, axis=1)
    selected_iou = np.max(iou, axis=1)
    selected_true = np.where(selected_iou > 0.0)[0]
    selected_pred = selected_pred[selected_true]
    total_inter = float(inter[selected_true, selected_pred].sum())
    total_union = float(union[selected_true, selected_pred].sum())
    used_true = set(selected_true.tolist())
    used_pred = set(selected_pred.tolist())
    true_area = np.asarray([(true == item).sum() for item in true_ids], dtype=np.float64)
    pred_area = np.asarray([(pred == item).sum() for item in pred_ids], dtype=np.float64)
    total_union += float(true_area[[i for i in range(len(true_ids)) if i not in used_true]].sum())
    total_union += float(pred_area[[i for i in range(len(pred_ids)) if i not in used_pred]].sum())
    return float(total_inter / total_union) if total_union else 1.0


def aji_plus(true: np.ndarray, pred: np.ndarray) -> float:
    true_ids, pred_ids, inter, union, iou = iou_matrix(true, pred)
    if not len(true_ids) and not len(pred_ids):
        return 1.0
    if not len(true_ids) or not len(pred_ids):
        return 0.0
    rows, cols = linear_sum_assignment(-iou)
    keep = iou[rows, cols] > 0.0
    rows, cols = rows[keep], cols[keep]
    total_inter = float(inter[rows, cols].sum())
    total_union = float(union[rows, cols].sum())
    used_true, used_pred = set(rows.tolist()), set(cols.tolist())
    true_area = np.asarray([(true == item).sum() for item in true_ids], dtype=np.float64)
    pred_area = np.asarray([(pred == item).sum() for item in pred_ids], dtype=np.float64)
    total_union += float(true_area[[i for i in range(len(true_ids)) if i not in used_true]].sum())
    total_union += float(pred_area[[i for i in range(len(pred_ids)) if i not in used_pred]].sum())
    return float(total_inter / total_union) if total_union else 1.0


def binary_dice(true: np.ndarray, pred: np.ndarray) -> float:
    left, right = np.asarray(true) > 0, np.asarray(pred) > 0
    denom = int(left.sum() + right.sum())
    return float(2 * (left & right).sum() / denom) if denom else 1.0


def instance_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    values: dict[str, float | int] = {
        "dice": binary_dice(true, pred),
        "aji": aji(true, pred),
        "aji_plus": aji_plus(true, pred),
    }
    values.update(pq_metrics(true, pred))
    return values


def coordinate_match(left: np.ndarray, right: np.ndarray, image_shape: tuple[int, int], radius_fraction: float) -> dict[str, object]:
    left, right = np.asarray(left, dtype=float).reshape(-1, 2), np.asarray(right, dtype=float).reshape(-1, 2)
    diagonal = float(np.hypot(*image_shape))
    threshold = radius_fraction * diagonal
    if not len(left) or not len(right):
        return {"left": [], "right": [], "distances": [], "threshold_px": threshold}
    distances = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=-1)
    rows, cols = linear_sum_assignment(distances)
    keep = distances[rows, cols] <= threshold
    return {
        "left": rows[keep].astype(int).tolist(),
        "right": cols[keep].astype(int).tolist(),
        "distances": distances[rows[keep], cols[keep]].astype(float).tolist(),
        "threshold_px": threshold,
    }


def correlation(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    left, right = np.asarray(left, dtype=float).reshape(-1), np.asarray(right, dtype=float).reshape(-1)
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan"), float("nan")
    return float(pearsonr(left, right).statistic), float(spearmanr(left, right).statistic)


def map_iou(left: np.ndarray, right: np.ndarray) -> float:
    left, right = np.asarray(left) > 0, np.asarray(right) > 0
    union = int((left | right).sum())
    return float((left & right).sum() / union) if union else 1.0


def split_merge_counts(reference: np.ndarray, candidate: np.ndarray, overlap_threshold: float = 0.10) -> tuple[int, int]:
    _, _, _, _, overlaps = iou_matrix(reference, candidate)
    # IoU alone is conservative; use the overlap graph only as a consistency flag.
    ref_edges = (overlaps >= overlap_threshold).sum(axis=1) if overlaps.size else np.asarray([])
    cand_edges = (overlaps >= overlap_threshold).sum(axis=0) if overlaps.size else np.asarray([])
    return int((ref_edges > 1).sum()), int((cand_edges > 1).sum())


def paired_bootstrap(values: Iterable[float], seed: int, replicates: int) -> dict[str, float | int]:
    values = np.asarray(list(values), dtype=float)
    if not len(values):
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    rng = np.random.default_rng(seed)
    means = np.asarray([values[rng.integers(0, len(values), len(values))].mean() for _ in range(replicates)])
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def per_group_mean(rows: Iterable[dict[str, object]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    out: list[dict[str, object]] = []
    for group, items in sorted(groups.items(), key=lambda pair: str(pair[0])):
        payload = dict(zip(keys, group, strict=True))
        numeric = {key for item in items for key, value in item.items() if isinstance(value, (int, float, np.number))}
        for key in numeric:
            payload[key] = float(np.mean([float(item[key]) for item in items if key in item]))
        payload["n"] = len(items)
        out.append(payload)
    return out
