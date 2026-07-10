"""Analyze StainPQR evaluation artifacts.

This script consumes files produced by:

    python main.py --eval --dump_eval_artifacts_dir <dir> ...

For every image it computes the standard segmentation metrics and a small
failure-mode decomposition for Stage 0 experiments:

  - TP / FP / FN under PQ matching
  - unmatched GT with best IoU near the 0.5 PQ cliff
  - weak matched pairs just above the 0.5 threshold
  - split-like and merge-like unmatched objects
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - optional dependency in light envs
    linear_sum_assignment = None


def _ids(label_map: np.ndarray) -> list[int]:
    return [int(v) for v in np.unique(label_map) if int(v) != 0]


def remap_label(label_map: np.ndarray, by_size: bool = False) -> np.ndarray:
    label_map = np.asarray(label_map)
    inst_ids = _ids(label_map)
    if not inst_ids:
        return label_map.astype(np.int32, copy=True)
    if by_size:
        inst_ids = sorted(inst_ids, key=lambda inst_id: int((label_map == inst_id).sum()), reverse=True)
    out = np.zeros(label_map.shape, dtype=np.int32)
    for new_id, old_id in enumerate(inst_ids, start=1):
        out[label_map == old_id] = new_id
    return out


def _safe_metric(fn, default: float = math.nan) -> float:
    try:
        value = fn()
    except Exception:
        return default
    if isinstance(value, np.generic):
        value = value.item()
    return float(value)


def _pairwise_stats(true: np.ndarray, pred: np.ndarray):
    true_ids = _ids(true)
    pred_ids = _ids(pred)
    inter = np.zeros((len(true_ids), len(pred_ids)), dtype=np.float64)
    true_area = np.zeros(len(true_ids), dtype=np.float64)
    pred_area = np.zeros(len(pred_ids), dtype=np.float64)

    true_index = {tid: idx for idx, tid in enumerate(true_ids)}
    pred_index = {pid: idx for idx, pid in enumerate(pred_ids)}

    for idx, tid in enumerate(true_ids):
        true_mask = true == tid
        true_area[idx] = float(true_mask.sum())
        overlap_ids, overlap_counts = np.unique(pred[true_mask], return_counts=True)
        for pid, count in zip(overlap_ids, overlap_counts):
            pid = int(pid)
            if pid == 0 or pid not in pred_index:
                continue
            inter[idx, pred_index[pid]] = float(count)

    for idx, pid in enumerate(pred_ids):
        pred_area[idx] = float((pred == pid).sum())

    union = true_area[:, None] + pred_area[None, :] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return true_ids, pred_ids, inter, true_area, pred_area, iou


def get_dice_1(true: np.ndarray, pred: np.ndarray) -> float:
    true_bin = np.asarray(true) > 0
    pred_bin = np.asarray(pred) > 0
    denom = float(true_bin.sum() + pred_bin.sum())
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(true_bin, pred_bin).sum() / denom)


def get_fast_dice_2(true: np.ndarray, pred: np.ndarray) -> float:
    _, _, inter, true_area, pred_area, _ = _pairwise_stats(true, pred)
    if inter.size == 0:
        return 1.0 if len(_ids(true)) == 0 and len(_ids(pred)) == 0 else 0.0
    hit_true, hit_pred = np.nonzero(inter > 0)
    if len(hit_true) == 0:
        return 0.0
    overall_inter = float(inter[hit_true, hit_pred].sum())
    overall_total = float((true_area[hit_true] + pred_area[hit_pred]).sum())
    if overall_total == 0:
        return 0.0
    return float(2.0 * overall_inter / overall_total)


def get_fast_aji(true: np.ndarray, pred: np.ndarray) -> float:
    true_ids, pred_ids, inter, true_area, pred_area, iou = _pairwise_stats(true, pred)
    if len(true_ids) == 0 and len(pred_ids) == 0:
        return 1.0
    if len(true_ids) == 0 or len(pred_ids) == 0:
        return 0.0

    paired_true_idx = []
    paired_pred_idx = []
    paired_inter = []
    paired_union = []
    for true_idx in range(len(true_ids)):
        pred_idx = int(np.argmax(iou[true_idx]))
        if iou[true_idx, pred_idx] > 0:
            paired_true_idx.append(true_idx)
            paired_pred_idx.append(pred_idx)
            paired_inter.append(inter[true_idx, pred_idx])
            paired_union.append(true_area[true_idx] + pred_area[pred_idx] - inter[true_idx, pred_idx])

    paired_true_set = set(paired_true_idx)
    paired_pred_set = set(paired_pred_idx)
    overall_inter = float(np.sum(paired_inter))
    overall_union = float(np.sum(paired_union))
    for idx in range(len(true_ids)):
        if idx not in paired_true_set:
            overall_union += float(true_area[idx])
    for idx in range(len(pred_ids)):
        if idx not in paired_pred_set:
            overall_union += float(pred_area[idx])
    if overall_union == 0:
        return 0.0
    return float(overall_inter / overall_union)


def get_fast_aji_plus(true: np.ndarray, pred: np.ndarray) -> float:
    if linear_sum_assignment is None:
        return math.nan
    true_ids, pred_ids, inter, true_area, pred_area, iou = _pairwise_stats(true, pred)
    if len(true_ids) == 0 and len(pred_ids) == 0:
        return 1.0
    if len(true_ids) == 0 or len(pred_ids) == 0:
        return 0.0

    paired_true_idx, paired_pred_idx = linear_sum_assignment(-iou)
    paired_iou = iou[paired_true_idx, paired_pred_idx]
    keep = paired_iou > 0
    paired_true_idx = paired_true_idx[keep]
    paired_pred_idx = paired_pred_idx[keep]

    paired_true_set = set(int(v) for v in paired_true_idx)
    paired_pred_set = set(int(v) for v in paired_pred_idx)
    paired_inter = inter[paired_true_idx, paired_pred_idx]
    paired_union = true_area[paired_true_idx] + pred_area[paired_pred_idx] - paired_inter
    overall_inter = float(paired_inter.sum())
    overall_union = float(paired_union.sum())
    for idx in range(len(true_ids)):
        if idx not in paired_true_set:
            overall_union += float(true_area[idx])
    for idx in range(len(pred_ids)):
        if idx not in paired_pred_set:
            overall_union += float(pred_area[idx])
    if overall_union == 0:
        return 0.0
    return float(overall_inter / overall_union)


def get_fast_pq(true: np.ndarray, pred: np.ndarray, match_iou: float = 0.5):
    true_ids, pred_ids, _, _, _, iou = _pairwise_stats(true, pred)
    if len(true_ids) == 0 and len(pred_ids) == 0:
        return [1.0, 1.0, 1.0], [[], [], [], []]

    if match_iou >= 0.5:
        exact_threshold = iou == match_iou
        if not np.any(exact_threshold):
            paired_true_idx, paired_pred_idx = np.nonzero(iou > match_iou)
        else:
            if linear_sum_assignment is None:
                raise RuntimeError("SciPy is required for inclusive PQ threshold matching")
            eligible = iou >= match_iou
            cardinality_bonus = float(min(iou.shape) + 1)
            weights = np.where(eligible, cardinality_bonus + iou, 0.0)
            paired_true_idx, paired_pred_idx = linear_sum_assignment(-weights)
            keep = eligible[paired_true_idx, paired_pred_idx]
            paired_true_idx = paired_true_idx[keep]
            paired_pred_idx = paired_pred_idx[keep]
    else:
        if linear_sum_assignment is None:
            raise RuntimeError("SciPy is required for PQ matching below IoU 0.5")
        paired_true_idx, paired_pred_idx = linear_sum_assignment(-iou)
        keep = iou[paired_true_idx, paired_pred_idx] > match_iou
        paired_true_idx = paired_true_idx[keep]
        paired_pred_idx = paired_pred_idx[keep]

    paired_iou = iou[paired_true_idx, paired_pred_idx] if len(paired_true_idx) else np.asarray([])
    paired_true = [true_ids[int(idx)] for idx in paired_true_idx]
    paired_pred = [pred_ids[int(idx)] for idx in paired_pred_idx]
    paired_true_set = set(paired_true)
    paired_pred_set = set(paired_pred)
    unpaired_true = [idx for idx in true_ids if idx not in paired_true_set]
    unpaired_pred = [idx for idx in pred_ids if idx not in paired_pred_set]

    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    denom = tp + 0.5 * fp + 0.5 * fn
    dq = float(tp / denom) if denom > 0 else 1.0
    sq = float(paired_iou.sum() / (tp + 1.0e-6)) if tp > 0 else 0.0
    return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]


def analyze_pair(
    name: str,
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    match_iou: float,
    near_low: float,
    weak_high: float,
    overlap_frac: float,
) -> dict:
    gt = remap_label(gt.astype(np.int32))
    pred = remap_label(pred.astype(np.int32))
    gt_ids, pred_ids, inter, gt_area, pred_area, iou = _pairwise_stats(gt, pred)

    if len(gt_ids) == 0 and len(pred_ids) == 0:
        paired_true, paired_pred, unpaired_true, unpaired_pred = [], [], [], []
        dq = sq = pq = 1.0
    else:
        [dq, sq, pq], pair_info = get_fast_pq(gt, pred, match_iou=match_iou)
        paired_true, paired_pred, unpaired_true, unpaired_pred = pair_info

    gt_to_idx = {tid: idx for idx, tid in enumerate(gt_ids)}
    pred_to_idx = {pid: idx for idx, pid in enumerate(pred_ids)}

    paired_iou = []
    for tid, pid in zip(paired_true, paired_pred):
        if int(tid) in gt_to_idx and int(pid) in pred_to_idx:
            paired_iou.append(float(iou[gt_to_idx[int(tid)], pred_to_idx[int(pid)]]))
    paired_iou_arr = np.asarray(paired_iou, dtype=np.float64)

    if iou.shape[1] > 0:
        best_iou_true = iou.max(axis=1)
    else:
        best_iou_true = np.zeros(len(gt_ids), dtype=np.float64)
    if iou.shape[0] > 0:
        best_iou_pred = iou.max(axis=0)
    else:
        best_iou_pred = np.zeros(len(pred_ids), dtype=np.float64)

    near_threshold_fn = 0
    missed_low_iou_fn = 0
    split_like_fn = 0
    for tid in unpaired_true:
        idx = gt_to_idx.get(int(tid))
        if idx is None:
            continue
        best_iou = float(best_iou_true[idx])
        if near_low <= best_iou <= match_iou:
            near_threshold_fn += 1
        if best_iou < near_low:
            missed_low_iou_fn += 1
        if gt_area[idx] > 0:
            covered_frac = inter[idx, :] / gt_area[idx]
            if int((covered_frac >= overlap_frac).sum()) >= 2:
                split_like_fn += 1

    merge_like_fp = 0
    for pid in unpaired_pred:
        idx = pred_to_idx.get(int(pid))
        if idx is None:
            continue
        if pred_area[idx] > 0:
            source_frac = inter[:, idx] / pred_area[idx]
            if int((source_frac >= overlap_frac).sum()) >= 2:
                merge_like_fp += 1

    row = {
        "image": name,
        "dice1": _safe_metric(lambda: get_dice_1(gt, pred)),
        "dice2": _safe_metric(lambda: get_fast_dice_2(gt, pred)),
        "aji": _safe_metric(lambda: get_fast_aji(gt, pred)),
        "aji_p": _safe_metric(lambda: get_fast_aji_plus(gt, pred)),
        "dq": float(dq),
        "sq": float(sq),
        "pq": float(pq),
        "gt_count": len(gt_ids),
        "pred_count": len(pred_ids),
        "tp": len(paired_true),
        "fp": len(unpaired_pred),
        "fn": len(unpaired_true),
        "near_threshold_fn": near_threshold_fn,
        "missed_low_iou_fn": missed_low_iou_fn,
        "weak_match": int(((paired_iou_arr > match_iou) & (paired_iou_arr <= weak_high)).sum()),
        "split_like_fn": split_like_fn,
        "merge_like_fp": merge_like_fp,
        "mean_paired_iou": float(np.nanmean(paired_iou_arr)) if paired_iou_arr.size else math.nan,
        "mean_best_unpaired_gt_iou": math.nan,
        "mean_best_unpaired_pred_iou": math.nan,
    }

    unpaired_true_idx = [gt_to_idx[int(tid)] for tid in unpaired_true if int(tid) in gt_to_idx]
    unpaired_pred_idx = [pred_to_idx[int(pid)] for pid in unpaired_pred if int(pid) in pred_to_idx]
    if unpaired_true_idx:
        row["mean_best_unpaired_gt_iou"] = float(np.mean(best_iou_true[unpaired_true_idx]))
    if unpaired_pred_idx:
        row["mean_best_unpaired_pred_iou"] = float(np.mean(best_iou_pred[unpaired_pred_idx]))
    return row


def summarize(rows: list[dict]) -> dict:
    metric_keys = ["dice1", "dice2", "aji", "aji_p", "dq", "sq", "pq"]
    count_keys = [
        "gt_count",
        "pred_count",
        "tp",
        "fp",
        "fn",
        "near_threshold_fn",
        "missed_low_iou_fn",
        "weak_match",
        "split_like_fn",
        "merge_like_fp",
    ]
    summary = {
        "num_images": len(rows),
        "mean_metrics": {},
        "totals": {},
    }
    for key in metric_keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary["mean_metrics"][key] = float(np.nanmean(values))
    for key in count_keys:
        summary["totals"][key] = int(sum(int(row[key]) for row in rows))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts_dir", required=True, type=Path)
    parser.add_argument("--out_prefix", default="", type=str)
    parser.add_argument("--match_iou", default=0.5, type=float)
    parser.add_argument("--near_low", default=0.3, type=float)
    parser.add_argument("--weak_high", default=0.6, type=float)
    parser.add_argument("--overlap_frac", default=0.1, type=float)
    args = parser.parse_args()

    artifact_dir = args.artifacts_dir
    pred_files = sorted(artifact_dir.glob("*_pred.npy"))
    if not pred_files:
        raise FileNotFoundError(f"No *_pred.npy files found in {artifact_dir}")

    rows = []
    for pred_path in pred_files:
        name = pred_path.name[: -len("_pred.npy")]
        gt_path = artifact_dir / f"{name}_gt.npy"
        if not gt_path.exists():
            raise FileNotFoundError(f"Missing GT artifact for {pred_path.name}: {gt_path}")
        rows.append(
            analyze_pair(
                name,
                np.load(gt_path),
                np.load(pred_path),
                match_iou=args.match_iou,
                near_low=args.near_low,
                weak_high=args.weak_high,
                overlap_frac=args.overlap_frac,
            )
        )

    out_prefix = Path(args.out_prefix) if args.out_prefix else artifact_dir / "stage0_error"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_path = out_prefix.with_suffix(".csv")
    json_path = out_prefix.with_suffix(".json")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Wrote per-image rows: {csv_path}")
    print(f"Wrote summary: {json_path}")


if __name__ == "__main__":
    main()
