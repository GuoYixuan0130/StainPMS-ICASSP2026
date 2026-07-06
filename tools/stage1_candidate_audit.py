"""Stage 1A audit for StainPQR corrective-action candidates.

This is a lightweight pre-oracle check. It reads Stage 0 artifacts and the
original test images, then asks whether simple inference-time signals can find
the residual errors that StainPQR wants to correct:

  - residual hematoxylin peaks outside current coverage vs remaining FN nuclei
  - multi-peak predicted masks vs merge-like predictions
  - raw decoder proxy scores vs weak/FP selected instances

It does not run the SAM2 decoder. Use it before spending GPU time on the full
oracle corrective-action simulation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from skimage.color import rgb2hed
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from skimage.io import imread
from skimage.morphology import binary_dilation, disk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analyze_eval_artifacts import _ids, _pairwise_stats, get_fast_pq  # noqa: E402


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _robust_normalize(values: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(values, [low, high])
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).clip(0.0, 1.0).astype(np.float32)


def _compute_h_evidence(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    rgb = np.asarray(image)[..., :3].astype(np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    hed = rgb2hed(rgb.clip(0.0, 1.0))
    h = _robust_normalize(hed[..., 0])
    if sigma > 0:
        h = gaussian(h, sigma=sigma, preserve_range=True)
    return h.astype(np.float32).clip(0.0, 1.0)


def _find_image(image_root: Path, name: str) -> Path:
    for ext in IMAGE_EXTS:
        path = image_root / f"{name}{ext}"
        if path.exists() and path.is_file():
            return path
    matches = [p for p in image_root.glob(f"{name}.*") if p.is_file()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No image file found for {name} under {image_root}")
    raise RuntimeError(f"Ambiguous image files for {name}: {matches}")


def _load_meta(artifact_dir: Path, name: str) -> dict:
    meta_path = artifact_dir / f"{name}_meta.json"
    if not meta_path.exists():
        return {"selected": [], "candidates": []}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _topk_peaks(score_map: np.ndarray, *, top_k: int, min_distance: int) -> np.ndarray:
    if score_map.size == 0 or float(score_map.max()) <= 0.0:
        return np.empty((0, 2), dtype=np.int64)
    coords_yx = peak_local_max(
        score_map.astype(np.float32),
        min_distance=min_distance,
        threshold_abs=0,
        exclude_border=False,
    )
    if len(coords_yx) == 0:
        return np.empty((0, 2), dtype=np.int64)
    scores = score_map[coords_yx[:, 0], coords_yx[:, 1]]
    order = np.argsort(-scores)
    if top_k > 0:
        order = order[:top_k]
    return coords_yx[order]


def _assign_gt_ids(gt: np.ndarray, coords_yx: np.ndarray, radius: int) -> list[int]:
    h, w = gt.shape
    hits: list[int] = []
    for y, x in coords_yx:
        y = int(y)
        x = int(x)
        if gt[y, x] > 0:
            hits.append(int(gt[y, x]))
            continue
        if radius <= 0:
            hits.append(0)
            continue
        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(w, x + radius + 1)
        window = gt[y0:y1, x0:x1]
        ys, xs = np.where(window > 0)
        if len(ys) == 0:
            hits.append(0)
            continue
        d2 = (ys - (y - y0)) ** 2 + (xs - (x - x0)) ** 2
        nearest = int(np.argmin(d2))
        hits.append(int(window[ys[nearest], xs[nearest]]))
    return hits


def _problem_sets(
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    match_iou: float,
    near_low: float,
    weak_high: float,
    overlap_frac: float,
) -> dict:
    gt_ids, pred_ids, inter, gt_area, pred_area, iou = _pairwise_stats(gt, pred)
    [_, _, _], pair_info = get_fast_pq(gt, pred, match_iou=match_iou)
    paired_true, paired_pred, unpaired_true, unpaired_pred = pair_info

    gt_to_idx = {tid: idx for idx, tid in enumerate(gt_ids)}
    pred_to_idx = {pid: idx for idx, pid in enumerate(pred_ids)}
    paired_iou_by_pred: dict[int, float] = {}
    paired_iou_by_true: dict[int, float] = {}
    for tid, pid in zip(paired_true, paired_pred):
        if int(tid) in gt_to_idx and int(pid) in pred_to_idx:
            value = float(iou[gt_to_idx[int(tid)], pred_to_idx[int(pid)]])
            paired_iou_by_true[int(tid)] = value
            paired_iou_by_pred[int(pid)] = value

    best_iou_true = iou.max(axis=1) if iou.shape[1] > 0 else np.zeros(len(gt_ids), dtype=np.float64)

    near_fn: set[int] = set()
    missed_fn: set[int] = set()
    split_like_fn: set[int] = set()
    for tid in unpaired_true:
        idx = gt_to_idx.get(int(tid))
        if idx is None:
            continue
        best_iou = float(best_iou_true[idx])
        if near_low <= best_iou <= match_iou:
            near_fn.add(int(tid))
        if best_iou < near_low:
            missed_fn.add(int(tid))
        if gt_area[idx] > 0:
            covered_frac = inter[idx, :] / gt_area[idx]
            if int((covered_frac >= overlap_frac).sum()) >= 2:
                split_like_fn.add(int(tid))

    merge_like_pred: set[int] = set()
    for pid in unpaired_pred:
        idx = pred_to_idx.get(int(pid))
        if idx is None:
            continue
        if pred_area[idx] > 0:
            source_frac = inter[:, idx] / pred_area[idx]
            if int((source_frac >= overlap_frac).sum()) >= 2:
                merge_like_pred.add(int(pid))

    weak_pred = {
        int(pid)
        for pid, value in paired_iou_by_pred.items()
        if match_iou < value <= weak_high
    }

    return {
        "gt_ids": set(gt_ids),
        "pred_ids": set(pred_ids),
        "paired_true": set(int(v) for v in paired_true),
        "paired_pred": set(int(v) for v in paired_pred),
        "unpaired_true": set(int(v) for v in unpaired_true),
        "unpaired_pred": set(int(v) for v in unpaired_pred),
        "near_fn": near_fn,
        "missed_fn": missed_fn,
        "split_like_fn": split_like_fn,
        "merge_like_pred": merge_like_pred,
        "weak_pred": weak_pred,
        "paired_iou_by_pred": paired_iou_by_pred,
    }


def _residual_peak_audit(
    gt: np.ndarray,
    pred: np.ndarray,
    evidence: np.ndarray,
    problem: dict,
    *,
    dilate_radius: int,
    top_k: int,
    min_distance: int,
    gt_radius: int,
) -> dict:
    pred_bin = pred > 0
    if dilate_radius > 0:
        pred_bin = binary_dilation(pred_bin, footprint=disk(dilate_radius))
    residual = evidence.copy()
    residual[pred_bin] = 0.0
    peaks_yx = _topk_peaks(residual, top_k=top_k, min_distance=min_distance)
    hit_ids = set(v for v in _assign_gt_ids(gt, peaks_yx, gt_radius) if v > 0)
    unpaired_true = problem["unpaired_true"]
    near_fn = problem["near_fn"]
    missed_fn = problem["missed_fn"]
    return {
        "coverage_peak_count": int(len(peaks_yx)),
        "coverage_peak_positive_gt": int(len(hit_ids)),
        "coverage_hit_fn": int(len(hit_ids & unpaired_true)),
        "coverage_hit_near_fn": int(len(hit_ids & near_fn)),
        "coverage_hit_missed_fn": int(len(hit_ids & missed_fn)),
    }


def _merge_peak_audit(
    pred: np.ndarray,
    evidence: np.ndarray,
    problem: dict,
    *,
    min_distance: int,
    max_peaks: int,
) -> dict:
    candidate_pred: set[int] = set()
    for pred_id in _ids(pred):
        mask = pred == pred_id
        if int(mask.sum()) == 0:
            continue
        masked = evidence * mask
        peaks_yx = _topk_peaks(masked, top_k=max_peaks, min_distance=min_distance)
        if len(peaks_yx) >= 2:
            candidate_pred.add(int(pred_id))
    merge_like = problem["merge_like_pred"]
    return {
        "multi_peak_pred_candidates": int(len(candidate_pred)),
        "multi_peak_hit_merge_like": int(len(candidate_pred & merge_like)),
    }


def _risk_proxy_rows(meta: dict, problem: dict) -> list[dict]:
    rows = []
    weak_pred = problem["weak_pred"]
    unpaired_pred = problem["unpaired_pred"]
    merge_like_pred = problem["merge_like_pred"]
    paired_iou_by_pred = problem["paired_iou_by_pred"]
    for record in meta.get("selected", []):
        pred_id = int(record.get("final_id", -1))
        pred_iou = float(record.get("predicted_iou", math.nan))
        stability = float(record.get("stability_score", math.nan))
        edge = bool(record.get("edge_penalized", False))
        is_weak = pred_id in weak_pred
        is_fp = pred_id in unpaired_pred
        is_merge = pred_id in merge_like_pred
        rows.append(
            {
                "pred_id": pred_id,
                "predicted_iou": pred_iou,
                "stability_score": stability,
                "edge_penalized": edge,
                "paired_iou": paired_iou_by_pred.get(pred_id, math.nan),
                "target": bool(is_weak or is_fp or is_merge),
                "weak": bool(is_weak),
                "fp": bool(is_fp),
                "merge_like": bool(is_merge),
                "score_low_iou": 1.0 - pred_iou if math.isfinite(pred_iou) else -math.inf,
                "score_low_stability": 1.0 - stability if math.isfinite(stability) else -math.inf,
                "score_combined": (
                    (1.0 - pred_iou if math.isfinite(pred_iou) else 0.0)
                    + 0.5 * (1.0 - stability if math.isfinite(stability) else 0.0)
                    + (0.25 if edge else 0.0)
                ),
            }
        )
    return rows


def _update_proxy_summary(summary: dict, rows: list[dict], budgets: list[int]) -> None:
    target_count = sum(1 for row in rows if row["target"])
    for score_key in ("score_low_iou", "score_low_stability", "score_combined"):
        bucket = summary.setdefault(score_key, {})
        ordered = sorted(rows, key=lambda row: row[score_key], reverse=True)
        for budget in budgets:
            chosen = ordered[:budget]
            stats = bucket.setdefault(
                str(budget),
                {"hits": 0, "chosen": 0, "targets": 0, "images_with_targets": 0},
            )
            hits = sum(1 for row in chosen if row["target"])
            stats["hits"] += int(hits)
            stats["chosen"] += int(len(chosen))
            stats["targets"] += int(target_count)
            if target_count > 0:
                stats["images_with_targets"] += 1


def _finalize_proxy_summary(summary: dict) -> dict:
    out = {}
    for score_key, budget_stats in summary.items():
        out[score_key] = {}
        for budget, stats in budget_stats.items():
            chosen = max(1, int(stats["chosen"]))
            targets = max(1, int(stats["targets"]))
            item = dict(stats)
            item["precision"] = float(stats["hits"] / chosen)
            item["recall"] = float(stats["hits"] / targets)
            out[score_key][budget] = item
    return out


def audit_image(
    artifact_dir: Path,
    image_root: Path,
    name: str,
    args: argparse.Namespace,
) -> tuple[dict, list[dict]]:
    gt = np.load(artifact_dir / f"{name}_gt.npy").astype(np.int32)
    pred = np.load(artifact_dir / f"{name}_pred.npy").astype(np.int32)
    image = imread(_find_image(image_root, name))[..., :3]
    evidence = _compute_h_evidence(image, sigma=args.stain_sigma)
    meta = _load_meta(artifact_dir, name)

    problem = _problem_sets(
        gt,
        pred,
        match_iou=args.match_iou,
        near_low=args.near_low,
        weak_high=args.weak_high,
        overlap_frac=args.overlap_frac,
    )
    coverage = _residual_peak_audit(
        gt,
        pred,
        evidence,
        problem,
        dilate_radius=args.coverage_dilate_radius,
        top_k=args.coverage_top_k,
        min_distance=args.coverage_min_distance,
        gt_radius=args.gt_match_radius,
    )
    merge = _merge_peak_audit(
        pred,
        evidence,
        problem,
        min_distance=args.merge_min_distance,
        max_peaks=args.merge_num_peaks,
    )
    risk_rows = _risk_proxy_rows(meta, problem)
    row = {
        "image": name,
        "gt_count": len(problem["gt_ids"]),
        "pred_count": len(problem["pred_ids"]),
        "fn": len(problem["unpaired_true"]),
        "fp": len(problem["unpaired_pred"]),
        "near_fn": len(problem["near_fn"]),
        "missed_fn": len(problem["missed_fn"]),
        "weak_pred": len(problem["weak_pred"]),
        "split_like_fn": len(problem["split_like_fn"]),
        "merge_like_pred": len(problem["merge_like_pred"]),
        **coverage,
        **merge,
        "risk_rows": len(risk_rows),
        "risk_targets": sum(1 for item in risk_rows if item["target"]),
    }
    return row, risk_rows


def summarize(rows: list[dict], proxy_summary: dict) -> dict:
    total_keys = [
        "gt_count",
        "pred_count",
        "fn",
        "fp",
        "near_fn",
        "missed_fn",
        "weak_pred",
        "split_like_fn",
        "merge_like_pred",
        "coverage_peak_count",
        "coverage_peak_positive_gt",
        "coverage_hit_fn",
        "coverage_hit_near_fn",
        "coverage_hit_missed_fn",
        "multi_peak_pred_candidates",
        "multi_peak_hit_merge_like",
        "risk_rows",
        "risk_targets",
    ]
    totals = {key: int(sum(int(row[key]) for row in rows)) for key in total_keys}

    def ratio(num_key: str, den_key: str):
        den = totals[den_key]
        return float(totals[num_key] / den) if den > 0 else None

    return {
        "num_images": len(rows),
        "totals": totals,
        "coverage_recall_fn": ratio("coverage_hit_fn", "fn"),
        "coverage_recall_near_fn": ratio("coverage_hit_near_fn", "near_fn"),
        "coverage_recall_missed_fn": ratio("coverage_hit_missed_fn", "missed_fn"),
        "merge_peak_recall": ratio("multi_peak_hit_merge_like", "merge_like_pred"),
        "proxy_topk": _finalize_proxy_summary(proxy_summary),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts_dir", required=True, type=Path)
    parser.add_argument("--data_path", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=["test", "train"])
    parser.add_argument("--out_prefix", default="", type=str)
    parser.add_argument("--match_iou", default=0.5, type=float)
    parser.add_argument("--near_low", default=0.3, type=float)
    parser.add_argument("--weak_high", default=0.6, type=float)
    parser.add_argument("--overlap_frac", default=0.1, type=float)
    parser.add_argument("--stain_sigma", default=1.0, type=float)
    parser.add_argument("--coverage_dilate_radius", default=5, type=int)
    parser.add_argument("--coverage_top_k", default=20, type=int)
    parser.add_argument("--coverage_min_distance", default=12, type=int)
    parser.add_argument("--gt_match_radius", default=8, type=int)
    parser.add_argument("--merge_min_distance", default=6, type=int)
    parser.add_argument("--merge_num_peaks", default=3, type=int)
    parser.add_argument("--budgets", nargs="+", default=[2, 4, 8, 12], type=int)
    args = parser.parse_args()

    image_root = args.data_path / ("test/images" if args.split == "test" else "train_12/images")
    pred_files = sorted(args.artifacts_dir.glob("*_pred.npy"))
    if not pred_files:
        raise FileNotFoundError(f"No *_pred.npy files found in {args.artifacts_dir}")

    rows = []
    proxy_summary: dict = {}
    for pred_path in pred_files:
        name = pred_path.name[: -len("_pred.npy")]
        row, risk_rows = audit_image(
            args.artifacts_dir,
            image_root,
            name,
            args,
        )
        rows.append(row)
        _update_proxy_summary(proxy_summary, risk_rows, args.budgets)

    out_prefix = Path(args.out_prefix) if args.out_prefix else args.artifacts_dir / "stage1a_candidate_audit"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    json_path = out_prefix.with_suffix(".json")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows, proxy_summary)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Wrote per-image rows: {csv_path}")
    print(f"Wrote summary: {json_path}")


if __name__ == "__main__":
    main()
