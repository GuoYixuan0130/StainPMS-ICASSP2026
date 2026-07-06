"""Analyze StainPQR oracle action labels.

Reads one or more Stage 1B `actions.csv` files and evaluates simple action
ranking rules. This is the bridge between decoder oracle labels and the next
utility/risk selector: if simple rules already rank useful actions well, use
them as baselines; otherwise train a calibrated selector.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def _to_float(value, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _to_int(value, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def _load_actions(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            lines = [line for line in f if line.strip()]
            reader = csv.DictReader(lines)
            for row in reader:
                item = dict(row)
                item["source_csv"] = str(path)
                item["action_rank"] = _to_int(row.get("action_rank"))
                item["x"] = _to_int(row.get("x"))
                item["y"] = _to_int(row.get("y"))
                for key in (
                    "evidence",
                    "residual_evidence",
                    "decoded_predicted_iou",
                    "decoded_stability_score",
                    "decoded_area",
                    "added_area",
                    "base_pq",
                    "next_pq",
                    "delta_pq",
                    "delta_dq",
                    "delta_sq",
                    "delta_aji",
                ):
                    item[key] = _to_float(row.get(key))
                item["positive"] = bool(item["delta_pq"] > 0)
                item["harmful"] = bool(item["delta_pq"] < 0)
                rows.append(item)
    return rows


def _group_by_image(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        key = f"{row.get('source_csv')}::{row.get('image')}"
        groups.setdefault(key, []).append(row)
    return groups


def _score(row: dict, method: str) -> float:
    if method == "oracle_delta":
        return row["delta_pq"]
    if method == "target_oracle":
        return 1.0 if row.get("target_error") in ("missed_fn", "near_fn", "fn") else 0.0
    if method == "missed_fn_oracle":
        return 1.0 if row.get("target_error") == "missed_fn" else 0.0
    if method == "rank_first":
        return -float(row["action_rank"])
    if method == "residual_evidence":
        return row["residual_evidence"]
    if method == "evidence":
        return row["evidence"]
    if method == "decoded_iou_high":
        return row["decoded_predicted_iou"]
    if method == "decoded_iou_low":
        return -row["decoded_predicted_iou"]
    if method == "stability_high":
        return row["decoded_stability_score"]
    if method == "stability_low":
        return -row["decoded_stability_score"]
    if method == "added_area":
        return row["added_area"]
    if method == "decoded_area":
        return row["decoded_area"]
    if method == "small_added_area":
        return -row["added_area"]
    if method == "missed_like_proxy":
        # Large new foreground with good decoder confidence and high residual stain.
        return (
            row["residual_evidence"]
            + 0.25 * row["decoded_predicted_iou"]
            + 0.25 * row["decoded_stability_score"]
            + 0.0005 * row["added_area"]
        )
    raise ValueError(f"Unknown method: {method}")


def _safe_auc(labels: np.ndarray, scores: np.ndarray):
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average tied ranks.
    unique_scores, inverse = np.unique(scores, return_inverse=True)
    for idx in range(len(unique_scores)):
        mask = inverse == idx
        if int(mask.sum()) > 1:
            ranks[mask] = float(ranks[mask].mean())
    rank_sum_pos = float(ranks[labels].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _safe_ap(labels: np.ndarray, scores: np.ndarray):
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return None
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    precision = tp / (np.arange(len(labels)) + 1)
    return float((precision * sorted_labels).sum() / n_pos)


def _feature_metrics(rows: list[dict], methods: list[str]) -> dict:
    labels = np.asarray([row["positive"] for row in rows], dtype=bool)
    out = {}
    for method in methods:
        scores = np.asarray([_score(row, method) for row in rows], dtype=np.float64)
        finite = np.isfinite(scores)
        if int(finite.sum()) == 0:
            out[method] = {"auroc": None, "ap": None}
            continue
        out[method] = {
            "auroc": _safe_auc(labels[finite], scores[finite]),
            "ap": _safe_ap(labels[finite], scores[finite]),
        }
    return out


def _budget_metrics(groups: dict[str, list[dict]], methods: list[str], budgets: list[int]) -> dict:
    total_positive = sum(1 for rows in groups.values() for row in rows if row["positive"])
    out = {}
    for method in methods:
        out[method] = {}
        for budget in budgets:
            chosen = []
            images_with_positive = 0
            image_delta_sums = []
            for rows in groups.values():
                if any(row["positive"] for row in rows):
                    images_with_positive += 1
                ordered = sorted(rows, key=lambda row: _score(row, method), reverse=True)
                selected = ordered[:budget]
                chosen.extend(selected)
                image_delta_sums.append(float(sum(row["delta_pq"] for row in selected)))

            hits = sum(1 for row in chosen if row["positive"])
            harmful = sum(1 for row in chosen if row["harmful"])
            delta_sum = float(sum(row["delta_pq"] for row in chosen))
            positive_delta_sum = float(sum(max(0.0, row["delta_pq"]) for row in chosen))
            harmful_delta_sum = float(sum(min(0.0, row["delta_pq"]) for row in chosen))
            out[method][str(budget)] = {
                "chosen": int(len(chosen)),
                "hits": int(hits),
                "harmful": int(harmful),
                "precision": float(hits / max(1, len(chosen))),
                "positive_recall": float(hits / max(1, total_positive)),
                "delta_pq_sum": delta_sum,
                "positive_delta_pq_sum": positive_delta_sum,
                "harmful_delta_pq_sum": harmful_delta_sum,
                "mean_selected_delta_pq": float(delta_sum / max(1, len(chosen))),
                "images_improved_individual_sum": int(sum(1 for value in image_delta_sums if value > 0)),
                "images_with_positive": int(images_with_positive),
            }
    return out


def _target_summary(rows: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for row in rows:
        key = str(row.get("target_error", "unknown"))
        item = out.setdefault(
            key,
            {
                "count": 0,
                "positive": 0,
                "harmful": 0,
                "delta_pq_sum": 0.0,
                "positive_delta_pq_sum": 0.0,
                "harmful_delta_pq_sum": 0.0,
            },
        )
        item["count"] += 1
        item["positive"] += int(row["positive"])
        item["harmful"] += int(row["harmful"])
        item["delta_pq_sum"] += float(row["delta_pq"])
        item["positive_delta_pq_sum"] += max(0.0, float(row["delta_pq"]))
        item["harmful_delta_pq_sum"] += min(0.0, float(row["delta_pq"]))
    for item in out.values():
        item["positive_rate"] = float(item["positive"] / max(1, item["count"]))
        item["harmful_rate"] = float(item["harmful"] / max(1, item["count"]))
        item["mean_delta_pq"] = float(item["delta_pq_sum"] / max(1, item["count"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions_csv", nargs="+", required=True, type=Path)
    parser.add_argument("--out_prefix", default="", type=str)
    parser.add_argument("--budgets", nargs="+", default=[1, 2, 4, 8, 12, 20], type=int)
    args = parser.parse_args()

    rows = _load_actions(args.actions_csv)
    if not rows:
        raise ValueError("No oracle action rows loaded")
    groups = _group_by_image(rows)
    methods = [
        "oracle_delta",
        "target_oracle",
        "missed_fn_oracle",
        "rank_first",
        "residual_evidence",
        "evidence",
        "decoded_iou_high",
        "decoded_iou_low",
        "stability_high",
        "stability_low",
        "added_area",
        "decoded_area",
        "small_added_area",
        "missed_like_proxy",
    ]
    feature_methods = [m for m in methods if not m.endswith("_oracle") and m != "target_oracle"]

    summary = {
        "num_actions": len(rows),
        "num_images": len(groups),
        "positive_actions": int(sum(1 for row in rows if row["positive"])),
        "harmful_actions": int(sum(1 for row in rows if row["harmful"])),
        "positive_rate": float(sum(1 for row in rows if row["positive"]) / len(rows)),
        "harmful_rate": float(sum(1 for row in rows if row["harmful"]) / len(rows)),
        "target_summary": _target_summary(rows),
        "feature_metrics": _feature_metrics(rows, feature_methods),
        "budget_metrics": _budget_metrics(groups, methods, args.budgets),
    }

    out_prefix = Path(args.out_prefix) if args.out_prefix else args.actions_csv[0].parent / "oracle_action_analysis"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_json = out_prefix.with_suffix(".json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Wrote summary: {out_json}")


if __name__ == "__main__":
    main()
