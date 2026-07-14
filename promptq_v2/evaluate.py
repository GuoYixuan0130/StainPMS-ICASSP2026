"""Offline score-only counterfactual assembly from a frozen deployment cache."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.stats import spearmanr
import torch

from .cache import _inside
from .data import load_label
from .protocol import INCLUSIVE_IOU_THRESHOLD, NMS_RADIUS, json_dump, paired_bootstrap, point_nms_indices, product_score, quality_only_score, verdict


Mode = Literal["baseline", "product", "quality_only", "oracle"]


def _metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    from sam2_train.modeling.stats_utils import get_dice_1, get_fast_aji, get_fast_aji_plus, get_fast_dice_2, get_fast_pq, remap_label

    gt = remap_label(gt)
    pred = remap_label(pred)
    pq_values, pairing = get_fast_pq(gt, pred, match_iou=INCLUSIVE_IOU_THRESHOLD)
    return {
        "dice": float(get_dice_1(gt, pred)), "dice2": float(get_fast_dice_2(gt, pred)),
        "aji": float(get_fast_aji(gt, pred)), "aji_p": float(get_fast_aji_plus(gt, pred)),
        "dq": float(pq_values[0]), "sq": float(pq_values[1]), "pq": float(pq_values[2]),
        "tp": int(len(pairing[0])), "fp": int(len(pairing[3])), "fn": int(len(pairing[2])),
    }


def _assembly_score(mask: dict, crop_box: np.ndarray, shape: tuple[int, int]) -> tuple[float, bool]:
    # Retain canonical validation's edge penalty and coordinate ordering exactly.
    bx1, by1, bx2, by2 = mask["bbox"]
    sx1, sy1, sx2, sy2 = crop_box.tolist()
    ori_h, ori_w = shape
    edge = (
        (bx1 > 7 and abs(bx1 - sx1) <= 7)
        or (abs(bx2 - ori_h) > 7 and abs(bx2 - sx2) <= 7)
        or (by1 > 7 and abs(by1 - sy1) <= 7)
        or (abs(by2 - ori_w) > 7 and abs(by2 - sy2) <= 7)
    )
    return float(mask["predicted_iou"] * (0.3 if edge else 1.0)), bool(edge)


def _scores(mode: Mode, objectness: np.ndarray, quality_logits: np.ndarray | None, oracle_iou: np.ndarray | None) -> np.ndarray:
    if mode == "baseline":
        return objectness.astype(np.float64)
    if mode == "product":
        if quality_logits is None:
            raise ValueError("product mode needs trained quality logits")
        return product_score(objectness, quality_logits)
    if mode == "quality_only":
        if quality_logits is None:
            raise ValueError("quality-only diagnostic needs trained quality logits")
        return quality_only_score(quality_logits)
    if mode == "oracle":
        if oracle_iou is None:
            raise ValueError("oracle mode needs separate GT label store")
        return oracle_iou.astype(np.float64)
    raise ValueError(mode)


def _reassemble(cache_path: Path, scores: np.ndarray) -> tuple[np.ndarray, dict]:
    from .assembly import assemble_instance_map, mask_process_eval

    with np.load(cache_path, allow_pickle=False) as cache:
        source_crop = np.asarray(cache["source_crop_id"], dtype=np.int64)
        points = np.asarray(cache["global_point"], dtype=np.float32)
        classes = np.asarray(cache["class_id"], dtype=np.int64)
        groups = np.asarray(cache["point_group"], dtype=np.int64)
        shape = tuple(int(value) for value in cache["image_shape"].tolist())
        all_masks, all_boxes, all_scores, all_inds, all_records = [], [], [], [], []
        trace: list[dict] = []
        for crop_id in range(int(cache["crop_count"])):
            seen = np.flatnonzero(source_crop <= crop_id)
            if not len(seen):
                trace.append({"crop_id": crop_id, "winner_source_ids": []})
                continue
            selected = seen[point_nms_indices(points[seen], scores[seen], NMS_RADIUS)]
            box = np.asarray(cache[f"crop_{crop_id:03d}_box"], dtype=np.int64)
            selected = selected[_inside(points[selected], tuple(int(value) for value in box.tolist()))]
            decode_ids = np.asarray(cache[f"crop_{crop_id:03d}_decode_source_ids"], dtype=np.int64)
            lookup = {int(source): index for index, source in enumerate(decode_ids.tolist())}
            if any(int(source) not in lookup for source in selected.tolist()):
                raise RuntimeError("offline path requested a candidate absent from frozen cache")
            logits = np.asarray(cache[f"crop_{crop_id:03d}_decoded_logits"], dtype=np.float32)
            iou = np.asarray(cache[f"crop_{crop_id:03d}_decoded_iou"], dtype=np.float32)
            decoded_index = np.asarray([lookup[int(source)] for source in selected], dtype=np.int64)
            local_points = points[selected] - box[:2].astype(np.float32)
            masks = mask_process_eval(
                classes[selected], torch.as_tensor(groups[selected]), box.tolist(), shape,
                torch.as_tensor(local_points, dtype=torch.float32), torch.as_tensor(logits[decoded_index]), torch.as_tensor(iou[decoded_index]),
            )
            for mask in masks:
                score, edge = _assembly_score(mask, box, shape)
                all_scores.append(score)
                all_masks.append(mask["segmentation"][:shape[0], :shape[1]])
                all_boxes.append(mask["bbox"])
                all_inds.append(mask["inds"])
                all_records.append({"crop_id": crop_id, "edge_penalized": edge, "source_group": int(mask["inds"])})
            trace.append({"crop_id": crop_id, "winner_source_ids": [int(value) for value in selected.tolist()]})
    prediction, selected_records = assemble_instance_map(all_boxes, all_scores, all_masks, all_inds, shape, INCLUSIVE_IOU_THRESHOLD, all_records=all_records, return_records=True)
    return prediction.astype(np.int32), {"nms": trace, "assembly_candidates": len(all_masks), "assembly_selected": len(selected_records), "selected_records": selected_records}


def _aggregate(rows: list[dict]) -> dict:
    metrics = ("dice", "dice2", "aji", "aji_p", "dq", "sq", "pq")
    result = {metric: float(np.mean([row[metric] for row in rows])) for metric in metrics}
    result.update({name: int(sum(int(row[name]) for row in rows)) for name in ("tp", "fp", "fn")})
    result["image_count"] = len(rows)
    return result


def _rank_diagnostics(objectness: np.ndarray, quality_logits: np.ndarray, oracle_iou: np.ndarray, point_groups: np.ndarray, base: dict, product: dict) -> dict:
    rank_base = np.argsort(np.argsort(-objectness, kind="mergesort"), kind="mergesort")
    rank_product = np.argsort(np.argsort(-product_score(objectness, quality_logits), kind="mergesort"), kind="mergesort")
    base_winners = {source for crop in base["nms"] for source in crop["winner_source_ids"]}
    product_winners = {source for crop in product["nms"] for source in crop["winner_source_ids"]}
    changed = sorted(base_winners.symmetric_difference(product_winners))
    changed_group_count = improved = lowered = unchanged = 0
    for group in np.unique(point_groups):
        members = set(np.flatnonzero(point_groups == group).tolist())
        before = sorted(members & base_winners)
        after = sorted(members & product_winners)
        if before == after or not before or not after:
            continue
        changed_group_count += 1
        delta = max(float(oracle_iou[item]) for item in after) - max(float(oracle_iou[item]) for item in before)
        if delta > 0:
            improved += 1
        elif delta < 0:
            lowered += 1
        else:
            unchanged += 1
    return {
        "rank_changed_candidate_count": int((rank_base != rank_product).sum()),
        "nms_winner_changed_candidate_count": len(changed),
        "nms_winner_added_by_product": len(product_winners - base_winners),
        "nms_winner_removed_by_product": len(base_winners - product_winners),
        "changed_winner_oracle_iou_mean": float(np.mean(oracle_iou[changed])) if changed else None,
        "nms_changed_comparable_groups": changed_group_count,
        "nms_changed_iou_improved_fraction": float(improved / changed_group_count) if changed_group_count else None,
        "nms_changed_iou_lowered_fraction": float(lowered / changed_group_count) if changed_group_count else None,
        "nms_changed_iou_unchanged_fraction": float(unchanged / changed_group_count) if changed_group_count else None,
    }


def _mechanism(objectness: np.ndarray, quality_logits: np.ndarray, target: np.ndarray, matched: np.ndarray, oracle_iou: np.ndarray) -> dict:
    quality = quality_only_score(quality_logits)
    product = product_score(objectness, quality_logits)
    observed = matched.astype(bool)
    def corr(left, right):
        if len(left) < 2 or np.all(left == left[0]) or np.all(right == right[0]):
            return None
        value = float(spearmanr(left, right).statistic)
        return value if np.isfinite(value) else None
    # ECE for decoded-IoU>=.5 is diagnostic only.
    bins = np.array_split(np.argsort(quality), 10)
    ece = sum((len(index) / len(quality)) * abs(float(quality[index].mean()) - float((oracle_iou[index] >= .5).mean())) for index in bins if len(index))
    return {
        "quality_target_spearman": corr(quality[observed], target[observed]),
        "raw_objectness_iou_spearman": corr(objectness[observed], oracle_iou[observed]),
        "product_iou_spearman": corr(product[observed], oracle_iou[observed]),
        "quality_ece_iou_ge_0_5": float(ece),
    }


def evaluate_development(cache_dir: Path, target_dir: Path, label_dir: Path, quality_logits_by_image: dict[str, np.ndarray], out_dir: Path) -> dict:
    """Run all four counterfactuals offline without model/decoder invocation."""
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite offline evaluation: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    for record in manifest["records"]:
        image_id = record["image_id"]
        with np.load(cache_dir / record["file"], allow_pickle=False) as cache, np.load(target_dir / f"{image_id}.npz", allow_pickle=False) as targets:
            objectness = np.asarray(cache["objectness"], dtype=np.float32)
            point_groups = np.asarray(cache["point_group"], dtype=np.int64)
            quality_logits = np.asarray(quality_logits_by_image[image_id], dtype=np.float32)
            oracle_iou = np.asarray(targets["oracle_iou"], dtype=np.float32)
            target = np.asarray(targets["utility_target"], dtype=np.float32)
            matched = np.asarray(targets["matched"], dtype=np.bool_)
        paths = {}
        predictions = {}
        for mode in ("baseline", "product", "quality_only", "oracle"):
            predictions[mode], paths[mode] = _reassemble(cache_dir / record["file"], _scores(mode, objectness, quality_logits, oracle_iou))
        baseline_replay, _ = _reassemble(cache_dir / record["file"], _scores("baseline", objectness, quality_logits, oracle_iou))
        if not np.array_equal(predictions["baseline"], baseline_replay):
            raise RuntimeError("baseline cache replay was not pixel-identical")
        gt = load_label(label_dir, image_id)
        metrics = {mode: _metrics(gt, prediction) for mode, prediction in predictions.items()}
        row = {"image_id": image_id, "patient": int(record["patient"])}
        for mode, value in metrics.items():
            row.update({f"{mode}_{key}": item for key, item in value.items()})
        row.update({
            "promptq_minus_baseline_aji": float(metrics["product"]["aji"] - metrics["baseline"]["aji"]),
            "promptq_minus_baseline_pq": float(metrics["product"]["pq"] - metrics["baseline"]["pq"]),
            "promptq_minus_baseline_dq": float(metrics["product"]["dq"] - metrics["baseline"]["dq"]),
            "unmatched_fn_change": int(metrics["product"]["fn"] - metrics["baseline"]["fn"]),
            "fp_change": int(metrics["product"]["fp"] - metrics["baseline"]["fp"]),
        })
        row.update(_rank_diagnostics(objectness, quality_logits, oracle_iou, point_groups, paths["baseline"], paths["product"]))
        rows.append(row)
        diagnostic_rows.append({"image_id": image_id, **_mechanism(objectness, quality_logits, target, matched, oracle_iou)})
        np.save(out_dir / f"{image_id}_baseline.npy", predictions["baseline"])
        np.save(out_dir / f"{image_id}_promptq_v2.npy", predictions["product"])
    fieldnames = sorted({key for row in rows for key in row})
    with (out_dir / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    with (out_dir / "mechanism_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in diagnostic_rows for key in row}))
        writer.writeheader(); writer.writerows(diagnostic_rows)
    path_rows = {mode: [] for mode in ("baseline", "product", "quality_only", "oracle")}
    for row in rows:
        for mode in path_rows:
            path_rows[mode].append({key: row[f"{mode}_{key}"] for key in ("dice", "dice2", "aji", "aji_p", "dq", "sq", "pq", "tp", "fp", "fn")})
    patient_report = {}
    for patient in (7, 8):
        patient_rows = [row for row in rows if int(row["patient"]) == patient]
        patient_report[str(patient)] = {mode: _aggregate([{key: row[f"{mode}_{key}"] for key in ("dice", "dice2", "aji", "aji_p", "dq", "sq", "pq", "tp", "fp", "fn")} for row in patient_rows]) for mode in path_rows}
    paired_rows = [{f"baseline_{metric}": row[f"baseline_{metric}"], f"promptq_{metric}": row[f"product_{metric}"]} for row in rows for metric in ()]
    bootstrap = {}
    for metric in ("aji", "pq", "dq"):
        bootstrap[metric] = paired_bootstrap([{f"baseline_{metric}": row[f"baseline_{metric}"], f"promptq_{metric}": row[f"product_{metric}"]} for row in rows], metric)
    summary = {mode: _aggregate(values) for mode, values in path_rows.items()}
    deltas = {metric: float(summary["product"][metric] - summary["baseline"][metric]) for metric in ("aji", "pq", "dq")}
    patient_deltas = [(float(patient_report[str(patient)]["product"]["aji"] - patient_report[str(patient)]["baseline"]["aji"]), float(patient_report[str(patient)]["product"]["pq"] - patient_report[str(patient)]["baseline"]["pq"])) for patient in (7, 8)]
    report = {
        "baseline_cache_reassembly": {"pixel_identical": True, "primary_metric_error": 0.0, "verification": "canonical candidate filtering, NMS, mask_process_eval, and _assemble_instance_map were replayed twice from immutable cache"},
        "metrics": summary, "patients": patient_report, "paired_delta": deltas, "paired_bootstrap_95ci": bootstrap,
        "oracle_minus_baseline": {metric: float(summary["oracle"][metric] - summary["baseline"][metric]) for metric in ("aji", "pq", "dq")},
        "verdict": verdict(deltas["aji"], deltas["pq"], patient_deltas),
        "notes": {"spearman_ece_quality_loss": "mechanism-only; not a GO/NO-GO gate", "oracle": "GT-IoU score-only diagnostic; never used for training or deployment", "inclusive_iou": ">= 0.5"},
    }
    json_dump(out_dir / "report.json", report)
    return report
