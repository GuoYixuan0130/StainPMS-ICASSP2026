"""Coverage-action decoder oracle for StainPQR Stage 1B."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from skimage.color import rgb2hed
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from skimage.io import imread
from skimage.morphology import binary_dilation, disk
from tqdm import tqdm

from run.run_on_epoch import inference, mask_process_eval
from tools.analyze_eval_artifacts import (
    _ids,
    _pairwise_stats,
    analyze_pair,
    get_fast_aji,
    get_fast_pq,
    remap_label,
    summarize,
)


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _safe_name(name) -> str:
    if isinstance(name, (list, tuple)):
        return "_".join(str(item) for item in name)
    return str(name)


def _ori_hw(ori_shape) -> tuple[int, int]:
    if torch.is_tensor(ori_shape):
        arr = ori_shape.detach().cpu().numpy()
    else:
        arr = np.asarray(ori_shape)
    arr = arr.reshape(-1)
    return int(arr[0]), int(arr[1])


def _find_image(image_root: Path, name: str) -> Path:
    for ext in IMAGE_EXTS:
        path = image_root / f"{name}{ext}"
        if path.exists() and path.is_file():
            return path
    matches = [p for p in image_root.glob(f"{name}.*") if p.is_file()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No image found for {name} under {image_root}")
    raise RuntimeError(f"Ambiguous image files for {name}: {matches}")


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


def _topk_peaks(score_map: np.ndarray, top_k: int, min_distance: int) -> np.ndarray:
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


def _assign_gt_id(gt: np.ndarray, y: int, x: int, radius: int) -> int:
    h, w = gt.shape
    if gt[y, x] > 0:
        return int(gt[y, x])
    if radius <= 0:
        return 0
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    window = gt[y0:y1, x0:x1]
    ys, xs = np.where(window > 0)
    if len(ys) == 0:
        return 0
    d2 = (ys - (y - y0)) ** 2 + (xs - (x - x0)) ** 2
    nearest = int(np.argmin(d2))
    return int(window[ys[nearest], xs[nearest]])


def _problem_sets(gt: np.ndarray, pred: np.ndarray) -> dict:
    gt_ids, pred_ids, _, _, _, iou = _pairwise_stats(gt, pred)
    [_, _, _], pair_info = get_fast_pq(gt, pred, match_iou=0.5)
    paired_true, paired_pred, unpaired_true, unpaired_pred = pair_info
    gt_to_idx = {tid: idx for idx, tid in enumerate(gt_ids)}
    best_iou_true = iou.max(axis=1) if iou.shape[1] > 0 else np.zeros(len(gt_ids), dtype=np.float64)

    near_fn: set[int] = set()
    missed_fn: set[int] = set()
    for tid in unpaired_true:
        idx = gt_to_idx.get(int(tid))
        if idx is None:
            continue
        best = float(best_iou_true[idx])
        if 0.3 <= best <= 0.5:
            near_fn.add(int(tid))
        if best < 0.3:
            missed_fn.add(int(tid))
    return {
        "paired_true": set(int(v) for v in paired_true),
        "paired_pred": set(int(v) for v in paired_pred),
        "unpaired_true": set(int(v) for v in unpaired_true),
        "unpaired_pred": set(int(v) for v in unpaired_pred),
        "near_fn": near_fn,
        "missed_fn": missed_fn,
    }


def _coverage_candidates(
    image: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    problem: dict,
    cfgs,
) -> list[dict]:
    evidence = _compute_h_evidence(image, sigma=1.0)
    pred_bin = pred > 0
    if int(cfgs.oracle_coverage_dilate_radius) > 0:
        pred_bin = binary_dilation(
            pred_bin,
            footprint=disk(int(cfgs.oracle_coverage_dilate_radius)),
        )
    residual = evidence.copy()
    residual[pred_bin] = 0.0
    peaks_yx = _topk_peaks(
        residual,
        top_k=int(cfgs.oracle_coverage_top_k),
        min_distance=int(cfgs.oracle_coverage_min_distance),
    )

    rows = []
    for rank, (y, x) in enumerate(peaks_yx):
        y = int(y)
        x = int(x)
        gt_id = _assign_gt_id(gt, y, x, int(cfgs.oracle_gt_match_radius))
        if gt_id in problem["near_fn"]:
            target = "near_fn"
        elif gt_id in problem["missed_fn"]:
            target = "missed_fn"
        elif gt_id in problem["unpaired_true"]:
            target = "fn"
        elif gt_id > 0:
            target = "covered_gt"
        else:
            target = "background"
        rows.append(
            {
                "action_rank": int(rank),
                "type": "coverage",
                "x": x,
                "y": y,
                "evidence": float(evidence[y, x]),
                "residual_evidence": float(residual[y, x]),
                "target_gt_id": int(gt_id),
                "target_error": target,
            }
        )
    return rows


def _crop_box_around_point(x: int, y: int, h: int, w: int, crop_size: int) -> list[int]:
    crop_w = min(int(crop_size), w)
    crop_h = min(int(crop_size), h)
    x1 = int(round(x - crop_w / 2))
    y1 = int(round(y - crop_h / 2))
    x1 = max(0, min(w - crop_w, x1))
    y1 = max(0, min(h - crop_h, y1))
    return [x1, y1, x1 + crop_w, y1 + crop_h]


def _metric_row(gt: np.ndarray, pred: np.ndarray) -> dict:
    gt_r = remap_label(gt.astype(np.int32))
    pred_r = remap_label(pred.astype(np.int32))
    [dq, sq, pq], _ = get_fast_pq(gt_r, pred_r, match_iou=0.5)
    return {
        "dq": float(dq),
        "sq": float(sq),
        "pq": float(pq),
        "aji": float(get_fast_aji(gt_r, pred_r)),
    }


def _apply_insert(pred: np.ndarray, candidate_mask: np.ndarray, min_added_area: int) -> tuple[np.ndarray, int]:
    candidate_mask = np.asarray(candidate_mask).astype(bool)
    add_region = candidate_mask & (pred == 0)
    added_area = int(add_region.sum())
    if added_area < int(min_added_area):
        return pred.copy(), added_area
    out = pred.copy()
    out[add_region] = int(out.max()) + 1
    return out, added_area


@torch.no_grad()
def _decode_action_mask(
    images_seg: torch.Tensor,
    action: dict,
    ori_shape,
    cfgs,
    net,
    point_encoder,
    memory_bank_list,
    device,
) -> dict | None:
    h, w = _ori_hw(ori_shape)
    crop_box = _crop_box_around_point(
        int(action["x"]),
        int(action["y"]),
        h,
        w,
        int(cfgs.crop_size),
    )
    x1, y1, x2, y2 = crop_box
    img = images_seg[..., y1:y2, x1:x2].to(device)
    sub_point = torch.tensor(
        [[[float(action["x"] - x1), float(action["y"] - y1)]]],
        dtype=torch.float32,
        device=device,
    )
    sub_label = torch.ones((1, 1), dtype=torch.int, device=device)

    pred, values, _, _, _ = inference(
        net,
        point_encoder,
        img,
        memory_bank_list,
        sub_point,
        sub_label,
        [(64, 64), (32, 32), (16, 16)],
        [],
        x1,
        y1,
        False,
        cfgs,
        device,
    )
    masks = mask_process_eval(
        np.ones(1, dtype=np.int64),
        torch.tensor([int(action["action_rank"])], dtype=torch.long, device=device),
        crop_box,
        ori_shape,
        sub_point,
        pred,
        values,
    )
    if not masks:
        return None
    mask = masks[0]["segmentation"][:h, :w]
    return {
        "mask": mask,
        "bbox": masks[0]["bbox"],
        "predicted_iou": float(masks[0]["predicted_iou"]),
        "stability_score": float(masks[0]["stability_score"]),
        "crop_box": crop_box,
    }


def _summarize(actions: list[dict], images: list[dict]) -> dict:
    positives = [a for a in actions if a["delta_pq"] > 0]
    harmful = [a for a in actions if a["delta_pq"] < 0]
    by_target: dict[str, dict] = {}
    for action in actions:
        item = by_target.setdefault(
            action["target_error"],
            {"count": 0, "positive": 0, "mean_delta_pq": 0.0, "oracle_delta_pq_sum": 0.0},
        )
        item["count"] += 1
        if action["delta_pq"] > 0:
            item["positive"] += 1
        item["mean_delta_pq"] += float(action["delta_pq"])
        item["oracle_delta_pq_sum"] += max(0.0, float(action["delta_pq"]))
    for item in by_target.values():
        item["mean_delta_pq"] = float(item["mean_delta_pq"] / max(1, item["count"]))
        item["positive_rate"] = float(item["positive"] / max(1, item["count"]))

    return {
        "num_images": len(images),
        "num_actions": len(actions),
        "positive_actions": len(positives),
        "harmful_actions": len(harmful),
        "positive_rate": float(len(positives) / max(1, len(actions))),
        "harmful_rate": float(len(harmful) / max(1, len(actions))),
        "mean_delta_pq": float(np.mean([a["delta_pq"] for a in actions])) if actions else None,
        "oracle_positive_delta_pq_sum": float(sum(max(0.0, a["delta_pq"]) for a in actions)),
        "by_target_error": by_target,
        "images": images,
    }


def run_coverage_oracle(
    cfgs,
    loader,
    net,
    point_encoder,
    memory_bank_list,
    device,
) -> dict:
    if not cfgs.oracle_artifacts_dir:
        raise ValueError("--oracle_artifacts_dir is required for --stage1_coverage_oracle")
    artifact_dir = Path(cfgs.oracle_artifacts_dir)
    out_dir = Path(cfgs.oracle_out_dir or artifact_dir / "stage1b_coverage_oracle")
    out_dir.mkdir(parents=True, exist_ok=True)
    image_root = Path(cfgs.data_path) / ("test/images" if cfgs.oracle_split == "test" else "train_12/images")

    net.eval()
    point_encoder.eval()
    actions_out: list[dict] = []
    image_rows: list[dict] = []
    max_images = int(getattr(cfgs, "oracle_max_images", 0) or 0)

    pbar = tqdm(total=len(loader), desc="Coverage oracle", unit="image")
    for image_idx, batch in enumerate(loader):
        if max_images > 0 and image_idx >= max_images:
            break
        img_seg, inst_maps, _, _, _, _, ori_shape, _, name = batch
        name_str = _safe_name(name)
        gt = np.asarray(inst_maps.numpy()[0]).astype(np.int32)
        pred_path = artifact_dir / f"{name_str}_pred.npy"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing Stage 0 pred artifact: {pred_path}")
        base_pred = np.load(pred_path).astype(np.int32)
        image = imread(_find_image(image_root, name_str))[..., :3]
        problem = _problem_sets(gt, base_pred)
        candidates = _coverage_candidates(image, gt, base_pred, problem, cfgs)
        base_metrics = _metric_row(gt, base_pred)
        images_seg = img_seg.to(device)
        image_positive = 0
        image_harmful = 0

        for action in candidates:
            decoded = _decode_action_mask(
                images_seg,
                action,
                ori_shape,
                cfgs,
                net,
                point_encoder,
                memory_bank_list,
                device,
            )
            if decoded is None:
                continue
            next_pred, added_area = _apply_insert(
                base_pred,
                decoded["mask"],
                int(cfgs.oracle_min_added_area),
            )
            next_metrics = _metric_row(gt, next_pred)
            delta_pq = float(next_metrics["pq"] - base_metrics["pq"])
            if delta_pq > 0:
                image_positive += 1
            if delta_pq < 0:
                image_harmful += 1
            actions_out.append(
                {
                    "image": name_str,
                    **action,
                    "decoded_predicted_iou": decoded["predicted_iou"],
                    "decoded_stability_score": decoded["stability_score"],
                    "decoded_bbox": decoded["bbox"],
                    "decoded_crop_box": decoded["crop_box"],
                    "decoded_area": int(np.asarray(decoded["mask"]).sum()),
                    "added_area": int(added_area),
                    "base_pq": base_metrics["pq"],
                    "next_pq": next_metrics["pq"],
                    "delta_pq": delta_pq,
                    "delta_dq": float(next_metrics["dq"] - base_metrics["dq"]),
                    "delta_sq": float(next_metrics["sq"] - base_metrics["sq"]),
                    "delta_aji": float(next_metrics["aji"] - base_metrics["aji"]),
                    "positive_utility": bool(delta_pq > 0),
                }
            )

        image_rows.append(
            {
                "image": name_str,
                "base_pq": base_metrics["pq"],
                "candidate_count": len(candidates),
                "decoded_count": sum(1 for a in actions_out if a["image"] == name_str),
                "positive_actions": image_positive,
                "harmful_actions": image_harmful,
                "fn": len(problem["unpaired_true"]),
                "near_fn": len(problem["near_fn"]),
                "missed_fn": len(problem["missed_fn"]),
            }
        )
        pbar.update()
    pbar.close()

    actions_csv = out_dir / "actions.csv"
    images_csv = out_dir / "images.csv"
    summary_json = out_dir / "summary.json"
    if actions_out:
        with open(actions_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(actions_out[0].keys()))
            writer.writeheader()
            writer.writerows(actions_out)
    else:
        with open(actions_csv, "w", newline="", encoding="utf-8") as f:
            f.write("")

    with open(images_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(image_rows[0].keys()) if image_rows else ["image"])
        writer.writeheader()
        writer.writerows(image_rows)

    summary = _summarize(actions_out, image_rows)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Wrote actions: {actions_csv}")
    print(f"Wrote images: {images_csv}")
    print(f"Wrote summary: {summary_json}")
    return summary


def _csv_rows(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        lines = [line for line in f if line.strip()]
        return list(csv.DictReader(lines))


def _csv_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def _csv_int(row: dict, key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def _load_selective_actions(cfgs, prediction_rows: list[dict]) -> dict[tuple[str, int], dict]:
    action_paths = [Path(path) for path in getattr(cfgs, "selective_actions_csv", []) or []]
    if not action_paths:
        inferred = sorted({str(row.get("source_csv", "")) for row in prediction_rows if row.get("source_csv")})
        action_paths = [Path(path) for path in inferred]
    if not action_paths:
        raise ValueError("--selective_actions_csv is required when predictions do not contain source_csv")

    actions: dict[tuple[str, int], dict] = {}
    for path in action_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing selective action CSV: {path}")
        for row in _csv_rows(path):
            image = str(row.get("image", ""))
            rank = _csv_int(row, "action_rank")
            item = dict(row)
            item["image"] = image
            item["action_rank"] = rank
            item["x"] = _csv_int(row, "x")
            item["y"] = _csv_int(row, "y")
            actions[(image, rank)] = item
    return actions


def _load_selective_predictions(cfgs) -> list[dict]:
    path = Path(cfgs.selective_predictions_csv)
    if not path.exists():
        raise FileNotFoundError(f"Missing selective prediction CSV: {path}")
    score_name = str(cfgs.selective_score)
    rows = []
    for row in _csv_rows(path):
        score = _csv_float(row, score_name, default=-math.inf)
        if score < float(cfgs.selective_min_score):
            continue
        item = dict(row)
        item["image"] = str(row.get("image", ""))
        item["action_rank"] = _csv_int(row, "action_rank")
        item["_selective_score"] = score
        rows.append(item)
    return rows


def _group_selected_actions(cfgs) -> dict[str, list[dict]]:
    prediction_rows = _load_selective_predictions(cfgs)
    action_lookup = _load_selective_actions(cfgs, prediction_rows)
    grouped: dict[str, list[dict]] = {}
    for pred_row in prediction_rows:
        key = (str(pred_row["image"]), int(pred_row["action_rank"]))
        action = action_lookup.get(key)
        if action is None:
            continue
        item = dict(action)
        item["_selective_score"] = float(pred_row["_selective_score"])
        item["_selector_action_rank"] = int(pred_row["action_rank"])
        grouped.setdefault(str(item["image"]), []).append(item)

    budget = max(0, int(cfgs.selective_budget))
    for image, items in list(grouped.items()):
        ordered = sorted(items, key=lambda row: float(row["_selective_score"]), reverse=True)
        grouped[image] = ordered[:budget] if budget > 0 else []
    return grouped


def _summary_delta(base_summary: dict, refined_summary: dict) -> dict:
    metric_delta = {}
    for key, base_value in base_summary["mean_metrics"].items():
        metric_delta[key] = float(refined_summary["mean_metrics"][key] - base_value)
    total_delta = {}
    for key, base_value in base_summary["totals"].items():
        total_delta[key] = int(refined_summary["totals"][key] - base_value)
    return {"mean_metrics": metric_delta, "totals": total_delta}


def _analyze_pair_default(name: str, gt: np.ndarray, pred: np.ndarray) -> dict:
    return analyze_pair(
        name,
        gt,
        pred,
        match_iou=0.5,
        near_low=0.3,
        weak_high=0.6,
        overlap_frac=0.1,
    )


@torch.no_grad()
def run_selective_coverage_refinement(
    cfgs,
    loader,
    net,
    point_encoder,
    memory_bank_list,
    device,
) -> dict:
    if not cfgs.selective_artifacts_dir:
        raise ValueError("--selective_artifacts_dir is required for --stage2_selective_refine")
    if not cfgs.selective_predictions_csv:
        raise ValueError("--selective_predictions_csv is required for --stage2_selective_refine")

    artifact_dir = Path(cfgs.selective_artifacts_dir)
    out_dir = Path(cfgs.selective_out_dir or artifact_dir / "stage2c_selective_refine")
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped_actions = _group_selected_actions(cfgs)
    net.eval()
    point_encoder.eval()

    base_rows: list[dict] = []
    refined_rows: list[dict] = []
    image_rows: list[dict] = []
    selected_rows: list[dict] = []

    pbar = tqdm(total=len(loader), desc="Selective refinement", unit="image")
    for batch in loader:
        img_seg, inst_maps, _, _, _, _, ori_shape, _, name = batch
        name_str = _safe_name(name)
        gt = np.asarray(inst_maps.numpy()[0]).astype(np.int32)
        pred_path = artifact_dir / f"{name_str}_pred.npy"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing Stage 0 pred artifact: {pred_path}")
        base_pred = np.load(pred_path).astype(np.int32)
        current_pred = base_pred.copy()
        images_seg = img_seg.to(device)

        base_row = _analyze_pair_default(name_str, gt, base_pred)
        decoded_count = 0
        applied_count = 0
        image_selected = grouped_actions.get(name_str, [])

        for local_rank, action in enumerate(image_selected):
            decoded = _decode_action_mask(
                images_seg,
                action,
                ori_shape,
                cfgs,
                net,
                point_encoder,
                memory_bank_list,
                device,
            )
            selected_record = {
                "image": name_str,
                "local_rank": int(local_rank),
                "action_rank": int(action["action_rank"]),
                "score": float(action["_selective_score"]),
                "x": int(action["x"]),
                "y": int(action["y"]),
                "decoded": bool(decoded is not None),
                "applied": False,
                "added_area": 0,
            }
            if decoded is not None:
                decoded_count += 1
                current_pred, added_area = _apply_insert(
                    current_pred,
                    decoded["mask"],
                    int(cfgs.oracle_min_added_area),
                )
                selected_record["added_area"] = int(added_area)
                selected_record["applied"] = bool(added_area >= int(cfgs.oracle_min_added_area))
                if selected_record["applied"]:
                    applied_count += 1
            selected_rows.append(selected_record)

        refined_row = _analyze_pair_default(name_str, gt, current_pred)
        base_rows.append(base_row)
        refined_rows.append(refined_row)
        np.save(out_dir / f"{name_str}_pred.npy", current_pred.astype(np.int32))

        image_item = {
            "image": name_str,
            "selected_count": int(len(image_selected)),
            "decoded_count": int(decoded_count),
            "applied_count": int(applied_count),
        }
        for key in ["dice1", "dice2", "aji", "aji_p", "dq", "sq", "pq"]:
            image_item[f"base_{key}"] = float(base_row[key])
            image_item[f"refined_{key}"] = float(refined_row[key])
            image_item[f"delta_{key}"] = float(refined_row[key] - base_row[key])
        image_rows.append(image_item)
        pbar.update()
    pbar.close()

    with open(out_dir / "image_metrics.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = list(image_rows[0].keys()) if image_rows else ["image"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(image_rows)

    with open(out_dir / "selected_actions.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = list(selected_rows[0].keys()) if selected_rows else [
            "image",
            "local_rank",
            "action_rank",
            "score",
            "x",
            "y",
            "decoded",
            "applied",
            "added_area",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)

    base_summary = summarize(base_rows)
    refined_summary = summarize(refined_rows)
    summary = {
        "num_images": len(refined_rows),
        "score": str(cfgs.selective_score),
        "budget": int(cfgs.selective_budget),
        "min_score": float(cfgs.selective_min_score),
        "base": base_summary,
        "refined": refined_summary,
        "delta": _summary_delta(base_summary, refined_summary),
        "selected_actions": int(len(selected_rows)),
        "decoded_actions": int(sum(1 for row in selected_rows if row["decoded"])),
        "applied_actions": int(sum(1 for row in selected_rows if row["applied"])),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Wrote image metrics: {out_dir / 'image_metrics.csv'}")
    print(f"Wrote selected actions: {out_dir / 'selected_actions.csv'}")
    print(f"Wrote summary: {out_dir / 'summary.json'}")
    return summary
