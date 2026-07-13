"""Single frozen StainCF-PMS Phase 0 audit execution and reporting."""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation
from skimage.feature import peak_local_max
from skimage.segmentation import find_boundaries

from .inference import DeploymentResult, FrozenStainPMS
from .metrics import (
    coordinate_match, correlation, inclusive_iou_pairs, instance_metrics, map_iou,
    paired_bootstrap, split_merge_counts,
)
from .prepare import _read_label, _read_rgb
from .protocol import (
    BOOTSTRAP_REPLICATES, POINT_MATCH_DIAGONAL_FRACTION, RESIDUAL_PEAK_MIN_DISTANCE,
    RESIDUAL_PEAK_THRESHOLD, SEED, ProtocolError, VIEW_NAMES, VIEWS, baseline_selection_payload,
    require_exact_sha256, sha256_file, write_json,
)
from .transforms import decompose


TNBC_SHA256 = "44a3cb3e93051301d789e44f93769588abfa727d7a174b0270f55305ef023781"
MONUSEG_SHA256 = "6616c24626a162580d79aa7035d0b6c1a4ae6240a6c2f4599960dd4efac95db1"


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _point_metrics(reference: DeploymentResult, candidate: DeploymentResult, gt: np.ndarray) -> dict[str, Any]:
    shape = tuple(gt.shape)
    gt_points = []
    for instance_id in np.unique(gt)[1:]:
        ys, xs = np.where(gt == instance_id)
        index = int(np.argmin((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2))
        gt_points.append([float(xs[index]), float(ys[index])])
    gt_points = np.asarray(gt_points, dtype=float).reshape(-1, 2)
    ref_gt = coordinate_match(reference.points, gt_points, shape, POINT_MATCH_DIAGONAL_FRACTION)
    cand_gt = coordinate_match(candidate.points, gt_points, shape, POINT_MATCH_DIAGONAL_FRACTION)
    matched = coordinate_match(reference.points, candidate.points, shape, POINT_MATCH_DIAGONAL_FRACTION)
    ref_index, cand_index = matched["left"], matched["right"]
    pearson, spearman = correlation(reference.point_scores[ref_index], candidate.point_scores[cand_index]) if ref_index else (float("nan"), float("nan"))
    clean = set(ref_gt["left"])
    retained = sum(index in set(ref_index) for index in clean)
    return {
        "point_count_v0": len(reference.points), "point_count_view": len(candidate.points),
        "point_count_delta": len(candidate.points) - len(reference.points),
        "gt_point_coverage_v0": len(ref_gt["right"]) / max(1, len(gt_points)),
        "gt_point_coverage_view": len(cand_gt["right"]) / max(1, len(gt_points)),
        "gt_prompt_recall_v0": len(ref_gt["right"]) / max(1, len(gt_points)),
        "gt_prompt_recall_view": len(cand_gt["right"]) / max(1, len(gt_points)),
        "false_prompts_v0": len(reference.points) - len(ref_gt["left"]),
        "false_prompts_view": len(candidate.points) - len(cand_gt["left"]),
        "hungarian_matched_point_ratio": len(ref_index) / max(1, len(reference.points)),
        "matched_coordinate_distance_px": float(np.mean(matched["distances"])) if matched["distances"] else float("nan"),
        "objectness_pearson": pearson, "objectness_spearman": spearman,
        "clean_point_retention": retained / max(1, len(clean)),
        "new_points": len(candidate.points) - len(cand_index), "disappeared_points": len(reference.points) - len(ref_index),
        "point_match_threshold_px": float(matched["threshold_px"]),
    }


def _boundary_fscore(left: np.ndarray, right: np.ndarray) -> float:
    left_b, right_b = find_boundaries(left, mode="outer"), find_boundaries(right, mode="outer")
    if not left_b.any() and not right_b.any():
        return 1.0
    precision = (left_b & binary_dilation(right_b, iterations=1)).sum() / max(1, left_b.sum())
    recall = (right_b & binary_dilation(left_b, iterations=1)).sum() / max(1, right_b.sum())
    return float(2 * precision * recall / max(1e-12, precision + recall))


def _fixed_prompt_metrics(reference: DeploymentResult, candidate: DeploymentResult, gt: np.ndarray) -> dict[str, Any]:
    hard_iou: list[float] = []; logit_mae: list[float] = []; boundary: list[float] = []; area_ratio: list[float] = []
    iou_delta: list[float] = []; gt_iou_v0: list[float] = []; gt_iou_view: list[float] = []; crosses = 0
    for point_id, observations in reference.prompt_observations.items():
        ref_lookup = {tuple(item["crop_box"]): item for item in observations}
        cand_lookup = {tuple(item["crop_box"]): item for item in candidate.prompt_observations.get(point_id, [])}
        for box in sorted(set(ref_lookup) & set(cand_lookup)):
            left, right = ref_lookup[box], cand_lookup[box]
            hard_iou.append(map_iou(left["hard_mask"], right["hard_mask"]))
            logit_mae.append(float(np.abs(left["logits"].astype(np.float32) - right["logits"].astype(np.float32)).mean()))
            boundary.append(_boundary_fscore(left["hard_mask"], right["hard_mask"]))
            area_ratio.append(float(right["hard_mask"].sum() / max(1, left["hard_mask"].sum())))
            x, y, _, _ = box
            # The prompt is at a known global coordinate. Pick an instance only when it lies in GT.
            # The local crop association is deterministic under the frozen coordinate test.
            # GT IoU is evaluated on the local crop to avoid padding/uncropping ambiguity.
            # point id can be absent only when the auto proposal set is empty.
            # The cross-threshold count uses decoder predicted-IoU, as registered.
            before, after = float(left["predicted_iou"]), float(right["predicted_iou"])
            iou_delta.append(after - before)
            crosses += int((before >= 0.5) != (after >= 0.5))
            point = reference.points[point_id]
            px, py = int(round(float(point[0]))), int(round(float(point[1])))
            if 0 <= py < gt.shape[0] and 0 <= px < gt.shape[1] and gt[py, px] > 0:
                x, y, _, _ = box
                target = gt[y : y + left["hard_mask"].shape[0], x : x + left["hard_mask"].shape[1]] == gt[py, px]
                gt_iou_v0.append(map_iou(left["hard_mask"], target))
                gt_iou_view.append(map_iou(right["hard_mask"], target))
    return {
        "fixed_prompt_coordinate_equivalent": True,
        "fixed_prompt_pair_count": len(hard_iou), "hard_mask_iou": float(np.mean(hard_iou)) if hard_iou else float("nan"),
        "mask_logit_mae": float(np.mean(logit_mae)) if logit_mae else float("nan"),
        "boundary_fscore": float(np.mean(boundary)) if boundary else float("nan"),
        "predicted_iou_delta": float(np.mean(iou_delta)) if iou_delta else float("nan"),
        "mask_area_ratio": float(np.mean(area_ratio)) if area_ratio else float("nan"),
        "matched_gt_iou_v0": float(np.mean(gt_iou_v0)) if gt_iou_v0 else float("nan"),
        "matched_gt_iou_view": float(np.mean(gt_iou_view)) if gt_iou_view else float("nan"),
        "matched_gt_iou_delta": float(np.mean(gt_iou_view) - np.mean(gt_iou_v0)) if gt_iou_v0 else float("nan"),
        "matched_gt_iou_count": len(gt_iou_v0), "iou_threshold_crossings_0_5": crosses,
    }


def _h_residual(rgb: np.ndarray, pred: np.ndarray, fallback_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = decompose(rgb, fallback_matrix=fallback_matrix).concentration[..., 0]
    normalized = (h - h.min()) / max(1e-12, float(h.max() - h.min()))
    uncovered = pred <= 0
    return h * uncovered, normalized * uncovered, uncovered


def _coverage_metrics(reference_rgb: np.ndarray, candidate_rgb: np.ndarray, reference: DeploymentResult, candidate: DeploymentResult, fallback_matrix: np.ndarray) -> dict[str, Any]:
    raw0, norm0, uncovered0 = _h_residual(reference_rgb, reference.pred, fallback_matrix)
    raw1, norm1, uncovered1 = _h_residual(candidate_rgb, candidate.pred, fallback_matrix)
    raw_pearson, raw_spearman = correlation(raw0, raw1)
    norm_pearson, norm_spearman = correlation(norm0, norm1)
    peaks0 = peak_local_max(norm0, min_distance=RESIDUAL_PEAK_MIN_DISTANCE, threshold_abs=RESIDUAL_PEAK_THRESHOLD)
    peaks1 = peak_local_max(norm1, min_distance=RESIDUAL_PEAK_MIN_DISTANCE, threshold_abs=RESIDUAL_PEAK_THRESHOLD)
    # coordinate_match expects x,y whereas peak_local_max returns y,x.
    matched = coordinate_match(peaks0[:, ::-1] if len(peaks0) else peaks0, peaks1[:, ::-1] if len(peaks1) else peaks1, reference.pred.shape, POINT_MATCH_DIAGONAL_FRACTION)
    return {
        "coverage_map_iou": map_iou(reference.pred, candidate.pred),
        "uncovered_h_evidence_area_v0": int((raw0 > 0).sum()), "uncovered_h_evidence_area_view": int((raw1 > 0).sum()),
        "uncovered_h_evidence_sum_v0": float(raw0.sum()), "uncovered_h_evidence_sum_view": float(raw1.sum()),
        "raw_residual_pearson": raw_pearson, "raw_residual_spearman": raw_spearman,
        "normalized_residual_pearson": norm_pearson, "normalized_residual_spearman": norm_spearman,
        "residual_peak_count_v0": len(peaks0), "residual_peak_count_view": len(peaks1),
        "peak_hungarian_match_rate": len(matched["left"]) / max(1, len(peaks0)),
        "peak_spatial_displacement_px": float(np.mean(matched["distances"])) if matched["distances"] else float("nan"),
        "clean_residual_candidate_retention": len(matched["left"]) / max(1, len(peaks0)),
    }


def _prediction_consistency(reference: DeploymentResult, candidate: DeploymentResult) -> dict[str, Any]:
    metrics = instance_metrics(reference.pred, candidate.pred)
    paired_true, paired_pred, deaths, births, matrix = inclusive_iou_pairs(reference.pred, candidate.pred)
    matched = [matrix[np.where(np.unique(reference.pred)[1:] == left)[0][0], np.where(np.unique(candidate.pred)[1:] == right)[0][0]] for left, right in zip(paired_true, paired_pred)] if paired_true else []
    splits, merges = split_merge_counts(reference.pred, candidate.pred)
    return {
        "prediction_dice": metrics["dice"], "prediction_aji": metrics["aji"], "prediction_aji_plus": metrics["aji_plus"],
        "prediction_dq": metrics["dq"], "prediction_sq": metrics["sq"], "prediction_pq": metrics["pq"],
        "matched_instance_mask_iou": float(np.mean(matched)) if matched else 0.0,
        "instance_births": len(births), "instance_deaths": len(deaths), "split_changes": splits, "merge_changes": merges,
    }


def _visualize(path: Path, images: dict[str, np.ndarray], results: dict[str, DeploymentResult], fallback_matrix: np.ndarray) -> None:
    selected = ["V0", "V2", "V3", "V4", "V5"]
    panels = []
    for view in selected:
        panel = Image.fromarray(images[view]).convert("RGB")
        draw = ImageDraw.Draw(panel)
        for x, y in results[view].points:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 0, 0))
        panels.append(panel)
    width, height = panels[0].size
    canvas = Image.new("RGB", (width * len(panels), height))
    for index, panel in enumerate(panels): canvas.paste(panel, (index * width, 0))
    path.parent.mkdir(parents=True, exist_ok=True); canvas.save(path)
    masks = Image.new("L", (width * len(panels), height))
    residuals = Image.new("L", (width * len(panels), height))
    for index, view in enumerate(selected):
        mask = (results[view].pred > 0).astype(np.uint8) * 255
        _, residual, _ = _h_residual(images[view], results[view].pred, fallback_matrix)
        residual = np.clip(np.rint(residual * 255), 0, 255).astype(np.uint8)
        masks.paste(Image.fromarray(mask), (index * width, 0))
        residuals.paste(Image.fromarray(residual), (index * width, 0))
    masks.save(path.with_name(path.stem + "_masks_coverage.png"))
    residuals.save(path.with_name(path.stem + "_residual.png"))


def _evidence_label(rows: list[dict[str, Any]]) -> str:
    """A descriptive evidence level, never an automatic appeal-proof GO/NO-GO."""
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["view"] in {"V2", "V3", "V4", "V5"}: by_dataset[str(row["dataset"])].append(row)
    strong_datasets = 0; combined = 0
    for dataset_rows in by_dataset.values():
        aji = min(float(row["delta_aji"]) for row in dataset_rows)
        pq = min(float(row["delta_pq"]) for row in dataset_rows)
        if aji <= -0.020 or pq <= -0.020: strong_datasets += 1
        if aji <= -0.010 and pq <= -0.010: combined += 1
    if strong_datasets >= 2: return "STRONG GAP evidence candidate (requires PI adjudication)"
    if strong_datasets >= 1 or combined >= 1: return "MODERATE / composite GAP evidence candidate (requires PI adjudication)"
    return "WEAK / no material GAP evidence candidate (requires PI adjudication)"


def _mechanism_summary(point_rows: list[dict[str, Any]], fixed_rows: list[dict[str, Any]], coverage_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank transparent diagnostic proxies; this is not a causal claim."""
    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in point_rows:
        grouped[(str(row["dataset"]), str(row["view"]))]["auto_point"].append(float(row["gt_point_coverage_v0"]) - float(row["gt_point_coverage_view"]))
    for row in fixed_rows:
        value = float(row["hard_mask_iou"])
        if np.isfinite(value): grouped[(str(row["dataset"]), str(row["view"]))]["fixed_prompt_mask"].append(1.0 - value)
    for row in coverage_rows:
        grouped[(str(row["dataset"]), str(row["view"]))]["coverage_residual"].append(1.0 - float(row["coverage_map_iou"]))
    output: list[dict[str, Any]] = []
    for (dataset, view), values in sorted(grouped.items()):
        means = {key: float(np.mean(value)) for key, value in values.items() if value}
        output.append({"dataset": dataset, "view": view, "diagnostic_proxy_means": means, "largest_proxy": max(means, key=means.get) if means else "undetermined", "note": "Proxy ranking only; assembly effects are inspected jointly with prediction consistency and TP/FP/FN."})
    return output


def run_audit(out_dir: str | Path, tnbc_checkpoint: str | Path, monuseg_checkpoint: str | Path, sam2_checkpoint: str | Path, device: str) -> None:
    out_dir = Path(out_dir).resolve()
    manifest = json.loads((out_dir / "fixed_audit_manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("frozen") or manifest["quality_gate"]["failed_samples"]:
        raise ProtocolError("fixed manifest or transform quality gate is not valid; inference forbidden")
    tnbc_checkpoint, monuseg_checkpoint, sam2_checkpoint = Path(tnbc_checkpoint), Path(monuseg_checkpoint), Path(sam2_checkpoint)
    before_files = {
        "tnbc": require_exact_sha256(tnbc_checkpoint, TNBC_SHA256, "TNBC StainPMS checkpoint"),
        "monuseg": require_exact_sha256(monuseg_checkpoint, MONUSEG_SHA256, "MoNuSeg StainPMS checkpoint"),
        "sam2": sha256_file(sam2_checkpoint),
    }
    models = {
        "tnbc": FrozenStainPMS(tnbc_checkpoint, sam2_checkpoint, overlap=32, device=device),
        "monuseg": FrozenStainPMS(monuseg_checkpoint, sam2_checkpoint, overlap=92, device=device),
    }
    before_models = {dataset: model.parameter_checksum() for dataset, model in models.items()}
    end_rows: list[dict[str, Any]] = []; point_rows: list[dict[str, Any]] = []; fixed_rows: list[dict[str, Any]] = []; coverage_rows: list[dict[str, Any]] = []; consistency_rows: list[dict[str, Any]] = []
    runtime: dict[str, Any] = {"device": device, "batch_size": 1, "tta": False, "seed": SEED, "samples": []}
    stats = json.loads((out_dir / "stain_statistics.json").read_text(encoding="utf-8"))
    started = time.perf_counter()
    for sample in manifest["samples"]:
        dataset, sample_id = sample["dataset"], sample["id"]
        model, gt = models[dataset], _read_label(sample["prepared_label"])
        images = {view: _read_rgb(sample["views"][view]) for view in VIEWS}
        fallback = np.asarray(stats[dataset]["reference"]["fallback_matrix"], dtype=np.float64)
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(model.device)
        sample_start = time.perf_counter()
        v0_auto, counts0 = model.deploy(images["V0"])
        v0_fixed, fixed_counts0 = model.deploy(images["V0"], fixed_points=v0_auto.points)
        per_view_results: dict[str, DeploymentResult] = {"V0": v0_auto}
        for view in VIEWS:
            if view == "V0":
                auto, fixed, counts, fixed_counts = v0_auto, v0_fixed, counts0, fixed_counts0
            else:
                auto, counts = model.deploy(images[view])
                fixed, fixed_counts = model.deploy(images[view], fixed_points=v0_auto.points)
                per_view_results[view] = auto
            metrics = instance_metrics(gt, auto.pred); baseline = instance_metrics(gt, v0_auto.pred)
            end_rows.append({"dataset": dataset, "sample_id": sample_id, "view": view, "view_name": VIEW_NAMES[view], **metrics,
                "predicted_point_count": len(auto.points), "assembled_instance_count": int(auto.pred.max()),
                "delta_aji": float(metrics["aji"] - baseline["aji"]), "delta_aji_plus": float(metrics["aji_plus"] - baseline["aji_plus"]), "delta_pq": float(metrics["pq"] - baseline["pq"]),
                "delta_dq": float(metrics["dq"] - baseline["dq"]), "delta_sq": float(metrics["sq"] - baseline["sq"]),
                "delta_tp": int(metrics["tp"] - baseline["tp"]), "delta_fp": int(metrics["fp"] - baseline["fp"]), "delta_fn": int(metrics["fn"] - baseline["fn"]),})
            if view != "V0":
                point_rows.append({"dataset": dataset, "sample_id": sample_id, "view": view, **_point_metrics(v0_auto, auto, gt)})
                fixed_rows.append({"dataset": dataset, "sample_id": sample_id, "view": view, **_fixed_prompt_metrics(v0_fixed, fixed, gt)})
                coverage_rows.append({"dataset": dataset, "sample_id": sample_id, "view": view, **_coverage_metrics(images["V0"], images[view], v0_auto, auto, fallback)})
                consistency_rows.append({"dataset": dataset, "sample_id": sample_id, "view": view, **_prediction_consistency(v0_auto, auto)})
            counts = {key: int(counts[key] + fixed_counts[key]) for key in counts}
            runtime["samples"].append({"dataset": dataset, "sample_id": sample_id, "view": view, **counts})
            prediction_path = out_dir / "predictions" / dataset / view / f"{sample_id}.npy"
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(prediction_path, auto.pred)
        _visualize(out_dir / "representative" / f"{dataset}_{sample_id}_views_points.png", images, per_view_results, fallback)
        runtime["samples"][-1]["sample_wall_time_seconds"] = time.perf_counter() - sample_start
        if torch.cuda.is_available(): runtime["samples"][-1]["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated(model.device))
    runtime["wall_time_seconds"] = time.perf_counter() - started
    after_models = {dataset: model.parameter_checksum() for dataset, model in models.items()}
    after_files = {"tnbc": sha256_file(tnbc_checkpoint), "monuseg": sha256_file(monuseg_checkpoint), "sam2": sha256_file(sam2_checkpoint)}
    if before_files != after_files or before_models != after_models:
        raise ProtocolError("checkpoint or model parameter checksum changed during audit")
    (out_dir / "predictions").mkdir(exist_ok=True)
    _write_csv(out_dir / "end_to_end_metrics.csv", end_rows); _write_csv(out_dir / "point_set_stability.csv", point_rows)
    _write_csv(out_dir / "fixed_prompt_mask_stability.csv", fixed_rows); _write_csv(out_dir / "coverage_residual_stability.csv", coverage_rows)
    _write_csv(out_dir / "prediction_consistency.csv", consistency_rows)
    _write_csv(out_dir / "per_sample_summary.csv", end_rows)
    summaries: list[dict[str, Any]] = []
    for dataset in ("tnbc", "monuseg"):
        for view in VIEWS:
            rows = [row for row in end_rows if row["dataset"] == dataset and row["view"] == view]
            payload: dict[str, Any] = {"dataset": dataset, "view": view, "n": len(rows)}
            for metric in ("aji", "aji_plus", "pq", "dq", "sq", "delta_aji", "delta_aji_plus", "delta_pq"):
                values = [float(row[metric]) for row in rows]
                payload[f"mean_{metric}"] = float(np.mean(values)); payload[f"median_{metric}"] = float(np.median(values))
                if metric.startswith("delta_"): payload[f"bootstrap_{metric}"] = paired_bootstrap(values, SEED, BOOTSTRAP_REPLICATES)
            if view != "V0":
                for metric in ("aji", "aji_plus", "pq"):
                    negatives = np.asarray([-min(0.0, float(row[f"delta_{metric}"])) for row in rows])
                    payload[f"max_single_sample_negative_{metric}_contribution"] = float(negatives.max() / negatives.sum()) if negatives.sum() else 0.0
                    payload[f"{metric}_decline_sample_fraction"] = float(np.mean([float(row[f"delta_{metric}"]) < 0 for row in rows]))
            summaries.append(payload)
    mechanisms = _mechanism_summary(point_rows, fixed_rows, coverage_rows)
    report = {"protocol": baseline_selection_payload(), "evidence_interpretation": _evidence_label(end_rows), "important": "AJI, AJI+, and PQ are co-primary evidence; no automatic GO/NO-GO is issued.", "dataset_summaries": summaries, "mechanism_proxy_summary": mechanisms}
    write_json(out_dir / "dataset_summary.json", {"summaries": summaries}); write_json(out_dir / "report.json", report)
    runtime.update({"checkpoint_sha256_before": before_files, "checkpoint_sha256_after": after_files, "model_parameter_sha256_before": before_models, "model_parameter_sha256_after": after_models})
    write_json(out_dir / "checkpoint_manifest.json", {"files": before_files, "expected": {"tnbc": TNBC_SHA256, "monuseg": MONUSEG_SHA256}})
    write_json(out_dir / "runtime_summary.json", runtime)
    (out_dir / "environment.txt").write_text(f"platform={platform.platform()}\npython={platform.python_version()}\ntorch={torch.__version__}\ncuda={torch.version.cuda}\ngit_base=2a1348cb7a1158a6f77aae2f92c168f9552d8068\n", encoding="utf-8")
    _write_sha256sums(out_dir)


def _write_sha256sums(out_dir: Path) -> None:
    paths = sorted(path for path in out_dir.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    (out_dir / "SHA256SUMS").write_text("".join(f"{sha256_file(path)}  {path.relative_to(out_dir).as_posix()}\n" for path in paths), encoding="utf-8")
