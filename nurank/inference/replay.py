"""Frozen cache replay for Single, existing IoU, NuRank and oracle token selection."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nuset.audit.metrics import assembly_metrics
from nuset.audit.runner import _append_path_masks, _assemble_instance_map, mask_process_eval
from nurank.analysis.metrics import ranking_metrics
from nurank.cache.data import resolve_nurank_images
from nurank.cache.io import group_feature_matrix, iter_groups, load_manifest
from nurank.model.ranker import NuRankSharedRanker


PATHS = ("baseline_single", "existing_all_pred", "nurank", "oracle_all")


@dataclass
class _Path:
    boxes: list[Any] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)
    inds: list[int] = field(default_factory=list)
    candidate_count: int = 0


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _select(group: dict[str, Any], ranker: NuRankSharedRanker, device: torch.device) -> dict[str, np.ndarray]:
    predicted = np.asarray(group["original_predicted_iou"], dtype=np.float32)
    truth = np.asarray(group["true_hard_iou"], dtype=np.float32)
    with torch.no_grad():
        rank_scores = ranker(torch.from_numpy(group_feature_matrix(group)).to(device)).detach().cpu().numpy()
    return {"baseline_single": np.zeros(len(predicted), dtype=np.int64), "existing_all_pred": predicted.argmax(axis=1), "nurank": rank_scores.argmax(axis=1), "oracle_all": truth.argmax(axis=1)}, rank_scores


def _append_group(path: _Path, group: dict[str, Any], selection: np.ndarray) -> None:
    logits = torch.from_numpy(np.asarray(group["mask_logits"], dtype=np.float32))
    predicted_iou = torch.from_numpy(np.asarray(group["original_predicted_iou"], dtype=np.float32))
    index = torch.as_tensor(selection, dtype=torch.long)
    batch = torch.arange(len(index), dtype=torch.long)
    chosen_logits, chosen_iou = logits[batch, index], predicted_iou[batch, index]
    box = tuple(int(value) for value in np.asarray(group["crop_box_xyxy"]).tolist())
    ori_shape = tuple(int(value) for value in np.asarray(group["ori_shape_hw"]).tolist())
    records = mask_process_eval(np.asarray(group["classes"]), np.asarray(group["prompt_ids"]), box, ori_shape, torch.from_numpy(np.asarray(group["coordinates_local_xy"], dtype=np.float32)).unsqueeze(1), chosen_logits, chosen_iou)
    path.candidate_count += len(records)
    _append_path_masks(path, records, box, ori_shape)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"paths": {}, "comparisons": {}}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows: grouped.setdefault(row["path"], []).append(row)
    metric_names = ("dice", "aji", "aji_plus", "dq", "sq", "pq", "tp", "fp", "fn", "matched_iou_sum", "instance_count")
    for path, values in grouped.items(): result["paths"][path] = {name: float(np.mean([row[name] for row in values])) for name in metric_names}
    base = {row["image_id"]: row for row in grouped["baseline_single"]}
    oracle_delta = result["paths"]["oracle_all"]["pq"] - result["paths"]["baseline_single"]["pq"]
    for path in ("existing_all_pred", "nurank", "oracle_all"):
        deltas = {row["image_id"]: float(row["pq"] - base[row["image_id"]]["pq"]) for row in grouped[path]}
        positive = sum(max(0.0, delta) for delta in deltas.values())
        result["comparisons"][path] = {"mean_delta": {name: result["paths"][path][name] - result["paths"]["baseline_single"][name] for name in ("dice", "aji", "aji_plus", "dq", "sq", "pq")}, "per_image_pq_delta": deltas, "pq_non_decreasing_images": int(sum(delta >= 0 for delta in deltas.values())), "largest_positive_image_contribution_fraction": float(max([max(0.0, value) for value in deltas.values()], default=0.0) / positive) if positive else 0.0}
    nr_delta = result["comparisons"]["nurank"]["mean_delta"]["pq"]
    result["comparisons"]["nurank_recovery_ratio"] = float(nr_delta / oracle_delta) if oracle_delta > 0 else None
    result["comparisons"]["nurank_minus_existing_pq"] = result["paths"]["nurank"]["pq"] - result["paths"]["existing_all_pred"]["pq"]
    return result


def _bootstrap(rows: list[dict[str, Any]], *, seed: int = 3407, samples: int = 2000) -> dict[str, Any]:
    by_path = {path: {row["image_id"]: row for row in rows if row["path"] == path} for path in PATHS}
    ids = sorted(by_path["baseline_single"])
    rng = np.random.default_rng(seed)
    result: dict[str, Any] = {"seed": seed, "resamples": samples, "image_count": len(ids), "comparisons": {}}
    for path in ("existing_all_pred", "nurank", "oracle_all"):
        result["comparisons"][path] = {}
        for metric in ("pq", "aji", "dq", "sq"):
            delta = np.asarray([by_path[path][image_id][metric] - by_path["baseline_single"][image_id][metric] for image_id in ids], dtype=np.float64)
            bootstrap = np.asarray([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(samples)])
            result["comparisons"][path][metric] = {"mean": float(delta.mean()), "ci95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))], "positive_image_fraction": float(np.mean(delta > 0)), "negative_image_fraction": float(np.mean(delta < 0)), "largest_positive_image_contribution_fraction": float(np.maximum(delta, 0).max() / np.maximum(delta, 0).sum()) if np.maximum(delta, 0).sum() else 0.0}
    return result


def _ranking_report(original: np.ndarray, rank_scores: np.ndarray, target: np.ndarray, matched: np.ndarray) -> dict[str, Any]:
    existing, nurank = ranking_metrics(original, target, matched), ranking_metrics(rank_scores, target, matched)
    changed = existing["selected_indices"] != nurank["selected_indices"]
    delta = nurank["selected_true_iou"] - existing["selected_true_iou"]
    confusion_rows = []
    for oracle in range(4):
        for old in range(4):
            for new in range(4):
                count = int(np.sum((existing["oracle_indices"] == oracle) & (existing["selected_indices"] == old) & (nurank["selected_indices"] == new)))
                if count: confusion_rows.append({"oracle_token": oracle, "existing_token": old, "nurank_token": new, "count": count})
    changed_report = {"changed_prompt_count": int(changed.sum()), "true_iou_improved_fraction": float(np.mean(delta[changed] > 0)) if changed.any() else 0.0, "true_iou_decreased_fraction": float(np.mean(delta[changed] < 0)) if changed.any() else 0.0, "mean_true_iou_delta": float(delta[changed].mean()) if changed.any() else 0.0, "crosses_iou_0_5_up": int(np.sum((existing["selected_true_iou"] < 0.5) & (nurank["selected_true_iou"] >= 0.5) & changed)), "crosses_iou_0_5_down": int(np.sum((existing["selected_true_iou"] >= 0.5) & (nurank["selected_true_iou"] < 0.5) & changed)), "unmatched_changed_fraction": float(np.mean(~matched[changed])) if changed.any() else 0.0}
    subgroup = {}
    for name, selection in (("matched", matched), ("unmatched", ~matched)):
        if selection.any():
            subgroup[name] = {"existing": {key: value for key, value in ranking_metrics(original[selection], target[selection]).items() if not isinstance(value, np.ndarray)}, "nurank": {key: value for key, value in ranking_metrics(rank_scores[selection], target[selection]).items() if not isinstance(value, np.ndarray)}}
    unmatched_fp_proxy = {"existing_selected_iou_below_0_5": int(np.sum((~matched) & (existing["selected_true_iou"] < 0.5))), "nurank_selected_iou_below_0_5": int(np.sum((~matched) & (nurank["selected_true_iou"] < 0.5)))}
    return {"existing": {key: value for key, value in existing.items() if not isinstance(value, np.ndarray)}, "nurank": {key: value for key, value in nurank.items() if not isinstance(value, np.ndarray)}, "matched_unmatched": subgroup, "unmatched_false_positive_proxy": unmatched_fp_proxy, "top1_accuracy_improvement_points": float((nurank["top1_accuracy"] - existing["top1_accuracy"]) * 100), "mean_regret_reduction_fraction": float((existing["mean_selection_regret"] - nurank["mean_selection_regret"]) / existing["mean_selection_regret"]) if existing["mean_selection_regret"] > 0 else None, "changed_existing_winner": changed_report, "confusion_rows": confusion_rows}


def evaluate_cached_development(*, development_cache_dir: Path, ranker: NuRankSharedRanker, data_root: Path, split_manifest_path: Path, out_dir: Path, device: torch.device) -> dict[str, Any]:
    """Replay four deterministic selectors. This invokes no frozen-model forward call."""
    if out_dir.exists(): raise FileExistsError(f"NuRank evaluation destination must be new: {out_dir}")
    manifest = load_manifest(development_cache_dir)
    if manifest["role"] != "development": raise ValueError("NuRank evaluation requires development cache only")
    out_dir.mkdir(parents=True)
    items = {item.image_id: item for item in resolve_nurank_images(data_root, split_manifest_path, "development")}
    paths = {image_id: {name: _Path() for name in PATHS} for image_id in items}
    original, scores, targets, matched, image_ids, prompt_rows = [], [], [], [], [], []
    started = time.perf_counter(); ranker.eval(); ranker_calls = 0; ranker_selection_seconds = 0.0; existing_selection_seconds = 0.0
    for group in iter_groups(development_cache_dir):
        rank_started = time.perf_counter()
        selected, rank_score = _select(group, ranker, device); ranker_selection_seconds += time.perf_counter() - rank_started; ranker_calls += 1
        existing_started = time.perf_counter(); np.asarray(group["original_predicted_iou"], dtype=np.float32).argmax(axis=1); existing_selection_seconds += time.perf_counter() - existing_started
        image_id = group["_entry"]["image_id"]
        for name in PATHS: _append_group(paths[image_id][name], group, selected[name])
        count = len(rank_score)
        original.append(np.asarray(group["original_predicted_iou"], dtype=np.float32)); scores.append(rank_score); targets.append(np.asarray(group["true_hard_iou"], dtype=np.float32)); matched.append(np.asarray(group["matched"], dtype=bool)); image_ids.extend([image_id] * count)
        for index in range(count): prompt_rows.append({"image_id": image_id, "crop_id": int(group["_entry"]["crop_id"]), "prompt_id": int(group["prompt_ids"][index]), "matched": bool(group["matched"][index]), "original_predicted_iou": np.asarray(group["original_predicted_iou"][index]).tolist(), "nurank_score": rank_score[index].tolist(), "true_hard_iou": np.asarray(group["true_hard_iou"][index]).tolist(), "existing_token": int(selected["existing_all_pred"][index]), "nurank_token": int(selected["nurank"][index]), "oracle_token": int(selected["oracle_all"][index])})
    per_image: list[dict[str, Any]] = []
    for image_id, item in items.items():
        for name, path in paths[image_id].items():
            prediction = _assemble_instance_map(path.boxes, path.scores, path.masks, path.inds, item.instance_map.shape, 0.5)
            per_image.append({"image_id": image_id, "path": name, "candidate_mask_count": path.candidate_count, "assembly_rejected_mask_count": path.candidate_count - len(path.masks), **assembly_metrics(item.instance_map, prediction)})
    original_array, rank_array, target_array, matched_array = np.concatenate(original), np.concatenate(scores), np.concatenate(targets), np.concatenate(matched)
    ranking = _ranking_report(original_array, rank_array, target_array, matched_array)
    summary = _summary(per_image); bootstrap = _bootstrap(per_image)
    with (out_dir / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in per_image for key in row})); writer.writeheader(); writer.writerows(per_image)
    with (out_dir / "per_prompt_ranking.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prompt_rows[0]) if prompt_rows else []); writer.writeheader(); writer.writerows(prompt_rows)
    with (out_dir / "token_confusion_matrices.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["oracle_token", "existing_token", "nurank_token", "count"]); writer.writeheader(); writer.writerows(ranking["confusion_rows"])
    _json(out_dir / "segmentation_summary.json", summary); _json(out_dir / "ranking_summary.json", {key: value for key, value in ranking.items() if key != "confusion_rows"}); _json(out_dir / "bootstrap_summary.json", bootstrap)
    extraction_seconds = float(manifest.get("elapsed_seconds", 0.0))
    shared_calls = {key: int(manifest["call_counts"][key]) for key in ("sam_image_encoder_calls", "sam_prompt_encoder_calls", "sam_mask_decoder_calls")}
    runtime = {"evaluation_replay_seconds": time.perf_counter() - started, "ranker_calls": ranker_calls, "ranker_selection_seconds": ranker_selection_seconds, "existing_iou_argmax_seconds": existing_selection_seconds, "full_path_runtime_overhead_ratio_vs_cached_baseline": float((extraction_seconds + ranker_selection_seconds) / extraction_seconds) if extraction_seconds > 0 else None, "frozen_model_calls_during_replay": {"sam_image_encoder_calls": 0, "sam_prompt_encoder_calls": 0, "sam_mask_decoder_calls": 0}, "cache_extraction_call_counts": manifest["call_counts"], "path_call_counts": {"baseline_single": shared_calls, "existing_all_pred": shared_calls, "nurank": shared_calls, "oracle_all": shared_calls}, "path_wall_time_estimate_seconds": {"baseline_single": extraction_seconds, "existing_all_pred": extraction_seconds, "nurank": extraction_seconds + ranker_selection_seconds, "oracle_all": extraction_seconds}, "ranker_parameter_count": ranker.parameter_count()}
    conflict_rows = [{"image_id": row["image_id"], "path": row["path"], "fp": row["fp"], "candidate_mask_count": row["candidate_mask_count"], "assembly_rejected_mask_count": row["assembly_rejected_mask_count"]} for row in per_image]
    with (out_dir / "conflict_analysis.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "path", "fp", "candidate_mask_count", "assembly_rejected_mask_count"]); writer.writeheader(); writer.writerows(conflict_rows)
    with (out_dir / "runtime_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(runtime)); writer.writeheader(); writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value for key, value in runtime.items()})
    _json(out_dir / "runtime_summary.json", runtime)
    return {"segmentation": summary, "ranking": ranking, "bootstrap": bootstrap, "runtime": runtime}
