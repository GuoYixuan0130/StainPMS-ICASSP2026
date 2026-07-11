"""NuSet Postmortem-A: frozen NuRank failure and fusion-feasibility audit."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from nuset.audit.data import BASELINE_V1_TNBC_SHA256, sha256_file
from nuset.audit.metrics import assembly_metrics
from nuset.audit.models import load_frozen_bundle, module_state_sha256
from nuset.audit.runner import _append_path_masks, _assemble_instance_map, mask_process_eval
from nuset.postmortem.fusion import FIXED_FUSIONS, fixed_fusions, is_one_hot, simplex_weights, upsample_logits
from nuset.postmortem.metrics import changed_winner_metrics, failure_mode, oracle_gain_summary, score_and_truth_gap, selector_metrics
from nurank.cache.builder import build_automatic_prompt_cache
from nurank.cache.data import resolve_nurank_images
from nurank.cache.io import group_feature_matrix, iter_groups, load_manifest
from nurank.model.ranker import NuRankSharedRanker
from nurank.stage1 import load_ranker_checkpoint


SEED = 3407
TIME_CAP_SECONDS = 45 * 60
TOKEN_PATHS = ("single", "existing_all_pred", "nurank", "token_oracle")
ALL_PATHS = TOKEN_PATHS + FIXED_FUSIONS + ("fixed_library_oracle", "convex_fusion_oracle")


class PostmortemTimeCap(RuntimeError):
    pass


@dataclass
class CandidatePath:
    boxes: list[Any] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)
    inds: list[int] = field(default_factory=list)
    candidate_count: int = 0


@dataclass
class ErrorAccumulator:
    fn0: int = 0
    fn_recoverable: int = 0
    fp0: int = 0
    fp_rejectable: int = 0
    boundary_recoverable: int = 0
    interior_recoverable: int = 0
    boundary_rejectable: int = 0
    interior_rejectable: int = 0
    simultaneous_fn_fp_improvement: int = 0
    prompt_count: int = 0
    correlations: dict[str, list[float]] = field(default_factory=lambda: {f"{a}_{b}": [] for a in range(4) for b in range(a + 1, 4)})
    fp_correlations: dict[str, list[float]] = field(default_factory=lambda: {f"{a}_{b}": [] for a in range(4) for b in range(a + 1, 4)})

    def merge(self, other: "ErrorAccumulator") -> None:
        for field_name in ("fn0", "fn_recoverable", "fp0", "fp_rejectable", "boundary_recoverable", "interior_recoverable", "boundary_rejectable", "interior_rejectable", "simultaneous_fn_fp_improvement", "prompt_count"):
            setattr(self, field_name, getattr(self, field_name) + getattr(other, field_name))
        for key in self.correlations:
            self.correlations[key].extend(other.correlations[key])
            self.fp_correlations[key].extend(other.fp_correlations[key])

    def as_dict(self) -> dict[str, Any]:
        return {
            "matched_prompt_count": self.prompt_count,
            "token0_fn_pixels": self.fn0,
            "token0_fn_recoverable_by_other_token_fraction": self.fn_recoverable / self.fn0 if self.fn0 else 0.0,
            "token0_fp_pixels": self.fp0,
            "token0_fp_rejectable_by_other_token_fraction": self.fp_rejectable / self.fp0 if self.fp0 else 0.0,
            "same_prompt_separate_fn_fp_improvement_fraction": self.simultaneous_fn_fp_improvement / self.prompt_count if self.prompt_count else 0.0,
            "recoverable_error_boundary_fraction": self.boundary_recoverable / (self.boundary_recoverable + self.interior_recoverable) if self.boundary_recoverable + self.interior_recoverable else 0.0,
            "rejectable_error_boundary_fraction": self.boundary_rejectable / (self.boundary_rejectable + self.interior_rejectable) if self.boundary_rejectable + self.interior_rejectable else 0.0,
            "fn_pairwise_correlation": {key: float(np.mean(value)) if value else None for key, value in self.correlations.items()},
            "fp_pairwise_correlation": {key: float(np.mean(value)) if value else None for key, value in self.fp_correlations.items()},
        }


def _set_seed() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _checksums(out_dir: Path) -> None:
    records = [f"{sha256_file(path)}  {path.relative_to(out_dir).as_posix()}" for path in sorted(path for path in out_dir.rglob("*") if path.is_file() and path.name != "SHA256SUMS")]
    (out_dir / "SHA256SUMS").write_text("\n".join(records) + "\n", encoding="utf-8")


def _enforce_time(started: float) -> None:
    if time.perf_counter() - started > TIME_CAP_SECONDS:
        raise PostmortemTimeCap("NuSet Postmortem-A reached the fixed 45 minute GPU wall-time cap")


def _cache_has_lowres(cache_dir: Path) -> bool:
    manifest = load_manifest(cache_dir)
    if not manifest.get("groups"):
        return False
    path = cache_dir / manifest["groups"][0]["path"]
    with np.load(path, allow_pickle=False) as payload:
        return "low_res_logits" in payload.files


def _verify_reextraction(original_dir: Path, reproduced_dir: Path) -> dict[str, Any]:
    if int(load_manifest(original_dir)["group_count"]) != int(load_manifest(reproduced_dir)["group_count"]):
        raise RuntimeError("Postmortem low-resolution cache group count does not reproduce formal NuRank cache")
    maximum = {"mask_logits": 0.0, "predicted_iou": 0.0, "coordinates": 0.0}
    for old, new in zip(iter_groups(original_dir), iter_groups(reproduced_dir)):
        if old["_entry"]["image_id"] != new["_entry"]["image_id"] or old["_entry"]["crop_id"] != new["_entry"]["crop_id"]:
            raise RuntimeError("Postmortem cache image/crop order differs from formal NuRank cache")
        if not np.array_equal(old["prompt_ids"], new["prompt_ids"]) or not np.array_equal(old["classes"], new["classes"]):
            raise RuntimeError("Postmortem cache automatic prompt order differs from formal NuRank cache")
        maximum["mask_logits"] = max(maximum["mask_logits"], float(np.abs(old["mask_logits"] - new["mask_logits"]).max()))
        maximum["predicted_iou"] = max(maximum["predicted_iou"], float(np.abs(old["original_predicted_iou"] - new["original_predicted_iou"]).max()))
        maximum["coordinates"] = max(maximum["coordinates"], float(np.abs(old["coordinates_local_xy"] - new["coordinates_local_xy"]).max()))
    if maximum["mask_logits"] > 1e-6 or maximum["predicted_iou"] > 1e-6 or maximum["coordinates"] > 0.0:
        raise RuntimeError(f"Postmortem re-extraction does not exactly reproduce formal NuRank cache: {maximum}")
    return maximum


def _prepare_caches(*, nurank_run_dir: Path, out_dir: Path, data_root: Path, split_manifest_path: Path, checkpoint: Path, config_path: Path, sam_config: str, device: torch.device, started: float) -> tuple[dict[str, Path], dict[str, Any]]:
    original = {role: nurank_run_dir / "cache" / role for role in ("train", "development")}
    for path in original.values():
        load_manifest(path)
    if all(_cache_has_lowres(path) for path in original.values()):
        return original, {"cache_source": "formal_nurank_cache_with_low_resolution_logits", "reextraction": None}
    _enforce_time(started)
    reextracted = {role: out_dir / "reextracted_lowres_cache" / role for role in ("train", "development")}
    bundle = load_frozen_bundle(config_path, sam_config, checkpoint, device)
    verification = {}
    for role in ("train", "development"):
        elapsed = time.perf_counter() - started
        result = build_automatic_prompt_cache(bundle=bundle, data_root=data_root, split_manifest_path=split_manifest_path, role=role, cache_dir=reextracted[role], prior_stage_seconds=elapsed, time_limit_seconds=TIME_CAP_SECONDS)
        verification[role] = {"elapsed_seconds": result.elapsed_seconds, "estimated_total_seconds": result.estimated_total_seconds, "formal_alignment": _verify_reextraction(original[role], reextracted[role])}
        _enforce_time(started)
    return reextracted, {"cache_source": "deterministic_low_resolution_reextraction", "reextraction": verification}


def _local_target(item, group: dict[str, Any], index: int) -> tuple[np.ndarray | None, list[np.ndarray]]:
    x1, y1, x2, y2 = (int(value) for value in group["crop_box_xyxy"])
    local = item.instance_map[y1:y2, x1:x2]
    target_id = int(group["target_instance_id"][index])
    ids = np.unique(local); ids = ids[ids != 0]
    all_masks = [local == instance_id for instance_id in ids]
    return (local == target_id if target_id else None), all_masks


def _group_targets(item, group: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    x1, y1, x2, y2 = (int(value) for value in group["crop_box_xyxy"])
    local = item.instance_map[y1:y2, x1:x2]
    ids = np.unique(local); ids = ids[ids != 0]
    all_masks = [local == instance_id for instance_id in ids]
    matched = np.asarray(group["matched"], dtype=bool)
    direct = np.zeros((len(matched), *local.shape), dtype=bool)
    for index, instance_id in enumerate(group["target_instance_id"]):
        if matched[index]: direct[index] = local == int(instance_id)
    return direct, matched, all_masks


def _candidate_iou(direct_targets: np.ndarray, matched: np.ndarray, all_masks: list[np.ndarray], logits: torch.Tensor) -> np.ndarray:
    """Hard IoU against associated GT, or maximum existing GT IoU for unmatched prompts."""
    hard = (logits > 0).detach().cpu().numpy()[:, : direct_targets.shape[1], : direct_targets.shape[2]]
    output = np.zeros(len(hard), dtype=np.float32)
    if matched.any():
        prediction, truth = hard[matched], direct_targets[matched]
        intersection = np.logical_and(prediction, truth).sum(axis=(1, 2), dtype=np.float64)
        union = np.logical_or(prediction, truth).sum(axis=(1, 2), dtype=np.float64)
        output[matched] = np.where(union > 0, intersection / union, 1.0)
    for index in np.flatnonzero(~matched):
        prediction = hard[index]
        candidates = all_masks
        if not candidates:
            output[index] = 0.0
            continue
        values = []
        for truth in candidates:
            intersection = np.logical_and(prediction, truth).sum(dtype=np.float64)
            union = np.logical_or(prediction, truth).sum(dtype=np.float64)
            values.append(intersection / union if union else 1.0)
        output[index] = max(values)
    return output


def _append(path: CandidatePath, group: dict[str, Any], logits: torch.Tensor, predicted_iou: torch.Tensor) -> None:
    box = tuple(int(value) for value in group["crop_box_xyxy"])
    shape = tuple(int(value) for value in group["ori_shape_hw"])
    points = torch.from_numpy(np.asarray(group["coordinates_local_xy"], dtype=np.float32)).unsqueeze(1)
    records = mask_process_eval(np.asarray(group["classes"]), np.asarray(group["prompt_ids"]), box, shape, points, logits.cpu(), predicted_iou.cpu())
    path.candidate_count += len(records)
    _append_path_masks(path, records, box, shape)


def _one_hot_logit_score(predicted_iou: torch.Tensor, selection: np.ndarray) -> torch.Tensor:
    index = torch.as_tensor(selection, dtype=torch.long)
    return predicted_iou[torch.arange(len(index)), index]


def _binary_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    a, b = first.astype(np.float64).reshape(-1), second.astype(np.float64).reshape(-1)
    denominator = np.sqrt(a.mean() * (1 - a.mean()) * b.mean() * (1 - b.mean()))
    return float((np.mean(a * b) - a.mean() * b.mean()) / denominator) if denominator > 0 else None


def _error_complement(item, group: dict[str, Any], token_logits: torch.Tensor) -> ErrorAccumulator:
    hard = (token_logits > 0).detach().cpu().numpy()
    result = ErrorAccumulator()
    for index in np.flatnonzero(np.asarray(group["matched"], dtype=bool)):
        truth, _ = _local_target(item, group, int(index))
        if truth is None:
            continue
        masks = hard[index, :, : truth.shape[0], : truth.shape[1]]
        fn = np.logical_and(truth, ~masks)
        fp = np.logical_and(~truth, masks)
        fn0, fp0 = fn[0], fp[0]
        recoverable = np.logical_and(fn0, masks[1:].any(axis=0))
        rejectable = np.logical_and(fp0, (~masks[1:]).any(axis=0))
        boundary = np.logical_or(distance_transform_edt(truth) <= 3, distance_transform_edt(~truth) <= 3)
        result.fn0 += int(fn0.sum()); result.fn_recoverable += int(recoverable.sum())
        result.fp0 += int(fp0.sum()); result.fp_rejectable += int(rejectable.sum())
        result.boundary_recoverable += int(np.logical_and(recoverable, boundary).sum()); result.interior_recoverable += int(np.logical_and(recoverable, ~boundary).sum())
        result.boundary_rejectable += int(np.logical_and(rejectable, boundary).sum()); result.interior_rejectable += int(np.logical_and(rejectable, ~boundary).sum())
        result.simultaneous_fn_fp_improvement += int(any(fn[token].sum() < fn0.sum() for token in range(1, 4)) and any(fp[token].sum() < fp0.sum() for token in range(1, 4)))
        result.prompt_count += 1
        for left in range(4):
            for right in range(left + 1, 4):
                key = f"{left}_{right}"
                correlation = _binary_correlation(fn[left], fn[right]); fp_correlation = _binary_correlation(fp[left], fp[right])
                if correlation is not None: result.correlations[key].append(correlation)
                if fp_correlation is not None: result.fp_correlations[key].append(fp_correlation)
    return result


def _aggregate_path_metrics(paths: dict[str, dict[str, CandidatePath]], items: dict[str, Any], split: str) -> list[dict[str, Any]]:
    rows = []
    for image_id, image_paths in paths.items():
        for name, path in image_paths.items():
            prediction = _assemble_instance_map(path.boxes, path.scores, path.masks, path.inds, items[image_id].instance_map.shape, 0.5)
            rows.append({"split": split, "image_id": image_id, "path": name, "candidate_mask_count": path.candidate_count, "assembly_rejected_mask_count": path.candidate_count - len(path.masks), **assembly_metrics(items[image_id].instance_map, prediction)})
    return rows


def _path_mean(rows: list[dict[str, Any]], split: str, path: str) -> dict[str, float]:
    selected = [row for row in rows if row["split"] == split and row["path"] == path]
    names = ("dice", "aji", "aji_plus", "dq", "sq", "pq", "tp", "fp", "fn", "matched_iou_sum", "instance_count")
    return {name: float(np.mean([row[name] for row in selected])) for name in names}


def _rank_summary(blocks: list[dict[str, Any]], path: str, *, matched_only: bool = True) -> dict[str, Any]:
    truth = np.concatenate([block["truth"] for block in blocks])
    matched = np.concatenate([block["matched"] for block in blocks])
    selection = np.concatenate([block["selection"][path] for block in blocks])
    active = matched if matched_only else np.ones(len(matched), dtype=bool)
    return selector_metrics(selection[active], truth[active])


def _rank_rows(split: str, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for image_id in sorted({block["image_id"] for block in blocks} | {"__overall__"}):
        scoped = blocks if image_id == "__overall__" else [block for block in blocks if block["image_id"] == image_id]
        for path in TOKEN_PATHS:
            truth = np.concatenate([block["truth"] for block in scoped]); matched = np.concatenate([block["matched"] for block in scoped]); active = matched
            metric = selector_metrics(np.concatenate([block["selection"][path] for block in scoped])[active], truth[active])
            changed = changed_winner_metrics(np.concatenate([block["selection"]["existing_all_pred"] for block in scoped])[active], np.concatenate([block["selection"][path] for block in scoped])[active], truth[active])
            score_margin = np.concatenate([block["score_margin"].get(path, np.full(len(block["matched"]), np.nan)) for block in scoped])[active]
            gap = np.concatenate([block["truth_gap"] for block in scoped])[active]
            rows.append({"split": split, "image_id": image_id, "scope": "matched", "path": path, **{key: value for key, value in metric.items() if not isinstance(value, np.ndarray)}, **changed, "mean_score_margin": float(np.nanmean(score_margin)) if np.isfinite(score_margin).any() else None, "mean_true_best_vs_second_gap": float(gap.mean()) if len(gap) else None})
    return rows


def _oracle_gain_rows(split: str, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for image_id in sorted({block["image_id"] for block in blocks} | {"__overall__"}):
        scoped = blocks if image_id == "__overall__" else [block for block in blocks if block["image_id"] == image_id]
        truth, matched = np.concatenate([block["truth"] for block in scoped]), np.concatenate([block["matched"] for block in scoped])
        summary = oracle_gain_summary(truth[matched])
        rows.append({"split": split, "image_id": image_id, **summary})
    return rows


def _fixed_fusion_signal(per_image: list[dict[str, Any]]) -> dict[str, Any]:
    paths = []
    baseline = {row["image_id"]: row for row in per_image if row["split"] == "development" and row["path"] == "single"}
    for name in FIXED_FUSIONS:
        rows = [row for row in per_image if row["split"] == "development" and row["path"] == name]
        deltas = np.asarray([row["pq"] - baseline[row["image_id"]]["pq"] for row in rows])
        positive = np.maximum(deltas, 0)
        condition = {
            "delta_pq_ge_0_003": float(deltas.mean()) >= .003,
            "aji_not_decreased": float(np.mean([row["aji"] - baseline[row["image_id"]]["aji"] for row in rows])) >= 0,
            "five_of_seven_pq_non_decreasing": int(np.sum(deltas >= 0)) >= 5,
            "fp_not_higher_than_single": float(np.mean([row["fp"] for row in rows])) <= float(np.mean([baseline[row["image_id"]]["fp"] for row in rows])),
            "largest_positive_contribution_le_60pct": float(positive.max() / positive.sum()) <= .60 if positive.sum() else True,
        }
        paths.append({"path": name, "conditions": condition, "verdict": "YES" if all(condition.values()) else "NO", "mean_delta_pq": float(deltas.mean()), "pq_non_decreasing_images": int(np.sum(deltas >= 0))})
    return {"verdict": "YES" if any(path["verdict"] == "YES" for path in paths) else "NO", "paths": paths}


def _oracle_assembly_structure(per_image: list[dict[str, Any]], split: str) -> dict[str, Any]:
    baseline = {row["image_id"]: row for row in per_image if row["split"] == split and row["path"] == "single"}
    oracle = [row for row in per_image if row["split"] == split and row["path"] == "token_oracle"]
    delta = {row["image_id"]: float(row["pq"] - baseline[row["image_id"]]["pq"]) for row in oracle}
    positive = sum(max(0.0, value) for value in delta.values())
    return {"per_image_delta_pq": delta, "mean_delta_pq": float(np.mean(list(delta.values()))) if delta else 0.0, "largest_positive_image_contribution_fraction": max([max(0.0, value) for value in delta.values()], default=0.0) / positive if positive else 0.0}


def _learned_fusion_headroom(per_image: list[dict[str, Any]], convex_prompt_delta: dict[str, Any]) -> dict[str, Any]:
    token = {row["image_id"]: row for row in per_image if row["split"] == "development" and row["path"] == "token_oracle"}
    convex = [row for row in per_image if row["split"] == "development" and row["path"] == "convex_fusion_oracle"]
    deltas = np.asarray([row["pq"] - token[row["image_id"]]["pq"] for row in convex])
    positive = np.maximum(deltas, 0)
    condition = {
        "extra_delta_pq_ge_0_003": float(deltas.mean()) >= .003,
        "extra_mean_prompt_iou_ge_0_005": convex_prompt_delta["mean_extra_iou"] >= .005,
        "non_one_hot_optimal_fraction_ge_25pct": convex_prompt_delta["non_one_hot_fraction"] >= .25,
        "five_of_seven_not_below_token_oracle": int(np.sum(deltas >= 0)) >= 5,
        "largest_extra_gain_contribution_le_60pct": float(positive.max() / positive.sum()) <= .60 if positive.sum() else True,
    }
    return {"verdict": "YES" if all(condition.values()) else "NO", "conditions": condition, "mean_extra_delta_pq": float(deltas.mean()), "per_image_delta_pq": {row["image_id"]: float(row["pq"] - token[row["image_id"]]["pq"]) for row in convex}, **convex_prompt_delta}


def _process_split(*, split: str, cache_dir: Path, items: dict[str, Any], ranker: NuRankSharedRanker, device: torch.device, started: float) -> dict[str, Any]:
    paths = {image_id: {name: CandidatePath() for name in ALL_PATHS} for image_id in items}
    blocks, fixed_prompt_rows, error_by_image = [], [], {image_id: ErrorAccumulator() for image_id in items}
    convex_weights = simplex_weights().to(device)
    convex_non_one_hot, convex_extra, convex_count = 0, [], 0
    convex_by_image: dict[str, dict[str, Any]] = {image_id: {"extra": [], "non_one_hot": 0, "count": 0} for image_id in items}
    max_token0_upsample_error, max_cached_truth_error = 0.0, 0.0
    for group in iter_groups(cache_dir):
        _enforce_time(started)
        image_id = group["_entry"]["image_id"]; item = items[image_id]
        low = torch.from_numpy(np.asarray(group["low_res_logits"], dtype=np.float32)).to(device)
        upsampled = upsample_logits(low)
        max_token0_upsample_error = max(max_token0_upsample_error, float((upsampled[:, 0].cpu() - torch.from_numpy(np.asarray(group["mask_logits"], dtype=np.float32))[:, 0]).abs().max()))
        if max_token0_upsample_error > 1e-5:
            raise RuntimeError("Postmortem low-resolution token-0 upsampling does not reproduce cached baseline logits")
        predicted = torch.from_numpy(np.asarray(group["original_predicted_iou"], dtype=np.float32))
        truth = np.asarray(group["true_hard_iou"], dtype=np.float32); matched = np.asarray(group["matched"], dtype=bool)
        direct_targets, matched_targets, all_masks = _group_targets(item, group)
        if not np.array_equal(matched_targets, matched):
            raise RuntimeError("Postmortem GT association differs from frozen cache")
        max_cached_truth_error = max(max_cached_truth_error, float(np.abs(_candidate_iou(direct_targets, matched, all_masks, upsampled[:, 0]) - truth[:, 0]).max()))
        if max_cached_truth_error > 1e-5:
            raise RuntimeError("Postmortem GT IoU reconstruction differs from frozen cache")
        existing = np.asarray(group["original_predicted_iou"], dtype=np.float32).argmax(axis=1)
        with torch.no_grad():
            rank_scores = ranker(torch.from_numpy(group_feature_matrix(group)).to(device)).detach().cpu().numpy()
        rank_selection = rank_scores.argmax(axis=1)
        token_oracle = np.where(matched, truth.argmax(axis=1), 0).astype(np.int64)
        selection = {"single": np.zeros(len(truth), dtype=np.int64), "existing_all_pred": existing, "nurank": rank_selection, "token_oracle": token_oracle}
        _, truth_gap = score_and_truth_gap(np.asarray(group["original_predicted_iou"], dtype=np.float32), truth)
        existing_margin, _ = score_and_truth_gap(np.asarray(group["original_predicted_iou"], dtype=np.float32), truth)
        rank_margin, _ = score_and_truth_gap(rank_scores, truth)
        blocks.append({"image_id": image_id, "truth": truth, "matched": matched, "selection": selection, "score_margin": {"existing_all_pred": existing_margin, "nurank": rank_margin}, "truth_gap": truth_gap})
        for name in TOKEN_PATHS:
            chosen = torch.as_tensor(selection[name], dtype=torch.long)
            logits = upsampled[torch.arange(len(chosen), device=device), chosen]
            _append(paths[image_id][name], group, logits, _one_hot_logit_score(predicted, selection[name]))
        fusion_logits = fixed_fusions(low)
        for name, logits in fusion_logits.items():
            prompt_iou = _candidate_iou(direct_targets, matched, all_masks, logits)
            fixed_prompt_rows.extend({"split": split, "image_id": image_id, "crop_id": int(group["_entry"]["crop_id"]), "prompt_id": int(group["prompt_ids"][index]), "path": name, "matched": bool(matched[index]), "true_hard_iou": float(prompt_iou[index])} for index in range(len(prompt_iou)))
            _append(paths[image_id][name], group, logits, predicted[:, 0])
        raw_candidates: list[tuple[str, torch.Tensor, torch.Tensor]] = [(f"token_{token}", upsampled[:, token], predicted[:, token]) for token in range(4)] + [(name, logits, predicted[:, 0]) for name, logits in fusion_logits.items()]
        library_iou = np.full(len(truth), -np.inf, dtype=np.float32); library_logits = upsampled[:, 0].clone(); library_score = predicted[:, 0].clone()
        for _, logits, score in raw_candidates:
            current = _candidate_iou(direct_targets, matched, all_masks, logits)
            replace = matched & (current > library_iou)
            if replace.any():
                indices = torch.as_tensor(np.flatnonzero(replace), dtype=torch.long, device=device)
                library_logits[indices] = logits[indices]; library_score[indices.cpu()] = score[indices.cpu()]
                library_iou[replace] = current[replace]
        _append(paths[image_id]["fixed_library_oracle"], group, library_logits, library_score)
        convex_iou = np.full(len(truth), -np.inf, dtype=np.float32); convex_logits = upsampled[:, 0].clone(); convex_score = predicted[:, 0].clone(); convex_choice = np.zeros(len(truth), dtype=np.int64)
        for weight_index, weight in enumerate(convex_weights):
            candidate_low = (low * weight[None, :, None, None]).sum(dim=1)
            candidate = upsample_logits(candidate_low[:, None])[:, 0]
            current = _candidate_iou(direct_targets, matched, all_masks, candidate)
            replace = matched & (current > convex_iou)
            if replace.any():
                indices = torch.as_tensor(np.flatnonzero(replace), dtype=torch.long, device=device)
                convex_logits[indices] = candidate[indices]
                if is_one_hot(weight):
                    token = int(weight.argmax())
                    convex_score[indices.cpu()] = predicted[indices.cpu(), token]
                convex_iou[replace] = current[replace]; convex_choice[replace] = weight_index
        _append(paths[image_id]["convex_fusion_oracle"], group, convex_logits, convex_score)
        if matched.any():
            active = np.flatnonzero(matched)
            convex_count += len(active); convex_non_one_hot += sum(not is_one_hot(convex_weights[index]) for index in convex_choice[active])
            extra = (convex_iou[active] - truth[active, token_oracle[active]]).tolist()
            convex_extra.extend(extra); convex_by_image[image_id]["extra"].extend(extra)
            convex_by_image[image_id]["count"] += len(active); convex_by_image[image_id]["non_one_hot"] += sum(not is_one_hot(convex_weights[index]) for index in convex_choice[active])
        error_by_image[image_id].merge(_error_complement(item, group, upsampled))
    convex_summary = {"mean_extra_iou": float(np.mean(convex_extra)) if convex_extra else 0.0, "median_extra_iou": float(np.median(convex_extra)) if convex_extra else 0.0, "non_one_hot_fraction": convex_non_one_hot / convex_count if convex_count else 0.0, "matched_prompt_count": convex_count, "per_image": {image_id: {"mean_extra_iou": float(np.mean(value["extra"])) if value["extra"] else 0.0, "non_one_hot_fraction": value["non_one_hot"] / value["count"] if value["count"] else 0.0, "matched_prompt_count": value["count"]} for image_id, value in convex_by_image.items()}}
    return {"paths": paths, "blocks": blocks, "fixed_prompt_rows": fixed_prompt_rows, "error_by_image": error_by_image, "convex_prompt": convex_summary, "token0_upsampled_max_abs_error": max_token0_upsample_error, "cached_true_iou_max_abs_error": max_cached_truth_error}


def run_postmortem_fusion_audit(*, data_root: Path, checkpoint: Path, nurank_run_dir: Path, out_dir: Path, split_manifest_path: Path, config_path: Path, sam_config: str, device_name: str = "cuda") -> dict[str, Any]:
    """Run the sole authorized read-only Postmortem-A audit and create a new artifact."""
    if out_dir.exists():
        raise FileExistsError(f"Postmortem artifact directory must be new: {out_dir}")
    if sha256_file(checkpoint) != BASELINE_V1_TNBC_SHA256:
        raise ValueError("Postmortem-A requires the frozen baseline-v1 TNBC checkpoint")
    formal_report_path = nurank_run_dir / "report.json"
    formal_report = json.loads(formal_report_path.read_text(encoding="utf-8"))
    if formal_report.get("verdict", {}).get("verdict") != "NO-GO":
        raise RuntimeError("Postmortem-A requires the immutable formal NuRank-v1 NO-GO artifact")
    if formal_report.get("reproducibility", {}).get("checkpoint_sha256") != BASELINE_V1_TNBC_SHA256:
        raise RuntimeError("Formal NuRank artifact checkpoint does not match frozen baseline-v1")
    out_dir.mkdir(parents=True)
    started, device = time.perf_counter(), torch.device(device_name)
    _set_seed()
    environment = {"git_sha": _git_sha(), "python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "device": str(device), "seed": SEED, "checkpoint_sha256": sha256_file(checkpoint), "time_cap_seconds": TIME_CAP_SECONDS, "tta": False, "batch_size": 1, "allowed_patients": {"train": [1, 2, 3, 4, 5, 6], "development": [7, 8], "closed": [9, 10, 11]}}
    (out_dir / "environment.txt").write_text("\n".join(f"{key}={json.dumps(value, sort_keys=True)}" for key, value in environment.items()) + "\n", encoding="utf-8")
    try:
        caches, cache_status = _prepare_caches(nurank_run_dir=nurank_run_dir, out_dir=out_dir, data_root=data_root, split_manifest_path=split_manifest_path, checkpoint=checkpoint, config_path=config_path, sam_config=sam_config, device=device, started=started)
        ranker_checkpoint = nurank_run_dir / "training" / "nurank_epoch_030.pt"
        if sha256_file(ranker_checkpoint) != formal_report.get("reproducibility", {}).get("ranker_checkpoint_sha256"):
            raise RuntimeError("Formal NuRank ranker checkpoint checksum does not match its immutable report")
        ranker, ranker_payload = load_ranker_checkpoint(ranker_checkpoint, device)
        if ranker_payload.get("seed") != SEED:
            raise RuntimeError("Postmortem-A must replay the frozen seed-3407 formal NuRank ranker")
        ranker_checksum_before = module_state_sha256(ranker)
        split_results, per_image, rank_rows, gain_rows, fixed_rows, error_report = {}, [], [], [], [], {}
        for split, role in (("train", "train"), ("development", "development")):
            items = {item.image_id: item for item in resolve_nurank_images(data_root, split_manifest_path, role)}
            result = _process_split(split=split, cache_dir=caches[role], items=items, ranker=ranker, device=device, started=started)
            split_results[split] = result
            per_image.extend(_aggregate_path_metrics(result["paths"], items, split))
            rank_rows.extend(_rank_rows(split, result["blocks"]))
            gain_rows.extend(_oracle_gain_rows(split, result["blocks"]))
            fixed_rows.extend(result["fixed_prompt_rows"])
            error_report[split] = {"overall": ErrorAccumulator(), "per_image": {image_id: accumulator.as_dict() for image_id, accumulator in result["error_by_image"].items()}}
            for accumulator in result["error_by_image"].values(): error_report[split]["overall"].merge(accumulator)
            error_report[split]["overall"] = error_report[split]["overall"].as_dict()
        _enforce_time(started)
        _write_csv(out_dir / "ranker_failure_decomposition.csv", rank_rows)
        _write_csv(out_dir / "per_image_path_metrics.csv", per_image)
        _write_csv(out_dir / "token_gain_distribution.csv", gain_rows)
        for split in ("train", "development"):
            for path in FIXED_FUSIONS:
                scoped = [row["true_hard_iou"] for row in fixed_rows if row["split"] == split and row["path"] == path and row["matched"]]
                fixed_rows.append({"record_type": "matched_prompt_summary", "split": split, "path": path, "matched_prompt_mean_iou": float(np.mean(scoped)) if scoped else None, **{f"assembly_{name}": value for name, value in _path_mean(per_image, split, path).items()}})
        _write_csv(out_dir / "fixed_fusion_results.csv", fixed_rows)
        _write_json(out_dir / "error_complementarity.json", error_report)
        train_existing, train_nurank = _rank_summary(split_results["train"]["blocks"], "existing_all_pred"), _rank_summary(split_results["train"]["blocks"], "nurank")
        dev_existing, dev_nurank, dev_single = _rank_summary(split_results["development"]["blocks"], "existing_all_pred"), _rank_summary(split_results["development"]["blocks"], "nurank"), _rank_summary(split_results["development"]["blocks"], "single")
        summary = {split: {path: _path_mean(per_image, split, path) for path in ALL_PATHS} for split in ("train", "development")}
        dev_pq_delta = summary["development"]["nurank"]["pq"] - summary["development"]["single"]["pq"]
        failure = failure_mode(train_existing=train_existing, train_nurank=train_nurank, development_existing=dev_existing, development_nurank=dev_nurank, development_single=dev_single, development_pq_delta=dev_pq_delta)
        fixed_signal = _fixed_fusion_signal(per_image)
        learned_headroom = _learned_fusion_headroom(per_image, split_results["development"]["convex_prompt"])
        continuation = "CONDITIONAL GO" if fixed_signal["verdict"] == "YES" or learned_headroom["verdict"] == "YES" else "NO-GO"
        fusion_oracle = {"fixed_library_oracle": {split: summary[split]["fixed_library_oracle"] for split in ("train", "development")}, "convex_fusion_oracle": {split: summary[split]["convex_fusion_oracle"] for split in ("train", "development")}, "convex_vs_token_oracle": learned_headroom, "convex_prompt_structure": split_results["development"]["convex_prompt"]}
        _write_json(out_dir / "fusion_oracle_results.json", fusion_oracle)
        ranker_checksum_after = module_state_sha256(ranker)
        if ranker_checksum_before != ranker_checksum_after:
            raise RuntimeError("Frozen formal NuRank ranker changed during Postmortem-A")
        report = {"title": "REPORT FOR PROJECT LEAD — NUSET POSTMORTEM-A", "status": "read-only diagnostic complete; NuRank-v1 remains NO-GO", "environment": environment, "formal_nurank_artifact": str(nurank_run_dir), "cache_status": cache_status, "frozen_ranker_checksum": {"before": ranker_checksum_before, "after": ranker_checksum_after}, "failure_attribution": failure, "ranker_train_matched": {"existing": {key: value for key, value in train_existing.items() if not isinstance(value, np.ndarray)}, "nurank": {key: value for key, value in train_nurank.items() if not isinstance(value, np.ndarray)}}, "ranker_development_matched": {"single": {key: value for key, value in dev_single.items() if not isinstance(value, np.ndarray)}, "existing": {key: value for key, value in dev_existing.items() if not isinstance(value, np.ndarray)}, "nurank": {key: value for key, value in dev_nurank.items() if not isinstance(value, np.ndarray)}}, "oracle_gain_structure": gain_rows, "token_oracle_assembly_structure": {split: _oracle_assembly_structure(per_image, split) for split in ("train", "development")}, "path_summary": summary, "fixed_fusion_signal": fixed_signal, "fusion_oracle": fusion_oracle, "error_complementarity": error_report, "baseline_equivalence": {split: {"token0_upsampled_max_abs_error": result["token0_upsampled_max_abs_error"], "cached_true_iou_max_abs_error": result["cached_true_iou_max_abs_error"]} for split, result in split_results.items()}, "verdicts": {"FIXED_FUSION_SIGNAL": fixed_signal["verdict"], "LEARNED_FUSION_HEADROOM": learned_headroom["verdict"], "MULTIMASK_CONTINUE": continuation}, "recommendation": "Design baseline-anchored set fusion only if MULTIMASK_CONTINUE is CONDITIONAL GO; otherwise terminate the multimask route. No implementation is performed by this audit.", "elapsed_seconds": time.perf_counter() - started}
        with (out_dir / "tests.txt").open("w", encoding="utf-8") as handle:
            tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests/nuset", "-v"], cwd=Path(__file__).resolve().parents[2], stdout=handle, stderr=subprocess.STDOUT, text=True)
        if tests.returncode:
            raise RuntimeError("NuSet Postmortem-A unit tests failed; artifact is not finalized")
        _write_json(out_dir / "report.json", report)
        _checksums(out_dir)
        return report
    except PostmortemTimeCap as error:
        _write_json(out_dir / "report.json", {"title": "REPORT FOR PROJECT LEAD — NUSET POSTMORTEM-A", "status": "stopped: 45 minute cap reached", "error": str(error), "elapsed_seconds": time.perf_counter() - started})
        _checksums(out_dir)
        raise
