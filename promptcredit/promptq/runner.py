"""Authorized PromptQ scalar-isolation smoke and TNBC development runner.

This is deliberately a bounded, GPU-only research runner.  It never opens
TNBC patients 9--11, never invokes MoNuSeg, never trains inherited StainPMS or
SAM2 parameters, and refuses to reuse an artifact directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
from scipy.stats import spearmanr
import torch
from torch.utils.data import DataLoader

from promptcredit.audit.guardrails import BASELINE_V1_TNBC_SHA256, sha256_file
from promptcredit.audit.runner import _decode_standard, _load_models, _prepare_decoder_features
from promptcredit.audit.data import iter_selected_tnbc_crops
from promptcredit.method import (
    build_quality_targets,
    configure_promptq_trainable,
    frozen_parameters_have_no_grad,
    gather_nearest_coordinates,
    module_state_sha256,
    module_state_sha256_excluding,
    optimizer_excludes_frozen,
    quality_focal_loss,
)
from promptcredit.metrics.utility import score_utility_summary
from promptcredit.promptq.cache import extract_cache
from promptcredit.promptq.data import PromptQCrop, PromptQDevelopmentDataset, iter_promptq_crops, resolve_promptq_images
from promptcredit.promptq.training import LR, WEIGHT_DECAY, train_quality_head


SEED = 3407
SMOKE_STEPS = 100
TIME_LIMIT_SECONDS = 6 * 60 * 60
NMS_RADIUS = 12.0


class PromptQBudgetExceeded(RuntimeError):
    """Raised only after the pre-registered first-10%-of-cache time forecast."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: _jsonable(value) for key, value in row.items()} for row in rows])


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _hard_iou(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    hard = logits.detach() > 0
    truth = masks.detach().bool()
    intersection = (hard & truth).sum(dim=(1, 2)).float()
    union = (hard | truth).sum(dim=(1, 2)).float()
    return torch.where(union > 0, intersection / union, torch.ones_like(union))


def _crop_targets(crop: PromptQCrop, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(crop.gt_centroids_xy, dtype=torch.float32, device=device),
        torch.as_tensor(crop.gt_masks, dtype=torch.float32, device=device).contiguous(),
    )


def _module_max_abs_error(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    if left.keys() != right.keys():
        raise RuntimeError("model outputs do not have identical keys")
    errors = [float((left[key] - right[key]).abs().max().detach().cpu()) for key in left]
    return max(errors, default=0.0)


def _quality_logits_from_features(quality_head: torch.nn.Module, features: torch.Tensor) -> torch.Tensor:
    # The online PromptQ path deliberately quantizes detached features to
    # FP16 and restores FP32 before the head.  Cache and online paths share it.
    return quality_head(features.to(dtype=torch.float16).to(dtype=torch.float32)).reshape(-1)


def _nms_source_ids(
    *,
    coordinates: np.ndarray,
    point_logits: np.ndarray,
    semantic_logits: np.ndarray,
    quality_logits: np.ndarray | None,
    mode: str,
) -> np.ndarray:
    """Exact crop-local point-NMS source IDs; only rank score is selectable."""
    points = np.asarray(coordinates, dtype=np.float32).copy()
    logits = np.asarray(point_logits, dtype=np.float64)
    classes = logits.argmax(axis=-1)
    probability = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probability /= probability.sum(axis=-1, keepdims=True)
    foreground = probability[:, 0]
    if mode == "objectness":
        ranking = foreground
    elif mode == "objectness_x_quality":
        if quality_logits is None:
            raise ValueError("PromptQ product ranking requires quality logits")
        ranking = foreground * (1.0 / (1.0 + np.exp(-np.asarray(quality_logits, dtype=np.float64))))
    else:
        raise ValueError(f"PromptQ only authorizes objectness or product score, got {mode}")
    height, width = semantic_logits.shape
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    valid = classes < (logits.shape[-1] - 1)
    source = np.flatnonzero(valid)
    points, ranking = points[valid], ranking[valid]
    semantic = semantic_logits > 0
    if len(points):
        semantic_keep = semantic[points.astype(np.int64)[:, 1], points.astype(np.int64)[:, 0]]
        source, points, ranking = source[semantic_keep], points[semantic_keep], ranking[semantic_keep]
    if not len(points):
        return np.empty(0, dtype=np.int64)
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    retained = np.ones(len(points), dtype=bool)
    for index in np.argsort(-ranking):
        if retained[index]:
            retained[distances[index] <= NMS_RADIUS] = False
    return source[retained]


def _spearman(left: Iterable[float], right: Iterable[float]) -> float | None:
    first, second = np.asarray(list(left), dtype=np.float64), np.asarray(list(right), dtype=np.float64)
    if len(first) < 2 or np.all(first == first[0]) or np.all(second == second[0]):
        return None
    value = float(spearmanr(first, second).statistic)
    return value if np.isfinite(value) else None


def _select_scalar_smoke_crops(data_root: Path) -> list[PromptQCrop]:
    # Must match the corrected paired smoke's first fixed Stage-0 image and
    # its first two nucleus-containing unclockwise crops exactly.
    crops = list(iter_selected_tnbc_crops(data_root, ["02_1"]))
    if len(crops) < 2:
        raise RuntimeError("Fixed Stage-0 image 02_1 has fewer than two valid crops")
    return crops[:2]


def _collect_scalar_crop(
    bundle: Any, crop: PromptQCrop, context_bank: list[Any], *, require_quality_features: bool
) -> dict[str, Any]:
    image = crop.image.unsqueeze(0).to(bundle.device)
    centroids, masks = _crop_targets(crop, bundle.device)
    with torch.no_grad():
        output, _, _, _ = bundle.point_net(image)
        if require_quality_features and "quality_roi_features" not in output:
            raise RuntimeError("PromptQ requires detached exported ROI features")
        selection = gather_nearest_coordinates(output["pred_coords"].detach(), [centroids])
        image_embed, high_res = _prepare_decoder_features(bundle, image, context_bank, crop)
        decoded_logits, _ = _decode_standard(bundle, image_embed, high_res, selection.coordinates.detach())
        hard_iou = _hard_iou(decoded_logits, masks)
        if require_quality_features:
            targets = build_quality_targets(output["pred_quality_logits"], selection.source_indices, hard_iou)
    record = {
        "crop": crop,
        "coordinates": output["pred_coords"][0].detach().cpu(),
        "point_logits": output["pred_logits"][0].detach().cpu(),
        "semantic_logits": output["pred_masks"][0, 0].detach().cpu(),
        "decoded_logits": decoded_logits.detach().cpu(),
    }
    if not require_quality_features:
        return record
    source_hard_iou = torch.zeros_like(targets.values[0])
    for source, value in zip(
        selection.source_indices[0].detach().cpu().tolist(), hard_iou.detach().cpu().tolist(), strict=True
    ):
        source_hard_iou[source] = max(source_hard_iou[source], float(value))
    record.update(
        {
            "features": output["quality_roi_features"][0].detach().cpu().to(torch.float16),
            "utility": targets.values[0].detach().cpu(),
            "matched": targets.matched_proposals[0].detach().cpu(),
            "hard_iou": source_hard_iou,
        }
    )
    return record


def _collect_scalar_records(
    bundle: Any, crops: list[PromptQCrop], *, require_quality_features: bool = True
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    context_bank: list[Any] = []
    current_image = None
    for crop in crops:
        if crop.image_id != current_image:
            context_bank, current_image = [], crop.image_id
        records.append(_collect_scalar_crop(bundle, crop, context_bank, require_quality_features=require_quality_features))
    return records


def _scalar_summary(quality_head: torch.nn.Module, records: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    quality_head.eval()
    raw_scores: list[float] = []
    product_scores: list[float] = []
    utilities: list[float] = []
    quality_targets: list[float] = []
    quality_probabilities: list[float] = []
    all_quality_logits: list[np.ndarray] = []
    selected_ious: list[float] = []
    with torch.no_grad():
        for record in records:
            features = record["features"].to(device)
            logits = _quality_logits_from_features(quality_head, features).detach().cpu().numpy()
            all_quality_logits.append(logits)
            point_logits = record["point_logits"].numpy()
            foreground = torch.softmax(record["point_logits"], dim=-1)[:, 0].numpy()
            matched = record["matched"].numpy().astype(bool)
            hard_iou = record["hard_iou"].numpy()
            if matched.any():
                raw_scores.extend(foreground[matched].tolist())
                product_scores.extend((foreground[matched] / (1.0 + np.exp(-logits[matched]))).tolist())
                utilities.extend(hard_iou[matched].tolist())
                quality_targets.extend(record["utility"].numpy()[matched].tolist())
                quality_probabilities.extend((1.0 / (1.0 + np.exp(-logits[matched]))).tolist())
            kept = _nms_source_ids(
                coordinates=record["coordinates"].numpy(),
                point_logits=point_logits,
                semantic_logits=record["semantic_logits"].numpy(),
                quality_logits=logits,
                mode="objectness_x_quality",
            )
            selected = record["matched"].numpy()[kept]
            selected_ious.extend(record["hard_iou"].numpy()[kept][selected].tolist())
    raw_metrics = score_utility_summary(raw_scores, utilities)
    product_metrics = score_utility_summary(product_scores, utilities)
    return {
        "quality_target_spearman": _spearman(quality_probabilities, quality_targets),
        "raw_objectness": raw_metrics,
        "product_score": product_metrics,
        "product_mean_mask_iou_after_point_nms": float(np.mean(selected_ious)) if selected_ious else None,
        "quality_prediction_std": float(np.std(np.concatenate(all_quality_logits))) if all_quality_logits else 0.0,
        "finite": bool(
            np.isfinite(raw_scores).all() and np.isfinite(product_scores).all() and np.isfinite(utilities).all()
        ),
    }


@torch.no_grad()
def _scalar_quality_loss(quality_head: torch.nn.Module, records: list[dict[str, Any]], device: torch.device) -> float:
    quality_head.eval()
    losses: list[float] = []
    for record in records:
        logits = _quality_logits_from_features(quality_head, record["features"].to(device)).unsqueeze(0)
        targets = type("Targets", (), {
            "values": record["utility"].to(device).unsqueeze(0),
            "matched_proposals": record["matched"].to(device).unsqueeze(0),
            "matched_count": int(record["matched"].sum()),
            "duplicate_source_events": 0,
        })()
        losses.append(float(quality_focal_loss(logits, targets).detach().cpu()))
    return float(np.mean(losses))


def _scalar_smoke(
    *,
    data_root: Path,
    checkpoint: Path,
    config_path: Path,
    sam_config: str,
    device: torch.device,
    out_dir: Path,
) -> dict[str, Any]:
    """Run only the authorized two-crop, quality-head-only isolation smoke."""
    crops = _select_scalar_smoke_crops(data_root)
    crop_manifest = {
        "image_id": "02_1",
        "selection_rule": "same first two nucleus-containing unclockwise crops as corrected paired smoke",
        "crops": [
            {
                "crop_id": int(crop.crop_id),
                "crop_box_xyxy": list(crop.crop_box_xyxy),
                "image_crop_sha256": crop.image_crop_sha256,
                "gt_crop_sha256": crop.gt_crop_sha256,
            }
            for crop in crops
        ],
    }
    _write_json(out_dir / "scalar_smoke_crop_selection.json", crop_manifest)
    _set_seed()
    frozen_baseline = _load_models(config_path, sam_config, checkpoint, device, enable_quality_head=False)
    for parameter in list(frozen_baseline.point_net.parameters()) + list(frozen_baseline.net.parameters()):
        parameter.requires_grad_(False)
        parameter.grad = None
    _set_seed()
    promptq = _load_models(
        config_path,
        sam_config,
        checkpoint,
        device,
        enable_quality_head=True,
        quality_head_dropout=0.0,
        quality_head_without_dropout=True,
        detach_quality_features=True,
        quantize_quality_features_fp16=True,
        export_quality_features=True,
    )
    manifest = configure_promptq_trainable(promptq.point_net, promptq.net)
    if manifest["quality_head_parameter_count"] >= 100_000:
        raise RuntimeError("PromptQ quality head exceeds 0.1M parameters")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in promptq.point_net.parameters() if parameter.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY
    )
    if not optimizer_excludes_frozen(optimizer):
        raise RuntimeError("PromptQ optimizer includes a frozen parameter")
    promptq.point_net.eval()
    promptq.net.eval()
    initial_records = _collect_scalar_records(promptq, crops)
    baseline_records = _collect_scalar_records(frozen_baseline, crops, require_quality_features=False)
    baseline_errors = {
        "pred_coords_max_abs_error": max(
            float((left["coordinates"] - right["coordinates"]).abs().max())
            for left, right in zip(initial_records, baseline_records, strict=True)
        ),
        "pred_logits_max_abs_error": max(
            float((left["point_logits"] - right["point_logits"]).abs().max())
            for left, right in zip(initial_records, baseline_records, strict=True)
        ),
        "decoded_mask_logits_max_abs_error": max(
            float((left["decoded_logits"] - right["decoded_logits"]).abs().max())
            for left, right in zip(initial_records, baseline_records, strict=True)
        ),
    }
    if any(value != 0.0 for value in baseline_errors.values()):
        raise RuntimeError(f"PromptQ scalar smoke baseline equivalence failed: {baseline_errors}")
    original_before = module_state_sha256_excluding(promptq.point_net, ("quality_head.",))
    sam2_before = module_state_sha256(promptq.net)
    baseline_summary = _scalar_summary(promptq.point_net.quality_head, initial_records, device)
    qloss_start = _scalar_quality_loss(promptq.point_net.quality_head, initial_records, device)
    rows: list[dict[str, Any]] = []
    promptq.point_net.eval()
    promptq.point_net.quality_head.train()
    for step in range(1, SMOKE_STEPS + 1):
        record = initial_records[(step - 1) % len(initial_records)]
        _cuda_sync(device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = _quality_logits_from_features(promptq.point_net.quality_head, record["features"].to(device)).unsqueeze(0)
        targets = type("Targets", (), {
            "values": record["utility"].to(device).unsqueeze(0),
            "matched_proposals": record["matched"].to(device).unsqueeze(0),
            "matched_count": int(record["matched"].sum()),
            "duplicate_source_events": 0,
        })()
        loss = quality_focal_loss(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"PromptQ scalar smoke non-finite loss at step {step}")
        loss.backward()
        if not frozen_parameters_have_no_grad(promptq.net):
            raise RuntimeError("Frozen SAM2 received a PromptQ scalar-smoke gradient")
        frozen_point_grads = [
            parameter.grad for name, parameter in promptq.point_net.named_parameters() if not name.startswith("quality_head.")
        ]
        if any(gradient is not None for gradient in frozen_point_grads):
            raise RuntimeError("PromptQ scalar loss reached an inherited point-model parameter")
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in promptq.point_net.quality_head.parameters()
        ):
            raise FloatingPointError("PromptQ scalar smoke non-finite quality-head gradient")
        optimizer.step()
        _cuda_sync(device)
        if step % 10 == 0:
            snapshot = _scalar_summary(promptq.point_net.quality_head, initial_records, device)
            rows.append(
                {
                    "step": step,
                    "quality_loss": float(loss.detach().cpu()),
                    "quality_target_spearman": snapshot["quality_target_spearman"],
                    "raw_objectness_iou_spearman": snapshot["raw_objectness"]["spearman_point_score_vs_hard_iou"],
                    "product_score_iou_spearman": snapshot["product_score"]["spearman_point_score_vs_hard_iou"],
                    "product_mean_mask_iou_after_point_nms": snapshot["product_mean_mask_iou_after_point_nms"],
                    "quality_prediction_std": snapshot["quality_prediction_std"],
                    "step_seconds": time.perf_counter() - started,
                    "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                }
            )
    final_summary = _scalar_summary(promptq.point_net.quality_head, initial_records, device)
    post_records = _collect_scalar_records(promptq, crops)
    stable_original_errors = {
        "pred_coords_max_abs_error": max(float((start["coordinates"] - end["coordinates"]).abs().max()) for start, end in zip(initial_records, post_records, strict=True)),
        "pred_logits_max_abs_error": max(float((start["point_logits"] - end["point_logits"]).abs().max()) for start, end in zip(initial_records, post_records, strict=True)),
        "decoded_mask_logits_max_abs_error": max(float((start["decoded_logits"] - end["decoded_logits"]).abs().max()) for start, end in zip(initial_records, post_records, strict=True)),
    }
    original_after = module_state_sha256_excluding(promptq.point_net, ("quality_head.",))
    sam2_after = module_state_sha256(promptq.net)
    raw_spearman = final_summary["raw_objectness"]["spearman_point_score_vs_hard_iou"]
    product_spearman = final_summary["product_score"]["spearman_point_score_vs_hard_iou"]
    qloss_end = _scalar_quality_loss(promptq.point_net.quality_head, initial_records, device)
    checks = {
        "frozen_original_checksum_unchanged": original_before == original_after,
        "frozen_sam2_checksum_unchanged": sam2_before == sam2_after,
        "baseline_outputs_exact": all(value == 0.0 for value in baseline_errors.values()),
        "post_training_original_outputs_exact": all(value == 0.0 for value in stable_original_errors.values()),
        "quality_nontrivial": final_summary["quality_prediction_std"] > 1e-8,
        "quality_loss_decreased": qloss_end is not None and qloss_end < qloss_start,
        "quality_target_spearman_ge_0_60": (final_summary["quality_target_spearman"] or -np.inf) >= 0.60,
        "product_spearman_gain_ge_0_20": raw_spearman is not None and product_spearman is not None and product_spearman - raw_spearman >= 0.20,
        "product_nms_mean_iou_not_lower_than_objectness": (
            final_summary["product_mean_mask_iou_after_point_nms"] is not None
            and baseline_summary["product_mean_mask_iou_after_point_nms"] is not None
            and final_summary["product_mean_mask_iou_after_point_nms"] >= baseline_summary["product_mean_mask_iou_after_point_nms"]
        ),
        "finite": final_summary["finite"],
    }
    report = {
        "method": "PromptQ scalar-isolation smoke",
        "seed": SEED,
        "steps": SMOKE_STEPS,
        "optimizer": {"type": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY},
        "trainable_manifest": manifest,
        "prompt_credit_grad_scale": 0.0,
        "mask_loss_in_backward": False,
        "roi_features_detached": True,
        "quality_head_dropout": 0.0,
        "baseline_equivalence": baseline_errors,
        "post_training_original_output_equivalence": stable_original_errors,
        "checksums": {
            "original_point_before": original_before,
            "original_point_after": original_after,
            "sam2_before": sam2_before,
            "sam2_after": sam2_after,
        },
        "step0": baseline_summary,
        "step100": final_summary,
        "quality_loss_start_measurement": qloss_start,
        "quality_loss_end_measurement": qloss_end,
        "checks": checks,
        "recommendation": "PASS" if all(checks.values()) else "NO-GO",
    }
    _write_rows(out_dir / "scalar_isolation_curve.csv", rows)
    _write_json(out_dir / "scalar_isolation_report.json", report)
    return report


def _environment_payload(device: torch.device) -> str:
    return "\n".join(
        [
            f"git_sha={_git_sha()}",
            f"python={sys.version}",
            f"platform={platform.platform()}",
            f"torch={torch.__version__}",
            f"cuda_available={torch.cuda.is_available()}",
            f"device={device}",
            f"cuda_device={torch.cuda.get_device_name(device) if device.type == 'cuda' else ''}",
        ]
    ) + "\n"


def _artifact_checksums(out_dir: Path) -> None:
    rows: list[str] = []
    for path in sorted(path for path in out_dir.rglob("*") if path.is_file() and path.name != "SHA256SUMS"):
        rows.append(f"{sha256_file(path)}  {path.relative_to(out_dir).as_posix()}")
    (out_dir / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _append_log(out_dir: Path, message: str) -> None:
    with (out_dir / "stdout.log").open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def _make_eval_cfg() -> SimpleNamespace:
    # Fixed frozen baseline-v1 evaluation contract.  No threshold is tuned.
    return SimpleNamespace(
        crop_size=256,
        overlap=32,
        out_size=256,
        tta=False,
        texture=True,
        context=True,
        texture_memory_bank_size=64,
        context_memory_bank_size=100,
        context_atten_k=1,
        vis=False,
        dump_eval_artifacts_dir="",
        dump_baseline_masks_dir="",
        coverage_accumulate=False,
        prompt_score_mode="objectness",
    )


def _metric_dict(metrics: tuple[float, ...]) -> dict[str, float]:
    names = ("dice1", "dice2", "aji", "aji_plus", "dq", "sq", "pq")
    result = {name: float(value) for name, value in zip(names, metrics, strict=True)}
    # The repository historically reports Dice1 and Dice2.  Keep both and
    # provide the requested unambiguous ``dice`` alias for the first standard
    # Dice aggregate instead of silently replacing either legacy metric.
    result["dice"] = result["dice1"]
    return result


def _candidate_key(record: dict[str, Any]) -> tuple[int, int, float, float]:
    return (
        int(record["crop_id"]),
        int(record["proposal_index"]),
        round(float(record["point"][0]), 5),
        round(float(record["point"][1]), 5),
    )


def _conflict_components(points: np.ndarray, radius: float = NMS_RADIUS) -> list[list[int]]:
    if len(points) < 2:
        return []
    pending = set(range(len(points)))
    components: list[list[int]] = []
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    while pending:
        seed = pending.pop()
        component, stack = {seed}, [seed]
        while stack:
            index = stack.pop()
            neighbors = set(np.flatnonzero((distances[index] <= radius) & (distances[index] > 0)).tolist()) & pending
            pending -= neighbors
            component |= neighbors
            stack.extend(neighbors)
        if len(component) > 1:
            components.append(sorted(component))
    return components


def _utility_by_candidate(details: dict[str, Any]) -> dict[tuple[int, int, float, float], float]:
    grouped: dict[tuple[int, int, float, float], list[float]] = {}
    for record in details.get("decoded_prompt_records", []):
        grouped.setdefault(_candidate_key(record), []).append(float(record["decoded_hard_mask_iou"]))
    # A prompt can be decoded in overlapping standard crops.  Maximum observed
    # standard-path IoU is reported explicitly; no extra decode is performed.
    return {key: max(values) for key, values in grouped.items()}


def _nms_conflict_analysis(
    baseline_details: list[dict[str, Any]], promptq_details: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_baseline = {str(item["image_id"]): item for item in baseline_details}
    by_promptq = {str(item["image_id"]): item for item in promptq_details}
    if by_baseline.keys() != by_promptq.keys():
        raise RuntimeError("baseline and PromptQ development image sets differ")
    rows: list[dict[str, Any]] = []
    changed = improved = declined = tied = comparable = 0
    for image_id in sorted(by_baseline):
        base, promptq = by_baseline[image_id], by_promptq[image_id]
        base_candidates, promptq_candidates = base["point_nms_candidates"], promptq["point_nms_candidates"]
        base_keys = [_candidate_key(item) for item in base_candidates]
        promptq_keys = [_candidate_key(item) for item in promptq_candidates]
        if base_keys != promptq_keys:
            raise RuntimeError(f"PromptQ altered a non-ranking candidate on {image_id}")
        base_utility, promptq_utility = _utility_by_candidate(base), _utility_by_candidate(promptq)
        points = np.asarray([item["point"] for item in base_candidates], dtype=np.float64)
        for group_id, indices in enumerate(_conflict_components(points)):
            baseline_winner = max(indices, key=lambda index: float(base_candidates[index]["objectness_score"]))
            promptq_winner = max(indices, key=lambda index: float(promptq_candidates[index]["ranking_score"]))
            key_base, key_promptq = base_keys[baseline_winner], base_keys[promptq_winner]
            was_changed = baseline_winner != promptq_winner
            baseline_iou = base_utility.get(key_base)
            promptq_iou = promptq_utility.get(key_promptq)
            relation = "unchanged" if not was_changed else "not_comparable"
            if was_changed:
                changed += 1
                if baseline_iou is not None and promptq_iou is not None:
                    comparable += 1
                    if promptq_iou > baseline_iou:
                        relation, improved = "improved", improved + 1
                    elif promptq_iou < baseline_iou:
                        relation, declined = "declined", declined + 1
                    else:
                        relation, tied = "tied", tied + 1
            rows.append(
                {
                    "image_id": image_id,
                    "conflict_group_id": group_id,
                    "group_size": len(indices),
                    "baseline_winner": _jsonable(key_base),
                    "promptq_winner": _jsonable(key_promptq),
                    "winner_changed": was_changed,
                    "baseline_winner_max_standard_path_hard_iou": baseline_iou,
                    "promptq_winner_max_standard_path_hard_iou": promptq_iou,
                    "utility_relation": relation,
                }
            )
    return rows, {
        "nms_conflict_groups": len(rows),
        "winner_changed_conflict_groups": changed,
        "winner_change_comparable_groups": comparable,
        "winner_utility_improved": improved,
        "winner_utility_declined": declined,
        "winner_utility_tied": tied,
        "winner_utility_improved_fraction_among_comparable_changes": improved / comparable if comparable else None,
        "winner_utility_declined_fraction_among_comparable_changes": declined / comparable if comparable else None,
        "utility_source": "maximum hard IoU among already-decoded standard-path appearances; no extra decoder calls",
    }


def _paired_bootstrap(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    deltas = np.asarray([float(row[f"promptq_{metric}"]) - float(row[f"baseline_{metric}"]) for row in rows], dtype=np.float64)
    if len(deltas) != 7:
        raise RuntimeError(f"PromptQ development must contain exactly 7 images, found {len(deltas)}")
    rng = np.random.default_rng(SEED)
    samples = np.asarray([rng.choice(deltas, size=len(deltas), replace=True).mean() for _ in range(2000)])
    largest_index = int(np.argmax(np.abs(deltas)))
    total_abs = float(np.abs(deltas).sum())
    return {
        "metric": metric,
        "seed": SEED,
        "resamples": 2000,
        "mean_difference": float(deltas.mean()),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "positive_image_fraction": float((deltas > 0).mean()),
        "negative_image_fraction": float((deltas < 0).mean()),
        "largest_image_id": rows[largest_index]["image_id"],
        "largest_image_contribution_fraction": float(abs(deltas[largest_index]) / total_abs) if total_abs else 0.0,
    }


def _run_development_inference(
    *,
    data_root: Path,
    split_manifest_path: Path,
    config_path: Path,
    checkpoint: Path,
    sam_config: str,
    device: torch.device,
    quality_head_state: dict[str, torch.Tensor],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    from mmengine.config import Config
    from run.run_on_epoch import validation_on_epoch

    args_config = Config.fromfile(str(config_path))
    dataset = PromptQDevelopmentDataset(data_root, split_manifest_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    cfg = _make_eval_cfg()
    args = SimpleNamespace(test=SimpleNamespace(filtering=bool(args_config.test.filtering), nms_thr=NMS_RADIUS))
    _set_seed()
    baseline = _load_models(config_path, sam_config, checkpoint, device, enable_quality_head=False)
    _set_seed()
    promptq = _load_models(
        config_path, sam_config, checkpoint, device, enable_quality_head=True,
        quality_head_dropout=0.0, quality_head_without_dropout=True,
        detach_quality_features=True, quantize_quality_features_fp16=True,
    )
    manifest = configure_promptq_trainable(promptq.point_net, promptq.net)
    promptq.point_net.quality_head.load_state_dict(quality_head_state, strict=True)
    baseline.point_net.eval()
    baseline.net.eval()
    promptq.point_net.eval()
    promptq.net.eval()
    baseline_original_checksum = module_state_sha256(baseline.point_net)
    promptq_original_before = module_state_sha256_excluding(promptq.point_net, ("quality_head.",))
    promptq_sam_before = module_state_sha256(promptq.net)
    torch.cuda.reset_peak_memory_stats(device)
    cfg.prompt_score_mode = "objectness"
    _cuda_sync(device)
    started = time.perf_counter()
    base_metrics, base_details = validation_on_epoch(
        cfg, args, loader, 20, baseline.point_net, baseline.point_encoder, baseline.net,
        "unclockwise", float(args_config.data.post.iou_threshold), list(baseline.texture_memory_bank), device, return_details=True,
    )
    _cuda_sync(device)
    baseline_seconds, baseline_memory = time.perf_counter() - started, int(torch.cuda.max_memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    cfg.prompt_score_mode = "objectness_x_quality"
    _cuda_sync(device)
    started = time.perf_counter()
    promptq_metrics, promptq_details = validation_on_epoch(
        cfg, args, loader, 20, promptq.point_net, promptq.point_encoder, promptq.net,
        "unclockwise", float(args_config.data.post.iou_threshold), list(promptq.texture_memory_bank), device, return_details=True,
    )
    _cuda_sync(device)
    promptq_seconds, promptq_memory = time.perf_counter() - started, int(torch.cuda.max_memory_allocated(device))
    if baseline_original_checksum != module_state_sha256(baseline.point_net):
        raise RuntimeError("Frozen baseline point model changed during inference")
    if promptq_original_before != module_state_sha256_excluding(promptq.point_net, ("quality_head.",)):
        raise RuntimeError("PromptQ inherited point model changed during inference")
    if promptq_sam_before != module_state_sha256(promptq.net):
        raise RuntimeError("PromptQ SAM2 model changed during inference")
    base_by_id = {str(item["image_id"]): item for item in base_details}
    promptq_by_id = {str(item["image_id"]): item for item in promptq_details}
    rows: list[dict[str, Any]] = []
    for image_id in sorted(base_by_id):
        base, q = base_by_id[image_id], promptq_by_id[image_id]
        if base["metrics"] is None or q["metrics"] is None:
            raise RuntimeError(f"Development image {image_id} did not yield valid segmentation metrics")
        row: dict[str, Any] = {"image_id": image_id}
        for metric, value in base["metrics"].items():
            row[f"baseline_{metric}"] = value
        for metric, value in q["metrics"].items():
            row[f"promptq_{metric}"] = value
            row[f"delta_{metric}"] = value - base["metrics"][metric]
        row.update(
            {
                "baseline_prompts_before_nms": int(base["prompts_before_point_nms"]),
                "promptq_prompts_before_nms": int(q["prompts_before_point_nms"]),
                "baseline_prompts_after_nms": int(base["prompts_after_point_nms"]),
                "promptq_prompts_after_nms": int(q["prompts_after_point_nms"]),
                "baseline_masks_decoded": int(base["masks_decoded"]),
                "promptq_masks_decoded": int(q["masks_decoded"]),
            }
        )
        rows.append(row)
    conflict_rows, conflict_summary = _nms_conflict_analysis(base_details, promptq_details)
    decoded = [record for detail in promptq_details for record in detail.get("decoded_prompt_records", [])]
    calibration = {
        "n_decoded_standard_path_prompts": len(decoded),
        "raw_objectness": score_utility_summary(
            [record["objectness_score"] for record in decoded], [record["decoded_hard_mask_iou"] for record in decoded]
        ),
        "quality": score_utility_summary(
            [record["quality_score"] for record in decoded], [record["decoded_hard_mask_iou"] for record in decoded]
        ),
        "product": score_utility_summary(
            [record["ranking_score"] for record in decoded], [record["decoded_hard_mask_iou"] for record in decoded]
        ),
    }
    runtime = {
        "baseline_seconds": baseline_seconds,
        "promptq_seconds": promptq_seconds,
        "runtime_ratio": promptq_seconds / max(baseline_seconds, np.finfo(float).eps),
        "baseline_peak_gpu_memory_bytes": baseline_memory,
        "promptq_peak_gpu_memory_bytes": promptq_memory,
        "mask_decoder_call_count_equal_per_image": all(
            row["baseline_masks_decoded"] == row["promptq_masks_decoded"] for row in rows
        ),
        "frozen_checksums": {
            "baseline_point_unchanged": True,
            "promptq_inherited_point_unchanged": True,
            "promptq_sam2_unchanged": True,
        },
        "trainable_manifest": manifest,
    }
    return rows, _metric_dict(base_metrics), _metric_dict(promptq_metrics), conflict_rows, {
        "conflicts": conflict_summary,
        "calibration": calibration,
        "runtime": runtime,
    }


def _development_verdict(
    *,
    rows: list[dict[str, Any]],
    calibration: dict[str, Any],
    conflicts: dict[str, Any],
    runtime: dict[str, Any],
) -> tuple[str, dict[str, bool]]:
    pq_bootstrap = _paired_bootstrap(rows, "pq")
    aji_mean = float(np.mean([row["delta_aji"] for row in rows]))
    pq_mean = pq_bootstrap["mean_difference"]
    raw = calibration["raw_objectness"]["spearman_point_score_vs_hard_iou"]
    product = calibration["product"]["spearman_point_score_vs_hard_iou"]
    improved = conflicts["winner_utility_improved"]
    declined = conflicts["winner_utility_declined"]
    checks = {
        "pq_delta_ge_0_003": pq_mean >= 0.003,
        "aji_not_negative": aji_mean >= 0.0,
        "at_least_4_of_7_pq_non_decreasing": sum(row["delta_pq"] >= 0.0 for row in rows) >= 4,
        "largest_image_contribution_le_60pct": pq_bootstrap["largest_image_contribution_fraction"] <= 0.60,
        "product_iou_spearman_gain_ge_0_20": raw is not None and product is not None and product - raw >= 0.20,
        "changed_winner_utility_improves_over_declines": improved > declined,
        "mask_decoder_calls_unchanged": bool(runtime["mask_decoder_call_count_equal_per_image"]),
        "runtime_overhead_le_5pct": runtime["runtime_ratio"] <= 1.05,
    }
    no_go = bool(
        pq_mean < 0.001
        or aji_mean < 0.0
        or conflicts["winner_changed_conflict_groups"] == 0
        or (conflicts["winner_change_comparable_groups"] > 0 and abs(improved - declined) <= 1)
        or pq_bootstrap["largest_image_contribution_fraction"] > 0.60
        or not checks["mask_decoder_calls_unchanged"]
        or not checks["runtime_overhead_le_5pct"]
    )
    if all(checks.values()):
        verdict = "GO"
    elif no_go:
        verdict = "NO-GO"
    else:
        verdict = "CONDITIONAL"
    return verdict, {**checks, "no_go_rule_triggered": no_go, "pq_bootstrap": pq_bootstrap}


def _cache_progress_guard(total_units: int):
    threshold = max(1, math.ceil(total_units / 10))

    def callback(completed: int, elapsed: float) -> None:
        if completed < threshold:
            return
        projected = elapsed / completed * total_units
        if projected > TIME_LIMIT_SECONDS:
            raise PromptQBudgetExceeded(
                f"PromptQ projected {projected / 3600:.2f} GPU hours from first 10% cache extraction, over 6 hour cap"
            )

    return callback


def _cache_online_logit_error(bundle: Any, crop: PromptQCrop, cache_path: Path) -> float:
    with np.load(cache_path, allow_pickle=False) as cached:
        features = torch.from_numpy(cached["features"].astype(np.float16)).to(bundle.device)
    bundle.point_net.eval()
    with torch.no_grad():
        online, _, _, _ = bundle.point_net(crop.image.unsqueeze(0).to(bundle.device))
        cached_logits = _quality_logits_from_features(bundle.point_net.quality_head, features)
        online_logits = online["pred_quality_logits"][0]
    return float((online_logits - cached_logits).abs().max().detach().cpu())


def run_promptq_development(
    *,
    data_root: Path,
    split_manifest_path: Path,
    checkpoint: Path,
    config_path: Path,
    sam_config: str,
    out_dir: Path,
    device_name: str,
) -> dict[str, Any]:
    """Run the authorized PromptQ scalar-only development protocol once."""
    if device_name != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("PromptQ is GPU-only; run the documented command on AutoDL 4090")
    if out_dir.exists():
        raise FileExistsError(f"PromptQ artifact directory already exists: {out_dir}")
    if data_root.name.lower() != "tnbc":
        raise ValueError("PromptQ authorizes TNBC only; MoNuSeg is prohibited")
    if sha256_file(checkpoint) != BASELINE_V1_TNBC_SHA256:
        raise ValueError("PromptQ requires the frozen TNBC StainPMS baseline v1 checkpoint")
    # Resolve the two explicit authorized roles before model construction.  The
    # resolver never enumerates/open patients 9--11.
    train_images = resolve_promptq_images(data_root, split_manifest_path, "train")
    development_images = resolve_promptq_images(data_root, split_manifest_path, "development")
    if len(development_images) != 7:
        raise RuntimeError(f"PromptQ development must have seven fixed images, got {len(development_images)}")
    out_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    _set_seed()
    torch.cuda.reset_peak_memory_stats(device)
    _append_log(out_dir, "PromptQ run created; immutable PromptCredit-v1 artifacts are not opened for writing.")
    (out_dir / "tests.txt").write_text("Run: python -m unittest discover -s tests/promptcredit -v\n", encoding="utf-8")
    _write_json(
        out_dir / "manifest.json",
        {
            "schema_version": 1,
            "method": "PromptQ: Frozen-Model Pre-Decode Mask-Utility Distillation",
            "git_sha": _git_sha(),
            "seed": SEED,
            "authorized_scope": "TNBC train patients 1-6 and development patients 7-8 only",
            "prohibited": ["TNBC patients 9-11", "MoNuSeg", "Directional Credit", "StainRoute", "threshold tuning", "second seed"],
            "output_contract": "non-overwritable artifact directory",
        },
    )
    (out_dir / "environment.txt").write_text(_environment_payload(device), encoding="utf-8")
    _write_json(
        out_dir / "checkpoint_manifest.json",
        {"path": str(checkpoint), "sha256": BASELINE_V1_TNBC_SHA256, "git_sha": _git_sha()},
    )
    _write_json(
        out_dir / "data_split_manifest.json",
        {
            "train_role": {"patients": [1, 2, 3, 4, 5, 6], "image_ids": [item.image_id for item in train_images]},
            "development_role": {"patients": [7, 8], "image_ids": [item.image_id for item in development_images]},
            "test_role": "closed; patients 9-11 were not enumerated or opened",
            "development_caveat": "patients 7-8 may have been seen by StainPMS initialization; not independent leakage-free validation",
        },
    )
    smoke_dir = out_dir / "scalar_isolation_smoke"
    smoke_dir.mkdir()
    smoke = _scalar_smoke(
        data_root=data_root, checkpoint=checkpoint, config_path=config_path, sam_config=sam_config,
        device=device, out_dir=smoke_dir,
    )
    _append_log(out_dir, f"Scalar-Isolation Smoke finished: {smoke['recommendation']}")
    if smoke["recommendation"] != "PASS":
        report = {
            "recommendation": "NO-GO",
            "stopped_after": "Scalar-Isolation Smoke",
            "scalar_isolation_smoke": smoke,
            "reason": "Pre-registered scalar-isolation smoke did not pass; TNBC development was not entered.",
        }
        _write_json(out_dir / "report.json", report)
        _append_log(out_dir, "Stopped before TNBC development because scalar-isolation smoke was NO-GO.")
        _artifact_checksums(out_dir)
        return report
    # Rebuild from the original checkpoint after smoke; no smoke state may leak
    # into cache extraction or the fixed 20-epoch cache training.
    _set_seed()
    bundle = _load_models(
        config_path, sam_config, checkpoint, device, enable_quality_head=True,
        quality_head_dropout=0.0, quality_head_without_dropout=True,
        detach_quality_features=True, quantize_quality_features_fp16=True,
        export_quality_features=True,
    )
    trainable_manifest = configure_promptq_trainable(bundle.point_net, bundle.net)
    inherited_before = module_state_sha256_excluding(bundle.point_net, ("quality_head.",))
    sam2_before = module_state_sha256(bundle.net)
    train_crops = list(iter_promptq_crops(data_root, split_manifest_path, "train"))
    development_crops = list(iter_promptq_crops(data_root, split_manifest_path, "development"))
    total_units = len(train_crops) + len(development_crops) * 3
    train_cache_dir = out_dir / "train_cache"
    development_cache_dir = out_dir / "development_cache"
    try:
        train_cache = extract_cache(
            bundle=bundle, crops=iter(train_crops), out_dir=train_cache_dir, role="train",
            progress_callback=_cache_progress_guard(total_units),
        )
        development_cache = extract_cache(
            bundle=bundle, crops=iter(development_crops), out_dir=development_cache_dir, role="development",
            progress_callback=None,
        )
        _append_log(out_dir, "Train and development caches extracted within the six-GPU-hour forecast guard.")
    except PromptQBudgetExceeded as error:
        partial = {
            "recommendation": "NO-GO",
            "stopped_after": "cache extraction time-cap guard",
            "reason": str(error),
            "time_limit_gpu_hours": 6,
            "saved_partial_cache_files": [str(path.relative_to(out_dir)) for path in out_dir.rglob("*.npz")],
        }
        _write_json(out_dir / "report.json", partial)
        _append_log(out_dir, f"Stopped at the six-hour forecast guard: {error}")
        _artifact_checksums(out_dir)
        return partial
    _write_json(out_dir / "train_cache_manifest.json", train_cache)
    _write_json(out_dir / "development_cache_manifest.json", development_cache)
    if inherited_before != module_state_sha256_excluding(bundle.point_net, ("quality_head.",)):
        raise RuntimeError("Cache extraction changed inherited point parameters")
    if sam2_before != module_state_sha256(bundle.net):
        raise RuntimeError("Cache extraction changed SAM2 parameters")
    training_dir = out_dir / "quality_training"
    training = train_quality_head(
        quality_head=bundle.point_net.quality_head,
        train_manifest_path=train_cache_dir / "manifest.json",
        development_manifest_path=development_cache_dir / "manifest.json",
        out_dir=training_dir,
        device=device,
    )
    _append_log(out_dir, "Fixed 20-epoch quality-head cache training completed.")
    first_crop = development_crops[0]
    first_cache = development_cache_dir / development_cache["files"][0]["file"]
    online_logit_error = _cache_online_logit_error(bundle, first_crop, first_cache)
    if online_logit_error >= 1e-6:
        raise RuntimeError(f"cache-vs-online quality logit error {online_logit_error} is not < 1e-6")
    inherited_after_training = module_state_sha256_excluding(bundle.point_net, ("quality_head.",))
    sam2_after_training = module_state_sha256(bundle.net)
    if inherited_before != inherited_after_training or sam2_before != sam2_after_training:
        raise RuntimeError("PromptQ cache training changed a frozen inherited parameter")
    quality_state = {name: tensor.detach().cpu().clone() for name, tensor in bundle.point_net.quality_head.state_dict().items()}
    per_image_rows, base_metrics, promptq_metrics, conflict_rows, inference = _run_development_inference(
        data_root=data_root, split_manifest_path=split_manifest_path, config_path=config_path, checkpoint=checkpoint,
        sam_config=sam_config, device=device, quality_head_state=quality_state,
    )
    _write_rows(out_dir / "per_image_metrics.csv", per_image_rows)
    _write_rows(out_dir / "nms_conflict_analysis.csv", conflict_rows)
    _write_json(out_dir / "calibration_metrics.json", {"cache_training": training, "full_inference": inference["calibration"]})
    bootstrap = {metric: _paired_bootstrap(per_image_rows, metric) for metric in ("pq", "aji", "aji_p", "dq", "sq", "dice1", "dice2")}
    _write_json(out_dir / "bootstrap_summary.json", bootstrap)
    _write_rows(
        out_dir / "runtime_summary.csv",
        [{"path": "baseline", "seconds": inference["runtime"]["baseline_seconds"], "peak_gpu_memory_bytes": inference["runtime"]["baseline_peak_gpu_memory_bytes"]},
         {"path": "promptq", "seconds": inference["runtime"]["promptq_seconds"], "peak_gpu_memory_bytes": inference["runtime"]["promptq_peak_gpu_memory_bytes"], "runtime_ratio": inference["runtime"]["runtime_ratio"]}],
    )
    verdict, checks = _development_verdict(
        rows=per_image_rows, calibration=inference["calibration"], conflicts=inference["conflicts"], runtime=inference["runtime"],
    )
    report = {
        "method": "PromptQ: Frozen-Model Pre-Decode Mask-Utility Distillation",
        "recommendation": verdict,
        "promptcredit_v1_status": "FAIL retained; Directional Credit retired",
        "scalar_isolation_smoke": smoke,
        "trainable_manifest": trainable_manifest,
        "frozen_checksums": {
            "inherited_point_before": inherited_before,
            "inherited_point_after_training": inherited_after_training,
            "sam2_before": sam2_before,
            "sam2_after_training": sam2_after_training,
        },
        "cache_vs_online_quality_logit_max_abs_error": online_logit_error,
        "cache_training": training,
        "development_metrics": {"baseline": base_metrics, "promptq": promptq_metrics},
        "nms_conflict_analysis": inference["conflicts"],
        "runtime": inference["runtime"],
        "decision_checks": checks,
        "test_access": "closed; no patients 9-11 paths were enumerated or opened",
        "monuseg_access": "prohibited; no MoNuSeg path was opened",
        "next_action": "Stop and await project-lead decision; no test or MoNuSeg run is authorized.",
    }
    _write_json(out_dir / "report.json", report)
    _append_log(out_dir, f"Development inference completed: {verdict}. Stopping for project-lead decision.")
    _artifact_checksums(out_dir)
    return report
