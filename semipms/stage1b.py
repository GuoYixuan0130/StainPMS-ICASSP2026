"""Anchored SemiPMS Stage 1B: frozen static pseudo-centre experiment.

This is intentionally separate from the rejected online Stage-1 route.  It
uses the approved 720-step supervised anchor as a *fixed* teacher, builds one
GT-free cache, and then updates only the auto-point head for 240 more steps.
There is no EMA, no cache refresh, no pseudo-mask loss, and no hidden-train-GT
read anywhere in this module.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import json
import math
import random
import shutil
import sys
import time
import types
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from scipy.ndimage import label as connected_components
from skimage.morphology import disk
from scipy.ndimage import binary_dilation
from torch.utils.data import DataLoader

from run.dataset.monuseg import MONUSEG
from semipms.anchor import SOURCE_STAGE1_SHA, _records_from_manifest, _seed_everything, _state_hash
from semipms.guards import ImageRecord, sha256_file, write_json
from semipms.phase0 import (
    CANONICAL_BASELINE, StepBudgetReached, _candidate_feature_records, _csv,
    _environment, _git, _infer_standard, _legacy_helpers, _read_image, _run_tests,
    _runtime_config,
)
from semipms.residual import propose_residual_points, residual_evidence
from semipms.stage1b_protocol import (
    PROPOSAL_BUDGETS, TARGET_CALIBRATION_PRECISION, VIEW_MATCH_IOU,
    one_to_one_cross_view as _one_to_one_cross_view,
    select_rule_lopo as _select_rule_lopo,
)
from semipms.stage1 import (
    MASK_DUPLICATE_IOU, MIN_BASE_AREA, PseudoCropCursor, Stage1Optimizer,
    _aggregate_development, _evaluate_development, _filter_base_instances,
    _load_checkpoint, _make_labeled_loader, _mask_iou, _resolve_residual_masks,
)
from semipms.stage1_guards import Stage1AccessGuard


SEED = 3407
CONTINUATION_STEPS = 240
EVAL_STEPS = (0, 60, 120, 180, 240)
BASE_PSEUDO_WEIGHT = 0.10
RESIDUAL_PSEUDO_WEIGHT = 0.05
RESIDUAL_RAMP_STEPS = 120
CENTROID_SUPPRESSION_PX = 12.0


def _load_rule(path: Path) -> tuple[dict[str, float], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rule = payload.get("rule", payload)
    required = {
        "min_view_iou", "max_centroid_displacement", "min_area_stability",
        "min_h_occupancy", "min_boundary_stability", "max_pseudo_conflict",
    }
    if not isinstance(rule, dict) or set(rule) != required:
        raise PermissionError("--phase0-rule must be the complete frozen Phase-0 acceptance-rule JSON.")
    values = {name: float(rule[name]) for name in required}
    return values, {"path": str(path.resolve()), "sha256": sha256_file(path), "source_rule": values}


def _freeze_for_point_only(point_net: torch.nn.Module, net: torch.nn.Module) -> list[dict[str, Any]]:
    """Freeze every module except the explicit auto-point proposal heads."""
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    for name, parameter in point_net.named_parameters():
        parameter.requires_grad_(not (name.startswith("backbone.") or name.startswith("mask_head.")))
    # ``helpers.train`` calls .train() on both modules.  Keep all frozen
    # normalization/dropout modules fixed even though the parent point model
    # enters train mode for its actual auto-point heads.
    original_point_train = point_net.train
    def point_train(self, mode: bool = True):
        original_point_train(mode)
        self.backbone.eval(); self.mask_head.eval()
        return self
    point_net.train = types.MethodType(point_train, point_net)
    def frozen_train(self, mode: bool = True):
        torch.nn.Module.train(self, False)
        return self
    net.train = types.MethodType(frozen_train, net)
    groups = []
    for name, parameter in point_net.named_parameters():
        groups.append({"name": name, "numel": int(parameter.numel()), "trainable": bool(parameter.requires_grad)})
    if not any(item["trainable"] for item in groups):
        raise AssertionError("Stage-1B has no trainable auto-point parameters.")
    if any(parameter.requires_grad for parameter in net.parameters()):
        raise AssertionError("Stage-1B attempted to train a SAM2 parameter.")
    return groups


def _point_optimizer(point_net: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW([p for p in point_net.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)


def _frozen_state_hash(point_net: torch.nn.Module, net: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(point_net.state_dict().items()):
        if name.startswith("backbone.") or name.startswith("mask_head."):
            digest.update(name.encode("utf-8")); digest.update(np.asarray(tensor.detach().cpu()).tobytes())
    for name, tensor in sorted(net.state_dict().items()):
        digest.update(name.encode("utf-8")); digest.update(np.asarray(tensor.detach().cpu()).tobytes())
    return digest.hexdigest()


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    value = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            value += float(parameter.grad.detach().float().pow(2).sum().item())
    return math.sqrt(value)


def _pseudo_point_loss(sample: Mapping[str, Any], point_net: torch.nn.Module, device: torch.device, weight: float) -> tuple[torch.Tensor, int]:
    """Positive point/objectness loss only: no pseudo mask / Dice / BCE loss."""
    images = sample["image"].to(device)
    points = sample["points"].to(device).float().reshape(-1, 2)
    labels = sample["labels"].to(device).long().reshape(-1)
    if not len(points):
        return images.sum() * 0.0, 0
    outputs, _, _, _ = point_net(images)
    distances = torch.cdist(outputs["pred_coords"][0], points)
    # One target centre can supervise one proposal only.  Ties are resolved by
    # target order, which is deterministic in PseudoCropCursor.
    selected: list[int] = []
    used: set[int] = set()
    for target in range(len(points)):
        for proposal in torch.argsort(distances[:, target]).tolist():
            if proposal not in used:
                selected.append(int(proposal)); used.add(int(proposal)); break
    if not selected:
        return images.sum() * 0.0, 0
    index = torch.as_tensor(selected, dtype=torch.long, device=device)
    target = points[:len(selected)]
    target_labels = labels[:len(selected)]
    coordinate = F.smooth_l1_loss(outputs["pred_coords"][0, index], target)
    positive_objectness = F.cross_entropy(outputs["pred_logits"][0, index], target_labels)
    return (coordinate * 0.1 + positive_objectness) * float(weight), len(selected)


def _component_ids(evidence: np.ndarray) -> np.ndarray:
    # Match the proposal generator's lower peak range so every plausible
    # high-H candidate belongs to an H component before its one-per-component
    # suppression is enforced.
    cutoff = float(np.percentile(evidence, 60))
    return connected_components(evidence >= cutoff)[0].astype(np.int32)


def _centroid_suppress(base: np.ndarray, rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, list[dict[str, Any]], Counter]:
    """Mask NMS plus centre-distance suppression after view matching."""
    selected: list[dict[str, Any]] = []
    points: list[tuple[float, float]] = []
    stats: Counter = Counter()
    for row in sorted(rows, key=lambda item: (-float(item["evidence"]), int(item["candidate_index"]))):
        item = dict(row)
        if item.get("status") != "cross_view_matched":
            selected.append(item); continue
        if any(math.hypot(float(item["x"]) - x, float(item["y"]) - y) < CENTROID_SUPPRESSION_PX for x, y in points):
            item["cross_view_accepted"] = False; item["status"] = "residual_centroid_duplicate"; stats[item["status"]] += 1
        else:
            points.append((float(item["x"]), float(item["y"])))
            item["cross_view_accepted"] = True
        selected.append(item)
    residual, resolved, nms_stats = _resolve_residual_masks(base, selected, duplicate_iou=MASK_DUPLICATE_IOU)
    stats.update(nms_stats)
    return residual, resolved, stats


def _candidate_csv_rows(stem: str, patient: int, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        features = dict(row.get("features", {}))
        out.append({
            "image": stem, "patient": patient, "candidate_index": int(row["candidate_index"]),
            "x": float(row["x"]), "y": float(row["y"]), "evidence": float(row["evidence"]),
            "source": row["source"], "status": row.get("status", "unclassified"),
            "cross_view_accepted": bool(row.get("cross_view_accepted", False)),
            "base_max_iou": float(row.get("base_max_iou", 0.0)),
            "residual_max_iou": float(row.get("residual_max_iou", 0.0)),
            "h_component": int(row.get("h_component", 0)),
            **{f"feature_{name}": float(value) for name, value in features.items()},
        })
    return out


def _candidate_truth(row: Mapping[str, Any], gt: np.ndarray) -> bool:
    y, x = int(round(float(row["y"]))), int(round(float(row["x"])))
    if not (0 <= y < gt.shape[0] and 0 <= x < gt.shape[1]):
        return False
    instance = int(gt[y, x])
    return bool(instance and _mask_iou(np.asarray(row["mask"], bool), gt == instance) >= 0.5)


@torch.inference_mode()
def _calibration_candidates(records: Sequence[ImageRecord], point_net, point_encoder, net, cfg, device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        raw, normal = _read_image(record)
        teacher = _infer_standard(normal.to(device), point_net, point_encoder, net, [], cfg, device)
        evidence = residual_evidence(raw, teacher)
        candidates = propose_residual_points(evidence, max_candidates=max(PROPOSAL_BUDGETS))
        features = _candidate_feature_records(raw, normal.to(device), teacher, candidates, point_net, point_encoder, net, cfg, device)
        gt = sio.loadmat(record.label_path)["inst_map"].astype(np.int32)  # labelled images only
        for row in features:
            rows.append({"patient": record.patient, "image": record.stem, "is_true": _candidate_truth(row, gt), **row})
    return rows


@torch.inference_mode()
def _build_static_cache(artifact: Path, records: Sequence[ImageRecord], point_net, point_encoder, net, texture_memory, cfg, device, rule: Mapping[str, float], budget: int) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]]]:
    root = artifact / "static_pseudo_cache"; base_dir = root / "base"; residual_dir = root / "residual"
    root.mkdir(); base_dir.mkdir(); residual_dir.mkdir()
    stats_rows: list[dict[str, Any]] = []; candidate_rows: list[dict[str, Any]] = []
    for record in records:
        raw, normal = _read_image(record)
        teacher = _infer_standard(normal.to(device), point_net, point_encoder, net, list(texture_memory), cfg, device)
        base, small = _filter_base_instances(teacher)
        evidence = residual_evidence(raw, teacher)
        proposed = propose_residual_points(evidence, max_candidates=budget)
        rows = _candidate_feature_records(raw, normal.to(device), teacher, proposed, point_net, point_encoder, net, cfg, device)
        matched, matching_stats = _one_to_one_cross_view(rows, rule, _component_ids(evidence))
        residual, resolved, dedup_stats = _centroid_suppress(base, matched)
        # Every accepted pseudo instance contributes precisely one mask-derived centre.
        combined = base.copy(); combined[residual > 0] = residual[residual > 0] + int(base.max())
        sio.savemat(base_dir / f"{record.stem}.mat", {"inst_map": base.astype(np.int32)})
        sio.savemat(residual_dir / f"{record.stem}.mat", {"inst_map": residual.astype(np.int32)})
        sio.savemat(root / f"{record.stem}.mat", {"inst_map": combined.astype(np.int32)})
        candidate_rows.extend(_candidate_csv_rows(record.stem, record.patient, resolved))
        stats_rows.append({
            "image": record.stem, "patient": record.patient, "raw_candidates": len(rows),
            "base_instances": int(base.max()), "base_small_rejected": int(small),
            "accepted_residual_instances": int(residual.max()), "combined_instances": int(combined.max()),
            "pseudo_centres": int(combined.max()),
            **{f"match_{key}": int(value) for key, value in matching_stats.items()},
            **{f"dedup_{key}": int(value) for key, value in dedup_stats.items()},
        })
    _csv(root / "dedup_statistics.csv", stats_rows); _csv(root / "candidate_statistics.csv", candidate_rows)
    score_summary = {}
    for name in ("feature_stain_inverse_iou", "feature_geometric_inverse_iou", "feature_area_stability", "feature_boundary_stability", "feature_h_occupancy", "feature_centroid_displacement", "feature_pseudo_conflict"):
        values = np.asarray([float(row[name]) for row in candidate_rows if name in row and math.isfinite(float(row[name]))], dtype=float)
        score_summary[name] = {"n": int(len(values)), "mean": float(values.mean()) if len(values) else 0.0, "median": float(np.median(values)) if len(values) else 0.0, "p10": float(np.percentile(values, 10)) if len(values) else 0.0, "p90": float(np.percentile(values, 90)) if len(values) else 0.0}
    write_json(root / "cross_view_score_distribution.json", score_summary)
    write_json(root / "static_cache_manifest.json", {
        "frozen": True, "refresh": "forbidden", "uses_hidden_train_gt": False,
        "teacher": "720-step anchor standard deployment", "proposal_budget": budget,
        "acceptance_rule": dict(rule), "mask_nms_iou": MASK_DUPLICATE_IOU,
        "centroid_suppression_px": CENTROID_SUPPRESSION_PX,
        "cross_view_one_to_one_iou": VIEW_MATCH_IOU,
        "same_h_component_max_instances": 1, "residual_centres": "one per accepted residual mask",
    })
    return root, stats_rows, candidate_rows


class StaticPseudoProvider:
    def __init__(self, base_dir: Path | None, residual_dir: Path | None, records, data_root, cfg, args_cfg, point_net, device, *, residual: bool, clip_grad: float):
        self.point_net = point_net; self.device = device; self.residual = residual
        self.clip_grad = float(clip_grad)
        self.base = PseudoCropCursor(base_dir, records, data_root, cfg, args_cfg, "base") if base_dir else None
        self.residual_cursor = PseudoCropCursor(residual_dir, records, data_root, cfg, args_cfg, "residual") if residual_dir else None

    def before_step(self, step: int, point_parameters: Sequence[torch.nn.Parameter]) -> dict[str, float]:
        # Preserve the deterministic labelled crop/augmentation random stream.
        py, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            values = {"base_pseudo_loss": 0.0, "residual_pseudo_loss": 0.0, "base_instances": 0.0, "residual_instances": 0.0, "residual_ramp": 0.0, "supervised_gradient_norm": _gradient_norm(point_parameters), "base_gradient_norm": 0.0, "residual_gradient_norm": 0.0, "combined_gradient_norm_preclip": 0.0, "combined_gradient_norm_postclip": 0.0}
            if self.base is not None:
                sample = self.base.next_crop()
                if sample is not None:
                    before = [p.grad.detach().clone() if p.grad is not None else None for p in point_parameters]
                    loss, count = _pseudo_point_loss(sample, self.point_net, self.device, BASE_PSEUDO_WEIGHT); loss.backward()
                    values["base_pseudo_loss"] = float(loss.detach().cpu()); values["base_instances"] = float(count)
                    if not torch.isfinite(loss): raise FloatingPointError("Non-finite static base pseudo-point loss.")
                    values["base_gradient_norm"] = math.sqrt(sum(float(((p.grad - old) if old is not None else p.grad).detach().float().pow(2).sum()) for p, old in zip(point_parameters, before) if p.grad is not None))
            if self.residual and self.residual_cursor is not None:
                ramp = min(1.0, step / RESIDUAL_RAMP_STEPS); values["residual_ramp"] = ramp
                sample = self.residual_cursor.next_crop()
                if sample is not None and ramp:
                    before = [p.grad.detach().clone() if p.grad is not None else None for p in point_parameters]
                    loss, count = _pseudo_point_loss(sample, self.point_net, self.device, RESIDUAL_PSEUDO_WEIGHT * ramp); loss.backward()
                    values["residual_pseudo_loss"] = float(loss.detach().cpu()); values["residual_instances"] = float(count)
                    if not torch.isfinite(loss): raise FloatingPointError("Non-finite static residual pseudo-point loss.")
                    values["residual_gradient_norm"] = math.sqrt(sum(float(((p.grad - old) if old is not None else p.grad).detach().float().pow(2).sum()) for p, old in zip(point_parameters, before) if p.grad is not None))
            values["combined_gradient_norm_preclip"] = _gradient_norm(point_parameters)
            if self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(point_parameters, self.clip_grad)
            values["combined_gradient_norm_postclip"] = _gradient_norm(point_parameters)
            return values
        finally:
            random.setstate(py); np.random.set_state(np_state); torch.set_rng_state(torch_state)
            if cuda_state is not None: torch.cuda.set_rng_state_all(cuda_state)


def _save_point_head(path: Path, point_net, *, anchor: Path, step: int, method: str, texture_memory: Sequence[Any]) -> dict[str, Any]:
    payload = {"point_head": {name: tensor.detach().cpu() for name, tensor in point_net.state_dict().items() if not name.startswith("backbone.") and not name.startswith("mask_head.")}, "anchor_checkpoint": str(anchor), "anchor_sha256": sha256_file(anchor), "step": step, "method": method, "texture_memory_bank_list": list(texture_memory)}
    torch.save(payload, path)
    return {"path": str(path), "sha256": sha256_file(path), "step": step, "method": method}


def _run_path(artifact: Path, method: str, anchor: Path, args, data_root, labeled, development, cfg, device, base_dir: Path | None, residual_dir: Path | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    _seed_everything(SEED)
    args_cfg, point_net, point_encoder, net, _, texture, payload = _load_checkpoint(anchor, args, device, require_optimizer=False)
    trainable = _freeze_for_point_only(point_net, net)
    frozen_before = _frozen_state_hash(point_net, net)
    optimizer = _point_optimizer(point_net); parameters = [p for p in point_net.parameters() if p.requires_grad]
    helpers = _legacy_helpers(); loader = _make_labeled_loader(cfg, args_cfg, data_root, labeled, args.num_workers); criterion, _ = helpers.build_criterion(args_cfg, device)
    # Cache cursors read only artifact-owned pseudo maps for the 24 images.
    provider = StaticPseudoProvider(base_dir, residual_dir, args._unlabeled, data_root, cfg, args_cfg, point_net, device, residual=residual_dir is not None, clip_grad=float(cfg.clip_grad)) if base_dir else None
    access = Stage1AccessGuard(); access.freeze_training_configuration()
    rows: list[dict[str, Any]] = []; dev_rows: list[dict[str, Any]] = []; pseudo_rows: list[dict[str, Any]] = []; latest_texture = list(texture)
    checkpoints = artifact / "checkpoints"; checkpoints.mkdir(exist_ok=True)
    started = time.monotonic(); best: dict[str, Any] | None = None
    evaluation_seconds: dict[int, float] = {}
    def evaluate(step: int):
        nonlocal best
        eval_started = time.monotonic()
        result = _evaluate_development(development, access, point_net, point_encoder, net, latest_texture, cfg, device, method=method, step=step)
        evaluation_seconds[step] = time.monotonic() - eval_started
        dev_rows.extend(result); aggregate = [row for row in _aggregate_development(result) if row["level"] == "all"][0]
        if best is None or float(aggregate["pq"]) > float(best["pq"]):
            best = dict(aggregate); best["checkpoint"] = _save_point_head(checkpoints / f"{method.lower().replace('-', '_')}_best_step_{step:04d}.pth", point_net, anchor=anchor, step=step, method=method, texture_memory=latest_texture)
    evaluate(0)
    bounded: Stage1Optimizer | None = None
    def before(step: int):
        values = provider.before_step(step, parameters) if provider else {"base_pseudo_loss": 0.0, "residual_pseudo_loss": 0.0, "base_instances": 0.0, "residual_instances": 0.0, "residual_ramp": 0.0, "supervised_gradient_norm": _gradient_norm(parameters), "base_gradient_norm": 0.0, "residual_gradient_norm": 0.0, "combined_gradient_norm_preclip": _gradient_norm(parameters), "combined_gradient_norm_postclip": _gradient_norm(parameters)}
        pseudo_rows.append({"method": method, "optimizer_step": step, **values}); return values
    def after(step: int, texture_memory):
        nonlocal latest_texture
        latest_texture = list(texture_memory)
        if step in EVAL_STEPS[1:]: evaluate(step)
    bounded = Stage1Optimizer(optimizer, CONTINUATION_STEPS, before_step=before, after_step=after)
    epoch = 0
    while bounded.steps < CONTINUATION_STEPS:
        texture_memory: list[Any] = []; bounded.texture_memory_bank = texture_memory; started_epoch = time.monotonic(); complete = True
        try: log = helpers.train(cfg, point_net, point_encoder, net, loader, criterion, bounded, epoch, texture_memory, device)
        except StepBudgetReached: complete = False; log = {"partial_epoch": True}
        rows.append({"method": method, "epoch": epoch, "optimizer_steps_completed": bounded.steps, "epoch_completed": complete, "seconds": time.monotonic()-started_epoch, **{k: float(v) for k,v in log.items()}}); epoch += 1
    if bounded.steps != CONTINUATION_STEPS: raise AssertionError(f"{method} ended at {bounded.steps}, expected 240.")
    final_checkpoint = _save_point_head(checkpoints / f"{method.lower().replace('-', '_')}_final_step_0240.pth", point_net, anchor=anchor, step=240, method=method, texture_memory=latest_texture)
    frozen_after = _frozen_state_hash(point_net, net)
    if frozen_before != frozen_after:
        raise AssertionError(f"{method} modified a frozen encoder/decoder parameter or buffer.")
    return rows, dev_rows, pseudo_rows, {"method": method, "best_development": best, "final_checkpoint": final_checkpoint, "runtime_seconds": time.monotonic()-started, "evaluation_seconds": evaluation_seconds, "trainable_parameters": trainable, "frozen_state_sha256": frozen_after}


def run_stage1b(args: argparse.Namespace) -> Path:
    started = time.monotonic(); repo = Path(__file__).resolve().parents[1]
    if _git(repo, "merge-base", "HEAD", SOURCE_STAGE1_SHA) != SOURCE_STAGE1_SHA: raise PermissionError("Stage-1B must descend from original Stage-1 code.")
    anchor = Path(args.anchor_checkpoint).resolve(); anchor_manifest = Path(args.anchor_artifact).resolve() / "report.json"
    report = json.loads(anchor_manifest.read_text(encoding="utf-8"))
    if not report.get("anchor_eligible"): raise PermissionError("Stage-1B requires the eligible 720-step anchor.")
    if anchor.name != "supervised_stainpms20_step_0720.pth" or not anchor.is_file(): raise PermissionError("--anchor-checkpoint must be the reconstructed 720-step supervised anchor.")
    source_data_root, labeled, development, source_manifest = _records_from_manifest(Path(args.stage1_manifest).resolve())
    # This finds only p1--6 filenames; no hidden label is opened or checksum-read.
    all_train, _ = __import__("semipms.stage1_guards", fromlist=["list_stage1_records"]).list_stage1_records(source_data_root)
    unlabeled = [record for record in all_train if record not in labeled]
    if len(unlabeled) != 24: raise AssertionError("Stage-1B requires exactly 24 unlabeled train images.")
    run_id = args.run_id or f"semipms_stage1b_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{_git(repo, 'rev-parse', '--short', 'HEAD')}"
    artifact = Path(args.output_root).resolve() / run_id
    if artifact.exists(): raise FileExistsError(f"Refusing to overwrite {artifact}")
    artifact.mkdir(parents=True)
    rule_base, rule_provenance = _load_rule(Path(args.phase0_rule).resolve())
    device = torch.device(f"cuda:{args.gpu_device}" if torch.cuda.is_available() else "cpu")
    _seed_everything(SEED); cfg = _runtime_config(args)
    args_cfg, teacher_point, teacher_encoder, teacher_net, _, teacher_texture, _ = _load_checkpoint(anchor, args, device, require_optimizer=False)
    _freeze_for_point_only(teacher_point, teacher_net)
    calibration = _calibration_candidates(labeled, teacher_point, teacher_encoder, teacher_net, cfg, device)
    frozen_rule, budget, lopo = _select_rule_lopo(calibration, rule_base)
    write_json(artifact / "frozen_stage1b_acceptance_rule.json", {"rule": frozen_rule, "proposal_budget": budget, "target_precision": TARGET_CALIBRATION_PRECISION, "lopo": lopo, "source_phase0_rule": rule_provenance})
    _csv(artifact / "lopo_calibration.csv", [{key: value for key, value in row.items() if key not in {"mask", "stain_mask", "geometry_mask", "features"}} for row in calibration])
    cache_dir, dedup_rows, candidate_rows = _build_static_cache(artifact, unlabeled, teacher_point, teacher_encoder, teacher_net, teacher_texture, cfg, device, frozen_rule, budget)
    cache_initial_hash = _tree_hash(cache_dir)
    teacher_state = {"anchor_checkpoint": str(anchor), "anchor_sha256": sha256_file(anchor), "teacher_point_state_sha256": _state_hash(teacher_point), "teacher_sam2_state_sha256": _state_hash(teacher_net), "static_cache_sha256": sha256_file(cache_dir / "static_cache_manifest.json")}
    del teacher_point, teacher_encoder, teacher_net
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    args._unlabeled = unlabeled
    results = {}
    all_training: list[dict[str, Any]] = []; all_dev: list[dict[str, Any]] = []; all_pseudo: list[dict[str, Any]] = []
    for method, base, residual in (("Labeled-Only Continuation", None, None), ("Static-Base Self-Training", cache_dir / "base", None), ("Anchored SemiPMS", cache_dir / "base", cache_dir / "residual")):
        train, dev, pseudo, summary = _run_path(artifact, method, anchor, args, source_data_root, labeled, development, cfg, device, base, residual)
        results[method] = summary; all_training.extend(train); all_dev.extend(dev); all_pseudo.extend(pseudo)
    _csv(artifact / "training_curve.csv", all_training); _csv(artifact / "pseudo_point_statistics.csv", all_pseudo); _csv(artifact / "per_image_metrics.csv", all_dev)
    aggregate = _aggregate_development(all_dev); _csv(artifact / "per_patient_metrics.csv", aggregate)
    cache_final_hash = _tree_hash(cache_dir)
    if cache_initial_hash != cache_final_hash:
        raise AssertionError("Static pseudo-cache changed after it was frozen.")
    write_json(artifact / "data_manifest.json", {"source_stage1_manifest": source_manifest, "labeled": [record.__dict__ for record in labeled], "unlabeled_train_images_only": [record.stem for record in unlabeled], "development": [record.__dict__ for record in development], "closed": [9,10,11], "monuseg": "forbidden"})
    write_json(artifact / "checkpoint_provenance.json", teacher_state)
    write_json(artifact / "trainable_parameter_manifest.json", {"only_trainable": "point_net excluding backbone and mask_head", "forbidden_trainable_modules": ["SAM2 image encoder", "SAM2 prompt encoder", "SAM2 mask decoder", "mask-quality/multimask modules"], "paths": {name: result["trainable_parameters"] for name, result in results.items()}})
    (artifact / "environment.txt").write_text(json.dumps(_environment(), indent=2) + "\n", encoding="utf-8")
    _run_tests(artifact)
    final = {name: {"best_development": result["best_development"], "final_checkpoint": result["final_checkpoint"], "runtime_seconds": result["runtime_seconds"], "evaluation_seconds": result["evaluation_seconds"], "frozen_state_sha256": result["frozen_state_sha256"]} for name, result in results.items()}
    final_dev = {row["method"]: row for row in aggregate if row["level"] == "all" and int(row["optimizer_steps"]) == CONTINUATION_STEPS}
    def delta(left: str, right: str) -> dict[str, float]:
        return {f"delta_{name}": float(final_dev[left][name]) - float(final_dev[right][name]) for name in ("dice", "aji", "aji_plus", "dq", "sq", "pq")} | {"delta_tp": int(final_dev[left]["tp"]) - int(final_dev[right]["tp"]), "delta_fp": int(final_dev[left]["fp"]) - int(final_dev[right]["fp"]), "delta_fn": int(final_dev[left]["fn"]) - int(final_dev[right]["fn"])}
    write_json(artifact / "report.json", {"phase": "Anchored SemiPMS Stage 1B", "git_sha": _git(repo, "rev-parse", "HEAD"), "anchor": teacher_state, "static_cache": str(cache_dir), "static_cache_frozen": True, "static_cache_tree_sha256": cache_final_hash, "ema": "forbidden", "pseudo_mask_loss": "forbidden", "continuation_steps": CONTINUATION_STEPS, "evaluation_steps": list(EVAL_STEPS), "pseudo_loss": {"base_weight": BASE_PSEUDO_WEIGHT, "residual_weight": RESIDUAL_PSEUDO_WEIGHT, "residual_ramp_steps": RESIDUAL_RAMP_STEPS, "labeled_to_pseudo_ratio": "1:1 base; Anchored adds one residual pseudo batch"}, "calibration": {"target_precision": TARGET_CALIBRATION_PRECISION, "folds_meeting_target": sum(bool(row["target_precision_met"]) for row in lopo), "frozen_rule": frozen_rule, "proposal_budget": budget}, "results": final, "development_final": final_dev, "comparisons": {"anchored_minus_labeled_only": delta("Anchored SemiPMS", "Labeled-Only Continuation"), "anchored_minus_static_base": delta("Anchored SemiPMS", "Static-Base Self-Training")}, "access": {"patients_1_to_6_train": True, "patients_7_to_8_development": True, "patients_9_to_11": "forbidden", "monuseg": "forbidden", "hidden_train_gt": "not read"}, "runtime_seconds": time.monotonic()-started})
    with (artifact / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in sorted(item for item in artifact.rglob("*") if item.is_file() and item.name != "SHA256SUMS"):
            handle.write(f"{sha256_file(path)}  {path.relative_to(artifact).as_posix()}\n")
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Anchored SemiPMS Stage 1B: static pseudo-centre TNBC development experiment")
    parser.add_argument("--anchor-checkpoint", required=True)
    parser.add_argument("--anchor-artifact", required=True)
    parser.add_argument("--stage1-manifest", required=True)
    parser.add_argument("--phase0-rule", required=True)
    parser.add_argument("--output-root", default="logs/semipms/stage1b_anchored_tnbc_dev")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--sam-config", default="sam2_hiera_l")
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    artifact = run_stage1b(build_parser().parse_args(argv))
    print(f"Anchored SemiPMS Stage 1B complete: {artifact}")
    return 0
