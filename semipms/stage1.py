"""SemiPMS Stage 1: fair 20%-label TNBC development experiment.

The runner has one shared supervised 240-step warm-up, followed by three
fixed-length paths:

* ``Supervised-StainPMS-20``: six labelled images only;
* ``MeanTeacher-PMS``: the same labelled updates plus EMA base pseudo masks;
* ``SemiPMS``: the same base pseudo masks plus frozen accepted residual masks.

Patients 7--8 are development-only.  The 24 remaining train-side labels are
not opened until all three models have completed their fixed 960 optimizer
updates, at which point they are used exactly once for retrospective cache and
false-positive diagnostics.  No model-selection decision uses those labels.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import math
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from mmengine.config import Config
from torch.utils.data import DataLoader

from run.dataset.monuseg import MONUSEG
from sam2_train.build_sam import build_sam2
from sam2_train.modeling.dpa_p2pnet import build_model
from sam2_train.modeling.stats_utils import get_fast_pq, remap_label
from semipms.guards import ImageRecord, deterministic_split, inspect_clean_initialization, sha256_file, validate_clean_checkpoint_name, write_json
from semipms.phase0 import (
    CANONICAL_BASELINE,
    StepBudgetReached,
    _aggregate,
    _assert_baseline,
    _build_models,
    _candidate_feature_records,
    _csv,
    _environment,
    _git,
    _infer_standard,
    _legacy_helpers,
    _metrics,
    _normalise,
    _read_image,
    _run_tests,
    _runtime_config,
    _verify_formal_baseline_equivalence,
)
from semipms.residual import frozen_accept, propose_residual_points, residual_evidence
from semipms.stage1_guards import Stage1AccessGuard, list_stage1_records, stage1_data_manifest


WARMUP_STEPS = 240
TOTAL_STEPS = 960
CACHE_REFRESH_STEPS = 240
EMA_DECAY = 0.99
MIN_BASE_AREA = 8
MASK_DUPLICATE_IOU = 0.50
RESIDUAL_RAMP_STEPS = 240


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _state_sha256(*modules: torch.nn.Module) -> str:
    """Stable, lightweight state fingerprint without materialising a checkpoint."""
    import hashlib

    digest = hashlib.sha256()
    for module in modules:
        for name, tensor in sorted(module.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(np.asarray(tensor.detach().cpu()).tobytes())
    return digest.hexdigest()


def _new_optimizer(point_net: torch.nn.Module, net: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [parameter for module in (point_net, net) for parameter in module.parameters() if parameter.requires_grad],
        lr=1e-4,
        weight_decay=1e-4,
    )


class Stage1Optimizer:
    """Fixed-step optimiser with an optional pseudo-loss accumulation callback."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        *,
        start_steps: int = 0,
        before_step: Callable[[int], Mapping[str, float]] | None = None,
        after_step: Callable[[int, Sequence[Any]], None] | None = None,
    ) -> None:
        self._optimizer = optimizer
        self.total_steps = int(total_steps)
        self.steps = int(start_steps)
        self.before_step = before_step
        self.after_step = after_step
        self.texture_memory_bank: list[Any] = []
        self.pseudo_step_rows: list[dict[str, float]] = []

    @property
    def param_groups(self):
        return self._optimizer.param_groups

    def zero_grad(self, *args, **kwargs):
        return self._optimizer.zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        next_step = self.steps + 1
        if self.before_step is not None:
            values = {key: float(value) for key, value in self.before_step(next_step).items()}
            values["optimizer_step"] = float(next_step)
            self.pseudo_step_rows.append(values)
        result = self._optimizer.step(*args, **kwargs)
        self.steps = next_step
        if self.after_step is not None:
            self.after_step(self.steps, self.texture_memory_bank)
        if self.steps >= self.total_steps:
            raise StepBudgetReached
        return result

    def __getattr__(self, name: str):
        return getattr(self._optimizer, name)


class ModelEMA:
    """EMA teacher over the same point head and SAM2 decoder as its student."""

    def __init__(self, point_net: torch.nn.Module, net: torch.nn.Module, decay: float) -> None:
        self.point_net = copy.deepcopy(point_net).eval()
        self.net = copy.deepcopy(net).eval()
        self.decay = float(decay)
        for module in (self.point_net, self.net):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, point_net: torch.nn.Module, net: torch.nn.Module) -> None:
        for teacher, student in ((self.point_net, point_net), (self.net, net)):
            teacher_state = teacher.state_dict()
            student_state = student.state_dict()
            for name, teacher_value in teacher_state.items():
                student_value = student_state[name].detach()
                if torch.is_floating_point(teacher_value):
                    teacher_value.mul_(self.decay).add_(student_value, alpha=1.0 - self.decay)
                else:
                    teacher_value.copy_(student_value)


def _save_checkpoint(
    path: Path,
    point_net: torch.nn.Module,
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    texture_memory_bank: Sequence[Any],
    *,
    step: int,
    role: str,
    initial_state_sha256: str,
    ema: ModelEMA | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": net.state_dict(),
        "model1": point_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "texture_memory_bank_list": list(texture_memory_bank),
        "semipms_stage1": {
            "optimizer_steps": int(step),
            "role": role,
            "initial_state_sha256": initial_state_sha256,
            "selection_rule": "fixed optimizer step; no train-side hidden-GT selection",
        },
    }
    if ema is not None:
        payload["ema_decay"] = ema.decay
        payload["ema_state_sha256"] = _state_sha256(ema.point_net, ema.net)
    torch.save(payload, path)
    return {"path": str(path), "sha256": sha256_file(path), "optimizer_steps": int(step), "role": role}


def _load_checkpoint(
    checkpoint: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Any, torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.optim.Optimizer, list[Any], Mapping[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu")
    if not {"model", "model1", "optimizer", "semipms_stage1"}.issubset(payload):
        raise PermissionError("Stage-1 continuation must be the shared Stage-1 warm-up checkpoint.")
    args_cfg = Config.fromfile("args.py")
    args_cfg.criterion.pms_loss_coef = 0.5
    args_cfg.criterion.pms_object_weight = 1.0
    args_cfg.criterion.pms_residual_mask_weight = 0.3
    args_cfg.criterion.pms_preserve_loss_coef = 1.0
    point_net, point_encoder = build_model(args_cfg)
    net = build_sam2(args.sam_config, None, device=device)
    point_net.load_state_dict(payload["model1"], strict=True)
    net.load_state_dict(payload["model"], strict=True)
    point_net.to(device); point_encoder.to(device); net.to(device)
    for name, parameter in net.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            parameter.requires_grad_(False)
    optimizer = _new_optimizer(point_net, net)
    optimizer.load_state_dict(payload["optimizer"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    return args_cfg, point_net, point_encoder, net, optimizer, list(payload.get("texture_memory_bank_list", []) or []), payload


def _make_labeled_loader(
    cfg: SimpleNamespace,
    args_cfg: Any,
    data_root: Path,
    labeled: Sequence[ImageRecord],
    num_workers: int,
) -> DataLoader:
    helpers = _legacy_helpers()
    dataset = MONUSEG(cfg, args_cfg, str(data_root), cfg.load, mode="train")
    dataset.paths = [f"{record.stem}{Path(record.image_path).suffix}" for record in labeled]
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=helpers.collate)


def _evaluate_development(
    records: Sequence[ImageRecord],
    access_guard: Stage1AccessGuard,
    point_net: torch.nn.Module,
    point_encoder: torch.nn.Module,
    net: torch.nn.Module,
    texture_memory_bank: Sequence[Any],
    cfg: SimpleNamespace,
    device: torch.device,
    *,
    method: str,
    step: int,
    shared_warmup: bool = False,
) -> list[dict[str, object]]:
    point_net.eval(); point_encoder.eval(); net.eval()
    rows: list[dict[str, object]] = []
    for record in records:
        _, image = _read_image(record)
        prediction = _infer_standard(image.to(device), point_net, point_encoder, net, list(texture_memory_bank), cfg, device)
        access_guard.allow_development_label_read(record)
        gt = sio.loadmat(record.label_path)["inst_map"].astype(np.int32)
        metrics, _ = _metrics(gt, prediction)
        rows.append({"method": method, "optimizer_steps": int(step), "shared_warmup": bool(shared_warmup), "image": record.stem, "patient": record.patient, **metrics})
    point_net.train(); net.train()
    return rows


def _aggregate_development(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    groups: dict[tuple[str, int, str, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["method"]), int(row["optimizer_steps"]), "patient", int(row["patient"]))].append(row)
        groups[(str(row["method"]), int(row["optimizer_steps"]), "all", 0)].append(row)
    for (method, step, level, patient), subset in sorted(groups.items()):
        out.append({
            "method": method,
            "optimizer_steps": step,
            "level": level,
            "patient": patient if level == "patient" else "all",
            "n_images": len(subset),
            **_aggregate(subset),
        })
    return out


def _filter_base_instances(instance_map: np.ndarray, min_area: int = MIN_BASE_AREA) -> tuple[np.ndarray, int]:
    out = np.zeros_like(instance_map, dtype=np.int32)
    next_id = 1
    rejected = 0
    for instance_id in sorted(int(value) for value in np.unique(instance_map) if value != 0):
        mask = instance_map == instance_id
        if int(mask.sum()) < int(min_area):
            rejected += 1
            continue
        out[mask] = next_id
        next_id += 1
    return out, rejected


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = int(np.logical_or(left, right).sum())
    return float(np.logical_and(left, right).sum() / union) if union else 0.0


def _resolve_residual_masks(
    base_instances: np.ndarray,
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    duplicate_iou: float = MASK_DUPLICATE_IOU,
) -> tuple[np.ndarray, list[dict[str, Any]], Counter]:
    """Mask-level base/residual deduplication without any ground-truth query."""
    residual = np.zeros_like(base_instances, dtype=np.int32)
    base_masks = [base_instances == instance_id for instance_id in np.unique(base_instances) if instance_id]
    accepted_masks: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    stats: Counter = Counter()
    next_id = 1
    for row in sorted(candidate_rows, key=lambda item: (-float(item["evidence"]), int(item["candidate_index"]))):
        item = dict(row)
        mask = np.asarray(item["mask"], dtype=bool)
        item["cross_view_accepted"] = bool(item.get("cross_view_accepted", False))
        item["base_max_iou"] = max((_mask_iou(mask, base) for base in base_masks), default=0.0)
        item["residual_max_iou"] = max((_mask_iou(mask, other) for other in accepted_masks), default=0.0)
        item["assembly_overlap_pixels"] = int(np.logical_and(mask, (base_instances > 0) | (residual > 0)).sum())
        if not item["cross_view_accepted"]:
            item["status"] = "cross_view_rejected"; stats[item["status"]] += 1
        elif not mask.any():
            item["status"] = "empty_decoder_mask"; stats[item["status"]] += 1
        elif item["base_max_iou"] >= duplicate_iou:
            item["status"] = "residual_teacher_duplicate"; stats[item["status"]] += 1
        elif item["residual_max_iou"] >= duplicate_iou:
            item["status"] = "residual_residual_duplicate"; stats[item["status"]] += 1
        else:
            uncovered = mask & (base_instances == 0) & (residual == 0)
            if int(uncovered.sum()) < MIN_BASE_AREA:
                item["status"] = "assembly_conflict_rejected"; stats[item["status"]] += 1
            else:
                residual[uncovered] = next_id
                item["status"] = "accepted_trimmed" if item["assembly_overlap_pixels"] else "accepted"
                item["final_residual_id"] = next_id
                next_id += 1
                accepted_masks.append(uncovered)
                stats["accepted"] += 1
        metadata.append(item)
    return residual, metadata, stats


def _write_candidate_archive(cache_dir: Path, stem: str, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, object]]:
    """Persist GT-free masks compactly so post-training diagnostics are auditable."""
    archive_dir = cache_dir / "candidate_masks"; archive_dir.mkdir(exist_ok=True)
    masks = np.asarray([np.asarray(row["mask"], dtype=np.uint8) for row in rows], dtype=np.uint8)
    np.savez_compressed(archive_dir / f"{stem}.npz", masks=masks)
    output: list[dict[str, object]] = []
    for row in rows:
        features = dict(row.get("features", {}))
        output.append({
            "image": stem,
            "candidate_index": int(row["candidate_index"]),
            "x": float(row["x"]), "y": float(row["y"]), "evidence": float(row["evidence"]), "source": row["source"],
            "predicted_iou": float(row["predicted_iou"]), "cross_view_accepted": bool(row.get("cross_view_accepted", False)),
            "status": row.get("status", "not_applicable"),
            "base_max_iou": float(row.get("base_max_iou", 0.0)),
            "residual_max_iou": float(row.get("residual_max_iou", 0.0)),
            "assembly_overlap_pixels": int(row.get("assembly_overlap_pixels", 0)),
            **{f"feature_{key}": float(value) for key, value in features.items()},
        })
    return output


class PseudoCache:
    def __init__(self, cache_dir: Path, method: str, source_step: int, rows: Sequence[Mapping[str, object]]) -> None:
        self.cache_dir = cache_dir
        self.method = method
        self.source_step = int(source_step)
        self.rows = list(rows)

    @property
    def base_dir(self) -> Path:
        return self.cache_dir / "base"

    @property
    def residual_dir(self) -> Path:
        return self.cache_dir / "residual"


@torch.inference_mode()
def _build_pseudo_cache(
    artifact: Path,
    method: str,
    source_step: int,
    records: Sequence[ImageRecord],
    ema: ModelEMA,
    cfg: SimpleNamespace,
    device: torch.device,
    frozen_rule: Mapping[str, float] | None,
) -> PseudoCache:
    cache_dir = artifact / "pseudo_cache" / method / f"update_{source_step:04d}"
    if cache_dir.exists():
        raise FileExistsError(f"Refusing to overwrite pseudo cache {cache_dir}")
    base_dir = cache_dir / "base"; residual_dir = cache_dir / "residual"
    base_dir.mkdir(parents=True); residual_dir.mkdir()
    started = time.monotonic()
    stats_rows: list[dict[str, object]] = []
    candidate_metadata: list[dict[str, object]] = []
    for record in records:
        raw, normal = _read_image(record)
        teacher = _infer_standard(normal.to(device), ema.point_net, ema.point_net.backbone, ema.net, [], cfg, device)
        base, base_small_rejected = _filter_base_instances(teacher)
        candidates: list[dict[str, Any]] = []
        residual = np.zeros_like(base, dtype=np.int32)
        dedupe_stats: Counter = Counter()
        if frozen_rule is not None:
            proposed = propose_residual_points(residual_evidence(raw, teacher), max_candidates=64)
            candidates = _candidate_feature_records(raw, normal.to(device), teacher, proposed, ema.point_net, ema.point_net.backbone, ema.net, cfg, device)
            for row in candidates:
                row["cross_view_accepted"] = frozen_accept(row["features"], dict(frozen_rule))
            residual, candidates, dedupe_stats = _resolve_residual_masks(base, candidates)
            candidate_metadata.extend(_write_candidate_archive(cache_dir, record.stem, candidates))
        combined = base.copy()
        combined[residual > 0] = residual[residual > 0] + int(base.max())
        sio.savemat(base_dir / f"{record.stem}.mat", {"inst_map": base.astype(np.int32)})
        sio.savemat(residual_dir / f"{record.stem}.mat", {"inst_map": residual.astype(np.int32)})
        sio.savemat(cache_dir / f"{record.stem}.mat", {"inst_map": combined.astype(np.int32)})
        stats_rows.append({
            "method": method, "cache_source_step": source_step, "image": record.stem, "patient": record.patient,
            "base_instances": int(base.max()), "base_small_rejected": base_small_rejected,
            "raw_residual_candidates": len(candidates), "cross_view_accepted": sum(bool(item.get("cross_view_accepted", False)) for item in candidates),
            "residual_instances": int(residual.max()), "combined_instances": int(combined.max()),
            **{f"status_{key}": int(value) for key, value in dedupe_stats.items()},
        })
    _csv(cache_dir / "cache_statistics.csv", stats_rows)
    _csv(cache_dir / "candidate_metadata.csv", candidate_metadata)
    write_json(cache_dir / "cache_manifest.json", {
        "method": method, "source_step": source_step, "image_count": len(records), "uses_train_gt": False,
        "base_pseudo": "EMA standard deployment instances, area filtered only",
        "residual_pseudo": "frozen cross-view accepted residual instances" if frozen_rule is not None else "disabled",
        "mask_duplicate_iou": MASK_DUPLICATE_IOU, "min_base_area": MIN_BASE_AREA,
        "runtime_seconds": time.monotonic() - started,
    })
    return PseudoCache(cache_dir, method, source_step, stats_rows)


class PseudoCropCursor:
    """Read artifact-owned pseudo maps, never the hidden train label directory."""

    def __init__(
        self,
        pseudo_label_root: Path,
        records: Sequence[ImageRecord],
        data_root: Path,
        cfg: SimpleNamespace,
        args_cfg: Any,
        source: str,
    ) -> None:
        helpers = _legacy_helpers()
        pseudo_cfg = copy.copy(cfg)
        pseudo_cfg.use_pms = False
        dataset = MONUSEG(pseudo_cfg, args_cfg, str(data_root), cfg.load, mode="train")
        # MONUSEG is used only for its existing crop/augmentation mechanics.
        # Its raw TNBC label root is replaced before iteration and its paths are
        # fixed to the artifact-owned pseudo maps.
        dataset.label_root = str(pseudo_label_root)
        paths = []
        for record in records:
            pseudo_path = pseudo_label_root / f"{record.stem}.mat"
            payload = sio.loadmat(pseudo_path)["inst_map"]
            if np.any(payload):
                paths.append(f"{record.stem}{Path(record.image_path).suffix}")
        dataset.paths = paths
        self.source = source
        self.loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, collate_fn=helpers.collate)
        self.iterator: Iterable[Any] | None = None
        self.batch: Any | None = None
        self.crop_index = 0

    def _next_batch(self) -> Any:
        if not len(self.loader.dataset):
            return None
        if self.iterator is None:
            self.iterator = iter(self.loader)
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)

    def next_crop(self) -> dict[str, Any] | None:
        """Return one crop with one or more pseudo instances, or ``None``."""
        if not len(self.loader.dataset):
            return None
        for _ in range(max(1, len(self.loader.dataset) * 4)):
            if self.batch is None or self.crop_index >= int(self.batch[0].shape[0]):
                self.batch = self._next_batch()
                self.crop_index = 0
            if self.batch is None:
                return None
            index = self.crop_index
            self.crop_index += 1
            counts = self.batch[6]
            count = int(counts[index].item())
            if count <= 0:
                continue
            offset = int(counts[:index].sum().item())
            return {
                "image": self.batch[0][index:index + 1],
                "masks": self.batch[1][offset:offset + count],
                "points": self.batch[2][index],
                "labels": self.batch[3][index],
                "source": self.source,
                "instances": count,
            }
        return None


def _pseudo_positive_loss(
    sample: Mapping[str, Any],
    point_net: torch.nn.Module,
    point_encoder: torch.nn.Module,
    net: torch.nn.Module,
    cfg: SimpleNamespace,
    device: torch.device,
    *,
    coefficient: float,
) -> dict[str, torch.Tensor]:
    """Positive-only pseudo supervision.

    No pseudo loss is evaluated over an unlabeled pixel outside an accepted
    mask.  This deliberately avoids turning unconfirmed tissue into a negative
    background target while still teaching point placement, positive objectness,
    and decoder support for accepted instances.
    """
    images = sample["image"].to(device)
    gt_masks = sample["masks"].to(device).bool()
    points = sample["points"].to(device).float().reshape(-1, 2)
    labels = sample["labels"].to(device).long().reshape(-1)
    if not len(points):
        zero = images.sum() * 0.0
        return {"total": zero, "point": zero, "class": zero, "mask": zero, "confidence": zero, "instances": zero}

    feats, _ = point_encoder(images)
    backbone_out, _ = net.forward_image(images, feats)
    _, vision_feats, vision_pos_embeds, _ = net._prepare_backbone_features(backbone_out)
    feat_sizes = [(64, 64), (32, 32), (16, 16)]
    reshaped = [
        feature.permute(1, 2, 0).view(1, -1, *feature_size)
        for feature, feature_size in zip(vision_feats[::-1], feat_sizes[::-1])
    ][::-1]
    image_embed, high_res_feats = reshaped[-1], reshaped[:-1]
    outputs, _, _, _ = point_net(images)
    distances = torch.cdist(outputs["pred_coords"][0], points)
    nearest_indices = torch.argmin(distances.detach(), dim=0)
    nearest_points = outputs["pred_coords"][0, nearest_indices].unsqueeze(1)
    # Existing MONUSEG labels encode the positive nucleus class as zero.
    positive_labels = labels.unsqueeze(1)
    with torch.no_grad():
        sparse, dense = net.sam_prompt_encoder(
            points=(nearest_points, positive_labels), boxes=None, masks=None, batch_size=1
        )
    low_res_masks, iou_predictions, _, object_logits = net.sam_mask_decoder(
        image_embeddings=image_embed,
        image_pe=net.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
        repeat_image=False,
        cell_nums=torch.as_tensor([len(points)], dtype=torch.long, device=device),
        high_res_features=high_res_feats,
    )
    predicted = F.interpolate(low_res_masks, size=(cfg.out_size, cfg.out_size), mode="bilinear", align_corners=False)[:, 0]
    point_loss = F.smooth_l1_loss(nearest_points.squeeze(1), points) * 0.1
    class_loss = F.cross_entropy(outputs["pred_logits"][0, nearest_indices], labels)
    per_mask = []
    for index in range(predicted.shape[0]):
        known_positive = gt_masks[index]
        if known_positive.any():
            per_mask.append(F.softplus(-predicted[index][known_positive]).mean())
    mask_loss = torch.stack(per_mask).mean() if per_mask else predicted.sum() * 0.0
    confidence_loss = F.softplus(-iou_predictions.max(dim=1).values).mean() * 0.1
    total = (point_loss + class_loss + mask_loss + confidence_loss) * float(coefficient)
    return {
        "total": total,
        "point": point_loss.detach(),
        "class": class_loss.detach(),
        "mask": mask_loss.detach(),
        "confidence": confidence_loss.detach(),
        "instances": torch.as_tensor(float(len(points)), device=device),
    }


class PseudoLossProvider:
    """Owns EMA refresh, cache construction, and base/residual loss accounting."""

    def __init__(
        self,
        artifact: Path,
        method: str,
        records: Sequence[ImageRecord],
        data_root: Path,
        args_cfg: Any,
        cfg: SimpleNamespace,
        point_net: torch.nn.Module,
        point_encoder: torch.nn.Module,
        net: torch.nn.Module,
        device: torch.device,
        frozen_rule: Mapping[str, float] | None,
        *,
        warmup_steps: int,
        refresh_steps: int,
        residual_ramp_steps: int,
        ema_decay: float,
        base_weight: float,
        residual_weight: float,
    ) -> None:
        self.artifact = artifact
        self.method = method
        self.records = records
        self.data_root = data_root
        self.args_cfg = args_cfg
        self.cfg = cfg
        self.point_net = point_net
        self.point_encoder = point_encoder
        self.net = net
        self.device = device
        self.frozen_rule = frozen_rule
        self.warmup_steps = int(warmup_steps)
        self.refresh_steps = int(refresh_steps)
        self.residual_ramp_steps = max(1, int(residual_ramp_steps))
        self.base_weight = float(base_weight)
        self.residual_weight = float(residual_weight)
        self.ema = ModelEMA(point_net, net, ema_decay)
        self.cache: PseudoCache | None = None
        self.base_cursor: PseudoCropCursor | None = None
        self.residual_cursor: PseudoCropCursor | None = None
        self.cache_rows: list[dict[str, object]] = []

    def _refresh_if_needed(self, completed_steps: int) -> None:
        if self.cache is not None and completed_steps % self.refresh_steps != 0:
            return
        self.cache = _build_pseudo_cache(
            self.artifact,
            self.method,
            completed_steps,
            self.records,
            self.ema,
            self.cfg,
            self.device,
            self.frozen_rule,
        )
        self.base_cursor = PseudoCropCursor(self.cache.base_dir, self.records, self.data_root, self.cfg, self.args_cfg, "base")
        self.residual_cursor = (
            PseudoCropCursor(self.cache.residual_dir, self.records, self.data_root, self.cfg, self.args_cfg, "residual")
            if self.frozen_rule is not None else None
        )
        for row in self.cache.rows:
            self.cache_rows.append(dict(row))

    def before_student_step(self, next_step: int) -> Mapping[str, float]:
        if next_step <= self.warmup_steps:
            return {"base_pseudo_loss": 0.0, "residual_pseudo_loss": 0.0, "residual_ramp": 0.0, "base_instances": 0.0, "residual_instances": 0.0}
        # Pseudo-image augmentation/dropout must not perturb the deterministic
        # labelled-image augmentation stream shared with path A.
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            self._refresh_if_needed(next_step - 1)
            totals: dict[str, float] = {"base_pseudo_loss": 0.0, "residual_pseudo_loss": 0.0, "residual_ramp": 0.0, "base_instances": 0.0, "residual_instances": 0.0}
            if self.base_cursor is not None:
                sample = self.base_cursor.next_crop()
                if sample is not None:
                    losses = _pseudo_positive_loss(sample, self.point_net, self.point_encoder, self.net, self.cfg, self.device, coefficient=self.base_weight)
                    losses["total"].backward()
                    totals["base_pseudo_loss"] = float(losses["total"].detach().cpu())
                    totals["base_instances"] = float(losses["instances"].detach().cpu())
            if self.residual_cursor is not None:
                sample = self.residual_cursor.next_crop()
                ramp = min(1.0, (next_step - self.warmup_steps) / self.residual_ramp_steps)
                totals["residual_ramp"] = ramp
                if sample is not None and ramp > 0:
                    losses = _pseudo_positive_loss(sample, self.point_net, self.point_encoder, self.net, self.cfg, self.device, coefficient=self.residual_weight * ramp)
                    losses["total"].backward()
                    totals["residual_pseudo_loss"] = float(losses["total"].detach().cpu())
                    totals["residual_instances"] = float(losses["instances"].detach().cpu())
            return totals
        finally:
            random.setstate(python_state); np.random.set_state(numpy_state); torch.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)

    def after_student_step(self, step: int) -> None:
        self.ema.update(self.point_net, self.net)


def _train_to_fixed_step(
    artifact: Path,
    method: str,
    point_net: torch.nn.Module,
    point_encoder: torch.nn.Module,
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args_cfg: Any,
    cfg: SimpleNamespace,
    data_root: Path,
    labeled: Sequence[ImageRecord],
    development: Sequence[ImageRecord],
    access_guard: Stage1AccessGuard,
    device: torch.device,
    *,
    start_steps: int,
    total_steps: int,
    eval_steps: set[int],
    initial_state_sha256: str,
    pseudo_provider: PseudoLossProvider | None = None,
    initial_texture_memory: Sequence[Any] = (),
    num_workers: int = 2,
    save_final: bool = True,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[Any], dict[str, object]]:
    """Continue one path to the predeclared fixed step budget.

    The existing CA-SAM2/StainPMS labelled update is intentionally called
    unchanged.  For B/C, ``PseudoLossProvider`` adds positive-only pseudo
    gradients immediately before the same optimiser update; it never changes
    deployment inference or reads a train-side hidden label.
    """
    helpers = _legacy_helpers()
    loader = _make_labeled_loader(cfg, args_cfg, data_root, labeled, num_workers)
    criterion, _ = helpers.build_criterion(args_cfg, device)
    dev_rows: list[dict[str, object]] = []
    checkpoint_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    started = time.monotonic()
    latest_texture = list(initial_texture_memory)

    def before_step(next_step: int) -> Mapping[str, float]:
        if pseudo_provider is None:
            return {"base_pseudo_loss": 0.0, "residual_pseudo_loss": 0.0, "residual_ramp": 0.0, "base_instances": 0.0, "residual_instances": 0.0}
        return pseudo_provider.before_student_step(next_step)

    def after_step(step: int, texture_memory: Sequence[Any]) -> None:
        nonlocal latest_texture
        latest_texture = list(texture_memory)
        if pseudo_provider is not None:
            pseudo_provider.after_student_step(step)
        if step in eval_steps:
            dev_rows.extend(_evaluate_development(
                development, access_guard, point_net, point_encoder, net, latest_texture, cfg, device,
                method=method, step=step,
            ))
        if save_final and step == total_steps:
            checkpoint_rows.append(_save_checkpoint(
                artifact / "checkpoints" / f"{method.lower().replace('-', '_')}_final_{step}.pth",
                point_net, net, optimizer, latest_texture, step=step, role=method,
                initial_state_sha256=initial_state_sha256, ema=pseudo_provider.ema if pseudo_provider else None,
            ))

    bounded = Stage1Optimizer(
        optimizer,
        total_steps,
        start_steps=start_steps,
        before_step=before_step if pseudo_provider is not None else None,
        after_step=after_step,
    )
    epoch = 0
    while bounded.steps < total_steps:
        texture_memory: list[Any] = []
        bounded.texture_memory_bank = texture_memory
        epoch_started = time.monotonic()
        pseudo_row_start = len(bounded.pseudo_step_rows)
        completed = True
        try:
            log_info = helpers.train(cfg, point_net, point_encoder, net, loader, criterion, bounded, epoch, texture_memory, device)
        except StepBudgetReached:
            completed = False
            log_info = {"partial_epoch": True}
        pseudo_epoch = bounded.pseudo_step_rows[pseudo_row_start:]
        training_rows.append({
            "method": method, "epoch": epoch, "optimizer_steps_completed": bounded.steps,
            "epoch_completed": completed, "seconds": time.monotonic() - epoch_started,
            **{key: float(value) for key, value in log_info.items()},
            "pseudo_base_loss_mean": float(np.mean([row["base_pseudo_loss"] for row in pseudo_epoch])) if pseudo_epoch else 0.0,
            "pseudo_residual_loss_mean": float(np.mean([row["residual_pseudo_loss"] for row in pseudo_epoch])) if pseudo_epoch else 0.0,
        })
        epoch += 1
    if bounded.steps != total_steps:
        raise AssertionError(f"{method} completed {bounded.steps}, expected fixed {total_steps} optimizer steps.")
    pseudo_rows = [{"method": method, **row} for row in bounded.pseudo_step_rows]
    summary = {
        "method": method,
        "start_steps": start_steps,
        "total_steps": total_steps,
        "additional_seconds": time.monotonic() - started,
        "cuda_max_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "pseudo_steps": len(pseudo_rows),
    }
    return training_rows, dev_rows, checkpoint_rows + pseudo_rows, latest_texture, summary


def _load_frozen_rule(path: Path) -> tuple[dict[str, float], dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"Phase-0 frozen acceptance rule not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rule = payload.get("rule")
    required = {"min_view_iou", "max_centroid_displacement", "min_area_stability", "min_h_occupancy", "min_boundary_stability", "max_pseudo_conflict"}
    if not isinstance(rule, dict) or set(rule) != required:
        raise PermissionError("Stage 1 accepts only the complete frozen LOPO cross-view rule from SemiPMS Phase 0.")
    return {key: float(value) for key, value in rule.items()}, {
        "source_path": str(path.resolve()), "source_sha256": sha256_file(path),
        "rule": {key: float(value) for key, value in rule.items()},
        "origin": "Phase-0 labeled-only LOPO calibration; frozen before all Stage-1 pseudo caches",
    }


def _cache_dirs(artifact: Path) -> list[tuple[str, int, Path]]:
    root = artifact / "pseudo_cache"
    if not root.is_dir():
        return []
    out = []
    for method_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for update_dir in sorted(path for path in method_dir.iterdir() if path.is_dir() and path.name.startswith("update_")):
            out.append((method_dir.name, int(update_dir.name.removeprefix("update_")), update_dir))
    return out


def _max_iou_to_instances(mask: np.ndarray, instance_map: np.ndarray) -> tuple[int, float]:
    best_id, best_iou = 0, 0.0
    for instance_id in (int(value) for value in np.unique(instance_map) if value):
        score = _mask_iou(mask, instance_map == instance_id)
        if score > best_iou:
            best_id, best_iou = instance_id, score
    return best_id, best_iou


def _offline_cache_audit(
    artifact: Path,
    records: Sequence[ImageRecord],
    access_guard: Stage1AccessGuard,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Post-training only diagnostics for pseudo purity and false-positive sources."""
    hidden_gt: dict[str, np.ndarray] = {}
    for record in records:
        access_guard.allow_hidden_train_audit_read(record)
        hidden_gt[record.stem] = sio.loadmat(record.label_path)["inst_map"].astype(np.int32)
    audit_rows: list[dict[str, object]] = []
    candidate_rows_out: list[dict[str, object]] = []
    fp_rows: list[dict[str, object]] = []
    record_by_stem = {record.stem: record for record in records}
    for method, source_step, cache_dir in _cache_dirs(artifact):
        for stem, record in sorted(record_by_stem.items()):
            base_path, residual_path = cache_dir / "base" / f"{stem}.mat", cache_dir / "residual" / f"{stem}.mat"
            if not base_path.is_file() or not residual_path.is_file():
                raise FileNotFoundError(f"Incomplete pseudo cache for {stem}: {cache_dir}")
            gt = hidden_gt[stem]
            base = sio.loadmat(base_path)["inst_map"].astype(np.int32)
            residual = sio.loadmat(residual_path)["inst_map"].astype(np.int32)
            combined = base.copy(); combined[residual > 0] = residual[residual > 0] + int(base.max())
            base_metrics, base_missed = _metrics(gt, base)
            residual_metrics, _ = _metrics(gt, residual)
            combined_metrics, _ = _metrics(gt, combined)
            metadata_path = cache_dir / "candidate_metadata.csv"
            metadata = []
            if metadata_path.is_file():
                metadata = [row for row in csv.DictReader(metadata_path.open(encoding="utf-8")) if row["image"] == stem]
            masks_path = cache_dir / "candidate_masks" / f"{stem}.npz"
            masks = np.load(masks_path)["masks"].astype(bool) if masks_path.is_file() else np.empty((0, *gt.shape), dtype=bool)
            decoded_ious: list[float] = []
            raw_hits: set[int] = set(); accepted_hits: set[int] = set()
            source_counts: Counter = Counter()
            seen_targets: set[int] = set()
            for index, row in enumerate(metadata):
                mask = masks[index] if index < len(masks) else np.zeros_like(gt, dtype=bool)
                x, y = int(round(float(row["x"]))), int(round(float(row["y"])))
                target = int(gt[y, x]) if 0 <= y < gt.shape[0] and 0 <= x < gt.shape[1] else 0
                _, mask_iou = _max_iou_to_instances(mask, gt)
                if target:
                    decoded_ious.append(mask_iou)
                target_is_teacher_fn = target in base_missed
                if target_is_teacher_fn:
                    raw_hits.add(target)
                status = row["status"]
                accepted = status in {"accepted", "accepted_trimmed"}
                if accepted and target_is_teacher_fn and mask_iou >= 0.5:
                    accepted_hits.add(target)
                reason = "accepted_true_positive" if accepted and target_is_teacher_fn and mask_iou >= 0.5 else ""
                if status == "residual_teacher_duplicate":
                    reason = "residual_teacher_duplicate"
                elif status == "residual_residual_duplicate":
                    reason = "residual_residual_duplicate"
                elif status == "assembly_conflict_rejected":
                    reason = "assembly_conflict"
                elif target and mask_iou < 0.5:
                    reason = "nucleus_point_mask_iou_lt_05"
                elif target == 0 and accepted:
                    reason = "cross_view_consistent_non_nucleus"
                elif target == 0:
                    reason = "stain_artifact"
                elif target_is_teacher_fn and mask_iou >= 0.5 and not accepted:
                    reason = "legitimate_candidate_wrong_nms_or_conflict"
                elif target in seen_targets and accepted:
                    reason = "posthoc_residual_duplicate"
                else:
                    reason = reason or "other"
                if target:
                    seen_targets.add(target)
                source_counts[reason] += 1
                candidate_rows_out.append({
                    "method": method, "cache_source_step": source_step, "image": stem, "patient": record.patient,
                    "candidate_index": row["candidate_index"], "status": status, "accepted": accepted,
                    "target_gt_id": target, "target_teacher_fn": target_is_teacher_fn,
                    "decoded_iou": mask_iou, "fp_source": reason,
                })
            raw_n = len(metadata)
            accepted_n = sum(row["status"] in {"accepted", "accepted_trimmed"} for row in metadata)
            audit_rows.append({
                "method": method, "cache_source_step": source_step, "image": stem, "patient": record.patient,
                "raw_proposal_count": raw_n, "accepted_candidate_count": accepted_n,
                "teacher_fn": len(base_missed),
                "proposal_precision_teacher_fn": len(raw_hits) / raw_n if raw_n else 0.0,
                "proposal_recall_teacher_fn": len(raw_hits) / len(base_missed) if base_missed else 0.0,
                "accepted_mask_precision": residual_metrics["tp"] / max(1, residual_metrics["tp"] + residual_metrics["fp"]),
                "accepted_mask_recall_teacher_fn": len(accepted_hits) / len(base_missed) if base_missed else 0.0,
                "pseudo_set_precision": combined_metrics["tp"] / max(1, combined_metrics["tp"] + combined_metrics["fp"]),
                "pseudo_set_recall": combined_metrics["tp"] / max(1, combined_metrics["tp"] + combined_metrics["fn"]),
                "decoded_iou_mean": float(np.mean(decoded_ious)) if decoded_ious else 0.0,
                "decoded_iou_median": float(np.median(decoded_ious)) if decoded_ious else 0.0,
                "duplicate_rate": (source_counts["residual_teacher_duplicate"] + source_counts["residual_residual_duplicate"] + source_counts["posthoc_residual_duplicate"]) / raw_n if raw_n else 0.0,
                **{f"base_{key}": value for key, value in base_metrics.items()},
                **{f"residual_{key}": value for key, value in residual_metrics.items()},
                **{f"pseudo_{key}": value for key, value in combined_metrics.items()},
            })
            for reason, count in source_counts.items():
                fp_rows.append({"method": method, "cache_source_step": source_step, "image": stem, "patient": record.patient, "source": reason, "count": count})
    return audit_rows, candidate_rows_out, fp_rows


def _cache_summary(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["method"]), int(row["cache_source_step"]))].append(row)
    metrics = (
        "proposal_precision_teacher_fn", "proposal_recall_teacher_fn", "accepted_mask_precision",
        "accepted_mask_recall_teacher_fn", "pseudo_set_precision", "pseudo_set_recall",
        "decoded_iou_mean", "decoded_iou_median", "duplicate_rate",
    )
    output = []
    for (method, step), subset in sorted(groups.items()):
        output.append({
            "method": method, "cache_source_step": step, "n_images": len(subset),
            **{name: float(np.mean([float(row[name]) for row in subset])) for name in metrics},
            **{name: int(sum(int(row[name]) for row in subset)) for name in ("raw_proposal_count", "accepted_candidate_count", "teacher_fn")},
        })
    return output


def _find_dev_aggregate(rows: Sequence[Mapping[str, object]], method: str, step: int) -> Mapping[str, object]:
    matches = [row for row in rows if row["method"] == method and int(row["optimizer_steps"]) == step and row["level"] == "all"]
    if len(matches) != 1:
        raise AssertionError(f"Missing unique development aggregate for {method} at step {step}.")
    return matches[0]


def _comparison(left: Mapping[str, object], right: Mapping[str, object], name: str) -> dict[str, object]:
    metrics = ("dice", "aji", "aji_plus", "dq", "sq", "pq")
    return {
        "comparison": name,
        "left": left["method"], "right": right["method"],
        **{f"delta_{metric}": float(left[metric]) - float(right[metric]) for metric in metrics},
        **{f"delta_{metric}": int(left[metric]) - int(right[metric]) for metric in ("tp", "fp", "fn")},
    }


def _write_sha256s(artifact: Path) -> None:
    with (artifact / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in sorted(item for item in artifact.rglob("*") if item.is_file() and item.name != "SHA256SUMS"):
            handle.write(f"{sha256_file(path)}  {path.relative_to(artifact).as_posix()}\n")


def run_stage1(args: argparse.Namespace) -> Path:
    started = time.monotonic()
    repo = Path(__file__).resolve().parents[1]
    _assert_baseline(repo)
    if args.monuseg or args.allow_closed_patients:
        raise PermissionError("SemiPMS Stage 1 allows only TNBC patients 1--8; patients 9--11 and MoNuSeg are forbidden.")
    if args.warmup_steps != WARMUP_STEPS:
        raise PermissionError(f"Stage 1 fixes the shared warm-up at {WARMUP_STEPS} optimizer steps.")
    if args.total_steps <= args.warmup_steps:
        raise ValueError("--total-steps must exceed the shared 240-step warm-up.")
    if args.cache_refresh_steps <= 0:
        raise ValueError("--cache-refresh-steps must be positive.")
    data_root = Path(args.data_root).resolve()
    init_checkpoint = Path(args.init_checkpoint).resolve()
    validate_clean_checkpoint_name(init_checkpoint)
    official_payload = torch.load(init_checkpoint, map_location="cpu")
    provenance = inspect_clean_initialization(init_checkpoint, official_payload)
    frozen_rule, frozen_rule_meta = _load_frozen_rule(Path(args.phase0_rule).resolve())
    train_records, development = list_stage1_records(data_root)
    labeled, unlabeled = deterministic_split(train_records)
    run_id = args.run_id or f"semipms_stage1_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{_git(repo, 'rev-parse', '--short', 'HEAD')}"
    artifact = Path(args.output_root).resolve() / run_id
    if artifact.exists():
        raise FileExistsError(f"Refusing to overwrite {artifact}")
    (artifact / "checkpoints").mkdir(parents=True)
    write_json(artifact / "data_manifest.json", stage1_data_manifest(data_root, labeled, unlabeled, development))
    write_json(artifact / "checkpoint_provenance.json", provenance)
    shutil.copy2(Path(args.phase0_rule).resolve(), artifact / "frozen_acceptance_rule.json")
    frozen_rule_copy_sha = sha256_file(artifact / "frozen_acceptance_rule.json")
    if frozen_rule_copy_sha != frozen_rule_meta["source_sha256"]:
        raise AssertionError("Copied frozen Phase-0 rule checksum does not match its source.")
    (artifact / "environment.txt").write_text(json.dumps(_environment(), indent=2) + "\n", encoding="utf-8")
    _run_tests(artifact)
    _seed_everything(3407)
    device = torch.device(f"cuda:{args.gpu_device}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    access_guard = Stage1AccessGuard()
    # All pseudo thresholds, cache cadence, and loss weights are now frozen.
    access_guard.freeze_training_configuration()
    cfg = _runtime_config(args)
    args_cfg, warm_point, warm_encoder, warm_net = _build_models(args, official_payload, device)
    del official_payload
    initial_state_sha256 = _state_sha256(warm_point, warm_net)
    warm_optimizer = _new_optimizer(warm_point, warm_net)
    warm_training, warm_dev, warm_aux, _, warm_summary = _train_to_fixed_step(
        artifact, "Shared-Warmup", warm_point, warm_encoder, warm_net, warm_optimizer, args_cfg, cfg,
        data_root, labeled, development, access_guard, device,
        start_steps=0, total_steps=args.warmup_steps, eval_steps={args.warmup_steps},
        initial_state_sha256=initial_state_sha256, num_workers=args.num_workers,
    )
    shared_checkpoint = artifact / "checkpoints" / f"shared_warmup_final_{args.warmup_steps}.pth"
    if not shared_checkpoint.is_file():
        raise AssertionError("Shared 240-step checkpoint was not written.")
    shared_sha = sha256_file(shared_checkpoint)

    dev_rows: list[dict[str, object]] = []
    for method in ("Supervised-StainPMS-20", "MeanTeacher-PMS", "SemiPMS"):
        for row in warm_dev:
            dev_rows.append({**row, "method": method, "shared_warmup": True})
    training_rows: list[dict[str, object]] = list(warm_training)
    pseudo_step_rows: list[dict[str, object]] = [row for row in warm_aux if "optimizer_step" in row]
    checkpoint_rows: list[dict[str, object]] = [row for row in warm_aux if "path" in row]
    method_summaries: list[dict[str, object]] = [warm_summary]
    cache_rows: list[dict[str, object]] = []
    # Four fixed reporting points.  The final checkpoint is the only selected
    # model for all three paths; development curves never trigger early stop.
    interval = (args.total_steps - args.warmup_steps) // 3
    evaluation_steps = {args.warmup_steps + interval, args.warmup_steps + interval * 2, args.total_steps}

    def continue_from_shared(method: str, with_residual: bool) -> None:
        loaded_cfg, point_net, point_encoder, net, optimizer, texture_memory, payload = _load_checkpoint(shared_checkpoint, args, device)
        metadata = payload["semipms_stage1"]
        if int(metadata["optimizer_steps"]) != args.warmup_steps or metadata["initial_state_sha256"] != initial_state_sha256:
            raise PermissionError("Continuation checkpoint does not prove the shared Stage-1 warm-up protocol.")
        provider = None
        if method != "Supervised-StainPMS-20":
            provider = PseudoLossProvider(
                artifact, method, unlabeled, data_root, loaded_cfg, cfg, point_net, point_encoder, net, device,
                frozen_rule if with_residual else None,
                warmup_steps=args.warmup_steps, refresh_steps=args.cache_refresh_steps,
                residual_ramp_steps=args.residual_ramp_steps, ema_decay=args.ema_decay,
                base_weight=args.base_pseudo_weight, residual_weight=args.residual_pseudo_weight,
            )
        train_rows, current_dev, aux_rows, _, summary = _train_to_fixed_step(
            artifact, method, point_net, point_encoder, net, optimizer, loaded_cfg, cfg, data_root, labeled,
            development, access_guard, device, start_steps=args.warmup_steps, total_steps=args.total_steps,
            eval_steps=evaluation_steps, initial_state_sha256=initial_state_sha256,
            pseudo_provider=provider, initial_texture_memory=texture_memory, num_workers=args.num_workers,
        )
        training_rows.extend(train_rows); dev_rows.extend(current_dev)
        checkpoint_rows.extend(row for row in aux_rows if "path" in row)
        pseudo_step_rows.extend(row for row in aux_rows if "optimizer_step" in row)
        method_summaries.append(summary)
        if provider is not None:
            cache_rows.extend(provider.cache_rows)
        del point_net, point_encoder, net, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    continue_from_shared("Supervised-StainPMS-20", False)
    continue_from_shared("MeanTeacher-PMS", False)
    continue_from_shared("SemiPMS", True)
    # No hidden p1--6 annotation has been opened at this point.
    access_guard.mark_training_finished()

    formal_equivalence: dict[str, bool] = {}
    for method in ("Supervised-StainPMS-20", "MeanTeacher-PMS", "SemiPMS"):
        final_path = artifact / "checkpoints" / f"{method.lower().replace('-', '_')}_final_{args.total_steps}.pth"
        loaded_cfg, point_net, point_encoder, net, _, texture_memory, _ = _load_checkpoint(final_path, args, device)
        formal_equivalence[method] = _verify_formal_baseline_equivalence(
            artifact / "baseline_equivalence" / method.lower().replace("-", "_"), development[0], data_root,
            point_net, point_encoder, net, texture_memory, cfg, device,
        )
        del loaded_cfg, point_net, point_encoder, net
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not all(formal_equivalence.values()):
        raise AssertionError("A Stage-1 final model diverged from the formal standard inference path.")

    cache_audit, candidate_audit, fp_breakdown = _offline_cache_audit(artifact, unlabeled, access_guard)
    if access_guard.hidden_train_label_reads != len(unlabeled):
        raise AssertionError(
            f"Expected one post-training hidden-GT read for each of {len(unlabeled)} unlabeled train images; "
            f"observed {access_guard.hidden_train_label_reads}."
        )
    if sha256_file(artifact / "frozen_acceptance_rule.json") != frozen_rule_copy_sha:
        raise AssertionError("Frozen Phase-0 acceptance rule changed during Stage-1 training.")
    cache_summary = _cache_summary(cache_audit)
    _csv(artifact / "training_curve.csv", training_rows)
    _csv(artifact / "pseudo_training_losses.csv", pseudo_step_rows)
    _csv(artifact / "pseudo_label_statistics.csv", cache_rows)
    _csv(artifact / "pseudo_cache_audit.csv", cache_audit)
    _csv(artifact / "pseudo_cache_audit_summary.csv", cache_summary)
    _csv(artifact / "pseudo_candidate_audit.csv", candidate_audit)
    _csv(artifact / "fp_source_breakdown.csv", fp_breakdown)
    _csv(artifact / "per_image_metrics.csv", dev_rows)
    dev_aggregate = _aggregate_development(dev_rows)
    _csv(artifact / "per_patient_metrics.csv", dev_aggregate)

    supervised_final = _find_dev_aggregate(dev_aggregate, "Supervised-StainPMS-20", args.total_steps)
    mean_teacher_final = _find_dev_aggregate(dev_aggregate, "MeanTeacher-PMS", args.total_steps)
    semipms_final = _find_dev_aggregate(dev_aggregate, "SemiPMS", args.total_steps)
    supervised_warmup = _find_dev_aggregate(dev_aggregate, "Supervised-StainPMS-20", args.warmup_steps)
    undertraining = {f"delta_{name}_240_to_{args.total_steps}": float(supervised_final[name]) - float(supervised_warmup[name]) for name in ("dice", "aji", "dq", "sq", "pq")}
    report = {
        "phase": "SemiPMS Stage 1 -- TNBC development fair comparison",
        "git_sha": _git(repo, "rev-parse", "HEAD"),
        "canonical_baseline": CANONICAL_BASELINE,
        "checkpoint_provenance": provenance,
        "shared_initialization": {
            "official_checkpoint_sha256": provenance["sha256"], "point_head": "random with seed 3407",
            "initial_state_sha256": initial_state_sha256, "shared_warmup_checkpoint": str(shared_checkpoint),
            "shared_warmup_checkpoint_sha256": shared_sha, "warmup_steps": args.warmup_steps,
            "total_optimizer_steps_each_path": args.total_steps,
        },
        "data_manifest": "data_manifest.json",
        "access_guard": {
            **access_guard.manifest(), "hidden_train_label_reads_after_training": access_guard.hidden_train_label_reads,
            "development_label_reads": access_guard.development_label_reads, "expected_hidden_train_images": 24,
        },
        "frozen_acceptance_rule": {**frozen_rule_meta, "stage1_copy_sha256": frozen_rule_copy_sha},
        "pseudo_protocol": {
            "ema_decay": args.ema_decay, "cache_refresh_steps": args.cache_refresh_steps,
            "base_pseudo": "EMA standard deployment instance maps; no hematoxylin residual proposals",
            "residual_pseudo": "H residual outside teacher coverage; frozen Phase-0 cross-view rule; mask-level dedupe",
            "unknown_region_supervision": "not used as negative background: pseudo loss is foreground-positive only",
            "base_preservation_prompts": "base pseudo points are trained as a separate preservation loss",
            "residual_ramp_steps": args.residual_ramp_steps,
        },
        "baseline_adequacy": {
            "supervised_240_step_development": supervised_warmup,
            "supervised_final_development": supervised_final,
            "change": undertraining,
            "interpretation": "continuous development evidence only; it does not auto-declare 240 steps undertrained.",
        },
        "development_final": {
            "supervised": supervised_final, "mean_teacher": mean_teacher_final, "semipms": semipms_final,
            "semipms_minus_supervised": _comparison(semipms_final, supervised_final, "SemiPMS - Supervised-StainPMS-20"),
            "semipms_minus_mean_teacher": _comparison(semipms_final, mean_teacher_final, "SemiPMS - MeanTeacher-PMS"),
        },
        "cache_post_training_diagnostics": cache_summary,
        "training_cost": method_summaries,
        "checkpoints": checkpoint_rows,
        "formal_standard_inference_equivalence": formal_equivalence,
        "tests": {"guard_and_residual_tests": "tests.txt", "baseline_inference_equivalence": formal_equivalence, "checksum_guard": True},
        "reference_interpretation": {
            "strong_success": "development DeltaPQ >= +0.020",
            "valuable": "development DeltaPQ from +0.010 to +0.020",
            "small": "development DeltaPQ from 0 to +0.010; analyse bottlenecks",
            "negative": "multi-patient sustained reverse movement warrants lead review",
            "automatic_verdict": "disabled; project lead decides from the complete evidence",
        },
        "runtime_seconds": time.monotonic() - started,
        "stop_condition": "Stage 1 development comparison complete; do not access TNBC patients 9--11, MoNuSeg, multi-seed, or start a further stage.",
    }
    write_json(artifact / "report.json", report)
    _write_sha256s(artifact)
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SemiPMS Stage 1 TNBC development experiment")
    parser.add_argument("--data-root", default="data/tnbc")
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--phase0-rule", required=True, help="frozen_acceptance_rule.json from the valid Phase-0 artifact")
    parser.add_argument("--output-root", default="logs/semipms/stage1_tnbc_dev")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--sam-config", default="sam2_hiera_l")
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--total-steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--cache-refresh-steps", type=int, default=CACHE_REFRESH_STEPS)
    parser.add_argument("--ema-decay", type=float, default=EMA_DECAY)
    parser.add_argument("--residual-ramp-steps", type=int, default=RESIDUAL_RAMP_STEPS)
    parser.add_argument("--base-pseudo-weight", type=float, default=0.25)
    parser.add_argument("--residual-pseudo-weight", type=float, default=0.25)
    parser.add_argument("--monuseg", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--allow-closed-patients", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    artifact = run_stage1(build_parser().parse_args(argv))
    print(f"SemiPMS Stage 1 complete: {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
