"""SemiPMS Phase 0: frozen low-label residual-support expansion audit.

This runner deliberately separates three states:

1. six-image supervised weak-teacher training;
2. labeled-only leave-one-patient-out acceptance-rule calibration;
3. one hidden-GT audit over the other 24 patient-1--6 images.

No closed-patient label can be read until ``HiddenGTGuard.freeze_acceptance_rule``
has been invoked. There is no EMA student, pseudo-label training, checkpoint
selection on unlabeled images, or deployment-path modification.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import albumentations as A
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from mmengine.config import Config
from scipy.ndimage import binary_dilation
from skimage import io
from torch.utils.data import DataLoader

from run.dataset.monuseg import MONUSEG
from sam2_train.build_sam import build_sam2
from sam2_train.modeling.dpa_p2pnet import build_model
from sam2_train.modeling.stats_utils import (
    get_dice_1,
    get_fast_aji,
    get_fast_aji_plus,
    get_fast_dice_2,
    get_fast_pq,
    remap_label,
)

from semipms.guards import (
    HiddenGTGuard,
    ImageRecord,
    data_manifest,
    deterministic_split,
    inspect_clean_initialization,
    list_allowed_images,
    sha256_file,
    validate_clean_checkpoint_name,
    write_json,
)
from semipms.residual import (
    DEFAULT_BUDGETS,
    acceptance_features,
    candidate_rows,
    frozen_accept,
    geometric_view,
    h_channel_evidence,
    inverse_stain_mask,
    inverse_geometric_mask,
    propose_residual_points,
    residual_evidence,
    stain_perturbation,
    transform_points_xy,
)


CANONICAL_BASELINE = "2a1348cb7a1158a6f77aae2f92c168f9552d8068"
TRAIN_STEPS = 240
CALIBRATION_HIDE_FRACTION = 0.20
RULE_GRID = tuple(round(value, 2) for value in np.arange(0.30, 0.91, 0.05))


class StepBudgetReached(RuntimeError):
    pass


class StepBudgetOptimizer:
    """Proxy optimiser that stops immediately after the fixed final update."""

    def __init__(self, optimizer: torch.optim.Optimizer, max_steps: int) -> None:
        self._optimizer = optimizer
        self.max_steps = int(max_steps)
        self.steps = 0

    @property
    def param_groups(self):
        return self._optimizer.param_groups

    def zero_grad(self, *args, **kwargs):
        return self._optimizer.zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        result = self._optimizer.step(*args, **kwargs)
        self.steps += 1
        if self.steps >= self.max_steps:
            raise StepBudgetReached
        return result

    def __getattr__(self, name: str):
        return getattr(self._optimizer, name)


def _legacy_helpers():
    """Load CA-SAM2 helpers without letting their legacy parser consume this CLI."""
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        from run.run_on_epoch import (  # pylint: disable=import-outside-toplevel
            _assemble_instance_map,
            _ori_hw,
            combine_mask,
            context_memory_attention,
            crop_with_overlap,
            inference,
            mask_process_eval,
            train_on_epoch,
            validation_on_epoch,
        )
        from sam2_train.modeling.utils import (  # pylint: disable=import-outside-toplevel
            collate_fn,
            point_nms,
            predict,
        )
        from sam2_train.modeling.criterion import build_criterion  # pylint: disable=import-outside-toplevel
    finally:
        sys.argv = original_argv
    return SimpleNamespace(
        assemble=_assemble_instance_map,
        ori_hw=_ori_hw,
        combine_mask=combine_mask,
        context=context_memory_attention,
        crops=crop_with_overlap,
        inference=inference,
        mask_process=mask_process_eval,
        train=train_on_epoch,
        validate=validation_on_epoch,
        collate=collate_fn,
        point_nms=point_nms,
        predict=predict,
        build_criterion=build_criterion,
    )


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _assert_baseline(repo: Path) -> None:
    if subprocess.run(["git", "merge-base", "--is-ancestor", CANONICAL_BASELINE, "HEAD"], cwd=repo).returncode:
        raise RuntimeError(f"SemiPMS must descend from canonical baseline {CANONICAL_BASELINE}.")


def _environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def _csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, default=_json_default) if isinstance(value, (dict, list, np.ndarray)) else value
                for key, value in row.items()
            })


def _normalise(image: np.ndarray) -> torch.Tensor:
    arr = A.Compose([A.Normalize()])(image=np.asarray(image)[..., :3])["image"]
    return torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0)


def _read_image(record: ImageRecord) -> tuple[np.ndarray, torch.Tensor]:
    raw = io.imread(record.image_path)[..., :3]
    return raw, _normalise(raw)


def _read_label(record: ImageRecord, hidden_guard: HiddenGTGuard | None = None, *, unlabeled: bool = False) -> np.ndarray:
    if unlabeled:
        if hidden_guard is None:
            raise AssertionError("Unlabeled GT requires an explicit guard.")
        hidden_guard.allow_unlabeled_label_read(record)
    return sio.loadmat(record.label_path)["inst_map"].astype(np.int32)


def _runtime_config(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        crop_size=256,
        out_size=256,
        overlap=32,
        load="unclockwise",
        b=1,
        print_freq=100,
        clip_grad=0.1,
        context=True,
        texture=True,
        texture_memory_bank_size=64,
        context_memory_bank_size=100,
        context_atten_k=1,
        tta=False,
        vis=False,
        use_pms=True,
        pms_loss_coef=0.5,
    )


def _freeze_encoder(net: torch.nn.Module) -> None:
    for name, parameter in net.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            parameter.requires_grad_(False)


def _build_models(args: argparse.Namespace, checkpoint_payload: Mapping[str, Any], device: torch.device):
    args_cfg = Config.fromfile("args.py")
    # These are the existing low-label StainPMS losses, applied only on the six
    # labeled images. No self-bootstrap map or unlabeled pseudo-label enters.
    args_cfg.criterion.pms_loss_coef = 0.5
    args_cfg.criterion.pms_object_weight = 1.0
    args_cfg.criterion.pms_residual_mask_weight = 0.3
    args_cfg.criterion.pms_preserve_loss_coef = 1.0
    point_net, point_encoder = build_model(args_cfg)
    net = build_sam2(args.sam_config, None, device=device)
    missing, unexpected = net.load_state_dict(checkpoint_payload["model"], strict=False)
    # This exactly mirrors CA-SAM2's official SAM2 initialization: the public
    # backbone weights do not contain every project-specific prompt/memory
    # module. Unexpected checkpoint tensors would indicate the wrong model,
    # whereas missing project modules are randomly initialized by design.
    if unexpected:
        raise RuntimeError(f"Official SAM2 contains incompatible unexpected keys: {len(unexpected)}")
    point_net.to(device)
    point_encoder.to(device)
    net.to(device)
    _freeze_encoder(net)
    args_cfg.semipms_init_load = {"missing_keys": len(missing), "unexpected_keys": len(unexpected), "strict": False}
    return args_cfg, point_net, point_encoder, net


def _train_weak_teacher(
    artifact: Path,
    labeled: Sequence[ImageRecord],
    data_root: Path,
    args: argparse.Namespace,
    checkpoint_payload: Mapping[str, Any],
    device: torch.device,
) -> tuple[Path, dict[str, Any]]:
    helpers = _legacy_helpers()
    args_cfg, point_net, point_encoder, net = _build_models(args, checkpoint_payload, device)
    cfg = _runtime_config(args)
    dataset = MONUSEG(cfg, args_cfg, str(data_root), cfg.load, mode="train")
    dataset.paths = [f"{record.stem}{Path(record.image_path).suffix}" for record in labeled]
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=helpers.collate)
    criterion, _ = helpers.build_criterion(args_cfg, device)
    base_optimizer = torch.optim.AdamW(
        [parameter for module in (point_net, net) for parameter in module.parameters() if parameter.requires_grad],
        lr=1e-4,
        weight_decay=1e-4,
    )
    optimizer = StepBudgetOptimizer(base_optimizer, args.train_steps)
    curve: list[dict[str, Any]] = []
    started = time.monotonic()
    epoch = 0
    texture_bank: list[Any] = []
    while optimizer.steps < args.train_steps:
        texture_bank = []
        epoch_start = time.monotonic()
        completed = True
        try:
            log_info = helpers.train(cfg, point_net, point_encoder, net, loader, criterion, optimizer, epoch, texture_bank, device)
        except StepBudgetReached:
            completed = False
            log_info = {"partial_epoch": True}
        curve.append({
            "epoch": epoch,
            "optimizer_steps_completed": optimizer.steps,
            "epoch_completed": completed,
            "seconds": time.monotonic() - epoch_start,
            **{key: float(value) for key, value in log_info.items()},
        })
        epoch += 1
    checkpoint = artifact / "weak_teacher_final_fixed_step.pth"
    torch.save(
        {
            "model": net.state_dict(),
            "model1": point_net.state_dict(),
            "texture_memory_bank_list": texture_bank,
            "semipms_training": {
                "labeled_stems": [record.stem for record in labeled],
                "fixed_optimizer_steps": args.train_steps,
                "selection_rule": "final_fixed_step_only_no_unlabeled_gt_or_early_stopping",
            },
        },
        checkpoint,
    )
    _csv(artifact / "training_curve.csv", curve)
    return checkpoint, {
        "steps": optimizer.steps,
        "epochs_started": epoch,
        "seconds": time.monotonic() - started,
        "cuda_max_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "selection_rule": "final fixed optimizer step; no hidden-GT evaluation, early stopping, or model selection",
        "official_sam2_load": dict(args_cfg.semipms_init_load),
    }


def _load_weak_teacher(checkpoint: Path, args: argparse.Namespace, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu")
    args_cfg = Config.fromfile("args.py")
    point_net, point_encoder = build_model(args_cfg)
    net = build_sam2(args.sam_config, None, device=device)
    net.load_state_dict(payload["model"], strict=True)
    point_net.load_state_dict(payload["model1"], strict=True)
    point_net.to(device).eval(); point_encoder.to(device).eval(); net.to(device).eval()
    for module in (point_net, point_encoder, net):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return point_net, point_encoder, net, list(payload.get("texture_memory_bank_list", []) or [])


def _append_texture_memory(net, vision_feats, image_embed, prompts, pred, values, memory_bank, cfg, helpers, device):
    if not cfg.texture or len(prompts) == 0:
        return
    feat_sizes = [(64, 64), (32, 32), (16, 16)]
    inst = helpers.combine_mask(torch.as_tensor([256, 256]), prompts, pred, values)
    high = torch.from_numpy(inst.astype(float)).float().unsqueeze(0).unsqueeze(0).to(device)
    features, positions = net._encode_new_memory(vision_feats, feat_sizes, high, is_mask_from_pts=True)
    features, positions = features.to(device), positions[0].to(device)
    mean_iou = values.mean()
    for batch_index in range(features.size(0)):
        record = [features[batch_index].unsqueeze(0), positions[batch_index].unsqueeze(0), mean_iou, image_embed[batch_index].reshape(-1).detach()]
        if len(memory_bank) < cfg.texture_memory_bank_size:
            memory_bank.append(record)
            continue
        flat = torch.stack([item[0].reshape(-1).to(device) for item in memory_bank])
        normalized = F.normalize(flat, p=2, dim=1)
        similarity = torch.mm(normalized, normalized.t())
        no_diag = similarity.clone(); no_diag[torch.arange(no_diag.size(0)), torch.arange(no_diag.size(0))] = float("-inf")
        new = F.normalize(features[batch_index].reshape(-1), p=2, dim=0).unsqueeze(1)
        scores = torch.mm(normalized, new).squeeze()
        least = torch.argmin(scores); replace = torch.argmax(no_diag[least])
        if scores[least] < no_diag[least][replace] and mean_iou > memory_bank[int(replace)][2] - 0.1:
            memory_bank.pop(int(replace)); memory_bank.append(record)


@torch.inference_mode()
def _infer_standard(image: torch.Tensor, point_net, point_encoder, net, seed_memory: list[Any], cfg, device) -> np.ndarray:
    """CA-SAM2 validation point/filter/NMS/assembly path without reading GT."""
    helpers = _legacy_helpers()
    height, width = image.shape[-2:]
    all_masks: list[np.ndarray] = []; all_boxes = []; all_scores = []; all_inds = []
    all_points = []; all_point_scores = []; all_classes = []; processed = []
    point_ids: dict[tuple[float, float], int] = {}; next_id = 0
    memory_bank = list(seed_memory); context_bank: list[Any] = []
    for crop_box in helpers.crops(image[0], cfg.crop_size, cfg.crop_size, cfg.overlap, cfg.load).tolist():
        x1, y1, x2, y2 = (int(value) for value in crop_box)
        crop = image[..., y1:y2, x1:x2].to(device)
        points, scores, classes, _, _, _, _ = helpers.predict(point_net, crop, ori_shape=np.asarray((y2-y1, x2-x1)), filtering=True, nms_thr=12)
        if len(points) == 0:
            processed.append(crop_box); continue
        points[:, 0] += x1; points[:, 1] += y1
        keep_new = np.ones(len(points), dtype=bool)
        for px1, py1, px2, py2 in processed:
            keep_new &= ~((points[:,0] >= px1+1) & (points[:,0] <= px2-1) & (points[:,1] >= py1+1) & (points[:,1] <= py2-1))
        processed.append(crop_box)
        points, scores, classes = points[keep_new], scores[keep_new], classes[keep_new]
        if len(points) == 0: continue
        all_points.append(points); all_point_scores.append(scores); all_classes.append(classes)
        current_points, current_scores, current_classes = helpers.point_nms(np.vstack(all_points), np.concatenate(all_point_scores), np.concatenate(all_classes), 12)
        current_inds = []
        for point in current_points:
            key = tuple(float(value) for value in point)
            if key not in point_ids:
                point_ids[key] = next_id; next_id += 1
            current_inds.append(point_ids[key])
        prompts = torch.from_numpy(current_points).unsqueeze(1)
        inside = ((prompts[...,0] >= x1) & (prompts[...,0] < x2) & (prompts[...,1] >= y1) & (prompts[...,1] < y2)).squeeze(1)
        if inside.sum() == 0: continue
        local = (prompts[inside] - torch.as_tensor([x1, y1])).to(device).float()
        labels = torch.ones(local.size(0), 1, dtype=torch.int, device=device)
        pred, values, _, vision_feats, image_embed = helpers.inference(net, point_encoder, crop, memory_bank, local, labels, [(64,64),(32,32),(16,16)], context_bank, x1, y1, True, cfg, device)
        self_masks = helpers.mask_process(current_classes[inside.cpu().numpy()], torch.as_tensor(current_inds).long()[inside], crop_box, np.asarray([height,width]), local, pred, values)
        _append_texture_memory(net, vision_feats, image_embed, local, pred, values, memory_bank, cfg, helpers, device)
        for item in self_masks:
            bx1, by1, bx2, by2 = item["bbox"]
            edge = ((bx1 > 7 and abs(bx1-x1) <= 7) or (abs(bx2-height) > 7 and abs(bx2-x2) <= 7) or (by1 > 7 and abs(by1-y1) <= 7) or (abs(by2-width) > 7 and abs(by2-y2) <= 7))
            all_masks.append(item["segmentation"][:height,:width]); all_boxes.append(item["bbox"]); all_scores.append(float(item["predicted_iou"] * .3 if edge else item["predicted_iou"])); all_inds.append(item["inds"])
    return helpers.assemble(all_boxes, all_scores, all_masks, all_inds, (height, width), 0.5)


@torch.inference_mode()
def _decode_candidate_view(image: torch.Tensor, candidates, point_net, point_encoder, net, cfg, device, *, transform_xy=None, inverse_mask=None) -> list[dict[str, Any]]:
    """One image embedding per crop and batched frozen SAM2 decoder calls."""
    helpers = _legacy_helpers()
    height, width = image.shape[-2:]
    outputs: dict[int, dict[str, Any]] = {}
    for crop_box in helpers.crops(image[0], cfg.crop_size, cfg.crop_size, cfg.overlap, cfg.load).tolist():
        x1, y1, x2, y2 = (int(value) for value in crop_box)
        active = []
        for index, candidate in enumerate(candidates):
            point = np.asarray([[candidate.x, candidate.y]], dtype=np.float32)
            if transform_xy is not None: point = transform_xy(point)
            if x1 <= point[0,0] < x2 and y1 <= point[0,1] < y2 and index not in outputs:
                active.append((index, point[0]))
        if not active: continue
        crop = image[..., y1:y2, x1:x2].to(device)
        local = torch.as_tensor([[point[0]-x1, point[1]-y1] for _, point in active], dtype=torch.float32, device=device).unsqueeze(1)
        labels = torch.ones(len(active), 1, dtype=torch.int, device=device)
        pred, values, _, _, _ = helpers.inference(net, point_encoder, crop, [], local, labels, [(64,64),(32,32),(16,16)], [], x1, y1, False, cfg, device)
        for local_index, (candidate_index, _) in enumerate(active):
            full = np.zeros((height, width), dtype=bool)
            mask = (pred[local_index] > 0).detach().cpu().numpy()
            full[y1:y2, x1:x2] = mask[:y2-y1, :x2-x1]
            outputs[candidate_index] = {"mask": inverse_mask(full) if inverse_mask else full, "predicted_iou": float(values[local_index])}
    return [outputs.get(index, {"mask": np.zeros((height,width), dtype=bool), "predicted_iou": 0.0}) for index in range(len(candidates))]


def _hidden_instances(label_map: np.ndarray, patient: int) -> set[int]:
    ids = [int(value) for value in np.unique(label_map) if value != 0]
    n_hide = max(1, int(math.ceil(len(ids) * CALIBRATION_HIDE_FRACTION)))
    return set(sorted(ids, key=lambda value: hashlib.sha256(f"3407:{patient}:{value}".encode()).hexdigest())[:n_hide])


def _candidate_feature_records(raw, normal, teacher_map, candidates, point_net, point_encoder, net, cfg, device) -> list[dict[str, Any]]:
    original = _decode_candidate_view(normal, candidates, point_net, point_encoder, net, cfg, device)
    stain = _decode_candidate_view(_normalise(stain_perturbation(raw)).to(device), candidates, point_net, point_encoder, net, cfg, device)
    geometry = _decode_candidate_view(_normalise(geometric_view(raw)).to(device), candidates, point_net, point_encoder, net, cfg, device, transform_xy=transform_points_xy, inverse_mask=inverse_geometric_mask)
    h = h_channel_evidence(raw)
    rows = []
    for index, candidate in enumerate(candidates):
        features = acceptance_features(original[index]["mask"], stain[index]["mask"], geometry[index]["mask"], h, teacher_map)
        rows.append({
            "candidate_index": index,
            "x": candidate.x, "y": candidate.y, "evidence": candidate.evidence, "source": candidate.source,
            "mask": original[index]["mask"], "predicted_iou": original[index]["predicted_iou"],
            "features": features,
        })
    return rows


def _verify_formal_baseline_equivalence(
    artifact: Path, record: ImageRecord, data_root: Path, point_net, point_encoder, net, seed_memory, cfg, device
) -> bool:
    """Compare the GT-free replay map with the repository's formal validation map.

    This is run only on one of the six labeled images. It never touches an
    unlabeled label and proves that weak-teacher coverage uses the established
    deployment filtering/NMS/assembly implementation.
    """
    helpers = _legacy_helpers()
    raw, image = _read_image(record)
    del raw
    replay = _infer_standard(image.to(device), point_net, point_encoder, net, seed_memory, cfg, device)
    args_cfg = Config.fromfile("args.py")
    dataset = MONUSEG(cfg, args_cfg, str(data_root), cfg.load, mode="test")
    # Phase 0 deliberately has no access to TNBC test/images (patients 9--11).
    # The equivalence fixture is one of the six labeled train_12 files.
    dataset.image_root = str(data_root / "train_12" / "images")
    dataset.label_root = str(data_root / "train_12" / "labels")
    dataset.paths = [f"{record.stem}{Path(record.image_path).suffix}"]
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    formal_dir = artifact / "baseline_equivalence"
    cfg.dump_eval_artifacts_dir = str(formal_dir)
    try:
        helpers.validate(cfg, args_cfg, loader, 0, point_net, point_encoder, net, cfg.load, args_cfg.data.post.iou_threshold, list(seed_memory), device)
    finally:
        cfg.dump_eval_artifacts_dir = ""
    formal = np.load(formal_dir / f"{record.stem}_pred.npy")
    return bool(np.array_equal(replay, formal))


def _base_rule() -> dict[str, float]:
    return {"min_view_iou": 0.50, "max_centroid_displacement": 6.0, "min_area_stability": 0.45, "min_h_occupancy": 0.10, "min_boundary_stability": 0.25, "max_pseudo_conflict": 0.35}


def _calibrate_rule(calibration_rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """LOPO selection of one scalar view-consistency threshold, then median freeze."""
    base = _base_rule(); folds = []
    patients = sorted({int(row["patient"]) for row in calibration_rows})
    for held_out in patients:
        train = [row for row in calibration_rows if int(row["patient"]) != held_out]
        scored = []
        for threshold in RULE_GRID:
            rule = dict(base, min_view_iou=threshold)
            selected = [row for row in train if frozen_accept(row["features"], rule)]
            positives = sum(bool(row["hidden_hit"]) for row in selected)
            all_positive = sum(bool(row["hidden_hit"]) for row in train)
            precision = positives / len(selected) if selected else 0.0
            recall = positives / all_positive if all_positive else 0.0
            scored.append((precision * recall, precision, recall, -threshold, threshold, len(selected)))
        _, precision, recall, _, threshold, selected_count = max(scored)
        folds.append({"held_out_patient": held_out, "train_selected_threshold": threshold, "train_precision": precision, "train_recall": recall, "train_selected_count": selected_count})
    frozen = dict(base, min_view_iou=float(np.median([row["train_selected_threshold"] for row in folds])))
    return frozen, folds


def _metrics(gt: np.ndarray, prediction: np.ndarray) -> tuple[dict[str, Any], set[int]]:
    true = remap_label(gt); pred = remap_label(prediction)
    pq, pairing = get_fast_pq(true, pred, match_iou=0.5)
    gt_ids = sorted(int(value) for value in np.unique(gt) if value != 0)
    missed = {gt_ids[int(index)-1] for index in pairing[2]}
    return {
        "dice": float(get_dice_1(true, pred)), "dice2": float(get_fast_dice_2(true, pred)),
        "aji": float(get_fast_aji(true, pred)), "aji_plus": float(get_fast_aji_plus(true, pred)),
        "dq": float(pq[0]), "sq": float(pq[1]), "pq": float(pq[2]),
        "tp": len(pairing[0]), "fp": len(pairing[3]), "fn": len(pairing[2]),
    }, missed


def _add_masks(base: np.ndarray, candidates: Sequence[Mapping[str, Any]], *, oracle: bool, missed: set[int], gt: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    out = np.asarray(base, dtype=np.int32).copy(); next_id = int(out.max()) + 1
    seen_targets: set[int] = set(); info = Counter()
    rows = sorted(candidates, key=lambda row: (-float(row["evidence"]), int(row["candidate_index"])))
    for row in rows:
        target = int(gt[int(row["y"]), int(row["x"])]) if 0 <= int(row["y"]) < gt.shape[0] and 0 <= int(row["x"]) < gt.shape[1] else 0
        if oracle and target not in missed: continue
        mask = np.asarray(row["mask"], dtype=bool)
        if not mask.any(): continue
        overlaps = set(int(value) for value in np.unique(gt[mask]) if value != 0)
        if len(overlaps) > 1: info["merge"] += 1
        if target and target in seen_targets: info["duplicate"] += 1; continue
        uncovered = mask & (out == 0)
        if uncovered.sum() < 8: info["conflict_rejected"] += 1; continue
        out[uncovered] = next_id; next_id += 1; info["added"] += 1
        if target: seen_targets.add(target)
    return out, dict(info)


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    names = ("dice", "dice2", "aji", "aji_plus", "dq", "sq", "pq")
    out = {name: float(np.mean([row[name] for row in rows])) for name in names}
    out.update({name: int(sum(row[name] for row in rows)) for name in ("tp", "fp", "fn")})
    return out


def _run_tests(artifact: Path) -> str:
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests/semipms", "-v"]
    result = subprocess.run(command, text=True, capture_output=True)
    text = result.stdout + result.stderr
    (artifact / "tests.txt").write_text(text, encoding="utf-8")
    if result.returncode:
        raise RuntimeError("SemiPMS guard/residual tests failed; see tests.txt")
    return text


def _resume_weak_teacher(artifact: Path, resume_from: Path, args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    """Reuse a completed fixed-step weak teacher after a post-training failure.

    This does not alter the interrupted artifact and never executes an
    optimiser step. It is intentionally narrow: only a checkpoint explicitly
    marked as the required final fixed-step SemiPMS teacher is accepted.
    """
    source = resume_from.resolve()
    checkpoint = source / "weak_teacher_final_fixed_step.pth"
    curve = source / "training_curve.csv"
    if not checkpoint.is_file() or not curve.is_file():
        raise FileNotFoundError("--resume-from must contain weak_teacher_final_fixed_step.pth and training_curve.csv")
    payload = torch.load(checkpoint, map_location="cpu")
    training_meta = payload.get("semipms_training", {})
    if int(training_meta.get("fixed_optimizer_steps", -1)) != int(args.train_steps):
        raise PermissionError("Resume checkpoint does not prove the preregistered fixed-step weak-teacher protocol.")
    if not {"model", "model1"}.issubset(payload):
        raise PermissionError("Resume checkpoint lacks frozen weak-teacher model state.")
    shutil.copy2(curve, artifact / "training_curve.csv")
    return checkpoint, {
        "resumed_without_optimizer_steps": True,
        "source_artifact": str(source),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "steps": int(training_meta["fixed_optimizer_steps"]),
        "selection_rule": "reused final fixed-step checkpoint after post-training infrastructure failure",
    }


def run_phase0(args: argparse.Namespace) -> Path:
    started = time.monotonic()
    repo = Path(__file__).resolve().parents[1]; _assert_baseline(repo)
    if args.ema_student or args.monuseg or args.allow_closed_patients:
        raise PermissionError("EMA student, MoNuSeg, and patients 7--11 are forbidden in SemiPMS Phase 0.")
    data_root = Path(args.data_root).resolve(); init_checkpoint = Path(args.init_checkpoint).resolve()
    # One CPU load supplies both provenance validation and model construction;
    # do not rescan a multi-GB official checkpoint.
    validate_clean_checkpoint_name(init_checkpoint)
    official_payload = torch.load(init_checkpoint, map_location="cpu")
    provenance = inspect_clean_initialization(init_checkpoint, official_payload)
    run_id = args.run_id or f"semipms_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{_git(repo, 'rev-parse', '--short', 'HEAD')}"
    artifact = Path(args.output_root).resolve() / run_id
    if artifact.exists(): raise FileExistsError(f"Refusing to overwrite {artifact}")
    artifact.mkdir(parents=True)
    records = list_allowed_images(data_root); labeled, unlabeled = deterministic_split(records)
    manifest = data_manifest(data_root, labeled, unlabeled)
    write_json(artifact / "data_manifest.json", manifest); write_json(artifact / "checkpoint_provenance.json", provenance)
    (artifact / "environment.txt").write_text(json.dumps(_environment(), indent=2) + "\n", encoding="utf-8")
    _run_tests(artifact)
    torch.manual_seed(3407); np.random.seed(3407)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(3407)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    device = torch.device(f"cuda:{args.gpu_device}" if torch.cuda.is_available() else "cpu")
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    if args.resume_from:
        weak_checkpoint, training = _resume_weak_teacher(artifact, Path(args.resume_from), args)
    else:
        weak_checkpoint, training = _train_weak_teacher(artifact, labeled, data_root, args, official_payload, device)
    del official_payload
    point_net, point_encoder, net, seed_memory = _load_weak_teacher(weak_checkpoint, args, device)
    cfg = _runtime_config(args); guard = HiddenGTGuard()
    # A repeat of the exact GT-free standard deployment path must produce the
    # same map before it can serve as the weak-teacher coverage reference.
    if not _verify_formal_baseline_equivalence(artifact, labeled[0], data_root, point_net, point_encoder, net, seed_memory, cfg, device):
        raise AssertionError("Baseline inference equivalence failed on the labeled preflight image.")

    calibration_rows: list[dict[str, Any]] = []
    for record in labeled:
        raw, normal = _read_image(record); normal = normal.to(device)
        teacher = _infer_standard(normal, point_net, point_encoder, net, seed_memory, cfg, device)
        gt = _read_label(record)
        hidden = _hidden_instances(gt, record.patient)
        simulated = teacher.copy()
        simulated[np.isin(gt, list(hidden))] = 0
        candidates = propose_residual_points(residual_evidence(raw, simulated), max_candidates=args.max_candidates)
        rows = _candidate_feature_records(raw, normal, simulated, candidates, point_net, point_encoder, net, cfg, device)
        for row in rows:
            target = int(gt[int(row["y"]), int(row["x"])])
            calibration_rows.append({"patient": record.patient, "image": record.stem, "hidden_hit": target in hidden, **row})
    rule, folds = _calibrate_rule(calibration_rows)
    rule_payload = {"rule": rule, "leave_one_patient_out": folds, "frozen_before_unlabeled_gt": True}
    write_json(artifact / "frozen_acceptance_rule.json", rule_payload)
    rule_payload["sha256"] = sha256_file(artifact / "frozen_acceptance_rule.json")
    guard.freeze_acceptance_rule()

    instance_rows: list[dict[str, Any]] = []; candidate_curve: list[dict[str, Any]] = []; per_image: list[dict[str, Any]] = []
    contributions: list[float] = []
    for record in unlabeled:
        raw, normal = _read_image(record); normal = normal.to(device)
        teacher = _infer_standard(normal, point_net, point_encoder, net, seed_memory, cfg, device)
        proposals = propose_residual_points(residual_evidence(raw, teacher), max_candidates=args.max_candidates)
        rows = _candidate_feature_records(raw, normal, teacher, proposals, point_net, point_encoder, net, cfg, device)
        for row in rows: row["accepted"] = frozen_accept(row["features"], rule)
        # The first and only hidden-GT access for this image occurs below.
        gt = _read_label(record, guard, unlabeled=True)
        baseline_metrics, missed = _metrics(gt, teacher)
        for row in rows:
            y, x = int(row["y"]), int(row["x"]); target = int(gt[y, x]) if 0 <= y < gt.shape[0] and 0 <= x < gt.shape[1] else 0
            target_mask = gt == target
            union = int(np.logical_or(row["mask"], target_mask).sum()) if target else 0
            mask_iou = float(np.logical_and(row["mask"], target_mask).sum() / union) if union else 0.0
            row.update({"image": record.stem, "patient": record.patient, "target_gt_id": target, "hits_teacher_fn": target in missed, "mask_iou_to_target": mask_iou})
            instance_rows.append({key: value for key, value in row.items() if key != "mask"})
        for budget in DEFAULT_BUDGETS:
            top = rows[:budget]; hits = {int(row["target_gt_id"]) for row in top if row["hits_teacher_fn"]}
            candidate_curve.append({"image": record.stem, "patient": record.patient, "budget": budget, "proposals": len(top), "teacher_fn": len(missed), "fn_recall": len(hits)/len(missed) if missed else 0.0, "proposal_precision": sum(bool(row["hits_teacher_fn"]) for row in top)/len(top) if top else 0.0})
        oracle_map, oracle_info = _add_masks(teacher, rows, oracle=True, missed=missed, gt=gt)
        selected_map, selected_info = _add_masks(teacher, [row for row in rows if row["accepted"]], oracle=False, missed=missed, gt=gt)
        oracle_metrics, _ = _metrics(gt, oracle_map); selected_metrics, _ = _metrics(gt, selected_map)
        contributions.append(max(0.0, selected_metrics["pq"] - baseline_metrics["pq"]))
        per_image.append({
            "image": record.stem, "patient": record.patient, **{f"teacher_{k}": v for k,v in baseline_metrics.items()},
            **{f"oracle_{k}": v for k,v in oracle_metrics.items()}, **{f"selected_{k}": v for k,v in selected_metrics.items()},
            "oracle_delta_pq": oracle_metrics["pq"]-baseline_metrics["pq"], "oracle_delta_aji": oracle_metrics["aji"]-baseline_metrics["aji"],
            "selected_delta_pq": selected_metrics["pq"]-baseline_metrics["pq"], "selected_delta_aji": selected_metrics["aji"]-baseline_metrics["aji"],
            "oracle_added_tp": oracle_metrics["tp"]-baseline_metrics["tp"], "oracle_added_fp": oracle_metrics["fp"]-baseline_metrics["fp"],
            "selected_added_tp": selected_metrics["tp"]-baseline_metrics["tp"], "selected_added_fp": selected_metrics["fp"]-baseline_metrics["fp"],
            "teacher_fn_instances": len(missed), "raw_proposals": len(rows), "accepted_proposals": sum(bool(row["accepted"]) for row in rows),
            **{f"oracle_{k}": v for k,v in oracle_info.items()}, **{f"selected_{k}": v for k,v in selected_info.items()},
        })
    if guard.hidden_gt_reads != 24:
        raise AssertionError(f"Hidden-GT guard expected 24 reads after freeze, observed {guard.hidden_gt_reads}.")
    if sha256_file(artifact / "frozen_acceptance_rule.json") != rule_payload["sha256"]:
        raise AssertionError("Frozen acceptance rule changed after unlabeled audit began.")
    _csv(artifact / "candidate_curve.csv", candidate_curve); _csv(artifact / "instance_audit.csv", instance_rows); _csv(artifact / "per_image_metrics.csv", per_image)
    per_patient = []
    for patient in sorted({row["patient"] for row in per_image}):
        subset = [row for row in per_image if row["patient"] == patient]
        per_patient.append({
            "patient": patient, "n_images": len(subset),
            **{key: float(np.mean([row[key] for row in subset])) for key in ("oracle_delta_pq", "oracle_delta_aji", "selected_delta_pq", "selected_delta_aji")},
            **{key: int(sum(row[key] for row in subset)) for key in ("oracle_added_tp", "oracle_added_fp", "selected_added_tp", "selected_added_fp")},
        })
    _csv(artifact / "per_patient_metrics.csv", per_patient)
    accepted = [row for row in instance_rows if row["accepted"]]
    raw_hits = [row for row in instance_rows if row["hits_teacher_fn"]]
    selected_hits = [row for row in accepted if row["hits_teacher_fn"]]
    teacher_rows = [{key.removeprefix("teacher_"): value for key,value in row.items() if key.startswith("teacher_")} for row in per_image]
    oracle_rows = [{key.removeprefix("oracle_"): value for key,value in row.items() if key.startswith("oracle_") and key not in {"oracle_delta_pq","oracle_delta_aji"}} for row in per_image]
    selected_rows = [{key.removeprefix("selected_"): value for key,value in row.items() if key.startswith("selected_") and key not in {"selected_delta_pq","selected_delta_aji"}} for row in per_image]
    total_fn = sum(row["teacher_fn_instances"] for row in per_image)
    recovered = len({(row["image"], row["target_gt_id"]) for row in raw_hits})
    positive_sum = sum(contributions)
    report = {
        "phase": "SemiPMS Phase 0 — Stain-Residual Support Expansion", "git_sha": _git(repo, "rev-parse", "HEAD"),
        "canonical_baseline": CANONICAL_BASELINE, "training": training, "checkpoint_provenance": provenance,
        "data_manifest": "data_manifest.json", "hidden_gt_guard": {"frozen": guard.frozen, "unlabeled_label_reads": guard.hidden_gt_reads, "expected": 24},
        "acceptance_rule": rule_payload, "inference_path_modified": False,
        "tests": {"baseline_inference_equivalence": True, "guard_unit_tests": "tests.txt", "checksum_guard": True},
        "teacher": _aggregate(teacher_rows), "oracle_addition": _aggregate(oracle_rows), "selected_expansion": _aggregate(selected_rows),
        "summary": {
            "oracle_delta_pq": float(np.mean([row["oracle_delta_pq"] for row in per_image])), "oracle_delta_aji": float(np.mean([row["oracle_delta_aji"] for row in per_image])),
            "selected_delta_pq": float(np.mean([row["selected_delta_pq"] for row in per_image])), "selected_delta_aji": float(np.mean([row["selected_delta_aji"] for row in per_image])),
            "raw_residual_teacher_fn_recall": recovered / total_fn if total_fn else 0.0,
            "raw_residual_proposal_precision": len(raw_hits) / len(instance_rows) if instance_rows else 0.0,
            "frozen_rule_precision": len(selected_hits) / len(accepted) if accepted else 0.0,
            "correct_missed_decoder_mask_iou": {
                "n": len(raw_hits),
                "mean": float(np.mean([row["mask_iou_to_target"] for row in raw_hits])) if raw_hits else 0.0,
                "median": float(np.median([row["mask_iou_to_target"] for row in raw_hits])) if raw_hits else 0.0,
                "p10": float(np.percentile([row["mask_iou_to_target"] for row in raw_hits], 10)) if raw_hits else 0.0,
                "p90": float(np.percentile([row["mask_iou_to_target"] for row in raw_hits], 90)) if raw_hits else 0.0,
            },
            "patients_with_positive_selected_pq": sum(any(row["selected_delta_pq"] > 0 for row in per_image if row["patient"] == patient) for patient in range(1,7)),
            "maximum_single_image_positive_gain_contribution": max(contributions, default=0.0) / positive_sum if positive_sum else 0.0,
            "oracle_added_tp": int(sum(row["oracle_added_tp"] for row in per_image)),
            "oracle_added_fp": int(sum(row["oracle_added_fp"] for row in per_image)),
            "selected_added_tp": int(sum(row["selected_added_tp"] for row in per_image)),
            "selected_added_fp": int(sum(row["selected_added_fp"] for row in per_image)),
            "oracle_merge": int(sum(row.get("oracle_merge", 0) for row in per_image)),
            "oracle_duplicate": int(sum(row.get("oracle_duplicate", 0) for row in per_image)),
            "selected_merge": int(sum(row.get("selected_merge", 0) for row in per_image)),
            "selected_duplicate": int(sum(row.get("selected_duplicate", 0) for row in per_image)),
        },
        "evidence_interpretation": "continuous curves reported; project lead decides signal level; no automatic GO/NO-GO boundary applied.",
        "runtime_seconds": time.monotonic() - started,
        "stop_condition": "Phase 0 complete; do not access patients 7--11/MoNuSeg or implement EMA student training.",
    }
    write_json(artifact / "report.json", report)
    with (artifact / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in sorted(item for item in artifact.rglob("*") if item.is_file() and item.name != "SHA256SUMS"):
            handle.write(f"{sha256_file(path)}  {path.relative_to(artifact).as_posix()}\n")
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SemiPMS Phase 0 frozen low-label audit")
    parser.add_argument("--data-root", default="data/tnbc")
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--output-root", default="logs/semipms/phase0")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--sam-config", default="sam2_hiera_l")
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--train-steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--resume-from", default="", help="Read-only interrupted SemiPMS artifact with completed fixed-step weak teacher")
    parser.add_argument("--ema-student", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--monuseg", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--allow-closed-patients", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    artifact = run_phase0(build_parser().parse_args(argv))
    print(f"SemiPMS Phase 0 complete: {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
