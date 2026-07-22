"""Run the owner-approved, no-gradient Phase 1 candidate-failure diagnosis.

Only an explicit manifest may be opened. TNBC p9--p11 and any MoNuSeg test
path are rejected before a dataset is constructed. The tool does not create an
optimizer or call backward().
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run.dataset.manifest import load_dataset_manifest
from run.dataset.monuseg import MONUSEG
from run.dataset.tnbc import TNBC
from run.run_on_epoch import _assemble_instance_map, combine_mask, crop_with_overlap, mask_process_eval
from sam2_train.build_sam import build_sam2
from sam2_train.modeling.dpa_p2pnet import build_model
from sam2_train.modeling.utils import point_nms, predict
from stainpms.phase1_decoder import (
    decode_all_native_mask_tokens,
    prepare_image_for_all_token_decode,
    select_standard_single_mask,
    update_validation_texture_memory,
)
from stainpms.phase1_metrics import (
    attach_gt_error_classes,
    choose_edt_interior_points,
    final_instance_overlap_table,
    final_max_iou_by_gt,
    iou_against_label,
    mask_iou,
    strict_final_pairing,
    structural_errors,
    summarize_gt_rows,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write a small audit record without exposing a partial JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def torch_save_atomic(path: Path, payload: Any) -> None:
    """Persist read-only inference state so an interrupted diagnosis resumes exactly."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_completed_image_outputs(
    directory: Path,
    *,
    expected_indices: list[int],
    manifest_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    """Load completed image records and reject gaps or foreign-manifest state."""

    rows: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    elapsed = 0.0
    for index in expected_indices:
        path = directory / f"{index:05d}.json"
        payload = load_json(path, f"completed image output {path}")
        if int(payload.get("record_index", -1)) != index:
            raise ValueError(f"completed image index mismatch: {path}")
        if payload.get("manifest_sha256") != manifest_sha256:
            raise ValueError(f"completed image belongs to a different manifest: {path}")
        child_rows = payload.get("gt_rows")
        image_record = payload.get("image_record")
        if not isinstance(child_rows, list) or not isinstance(image_record, dict):
            raise ValueError(f"invalid completed image payload: {path}")
        rows.extend(child_rows)
        images.append(image_record)
        elapsed += float(payload.get("wall_seconds", 0.0))
    return rows, images, elapsed


def validate_scope(dataset: str, manifest: dict[str, Any], records: list[dict[str, Any]]) -> None:
    if dataset == "tnbc":
        allowed = {int(value) for value in manifest.get("allowed_patients", [])}
        if not allowed or allowed & {9, 10, 11} or not allowed <= set(range(1, 9)):
            raise ValueError("TNBC Phase 1 manifest must declare only p1--p8")
        for record in records:
            patient = int(record.get("patient", -1))
            if patient not in allowed or patient in {9, 10, 11}:
                raise ValueError(f"TNBC sealed patient rejected before sample use: {patient}")
    else:
        if len(records) != 37:
            raise ValueError("MoNuSeg Phase 1 is limited to exactly 37 training records")
        if manifest.get("role") != "phase1_training_set_mechanism_diagnosis":
            raise ValueError("MoNuSeg manifest is not the authorised Phase 1 train-only manifest")
        for record in records:
            for field in ("image_path", "label_path"):
                path = Path(record[field])
                if any(part.lower() == "test" or "test14" in part.lower() for part in path.parts):
                    raise ValueError(f"MoNuSeg test path rejected before sample use: {path}")


def load_checkpoint_declaration(path: Path, checkpoint: Path, dataset: str) -> dict[str, Any]:
    declaration = load_json(path, "checkpoint declaration")
    if declaration.get("dataset") != dataset:
        raise ValueError("checkpoint declaration dataset does not match --dataset")
    if declaration.get("classification") not in {"historical_exploratory", "clean_authorised"}:
        raise ValueError("checkpoint declaration classification must be historical_exploratory or clean_authorised")
    declared_path = declaration.get("checkpoint_path")
    if declared_path and Path(str(declared_path)).resolve() != checkpoint.resolve():
        raise ValueError("checkpoint declaration path does not match --checkpoint")
    observed_sha = sha256_file(checkpoint)
    declared_sha = declaration.get("checkpoint_sha256")
    if declared_sha and declared_sha != observed_sha:
        raise ValueError("checkpoint declaration SHA256 does not match checkpoint")
    declaration = dict(declaration)
    declaration["checkpoint_path"] = str(checkpoint.resolve())
    declaration["checkpoint_sha256"] = observed_sha
    return declaration


def runtime_cfg(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        crop_size=args.crop_size,
        overlap=args.overlap,
        load=args.load,
        data_path=args.data_path,
        test_nms_thr=args.point_nms_thr,
        seed=args.seed,
    )


def update_best(row: dict[str, Any], prefix: str, score: float, token: int, quality: float, point: tuple[float, float], crop: list[int]) -> None:
    key = f"{prefix}_best_candidate_iou"
    if row.get(key) is None or score > float(row[key]):
        row[key] = float(score)
        row[f"{prefix}_best_token"] = int(token)
        row[f"{prefix}_best_quality"] = float(quality)
        row[f"{prefix}_best_point_xy"] = [float(point[0]), float(point[1])]
        row[f"{prefix}_best_crop_xyxy"] = [int(value) for value in crop]


def update_selected(
    row: dict[str, Any], score: float, quality: float, token: int, point: tuple[float, float], crop: list[int]
) -> None:
    key = "auto_selected_candidate_iou"
    if row.get(key) is None or score > float(row[key]):
        row[key] = float(score)
        row["auto_selected_candidate_quality"] = float(quality)
        row["auto_selected_candidate_token"] = int(token)
        row["auto_selected_candidate_point_xy"] = [float(point[0]), float(point[1])]
        row["auto_selected_candidate_crop_xyxy"] = [int(value) for value in crop]


def decode_chunks(net, prepared, points: np.ndarray, out_size: int, device: torch.device, chunk_size: int):
    low_logits_chunks = []
    logits_chunks = []
    quality_chunks = []
    for start in range(0, len(points), chunk_size):
        coords = torch.as_tensor(points[start : start + chunk_size], device=device, dtype=torch.float32).unsqueeze(1)
        labels = torch.ones((coords.shape[0], 1), device=device, dtype=torch.int)
        low_logits, logits, quality = decode_all_native_mask_tokens(
            net=net,
            prepared=prepared,
            prompt_points=coords,
            prompt_labels=labels,
            out_size=out_size,
            device=device,
        )
        low_logits_chunks.append(low_logits.detach().cpu())
        logits_chunks.append(logits.detach().cpu())
        quality_chunks.append(quality.detach().cpu())
    if not logits_chunks:
        return (
            torch.empty((0, 4, out_size // 4, out_size // 4)),
            torch.empty((0, 4, out_size, out_size)),
            torch.empty((0, 4)),
        )
    return torch.cat(low_logits_chunks, dim=0), torch.cat(logits_chunks, dim=0), torch.cat(quality_chunks, dim=0)


def make_gt_rows(inst_map: np.ndarray, *, sample_id: str, group: dict[str, Any], gt_points: dict[int, tuple[int, int]]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for instance_id, (x, y) in gt_points.items():
        rows[instance_id] = {
            "sample_id": sample_id,
            "gt_instance_id": int(instance_id),
            "gt_area": int((inst_map == instance_id).sum()),
            "gt_point_xy": [int(x), int(y)],
            **group,
            "auto_point_count": 0,
            "gt_point_best_candidate_iou": None,
            "auto_best_candidate_iou": None,
            "auto_selected_candidate_iou": None,
        }
    return rows


@torch.no_grad()
def diagnose_image(
    *,
    image: torch.Tensor,
    inst_map: np.ndarray,
    sample_id: str,
    group: dict[str, Any],
    point_net,
    point_encoder,
    net,
    texture_memory_bank: list,
    args: argparse.Namespace,
    main_match_iou: float,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    full_h, full_w = inst_map.shape
    gt_points = choose_edt_interior_points(inst_map)
    rows_by_id = make_gt_rows(inst_map, sample_id=sample_id, group=group, gt_points=gt_points)
    gt_areas = {instance_id: row["gt_area"] for instance_id, row in rows_by_id.items()}
    point_gt: dict[int, int] = {}
    point_coords: dict[int, tuple[float, float]] = {}
    all_points: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_classes: list[np.ndarray] = []
    point_id_map: dict[tuple[float, float], int] = {}
    next_point_id = 0
    processed_boxes: list[list[int]] = []
    all_masks: list[np.ndarray] = []
    all_boxes: list[list[float]] = []
    all_assembly_scores: list[float] = []
    all_inds: list[int] = []
    context_memory_bank: list = []
    margin = 7
    image_batch = image.unsqueeze(0).to(device)
    crop_boxes = crop_with_overlap(image_batch[0], args.crop_size, args.crop_size, args.overlap, args.load).tolist()

    for crop_box in crop_boxes:
        x1, y1, x2, y2 = [int(value) for value in crop_box]
        crop = image_batch[..., y1:y2, x1:x2]
        pd_points, pd_scores, pd_classes, _, _, _, _ = predict(
            point_net,
            crop,
            ori_shape=np.array((y2 - y1, x2 - x1)),
            filtering=args.point_filtering,
            nms_thr=args.point_nms_thr,
        )
        auto_points = np.empty((0, 2), dtype=np.float32)
        auto_classes = np.empty((0,), dtype=np.int64)
        auto_inds = np.empty((0,), dtype=np.int64)
        if len(pd_points):
            pd_points[:, 0] += x1
            pd_points[:, 1] += y1
            keep_new = np.ones(len(pd_points), dtype=bool)
            for px1, py1, px2, py2 in processed_boxes:
                keep_new &= ~(
                    (pd_points[:, 0] >= px1 + 1)
                    & (pd_points[:, 0] <= px2 - 1)
                    & (pd_points[:, 1] >= py1 + 1)
                    & (pd_points[:, 1] <= py2 - 1)
                )
            processed_boxes.append([x1, y1, x2, y2])
            pd_points = pd_points[keep_new]
            pd_scores = pd_scores[keep_new]
            pd_classes = pd_classes[keep_new]
            if len(pd_points):
                all_points.append(pd_points)
                all_scores.append(pd_scores)
                all_classes.append(pd_classes)
                current_points, current_scores, current_classes = point_nms(
                    np.vstack(all_points), np.concatenate(all_scores), np.concatenate(all_classes), args.point_nms_thr
                )
                current_inds = []
                for point in current_points:
                    key = (float(point[0]), float(point[1]))
                    if key not in point_id_map:
                        point_id_map[key] = next_point_id
                        next_point_id += 1
                    current_inds.append(point_id_map[key])
                inside = (
                    (current_points[:, 0] >= x1)
                    & (current_points[:, 0] < x2)
                    & (current_points[:, 1] >= y1)
                    & (current_points[:, 1] < y2)
                )
                auto_points = current_points[inside]
                auto_classes = current_classes[inside]
                auto_inds = np.asarray(current_inds, dtype=np.int64)[inside]
        else:
            # Match validation_on_epoch: an empty point crop still becomes a
            # processed window, preventing later overlap duplicates.
            processed_boxes.append([x1, y1, x2, y2])

        gt_items = [
            (instance_id, x - x1, y - y1)
            for instance_id, (x, y) in gt_points.items()
            if x1 <= x < x2 and y1 <= y < y2
        ]
        if len(auto_points) == 0 and not gt_items:
            continue

        prepared = prepare_image_for_all_token_decode(
            net=net,
            point_encoder=point_encoder,
            image=crop,
            texture_memory_bank=texture_memory_bank,
            context_memory_bank=context_memory_bank,
            x1=x1,
            y1=y1,
            texture=args.texture,
            context=args.context,
            context_atten_k=args.context_atten_k,
            device=device,
        )

        if len(auto_points):
            local_auto = auto_points - np.asarray([x1, y1], dtype=np.float32)
            auto_low_logits, auto_logits, auto_quality = decode_chunks(
                net, prepared, local_auto, args.out_size, device, args.prompt_chunk_size
            )
            auto_masks = auto_logits.numpy() > 0.0
            auto_quality_np = auto_quality.numpy()
            standard_logits, standard_quality, standard_tokens = select_standard_single_mask(
                net=net,
                low_res_logits=auto_low_logits.to(device),
                high_res_logits=auto_logits.to(device),
                quality_predictions=auto_quality.to(device),
            )
            standard_masks = standard_logits.detach().cpu().numpy() > 0.0
            standard_quality_np = standard_quality.detach().cpu().numpy()
            standard_tokens_np = standard_tokens.detach().cpu().numpy()
            for prompt_index, (global_point, point_id) in enumerate(zip(auto_points, auto_inds, strict=True)):
                px = min(max(int(global_point[0]), 0), full_w - 1)
                py = min(max(int(global_point[1]), 0), full_h - 1)
                associated_gt = int(inst_map[py, px])
                point_gt[int(point_id)] = associated_gt
                point_coords[int(point_id)] = (float(global_point[0]), float(global_point[1]))
                if associated_gt:
                    local_gt = inst_map[y1:y2, x1:x2] == associated_gt
                    for token in range(4):
                        score = mask_iou(auto_masks[prompt_index, token], local_gt)
                        # The crop-local IoU needs the full GT denominator.
                        candidate_area = int(auto_masks[prompt_index, token].sum())
                        intersection = int(np.logical_and(auto_masks[prompt_index, token], local_gt).sum())
                        union = candidate_area + gt_areas[associated_gt] - intersection
                        score = float(intersection / union) if union else 0.0
                        update_best(
                            rows_by_id[associated_gt],
                            "auto",
                            score,
                            token,
                            float(auto_quality_np[prompt_index, token]),
                            (float(global_point[0]), float(global_point[1])),
                            crop_box,
                        )
                    candidate_area = int(standard_masks[prompt_index].sum())
                    intersection = int(np.logical_and(standard_masks[prompt_index], local_gt).sum())
                    union = candidate_area + gt_areas[associated_gt] - intersection
                    selected_score = float(intersection / union) if union else 0.0
                    update_selected(
                        rows_by_id[associated_gt],
                        selected_score,
                        float(standard_quality_np[prompt_index]),
                        int(standard_tokens_np[prompt_index]),
                        (float(global_point[0]), float(global_point[1])),
                        crop_box,
                    )

            default_logits = standard_logits
            default_quality = standard_quality
            local_points_t = torch.as_tensor(local_auto, device=device, dtype=torch.float32).unsqueeze(1)
            masks = mask_process_eval(
                auto_classes,
                torch.as_tensor(auto_inds, dtype=torch.long),
                crop_box,
                np.asarray([full_h, full_w]),
                local_points_t,
                default_logits,
                default_quality,
            )
            for mask_data in masks:
                bx1, by1, bx2, by2 = mask_data["bbox"]
                edge_penalized = (
                    (bx1 > margin and abs(bx1 - x1) <= margin)
                    or (abs(bx2 - full_h) > margin and abs(bx2 - x2) <= margin)
                    or (by1 > margin and abs(by1 - y1) <= margin)
                    or (abs(by2 - full_w) > margin and abs(by2 - y2) <= margin)
                )
                all_masks.append(mask_data["segmentation"][:full_h, :full_w])
                all_boxes.append(mask_data["bbox"])
                all_assembly_scores.append(float(mask_data["predicted_iou"]) * (0.3 if edge_penalized else 1.0))
                all_inds.append(int(mask_data["inds"]))
            if args.texture:
                crop_default_map = combine_mask(
                    np.asarray([y2 - y1, x2 - x1]), local_points_t, default_logits, default_quality
                )
                memory_mask = torch.from_numpy(crop_default_map.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
                update_validation_texture_memory(
                    net=net,
                    prepared=prepared,
                    default_mask_logits=memory_mask,
                    default_quality=default_quality,
                    texture_memory_bank=texture_memory_bank,
                    texture_memory_bank_size=args.texture_memory_bank_size,
                    device=device,
                )

        if gt_items:
            gt_ids = [item[0] for item in gt_items]
            local_gt_points = np.asarray([[item[1], item[2]] for item in gt_items], dtype=np.float32)
            _, gt_logits, gt_quality = decode_chunks(
                net, prepared, local_gt_points, args.out_size, device, args.prompt_chunk_size
            )
            gt_masks = gt_logits.numpy() > 0.0
            gt_quality_np = gt_quality.numpy()
            for prompt_index, instance_id in enumerate(gt_ids):
                local_gt = inst_map[y1:y2, x1:x2] == instance_id
                point = gt_points[instance_id]
                for token in range(4):
                    candidate_area = int(gt_masks[prompt_index, token].sum())
                    intersection = int(np.logical_and(gt_masks[prompt_index, token], local_gt).sum())
                    union = candidate_area + gt_areas[instance_id] - intersection
                    score = float(intersection / union) if union else 0.0
                    update_best(
                        rows_by_id[instance_id],
                        "gt_point",
                        score,
                        token,
                        float(gt_quality_np[prompt_index, token]),
                        point,
                        crop_box,
                    )

        # In regular inference the context entry is created in this crop but
        # only affects future crops. Appending after both diagnostic decodes is
        # therefore equivalent and keeps GT-point probes side-effect free.
        # The standard route reaches the decoder only when this crop retains
        # automatic prompts.  A GT-only diagnostic probe must not alter the
        # context available to later automatic-prompt crops.
        if args.context and len(auto_points) and prepared.context_entry is not None:
            context_memory_bank.append(prepared.context_entry)

    pred_map = _assemble_instance_map(
        all_boxes,
        all_assembly_scores,
        all_masks,
        all_inds,
        inst_map.shape,
        args.instance_nms_iou,
    )
    final_overlap = final_instance_overlap_table(inst_map, pred_map)
    final_iou_by_gt = final_max_iou_by_gt(final_overlap)
    final_info = strict_final_pairing(inst_map, pred_map, main_match_iou)
    pairs = final_info["pairs"]
    point_counts = Counter(value for value in point_gt.values() if value != 0)
    for instance_id, row in rows_by_id.items():
        row["auto_point_count"] = int(point_counts.get(instance_id, 0))
        row["final_matched"] = bool(instance_id in pairs)
        final_max_iou, final_best_pred_id = final_iou_by_gt.get(instance_id, (0.0, None))
        row["final_max_iou"] = float(final_max_iou)
        row["final_best_pred_id"] = final_best_pred_id
        row["final_matched_pred_id"] = pairs.get(instance_id)
    rows = attach_gt_error_classes(rows_by_id.values(), main_match_iou)
    auto_point_total = len(point_gt)
    background_points = sum(value == 0 for value in point_gt.values())
    image_record = {
        "sample_id": sample_id,
        **group,
        "gt_instance_count": len(rows),
        "auto_decoder_point_count": auto_point_total,
        "background_auto_point_count": background_points,
        "background_auto_point_fraction": float(background_points / auto_point_total) if auto_point_total else None,
        "final_metrics": final_info["evaluator"],
        "structural_errors": structural_errors(
            inst_map,
            pred_map,
            main_match_iou,
            pairing_info=final_info,
            overlap=final_overlap,
            best_iou_by_gt=final_iou_by_gt,
        ),
        "error_classes": dict(Counter(row["error_class"] for row in rows)),
    }
    return rows, image_record


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def aggregate_rows(rows: list[dict[str, Any]], image_records: list[dict[str, Any]], thresholds: list[float], match_iou: float) -> dict[str, Any]:
    summary = summarize_gt_rows(rows, thresholds=thresholds, match_iou=match_iou)
    point_count = sum(int(record["auto_decoder_point_count"]) for record in image_records)
    background_count = sum(int(record["background_auto_point_count"]) for record in image_records)
    structural_keys = ("tp", "fp", "fn", "split_unmatched_gt_count", "merge_unmatched_pred_count", "boundary_localization_unmatched_gt_count")
    summary.update(
        {
            "image_count": len(image_records),
            "auto_points": {
                "count": point_count,
                "background_count": background_count,
                "background_fraction": float(background_count / point_count) if point_count else None,
                "per_gt_count_mean": mean_or_none([float(row["auto_point_count"]) for row in rows]),
            },
            "final_structural_errors": {
                key: sum(int(record["structural_errors"][key]) for record in image_records)
                for key in structural_keys
            },
        }
    )
    return summary


def group_summaries(rows: list[dict[str, Any]], images: list[dict[str, Any]], thresholds: list[float], match_iou: float) -> dict[str, Any]:
    groups: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for label, key in (("split", "split"), ("patient", "patient"), ("case", "case_id")):
        values = sorted({str(row[key]) for row in rows if row.get(key) not in (None, "")})
        if not values:
            continue
        for value in values:
            group_rows = [row for row in rows if str(row.get(key)) == value]
            group_images = [record for record in images if str(record.get(key)) == value]
            groups[f"{label}:{value}"] = (group_rows, group_images)
    return {key: aggregate_rows(group_rows, group_images, thresholds, match_iou) for key, (group_rows, group_images) in groups.items()}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def write_summary_markdown(path: Path, report: dict[str, Any]) -> None:
    overall = report["overall"]
    lines = [
        "# Phase 1 candidate-failure diagnosis",
        "",
        f"- Status: `{report['status']}`",
        f"- Dataset/scope: `{report['dataset']}` / `{report['scope_label']}`",
        f"- Checkpoint class: `{report['checkpoint']['classification']}`",
        f"- GT instances: `{overall['gt_instance_count']}` across `{overall['image_count']}` images",
        f"- Automatic point recall: `{overall['auto_point_recall']['value']}`",
        f"- Background automatic-point fraction: `{overall['auto_points']['background_fraction']}`",
        "",
        "## Coverage",
        "",
        "| threshold | GT-point CCR | auto CCR given point | auto end-to-end CCR |",
        "|---:|---:|---:|---:|",
    ]
    for idx, ccr in enumerate(overall["ccr_gt_point"]):
        lines.append(
            f"| {ccr['threshold']:.1f} | {ccr['value']} | {overall['ccr_auto_given_point'][idx]['value']} | {overall['ccr_auto_e2e'][idx]['value']} |"
        )
    lines.extend(["", "## Mutually exclusive GT error classes", "", "| class | count |", "|---|---:|"])
    for name, value in overall["error_classes"].items():
        lines.append(f"| {name} | {value} |")
    lines.extend(["", "## Final supplementary errors", "", "| metric | count |", "|---|---:|"])
    for name, value in overall["final_structural_errors"].items():
        lines.append(f"| {name} | {value} |")
    lines.extend(["", "This is a descriptive read-only diagnosis. It makes no significance claim.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["tnbc", "monuseg"], required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-declaration", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metrics-spec", default="configs/phase1/metrics_frozen_v1.json")
    parser.add_argument("--model-config", default="args.py")
    parser.add_argument("--sam-config", default="sam2_hiera_l")
    parser.add_argument("--scope-label", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--out-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--load", choices=["sequence", "unsequence", "clockwise", "unclockwise"], default="unclockwise")
    parser.add_argument("--point-nms-thr", type=int, default=12)
    parser.add_argument("--instance-nms-iou", type=float, default=0.5)
    parser.add_argument("--prompt-chunk-size", type=int, default=64)
    parser.add_argument("--texture-memory-bank-size", type=int, default=64)
    parser.add_argument("--context-atten-k", type=int, default=1)
    parser.add_argument("--point-filtering", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--texture", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--context", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Resume only from this tool's manifest-matched per-image state.")
    parser.add_argument("--max-images", type=int, default=0, help="Nonzero produces a labelled smoke-only output, never a Phase 1 result.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.prompt_chunk_size <= 0:
        raise ValueError("--prompt-chunk-size must be positive")
    spec_path = Path(args.metrics_spec).resolve()
    spec = load_json(spec_path, "metric specification")
    main_match_iou = float(spec["evaluator"]["main_match_iou"])
    if args.instance_nms_iou != 0.5 or main_match_iou != 0.5:
        raise ValueError("Phase 1 freezes the existing 0.5 instance NMS and strict match IoU")
    thresholds = [float(value) for value in spec["evaluator"]["sensitivity_ccr_iou"]]
    manifest_path = Path(args.manifest).resolve()
    manifest, records = load_dataset_manifest(manifest_path, expected_dataset=args.dataset, require_labels=True, verify_hashes=True)
    validate_scope(args.dataset, manifest, records)
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    declaration = load_checkpoint_declaration(Path(args.checkpoint_declaration).resolve(), checkpoint, args.dataset)
    set_determinism(args.seed)
    cuda_device_index: int | None = None
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_device)
        cuda_device_index = int(torch.cuda.current_device())
        device = torch.device("cuda", cuda_device_index)
    else:
        device = torch.device("cpu")
    if torch.cuda.is_available():
        # PyTorch 2.7's CUDA memory API on this environment accepts an integer
        # device index here, whereas a torch.device object is rejected.
        torch.cuda.reset_peak_memory_stats(cuda_device_index)

    model_config = Config.fromfile(str(Path(args.model_config).resolve()))
    net = build_sam2(args.sam_config, str(checkpoint), device=device)
    point_net, point_encoder = build_model(model_config)
    point_net.to(device).eval()
    point_encoder.to(device).eval()
    checkpoint_payload = torch.load(checkpoint, map_location="cpu")
    if "model1" not in checkpoint_payload:
        raise ValueError("checkpoint has no CA-SAM2/StainPMS point-head state ('model1')")
    missing, unexpected = point_net.load_state_dict(checkpoint_payload["model1"], strict=False)
    texture_memory_bank = list(checkpoint_payload.get("texture_memory_bank_list", []) or [])
    runtime = runtime_cfg(args)
    dataset_cls = TNBC if args.dataset == "tnbc" else MONUSEG
    dataset = dataset_cls(
        runtime,
        model_config,
        args.data_path,
        args.load,
        mode="test",
        manifest_path=str(manifest_path),
        data_split="train",
        verify_manifest_hashes=True,
    )
    maximum = len(dataset) if args.max_images == 0 else min(int(args.max_images), len(dataset))
    output_dir = Path(args.output_dir).resolve()
    completed_dir = output_dir / "completed_images"
    progress_path = output_dir / "progress.json"
    texture_state_path = output_dir / "texture_memory_bank.pt"
    fingerprint_payload = {
        "dataset": args.dataset,
        "scope_label": args.scope_label,
        "manifest_sha256": manifest["manifest_sha256"],
        "checkpoint_sha256": declaration["checkpoint_sha256"],
        "metric_spec_sha256": sha256_file(spec_path),
        "maximum": maximum,
        "frozen_inference": {
            "sam_config": args.sam_config,
            "crop_size": args.crop_size,
            "out_size": args.out_size,
            "overlap": args.overlap,
            "load": args.load,
            "point_nms_thr": args.point_nms_thr,
            "instance_nms_iou": args.instance_nms_iou,
            "prompt_chunk_size": args.prompt_chunk_size,
            "texture": args.texture,
            "context": args.context,
            "context_atten_k": args.context_atten_k,
        },
    }
    run_fingerprint = json_sha256(fingerprint_payload)
    completed_indices: list[int] = []
    previous_elapsed = 0.0
    gt_rows: list[dict[str, Any]] = []
    image_records: list[dict[str, Any]] = []
    if args.resume:
        progress = load_json(progress_path, "Phase 1 progress state")
        if progress.get("run_fingerprint") != run_fingerprint:
            raise ValueError("--resume state does not match this manifest/checkpoint/frozen inference configuration")
        completed_indices = [int(value) for value in progress.get("completed_record_indices", [])]
        if completed_indices != list(range(len(completed_indices))) or any(index >= maximum for index in completed_indices):
            raise ValueError("--resume progress has non-contiguous or out-of-range completed indices")
        gt_rows, image_records, previous_elapsed = load_completed_image_outputs(
            completed_dir,
            expected_indices=completed_indices,
            manifest_sha256=manifest["manifest_sha256"],
        )
        if completed_indices:
            if not texture_state_path.is_file():
                raise FileNotFoundError("--resume requires texture_memory_bank.pt after completed images")
            texture_memory_bank = list(torch.load(texture_state_path, map_location="cpu", weights_only=False))
    else:
        if progress_path.exists() or any(completed_dir.glob("*.json")) or (output_dir / "summary.json").exists():
            raise FileExistsError("output directory already contains Phase 1 state; use --resume or choose a new directory")
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            progress_path,
            {
                "schema_version": 1,
                "phase": 1,
                "status": "in_progress",
                "run_fingerprint": run_fingerprint,
                "fingerprint": fingerprint_payload,
                "completed_record_indices": [],
                "completed_wall_seconds": 0.0,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    for index in range(len(completed_indices), maximum):
        image_started = time.perf_counter()
        image, inst_map, _, _, _, _, _, _, sample_id = dataset[index]
        inst_np = np.asarray(inst_map.cpu().numpy() if torch.is_tensor(inst_map) else inst_map, dtype=np.int32)
        record = records[index]
        group = {"split": args.scope_label, "patient": record.get("patient"), "case_id": record.get("case_id")}
        rows, image_record = diagnose_image(
            image=image,
            inst_map=inst_np,
            sample_id=str(sample_id),
            group=group,
            point_net=point_net,
            point_encoder=point_encoder,
            net=net,
            texture_memory_bank=texture_memory_bank,
            args=args,
            main_match_iou=main_match_iou,
            device=device,
        )
        image_wall_seconds = time.perf_counter() - image_started
        image_record["wall_seconds"] = image_wall_seconds
        gt_rows.extend(rows)
        image_records.append(image_record)
        write_json_atomic(
            completed_dir / f"{index:05d}.json",
            {
                "schema_version": 1,
                "phase": 1,
                "record_index": index,
                "sample_id": str(sample_id),
                "manifest_sha256": manifest["manifest_sha256"],
                "wall_seconds": image_wall_seconds,
                "gt_rows": rows,
                "image_record": image_record,
            },
        )
        torch_save_atomic(texture_state_path, texture_memory_bank)
        completed_indices.append(index)
        previous_elapsed += image_wall_seconds
        write_json_atomic(
            progress_path,
            {
                "schema_version": 1,
                "phase": 1,
                "status": "in_progress",
                "run_fingerprint": run_fingerprint,
                "fingerprint": fingerprint_payload,
                "completed_record_indices": completed_indices,
                "completed_wall_seconds": previous_elapsed,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        print(
            f"[phase1] {index + 1}/{maximum} {sample_id}: gt={len(rows)} "
            f"auto_points={image_record['auto_decoder_point_count']} wall_s={image_wall_seconds:.2f}",
            flush=True,
        )

    elapsed = previous_elapsed
    write_csv(output_dir / "gt_instances.csv", gt_rows)
    write_csv(output_dir / "images.csv", image_records)
    write_json_atomic(output_dir / "images.json", image_records)
    examples: dict[str, list[dict[str, Any]]] = {}
    for error_class in sorted({row["error_class"] for row in gt_rows}):
        candidates = sorted(
            [row for row in gt_rows if row["error_class"] == error_class],
            key=lambda row: (row["sample_id"], int(row["gt_instance_id"])),
        )
        examples[error_class] = candidates[:3]
    report = {
        "schema_version": 1,
        "phase": 1,
        "status": "complete" if args.max_images == 0 else "smoke_only_partial",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "scope_label": args.scope_label,
        "diagnosis_scope": "read_only_no_grad_no_optimizer",
        "metric_spec": {
            "path": str(spec_path),
            "sha256": sha256_file(spec_path),
            "protocol_id": spec["protocol_id"],
            "main_match_iou": main_match_iou,
            "ccr_thresholds": thresholds,
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": manifest["manifest_sha256"],
            "protocol_id": manifest.get("protocol_id"),
            "record_count": len(records),
            "processed_record_count": maximum,
            "hashes_verified": True,
        },
        "checkpoint": {**declaration, "load_model1_missing_key_count": len(missing), "load_model1_unexpected_key_count": len(unexpected)},
        "frozen_inference": {
            "sam_config": args.sam_config,
            "crop_size": args.crop_size,
            "out_size": args.out_size,
            "overlap": args.overlap,
            "load": args.load,
            "point_nms_threshold": args.point_nms_thr,
            "instance_nms_iou": args.instance_nms_iou,
            "point_filtering": args.point_filtering,
            "texture": args.texture,
            "context": args.context,
            "context_atten_k": args.context_atten_k,
            "native_mask_token_count": 4,
            "standard_single_mask_selection": "token0_with_existing_dynamic_stability_fallback",
        },
        "repository": {"branch": git_value("branch", "--show-current"), "commit": git_value("rev-parse", "HEAD")},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(cuda_device_index) if torch.cuda.is_available() else None,
            "seed": args.seed,
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        },
        "runtime": {
            "wall_seconds": elapsed,
            "wall_seconds_per_image": elapsed / maximum if maximum else None,
            "peak_memory_allocated_mib": torch.cuda.max_memory_allocated(cuda_device_index) / (1024 ** 2) if torch.cuda.is_available() else 0.0,
            "peak_memory_reserved_mib": torch.cuda.max_memory_reserved(cuda_device_index) / (1024 ** 2) if torch.cuda.is_available() else 0.0,
        },
        "overall": aggregate_rows(gt_rows, image_records, thresholds, main_match_iou),
        "groups": group_summaries(gt_rows, image_records, thresholds, main_match_iou),
        "fixed_examples": examples,
        "outputs": {"gt_instances_csv": "gt_instances.csv", "images_csv": "images.csv", "images_json": "images.json"},
    }
    write_json_atomic(output_dir / "summary.json", report)
    write_summary_markdown(output_dir / "summary.md", report)
    write_json_atomic(
        progress_path,
        {
            "schema_version": 1,
            "phase": 1,
            "status": "complete",
            "run_fingerprint": run_fingerprint,
            "fingerprint": fingerprint_payload,
            "completed_record_indices": completed_indices,
            "completed_wall_seconds": previous_elapsed,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(json.dumps({"status": report["status"], "output_dir": str(output_dir), "summary": str(output_dir / "summary.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
