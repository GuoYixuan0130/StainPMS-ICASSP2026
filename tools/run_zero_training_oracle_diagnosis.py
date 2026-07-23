"""Read-only TNBC p7/p8 four-level candidate-pool oracle diagnosis.

This runner intentionally creates no optimizer and never calls ``backward``.
It mirrors the frozen Phase-1 automatic-prompt inference route, exports every
native candidate token as compact RLE, and stops if native strict metrics do
not reproduce the fixed-epoch values in the frozen performance summary.

The accepted checkpoint is an audited epoch-5 full state from an approved
C0/C1 or C2-AR warm-start run.  It is loaded with explicit
``weights_only=False`` only after its declaration SHA256 is verified.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
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
from run.dataset.tnbc import TNBC
from run.run_on_epoch import _assemble_instance_map, combine_mask, crop_with_overlap, mask_process_eval
from sam2_train.build_sam import build_sam2
from sam2_train.modeling.dpa_p2pnet import build_model
from sam2_train.modeling.utils import point_nms, predict
from stainpms.evaluator import evaluate_instance_pair
from stainpms.phase1_decoder import (
    decode_all_native_mask_tokens,
    prepare_image_for_all_token_decode,
    select_standard_single_mask,
    update_validation_texture_memory,
)
from stainpms.phase1_metrics import instance_ids
from stainpms.zero_training_oracle import (
    ORACLE_MATCH_IOU,
    annotate_pool_ious,
    encode_binary_rle,
    error_partition,
    final_pool_oracle_stage,
    native_final_stage,
    normalize_point_xy,
    oracle_pool_stage,
    pool_gt_maxima,
)


TASK_NAMES = ("dice1", "dice2", "aji", "dq", "sq", "pq")
DIAGNOSIS_SEEDS = (2027, 1337)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_gzip_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
    os.replace(temporary, path)


def read_gzip_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


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


def validate_scope(manifest: dict[str, Any], records: list[dict[str, Any]]) -> None:
    allowed = {int(value) for value in manifest.get("allowed_patients", [])}
    observed = {int(record.get("patient", -1)) for record in records}
    if allowed != {7, 8} or observed != {7, 8}:
        raise ValueError("zero-training oracle diagnosis is limited to exactly TNBC patients 7 and 8")
    if any(patient in {9, 10, 11} for patient in observed):
        raise ValueError("sealed TNBC patient path rejected before dataset construction")
    if len(records) != 7:
        raise ValueError(f"TNBC p7/p8 diagnosis expects 7 images, got {len(records)}")


def validate_declaration(declaration_path: Path, checkpoint: Path, *, seed: int, arm: str) -> dict[str, Any]:
    declaration = read_json(declaration_path, "checkpoint declaration")
    observed_sha = sha256_file(checkpoint)
    if declaration.get("checkpoint_sha256") != observed_sha:
        raise ValueError("checkpoint SHA256 does not match its declaration")
    if declaration.get("dataset") != "tnbc" or declaration.get("classification") != "historical_exploratory":
        raise ValueError("only declared TNBC historical-exploratory warm-start states are permitted")
    if declaration.get("arm") != arm:
        raise ValueError("checkpoint declaration arm does not match command arm")
    protocol = str(declaration.get("protocol", ""))
    expected_protocol = (
        "tnbc_c2_ar_two_seed_v1"
        if arm == "c2_ar"
        else {
            2027: "tnbc_c0_c1_second_seed_2027_v1",
            1337: "tnbc_c0_c1_third_seed_1337_v1",
        }.get(seed)
    )
    if seed not in DIAGNOSIS_SEEDS or protocol != expected_protocol:
        raise ValueError(f"unexpected seed-{seed} protocol: {protocol}")
    if int(declaration.get("epoch", -1)) != 5:
        raise ValueError("zero-training diagnosis accepts only the fixed epoch-5 state")
    return {**declaration, "checkpoint_path": str(checkpoint.resolve()), "checkpoint_sha256": observed_sha}


def reference_metrics(reference: dict[str, Any], *, seed: int, arm: str) -> dict[int, dict[str, float]]:
    arm_key = "c0" if arm == "c0" else "c1_full"
    for record in reference.get("per_seed", []):
        if int(record.get("seed", -1)) != seed:
            continue
        patients = record.get(arm_key, {}).get("patients", {})
        output: dict[int, dict[str, float]] = {}
        for patient in (7, 8):
            metrics = patients.get(str(patient), {}).get("task_metrics_image_macro")
            if not isinstance(metrics, dict):
                raise ValueError(f"frozen reference lacks seed={seed} arm={arm} patient={patient}")
            output[patient] = {name: float(metrics[name]) for name in TASK_NAMES}
        return output
    raise ValueError(f"frozen reference lacks seed {seed}")


def runtime_cfg(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        crop_size=args.crop_size,
        overlap=args.overlap,
        load=args.load,
        data_path=args.data_path,
        test_nms_thr=args.point_nms_thr,
        seed=args.seed,
    )


def decode_chunks(net, prepared, points: np.ndarray, out_size: int, device: torch.device, chunk_size: int):
    lows: list[torch.Tensor] = []
    highs: list[torch.Tensor] = []
    qualities: list[torch.Tensor] = []
    for start in range(0, len(points), chunk_size):
        coords = torch.as_tensor(points[start : start + chunk_size], device=device, dtype=torch.float32).unsqueeze(1)
        labels = torch.ones((coords.shape[0], 1), device=device, dtype=torch.int)
        low, high, quality = decode_all_native_mask_tokens(
            net=net,
            prepared=prepared,
            prompt_points=coords,
            prompt_labels=labels,
            out_size=out_size,
            device=device,
        )
        lows.append(low.detach().cpu())
        highs.append(high.detach().cpu())
        qualities.append(quality.detach().cpu())
    if not highs:
        return (
            torch.empty((0, 4, out_size // 4, out_size // 4)),
            torch.empty((0, 4, out_size, out_size)),
            torch.empty((0, 4)),
        )
    return torch.cat(lows, dim=0), torch.cat(highs, dim=0), torch.cat(qualities, dim=0)


def _edge_penalized(mask_data: dict[str, Any], crop_box: list[int], full_h: int, full_w: int, margin: int = 7) -> bool:
    # This intentionally preserves the Phase-1 / validation assembly rule.
    bx1, by1, bx2, by2 = [float(value) for value in mask_data["bbox"]]
    x1, y1, x2, y2 = [int(value) for value in crop_box]
    return bool(
        (bx1 > margin and abs(bx1 - x1) <= margin)
        or (abs(bx2 - full_h) > margin and abs(bx2 - x2) <= margin)
        or (by1 > margin and abs(by1 - y1) <= margin)
        or (abs(by2 - full_w) > margin and abs(by2 - y2) <= margin)
    )


def _record_from_mask_data(
    mask_data: dict[str, Any],
    *,
    record_index: int,
    token: int,
    crop_index: int,
    edge_penalized: bool,
) -> dict[str, Any]:
    prompt_group = int(mask_data["inds"])
    quality = float(mask_data["predicted_iou"])
    return {
        "record_index": int(record_index),
        "prompt_group_id": prompt_group,
        "token": int(token),
        "crop_index": int(crop_index),
        "point_xy": normalize_point_xy(mask_data["point"]),
        "bbox_xyxy": [float(value) for value in mask_data["bbox"]],
        "quality": quality,
        "assembly_score": quality * (0.3 if edge_penalized else 1.0),
        "edge_penalized": bool(edge_penalized),
        "mask": np.asarray(mask_data["segmentation"], dtype=bool),
    }


def _serialize_mask_record(record: dict[str, Any]) -> dict[str, Any]:
    output = {key: value for key, value in record.items() if key not in {"mask", "gt_ious"}}
    output["mask_rle"] = encode_binary_rle(record["mask"])
    output["gt_ious"] = record.get("gt_ious", {})
    return output


def _serialize_final_map(pred_map: np.ndarray) -> list[dict[str, Any]]:
    return [
        {"final_instance_id": int(pred_id), "mask_rle": encode_binary_rle(pred_map == pred_id)}
        for pred_id in instance_ids(pred_map)
    ]


def _save_display_image(image: torch.Tensor, path: Path) -> None:
    from PIL import Image

    array = image.detach().cpu().permute(1, 2, 0).numpy()
    if array.ndim != 3 or array.shape[2] not in {1, 3}:
        raise ValueError(f"unexpected input display image shape: {array.shape}")
    if float(array.min()) < 0.0 or float(array.max()) > 1.0:
        minimum, maximum = float(array.min()), float(array.max())
        array = (array - minimum) / max(maximum - minimum, 1.0e-12)
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    if array.shape[2] == 1:
        array = array[..., 0]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


@torch.no_grad()
def diagnose_image(
    *,
    image: torch.Tensor,
    inst_map: np.ndarray,
    sample_id: str,
    patient: int,
    point_net,
    point_encoder,
    net,
    texture_memory_bank: list,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Mirror native Phase-1 inference and retain all four-token pools."""

    full_h, full_w = inst_map.shape
    image_batch = image.unsqueeze(0).to(device)
    crop_boxes = crop_with_overlap(image_batch[0], args.crop_size, args.crop_size, args.overlap, args.load).tolist()
    processed_boxes: list[list[int]] = []
    point_id_map: dict[tuple[float, float], int] = {}
    next_point_id = 0
    all_points: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_classes: list[np.ndarray] = []
    context_memory_bank: list = []
    selected_records: list[dict[str, Any]] = []
    all_candidate_records: list[dict[str, Any]] = []
    assembly_masks: list[np.ndarray] = []
    assembly_boxes: list[list[float]] = []
    assembly_scores: list[float] = []
    assembly_inds: list[int] = []
    crop_auto_prompt_count = 0
    background_point_count = 0

    for crop_index, raw_crop_box in enumerate(crop_boxes):
        crop_box = [int(value) for value in raw_crop_box]
        x1, y1, x2, y2 = crop_box
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
            processed_boxes.append(crop_box)
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
                current_inds: list[int] = []
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
            processed_boxes.append(crop_box)
        if not len(auto_points):
            continue

        crop_auto_prompt_count += int(len(auto_points))
        for point in auto_points:
            px = min(max(int(point[0]), 0), full_w - 1)
            py = min(max(int(point[1]), 0), full_h - 1)
            background_point_count += int(inst_map[py, px] == 0)

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
        local_auto = auto_points - np.asarray([x1, y1], dtype=np.float32)
        auto_low_logits, auto_logits, auto_quality = decode_chunks(net, prepared, local_auto, args.out_size, device, args.prompt_chunk_size)
        standard_logits, standard_quality, standard_tokens = select_standard_single_mask(
            net=net,
            low_res_logits=auto_low_logits.to(device),
            high_res_logits=auto_logits.to(device),
            quality_predictions=auto_quality.to(device),
        )
        local_points_t = torch.as_tensor(local_auto, device=device, dtype=torch.float32).unsqueeze(1)

        # C: the standard native selected candidate before global group/NMS/conflict assembly.
        selected_masks = mask_process_eval(
            auto_classes,
            torch.as_tensor(auto_inds, dtype=torch.long),
            crop_box,
            np.asarray([full_h, full_w]),
            local_points_t,
            standard_logits,
            standard_quality,
        )
        standard_token_by_group = {
            int(group): int(standard_tokens[local_index].item())
            for local_index, group in enumerate(auto_inds)
        }
        for mask_data in selected_masks:
            edge = _edge_penalized(mask_data, crop_box, full_h, full_w)
            row = _record_from_mask_data(
                mask_data,
                record_index=len(selected_records),
                token=standard_token_by_group[int(mask_data["inds"])],
                crop_index=crop_index,
                edge_penalized=edge,
            )
            selected_records.append(row)
            assembly_masks.append(row["mask"])
            assembly_boxes.append(row["bbox_xyxy"])
            assembly_scores.append(float(row["assembly_score"]))
            assembly_inds.append(int(row["prompt_group_id"]))

        # D: process every native token with the same threshold/uncrop route,
        # but intentionally do not call native global conflict filtering.
        for token in range(4):
            token_masks = mask_process_eval(
                auto_classes,
                torch.as_tensor(auto_inds, dtype=torch.long),
                crop_box,
                np.asarray([full_h, full_w]),
                local_points_t,
                auto_logits[:, token].to(device),
                auto_quality[:, token].to(device),
            )
            for mask_data in token_masks:
                edge = _edge_penalized(mask_data, crop_box, full_h, full_w)
                all_candidate_records.append(
                    _record_from_mask_data(
                        mask_data,
                        record_index=len(all_candidate_records),
                        token=token,
                        crop_index=crop_index,
                        edge_penalized=edge,
                    )
                )

        if args.texture:
            crop_default_map = combine_mask(
                np.asarray([y2 - y1, x2 - x1]), local_points_t, standard_logits, standard_quality
            )
            memory_mask = torch.from_numpy(crop_default_map.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            update_validation_texture_memory(
                net=net,
                prepared=prepared,
                default_mask_logits=memory_mask,
                default_quality=standard_quality,
                texture_memory_bank=texture_memory_bank,
                texture_memory_bank_size=args.texture_memory_bank_size,
                device=device,
            )
        if args.context and prepared.context_entry is not None:
            context_memory_bank.append(prepared.context_entry)

    final_map = _assemble_instance_map(
        assembly_boxes,
        assembly_scores,
        assembly_masks,
        assembly_inds,
        inst_map.shape,
        args.instance_nms_iou,
    )
    annotated_selected = annotate_pool_ious(selected_records, inst_map)
    annotated_all = annotate_pool_ious(all_candidate_records, inst_map)
    native = native_final_stage(inst_map, final_map, threshold=ORACLE_MATCH_IOU)
    final_oracle = final_pool_oracle_stage(inst_map, final_map, threshold=ORACLE_MATCH_IOU)
    selected_oracle = oracle_pool_stage(annotated_selected, inst_map, threshold=ORACLE_MATCH_IOU)
    all_oracle = oracle_pool_stage(annotated_all, inst_map, threshold=ORACLE_MATCH_IOU)
    errors = error_partition(
        gt_map=inst_map,
        all_candidate_records=annotated_all,
        selected_records=annotated_selected,
        native_final=native,
        final_map=final_map,
        threshold=ORACLE_MATCH_IOU,
    )
    all_maxima = pool_gt_maxima(annotated_all, instance_ids(inst_map))
    selected_maxima = pool_gt_maxima(annotated_selected, instance_ids(inst_map))
    gt_rles = [
        {"gt_instance_id": int(gt_id), "mask_rle": encode_binary_rle(inst_map == gt_id)}
        for gt_id in instance_ids(inst_map)
    ]
    artifact = {
        "schema_version": 1,
        "sample_id": sample_id,
        "patient": patient,
        "image_shape": [int(full_h), int(full_w)],
        "gt_instances": gt_rles,
        "native_selected_before_assembly": [_serialize_mask_record(row) for row in annotated_selected],
        "all_native_candidates": [_serialize_mask_record(row) for row in annotated_all],
        "native_final_instances": _serialize_final_map(final_map),
        "oracle_matches": {
            "final_pool": final_oracle["matched"],
            "native_selected_pool": selected_oracle["matched"],
            "all_candidate_pool": all_oracle["matched"],
        },
    }
    image_record = {
        "sample_id": sample_id,
        "patient": patient,
        "gt_instance_count": len(instance_ids(inst_map)),
        "auto_prompt_group_count": len({int(row["prompt_group_id"]) for row in annotated_all}),
        "auto_prompt_crop_occurrence_count": crop_auto_prompt_count,
        "background_auto_prompt_count": background_point_count,
        "native_selected_mask_count_before_assembly": len(annotated_selected),
        "all_candidate_mask_count": len(annotated_all),
        "native_final_prediction_count": len(instance_ids(final_map)),
        "stages": {
            "native_final": native,
            "final_pool_oracle": {key: value for key, value in final_oracle.items() if key != "retained_final_map"},
            "native_selected_pool_oracle": selected_oracle,
            "all_candidate_pool_oracle": all_oracle,
        },
        "candidate_quality": {
            "all_candidate_best_iou": errors["all_candidate_coverage"],
            "selected_candidate_iou": errors["selected_candidate_coverage"],
            "selection_regret_mean": _selection_regret(errors["per_gt"]),
            "all_candidate_coverage_recall_at_0_5": all_oracle["coverage_recall_at_0_5"],
            "native_selected_coverage_recall_at_0_5": selected_oracle["coverage_recall_at_0_5"],
        },
        "errors": {key: value for key, value in errors.items() if key != "per_gt"},
        "per_gt": errors["per_gt"],
    }
    return image_record, artifact


def _selection_regret(per_gt: list[dict[str, Any]]) -> float | None:
    values = [float(row["all_candidate_best_iou"]) - float(row["selected_pool_best_iou"]) for row in per_gt]
    return float(np.mean(values)) if values else None


def _aggregate_stage(image_records: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    rows = [record["stages"][stage] for record in image_records]
    fields = ("tp", "fp", "fn", "dq", "sq", "pq")
    aggregate: dict[str, Any] = {"image_count": len(rows)}
    for field in fields:
        values = [row.get(field) for row in rows]
        numeric = [float(value) for value in values if value is not None]
        aggregate[field] = int(sum(numeric)) if field in {"tp", "fp", "fn"} else (float(np.mean(numeric)) if numeric else None)
    if stage in {"native_final", "final_pool_oracle"}:
        metric_key = "strict_metrics" if stage == "native_final" else "strict_metrics_after_filtering"
        aggregate["task_metrics_image_macro"] = {
            name: float(np.mean([record[metric_key][name] for record in rows])) for name in TASK_NAMES
        }
    else:
        aggregate["task_metrics_image_macro"] = {name: None for name in TASK_NAMES}
    return aggregate


def _aggregate_image_records(image_records: list[dict[str, Any]]) -> dict[str, Any]:
    by_patient: dict[str, list[dict[str, Any]]] = {}
    for patient in (7, 8):
        by_patient[str(patient)] = [record for record in image_records if int(record["patient"]) == patient]
    stage_names = ("native_final", "final_pool_oracle", "native_selected_pool_oracle", "all_candidate_pool_oracle")
    patients = {
        patient: {
            "image_count": len(rows),
            "stages": {stage: _aggregate_stage(rows, stage) for stage in stage_names},
            "candidate_quality": _aggregate_candidate_quality(rows),
            "errors": _aggregate_errors(rows),
        }
        for patient, rows in by_patient.items()
    }
    macro = {
        "aggregation": "equal_weight_mean_of_patient_7_and_patient_8",
        "stages": {stage: _mean_stage([patients["7"]["stages"][stage], patients["8"]["stages"][stage]]) for stage in stage_names},
        "candidate_quality": _mean_mapping([patients["7"]["candidate_quality"], patients["8"]["candidate_quality"]]),
        "errors": _mean_mapping([patients["7"]["errors"], patients["8"]["errors"]]),
    }
    return {"patients": patients, "patient_macro": macro}


def _mean_mapping(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float)) and not isinstance(value, bool)})
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys if all(row.get(key) is not None for row in rows)}


def _mean_stage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = _mean_mapping(rows)
    task_rows = [row.get("task_metrics_image_macro", {}) for row in rows]
    result["task_metrics_image_macro"] = {
        name: (
            float(np.mean([float(row[name]) for row in task_rows]))
            if all(row.get(name) is not None for row in task_rows)
            else None
        )
        for name in TASK_NAMES
    }
    return result


def _aggregate_candidate_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[float]] = {
        "all_candidate_coverage_recall_at_0_5": [],
        "native_selected_coverage_recall_at_0_5": [],
        "selection_regret_mean": [],
    }
    all_candidate_ious: list[float] = []
    selected_candidate_ious: list[float] = []
    for row in rows:
        for key in values:
            value = row["candidate_quality"].get(key)
            if value is not None:
                values[key].append(float(value))
        all_candidate_ious.extend(float(item["all_candidate_best_iou"]) for item in row["per_gt"])
        selected_candidate_ious.extend(float(item["selected_pool_best_iou"]) for item in row["per_gt"])

    def distribution(prefix: str, numeric: list[float]) -> dict[str, float | None]:
        if not numeric:
            return {f"{prefix}_{name}": None for name in ("mean", "median", "q10", "q25", "q75", "q90")}
        array = np.asarray(numeric, dtype=np.float64)
        return {
            f"{prefix}_mean": float(array.mean()),
            f"{prefix}_median": float(np.quantile(array, 0.50)),
            f"{prefix}_q10": float(np.quantile(array, 0.10)),
            f"{prefix}_q25": float(np.quantile(array, 0.25)),
            f"{prefix}_q75": float(np.quantile(array, 0.75)),
            f"{prefix}_q90": float(np.quantile(array, 0.90)),
        }

    return {
        **{key: float(np.mean(value)) if value else None for key, value in values.items()},
        **distribution("all_candidate_best_iou", all_candidate_ious),
        **distribution("selected_candidate_iou", selected_candidate_ious),
        "gt_instance_count_for_iou_distribution": len(all_candidate_ious),
    }


def _aggregate_errors(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "generation_miss",
        "selection_miss",
        "assembly_loss",
        "native_final_tp",
        "native_final_false_positive_count",
        "native_final_false_negative_count",
        "all_candidate_one_to_one_conflict_gt_count",
        "selected_one_to_one_conflict_gt_count",
    )
    output = {key: sum(int(row["errors"]["counts"].get(key, 0)) for row in rows) for key in keys}
    output["duplicate_unmatched_prediction_count"] = sum(int(row["errors"]["native_final_structural_errors"]["duplicate_unmatched_prediction_count"]) for row in rows)
    for sensitivity in ("overlap_fraction_gt_or_pred_gt_0", "overlap_fraction_gt_or_pred_gt_0.1"):
        for key in ("split", "merge"):
            output[f"{key}_{sensitivity}"] = sum(
                int(row["errors"]["native_final_structural_errors"]["sensitivity"][sensitivity][key]) for row in rows
            )
    return output


def _reproduction_check(aggregate: dict[str, Any], reference: dict[int, dict[str, float]], tolerance: float) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    observed: dict[str, Any] = {}
    for patient in (7, 8):
        current = aggregate["patients"][str(patient)]["stages"]["native_final"]["task_metrics_image_macro"]
        observed[str(patient)] = current
        for name in TASK_NAMES:
            diff = abs(float(current[name]) - float(reference[patient][name]))
            if diff > tolerance:
                mismatches.append({"patient": patient, "metric": name, "observed": current[name], "reference": reference[patient][name], "absolute_difference": diff})
    return {"status": "pass" if not mismatches else "fail", "tolerance": tolerance, "observed": observed, "reference": reference, "mismatches": mismatches}


def _write_csv(path: Path, image_records: list[dict[str, Any]]) -> None:
    fields = [
        "sample_id", "patient", "gt_instance_count", "auto_prompt_group_count", "native_selected_mask_count_before_assembly", "all_candidate_mask_count", "native_final_prediction_count", "stage",
        "tp", "fp", "fn", "dq", "sq", "pq", "dice1", "dice2", "aji",
        "coverage_recall_at_0_5", "raw_prediction_group_count", "raw_prediction_mask_count",
        "all_candidate_coverage_recall_at_0_5", "native_selected_coverage_recall_at_0_5", "selection_regret_mean",
        "all_candidate_best_iou_mean", "all_candidate_best_iou_median", "all_candidate_best_iou_q10", "all_candidate_best_iou_q25", "all_candidate_best_iou_q75", "all_candidate_best_iou_q90",
        "selected_candidate_iou_mean", "selected_candidate_iou_median", "selected_candidate_iou_q10", "selected_candidate_iou_q25", "selected_candidate_iou_q75", "selected_candidate_iou_q90",
        "generation_miss", "selection_miss", "assembly_loss", "native_final_tp", "native_final_false_positive_count", "native_final_false_negative_count",
        "all_candidate_one_to_one_conflict_gt_count", "selected_one_to_one_conflict_gt_count", "duplicate_unmatched_prediction_count",
        "split_overlap_fraction_gt_or_pred_gt_0", "merge_overlap_fraction_gt_or_pred_gt_0",
        "split_overlap_fraction_gt_or_pred_gt_0.1", "merge_overlap_fraction_gt_or_pred_gt_0.1",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for image in image_records:
            candidate = image["candidate_quality"]
            all_distribution = candidate["all_candidate_best_iou"]
            selected_distribution = candidate["selected_candidate_iou"]
            counts = image["errors"]["counts"]
            structural = image["errors"]["native_final_structural_errors"]
            sensitivity = structural["sensitivity"]
            for stage, stage_record in image["stages"].items():
                strict = stage_record.get("strict_metrics") or stage_record.get("strict_metrics_after_filtering") or {}
                writer.writerow({
                    "sample_id": image["sample_id"],
                    "patient": image["patient"],
                    "gt_instance_count": image["gt_instance_count"],
                    "auto_prompt_group_count": image["auto_prompt_group_count"],
                    "native_selected_mask_count_before_assembly": image["native_selected_mask_count_before_assembly"],
                    "all_candidate_mask_count": image["all_candidate_mask_count"],
                    "native_final_prediction_count": image["native_final_prediction_count"],
                    "stage": stage,
                    **{key: stage_record.get(key) for key in ("tp", "fp", "fn", "dq", "sq", "pq", "coverage_recall_at_0_5", "raw_prediction_group_count", "raw_prediction_mask_count")},
                    **{key: strict.get(key) for key in ("dice1", "dice2", "aji")},
                    "all_candidate_coverage_recall_at_0_5": candidate["all_candidate_coverage_recall_at_0_5"],
                    "native_selected_coverage_recall_at_0_5": candidate["native_selected_coverage_recall_at_0_5"],
                    "selection_regret_mean": candidate["selection_regret_mean"],
                    **{f"all_candidate_best_iou_{key}": all_distribution[key] for key in ("mean", "median", "q10", "q25", "q75", "q90")},
                    **{f"selected_candidate_iou_{key}": selected_distribution[key] for key in ("mean", "median", "q10", "q25", "q75", "q90")},
                    **{key: counts.get(key, 0) for key in ("generation_miss", "selection_miss", "assembly_loss", "native_final_tp", "native_final_false_positive_count", "native_final_false_negative_count", "all_candidate_one_to_one_conflict_gt_count", "selected_one_to_one_conflict_gt_count")},
                    "duplicate_unmatched_prediction_count": structural["duplicate_unmatched_prediction_count"],
                    "split_overlap_fraction_gt_or_pred_gt_0": sensitivity["overlap_fraction_gt_or_pred_gt_0"]["split"],
                    "merge_overlap_fraction_gt_or_pred_gt_0": sensitivity["overlap_fraction_gt_or_pred_gt_0"]["merge"],
                    "split_overlap_fraction_gt_or_pred_gt_0.1": sensitivity["overlap_fraction_gt_or_pred_gt_0.1"]["split"],
                    "merge_overlap_fraction_gt_or_pred_gt_0.1": sensitivity["overlap_fraction_gt_or_pred_gt_0.1"]["merge"],
                })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-declaration", required=True)
    parser.add_argument("--reference-performance-summary", "--reference-three-seed-summary", dest="reference_performance_summary", default="")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", required=True, type=int, choices=DIAGNOSIS_SEEDS)
    parser.add_argument("--arm", required=True, choices=["c0", "c1", "c2_ar"])
    parser.add_argument("--model-config", default="args.py")
    parser.add_argument("--sam-config", default="sam2_hiera_l")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--out-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--load", choices=["sequence", "unsequence", "clockwise", "unclockwise"], default="unclockwise")
    parser.add_argument("--point-nms-thr", type=int, default=12)
    parser.add_argument("--instance-nms-iou", type=float, default=0.5)
    parser.add_argument("--prompt-chunk-size", type=int, default=64)
    parser.add_argument("--texture-memory-bank-size", type=int, default=64)
    parser.add_argument("--context-atten-k", type=int, default=1)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--reproduction-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--point-filtering", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--texture", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--context", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.crop_size != 256 or args.out_size != 256 or args.overlap != 32 or args.load != "unclockwise":
        raise ValueError("zero-training diagnosis freezes the Phase-1 TNBC inference geometry")
    if args.point_nms_thr != 12 or args.instance_nms_iou != 0.5 or args.prompt_chunk_size != 64:
        raise ValueError("zero-training diagnosis freezes point NMS, instance NMS, and decoder chunking")
    if not args.point_filtering or not args.texture or not args.context or args.context_atten_k != 1:
        raise ValueError("zero-training diagnosis freezes filtering, texture, and context enabled with context_atten_k=1")
    manifest_path = Path(args.manifest).resolve()
    manifest, records = load_dataset_manifest(manifest_path, expected_dataset="tnbc", require_labels=True, verify_hashes=True)
    validate_scope(manifest, records)
    checkpoint = Path(args.checkpoint).resolve()
    declaration = validate_declaration(Path(args.checkpoint_declaration).resolve(), checkpoint, seed=args.seed, arm=args.arm)
    if args.arm in {"c0", "c1"} and not args.reference_performance_summary:
        raise ValueError("C0/C1 diagnosis requires the frozen performance reference summary")
    if args.arm == "c2_ar" and args.reference_performance_summary:
        raise ValueError("C2-AR is a new epoch-5 checkpoint and must not be forced to reproduce a C0/C1 reference")
    targets = None
    reference_sha = None
    if args.reference_performance_summary:
        reference_path = Path(args.reference_performance_summary).resolve()
        reference = read_json(reference_path, "frozen performance reference summary")
        targets = reference_metrics(reference, seed=args.seed, arm=args.arm)
        reference_sha = sha256_file(reference_path)
    set_determinism(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("zero-training oracle diagnosis requires CUDA inference; CPU execution is intentionally unsupported")
    torch.cuda.set_device(args.gpu_device)
    device_index = int(torch.cuda.current_device())
    device = torch.device("cuda", device_index)
    torch.cuda.reset_peak_memory_stats(device_index)
    model_config = Config.fromfile(str(Path(args.model_config).resolve()))
    net = build_sam2(args.sam_config, str(checkpoint), device=device, checkpoint_has_training_state=True)
    point_net, point_encoder = build_model(model_config)
    point_net.to(device).eval()
    point_encoder.to(device).eval()
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "model1" not in checkpoint_payload:
        raise ValueError("formal state is missing CA-SAM2 point-head model1")
    missing, unexpected = point_net.load_state_dict(checkpoint_payload["model1"], strict=False)
    if missing or unexpected:
        raise ValueError(f"model1 state mismatch: missing={len(missing)}, unexpected={len(unexpected)}")
    if list(checkpoint_payload.get("texture_memory_bank_list", []) or []):
        raise ValueError("approved warm-start diagnosis requires an empty embedded texture bank")

    runtime = runtime_cfg(args)
    dataset = TNBC(
        runtime,
        model_config,
        args.data_path,
        args.load,
        mode="test",
        manifest_path=str(manifest_path),
        data_split="train",
        verify_manifest_hashes=True,
    )
    output_dir = Path(args.output_dir).resolve()
    completed_dir = output_dir / "completed_images"
    progress_path = output_dir / "progress.json"
    texture_state_path = output_dir / "texture_memory_bank.pt"
    fingerprint = {
        "protocol": "tnbc_zero_training_oracle_diagnosis_two_seed_v1",
        "seed": args.seed,
        "arm": args.arm,
        "manifest_sha256": manifest["manifest_sha256"],
        "checkpoint_sha256": declaration["checkpoint_sha256"],
        "reference_performance_summary_sha256": reference_sha,
        "frozen_inference": {"crop_size": args.crop_size, "out_size": args.out_size, "overlap": args.overlap, "load": args.load, "point_nms": args.point_nms_thr, "instance_nms": args.instance_nms_iou, "prompt_chunk": args.prompt_chunk_size, "texture": True, "context": True},
    }
    fingerprint_sha = json_sha256(fingerprint)
    completed: list[int] = []
    image_records: list[dict[str, Any]] = []
    texture_memory_bank: list = []
    if args.resume:
        progress = read_json(progress_path, "oracle progress")
        if progress.get("fingerprint_sha256") != fingerprint_sha:
            raise ValueError("resume output does not match this fixed checkpoint/manifest/inference fingerprint")
        completed = [int(value) for value in progress.get("completed_indices", [])]
        if completed != list(range(len(completed))):
            raise ValueError("resume progress has a non-contiguous image prefix")
        for index in completed:
            image_records.append(read_gzip_json(completed_dir / f"{index:05d}.json.gz", "completed oracle image")["image_record"])
        if completed:
            texture_memory_bank = list(torch.load(texture_state_path, map_location="cpu", weights_only=False))
    else:
        if progress_path.exists() or any(completed_dir.glob("*.json.gz")) or (output_dir / "summary.json").exists():
            raise FileExistsError("oracle output already exists; use --resume or choose a new output directory")
        write_json_atomic(progress_path, {"schema_version": 1, "status": "in_progress", "fingerprint": fingerprint, "fingerprint_sha256": fingerprint_sha, "completed_indices": []})

    started = time.perf_counter()
    for index in range(len(completed), len(dataset)):
        image_started = time.perf_counter()
        image, inst_map, _, _, _, _, _, _, sample_id = dataset[index]
        record = records[index]
        patient = int(record["patient"])
        inst_np = np.asarray(inst_map.cpu().numpy() if torch.is_tensor(inst_map) else inst_map, dtype=np.int32)
        image_record, artifact = diagnose_image(
            image=image,
            inst_map=inst_np,
            sample_id=str(sample_id),
            patient=patient,
            point_net=point_net,
            point_encoder=point_encoder,
            net=net,
            texture_memory_bank=texture_memory_bank,
            args=args,
            device=device,
        )
        image_record["wall_seconds"] = time.perf_counter() - image_started
        artifact["source_image_png"] = f"images/{sample_id}.png"
        _save_display_image(image, output_dir / artifact["source_image_png"])
        write_gzip_json_atomic(completed_dir / f"{index:05d}.json.gz", {"schema_version": 1, "record_index": index, "sample_id": str(sample_id), "manifest_sha256": manifest["manifest_sha256"], "image_record": image_record, "artifact": artifact})
        torch.save(texture_memory_bank, texture_state_path)
        image_records.append(image_record)
        completed.append(index)
        write_json_atomic(progress_path, {"schema_version": 1, "status": "in_progress", "fingerprint": fingerprint, "fingerprint_sha256": fingerprint_sha, "completed_indices": completed})
        print(f"[zero-oracle] seed={args.seed} arm={args.arm} {index + 1}/{len(dataset)} {sample_id} wall_s={image_record['wall_seconds']:.2f}", flush=True)

    aggregate = _aggregate_image_records(image_records)
    reproduction = (
        _reproduction_check(aggregate, targets, args.reproduction_tolerance)
        if targets is not None
        else {
            "status": "not_applicable_new_c2_checkpoint",
            "reason": "C2-AR epoch-5 is a new checkpoint; its metrics are not expected to reproduce a C0/C1 frozen report.",
            "observed": {
                str(patient): aggregate["patients"][str(patient)]["stages"]["native_final"]["task_metrics_image_macro"]
                for patient in (7, 8)
            },
        }
    )
    if reproduction["status"] == "fail":
        write_json_atomic(output_dir / "reproduction_failure.json", reproduction)
        raise RuntimeError("native final metrics do not reproduce the frozen fixed-epoch report; oracle attribution is stopped")
    _write_csv(output_dir / "per_image.csv", image_records)
    torch.cuda.synchronize(device_index)
    report = {
        "schema_version": 1,
        "protocol": "tnbc_zero_training_oracle_diagnosis_v1",
        "status": "complete",
        "diagnosis_scope": "read_only_no_grad_no_optimizer",
        "seed": args.seed,
        "arm": args.arm,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": {"path": str(manifest_path), "sha256": manifest["manifest_sha256"], "protocol_id": manifest.get("protocol_id"), "record_count": len(records), "patients": [7, 8]},
        "checkpoint": declaration,
        "reference_reproduction": reproduction,
        "frozen_inference": fingerprint["frozen_inference"],
        "export": {"all_four_candidate_masks": "RLE in completed_images/*.json.gz", "native_selected_masks_preassembly": "RLE in completed_images/*.json.gz", "native_final_instances": "RLE in completed_images/*.json.gz", "prompt_group_id": "stored on every candidate record", "quality_score": "stored as quality on every candidate record"},
        "summary": aggregate,
        "runtime": {"wall_seconds_this_process": time.perf_counter() - started, "peak_memory_allocated_mib": torch.cuda.max_memory_allocated(device_index) / (1024**2), "peak_memory_reserved_mib": torch.cuda.max_memory_reserved(device_index) / (1024**2)},
        "repository": {"branch": git_value("branch", "--show-current"), "commit": git_value("rev-parse", "HEAD")},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "torch_cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device_index), "cudnn_deterministic": bool(torch.backends.cudnn.deterministic), "cudnn_benchmark": bool(torch.backends.cudnn.benchmark)},
    }
    write_json_atomic(output_dir / "summary.json", report)
    write_json_atomic(progress_path, {"schema_version": 1, "status": "complete", "fingerprint": fingerprint, "fingerprint_sha256": fingerprint_sha, "completed_indices": completed})
    print(json.dumps({"status": "complete", "output_dir": str(output_dir), "summary": str(output_dir / "summary.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
