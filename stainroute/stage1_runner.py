"""Frozen train/calibration ADD+SPLIT oracle runner for StainRoute Stage 1.

Candidate generation receives only image pixels and first-pass predictions.
Ground truth is read only after a decoded action has been assembled, to compute
its global oracle utility and diagnostics.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from skimage.io import imread
from torch.utils.data import DataLoader

from stainroute.actions import (
    ActionCandidate,
    ActionType,
    AddCandidateConfig,
    SplitAssemblyConfig,
    SplitCandidateConfig,
    apply_add_action,
    apply_split_action,
    build_conflict_graph,
    generate_add_candidates,
    generate_split_candidates,
)
from stainroute.inference.cached_decode import (
    EncodedCrop,
    component_containing_point,
    decode_prompts_from_features,
    encode_crop,
    max_abs_logit_error,
)
from stainroute.inference.coordinates import global_to_crop
from stainroute.metrics import PQEvaluation, evaluate_pq
from stainroute.oracle_actions import (
    ActionUtility,
    compute_action_utility,
    exact_joint_oracle,
    normalized_oracle_recovery,
    utility_guided_beam_joint_oracle,
)
from stainroute.utils import sha256_file

from run.run_on_epoch import _assemble_instance_map, combine_mask, crop_with_overlap, inference, mask_process_eval
from sam2_train.modeling.utils import point_nms, predict


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _command_output(*command: str) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _require_frozen_baseline(cfgs: Any, split_manifest_path: Path) -> dict[str, Any]:
    """Fail closed unless the command is tied to Baseline v1 and one split.

    This prevents an accidental Stage 1 invocation with a changed checkpoint,
    non-canonical NMS/TTA setting, or an unrecorded train/calibration manifest.
    It deliberately does not inspect GT data.
    """

    baseline_path = Path(cfgs.stainroute_baseline_manifest)
    if not baseline_path.is_file():
        raise FileNotFoundError(
            "Stage 1 requires the frozen Baseline v1 manifest. Run "
            "tools/stainroute_freeze_baseline.py first: "
            f"{baseline_path}"
        )
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline.get("baseline_name") != "StainRoute Development Baseline v1":
        raise ValueError(f"Unexpected baseline manifest: {baseline_path}")
    resolved = baseline.get("config", {}).get("resolved", {})
    expected_eval = resolved.get("evaluation", {})
    if int(getattr(cfgs, "test_nms_thr", -1)) != int(expected_eval.get("test_nms_thr", -1)):
        raise ValueError("Stage 1 NMS threshold differs from frozen Baseline v1")
    if bool(getattr(cfgs, "tta", False)) != bool(expected_eval.get("tta", False)):
        raise ValueError("Stage 1 TTA setting differs from frozen Baseline v1")
    if int(getattr(cfgs, "seed", -1)) != int(expected_eval.get("seed", -1)):
        raise ValueError("Stage 1 seed differs from frozen Baseline v1")
    if int(getattr(cfgs, "b", -1)) != int(expected_eval.get("decoder_batch_size", -1)):
        raise ValueError("Stage 1 decoder batch size differs from frozen Baseline v1")
    if not bool(getattr(cfgs, "texture", False)) or not bool(getattr(cfgs, "context", False)):
        raise ValueError("Stage 1 requires frozen texture/context enabled settings")
    if not split_manifest_path.is_file():
        raise FileNotFoundError(f"Stage 1 split manifest does not exist: {split_manifest_path}")

    observed_checkpoint = sha256_file(cfgs.sam_ckpt)
    root_name = Path(cfgs.data_path).name.lower()
    expected_key = "monuseg_stainpms" if "monuseg" in root_name else "tnbc_stainpms" if "tnbc" in root_name else None
    if expected_key is None:
        raise ValueError(f"Cannot associate Stage 1 data root with Baseline v1: {cfgs.data_path}")
    expected_checkpoint = baseline.get("checkpoints", {}).get(expected_key, {})
    if observed_checkpoint.lower() != str(expected_checkpoint.get("sha256", "")).lower():
        raise RuntimeError(
            f"Stage 1 checkpoint SHA256 does not match {expected_key}: "
            f"{observed_checkpoint} != {expected_checkpoint.get('sha256')}"
        )
    split_key = "monuseg" if expected_key.startswith("monuseg") else "tnbc"
    expected_split = baseline.get("splits", {}).get(split_key, {})
    observed_split = sha256_file(split_manifest_path)
    if observed_split.lower() != str(expected_split.get("sha256", "")).lower():
        raise RuntimeError(
            f"Stage 1 split manifest SHA256 does not match frozen Baseline v1: "
            f"{observed_split} != {expected_split.get('sha256')}"
        )
    return baseline


def _safe_name(name: Any) -> str:
    if isinstance(name, (list, tuple)):
        return "_".join(str(item) for item in name)
    return str(name)


def _ori_hw(ori_shape: Any) -> tuple[int, int]:
    if torch.is_tensor(ori_shape):
        values = ori_shape.detach().cpu().numpy().reshape(-1)
    else:
        values = np.asarray(ori_shape).reshape(-1)
    return int(values[0]), int(values[1])


def _find_image(image_root: Path, stem: str) -> Path:
    for extension in IMAGE_EXTENSIONS:
        path = image_root / f"{stem}{extension}"
        if path.is_file():
            return path
    paths = [path for path in image_root.glob(f"{stem}.*") if path.is_file()]
    if len(paths) != 1:
        raise FileNotFoundError(f"Cannot uniquely find image '{stem}' under {image_root}")
    return paths[0]


def _json_config(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Stage 1 config must be JSON-compatible YAML: {path}") from exc


def _subset_loader(cfgs: Any, test_dataset: Any, split_manifest_path: Path, split_name: str) -> tuple[DataLoader, Path, dict[str, Any]]:
    manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    if split_name not in {"router_train", "calibration"}:
        raise ValueError("Stage 1 permits only router_train or calibration")
    if split_name not in manifest:
        raise ValueError(f"Split '{split_name}' missing from {split_manifest_path}")
    selected_stems = list(manifest[split_name])
    if not selected_stems:
        raise ValueError(f"Split '{split_name}' is empty")
    data_root = Path(cfgs.data_path)
    image_root = data_root / "train_12" / "images"
    label_root = data_root / "train_12" / "labels"
    filenames = {path.stem: path.name for path in image_root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS}
    unknown = sorted(set(selected_stems) - set(filenames))
    if unknown:
        raise FileNotFoundError(f"Split references images absent from train_12: {unknown[:8]}")

    dataset = copy.copy(test_dataset)
    dataset.image_root = str(image_root)
    dataset.label_root = str(label_root)
    dataset.paths = [filenames[stem] for stem in sorted(selected_stems)]
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfgs.num_workers, pin_memory=True)
    return loader, image_root, manifest


@dataclass(frozen=True)
class DecodedAction:
    candidate: ActionCandidate
    add_mask: np.ndarray | None = None
    child_first: np.ndarray | None = None
    child_second: np.ndarray | None = None
    child_first_logits: np.ndarray | None = None
    child_second_logits: np.ndarray | None = None
    decode_reason: str = "decoded"

    def apply(self, prediction: np.ndarray, split_config: SplitAssemblyConfig, min_added_area: int):
        if self.candidate.action_type is ActionType.ADD:
            if self.add_mask is None:
                from stainroute.actions.assembly import AssemblyResult

                return AssemblyResult(prediction.copy(), False, self.decode_reason, {})
            return apply_add_action(prediction, self.add_mask, min_added_area=min_added_area)
        if self.child_first is None or self.child_second is None:
            from stainroute.actions.assembly import AssemblyResult

            return AssemblyResult(prediction.copy(), False, self.decode_reason, {})
        return apply_split_action(
            prediction,
            parent_id=self.candidate.affected_instance_ids[0],
            child_first=self.child_first,
            child_second=self.child_second,
            first_point=self.candidate.positive_points[0],
            second_point=self.candidate.positive_points[1],
            first_logits=self.child_first_logits,
            second_logits=self.child_second_logits,
            config=split_config,
        )


def _uncrop_logits(logits: torch.Tensor, crop_box: tuple[int, int, int, int], shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = crop_box
    output = np.full((height, width), -np.inf, dtype=np.float32)
    local = logits.detach().float().cpu().numpy()
    output[y1:y2, x1:x2] = local[: y2 - y1, : x2 - x1]
    return output


def _mask_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _assign_tiles(actions: list[ActionCandidate], caches: dict[tuple[int, int, int, int], EncodedCrop]) -> list[ActionCandidate]:
    output: list[ActionCandidate] = []
    ordered_boxes = list(caches)
    for action in actions:
        matching = [
            box
            for box in ordered_boxes
            if all(box[0] <= point.x < box[2] and box[1] <= point.y < box[3] for point in action.positive_points)
        ]
        if matching:
            output.append(replace(action, tile_box=matching[0]))
        else:
            output.append(replace(action, decoded_features={"decode_status": "no_common_cached_tile"}))
    return output


def _smoke_candidate_subset(actions: list[ActionCandidate]) -> list[ActionCandidate]:
    """Bound a smoke run without changing the frozen formal action space.

    The smoke objective is encoder-cache equivalence and ADD/SPLIT execution
    plumbing.  It is explicitly not an oracle-headroom measurement, so it
    uses the first two deterministic proposals from each family and a tiny
    control/search budget.  Formal runs (``max_images == 0``) never call this.
    """

    additions = [action for action in actions if action.action_type is ActionType.ADD]
    splits = [action for action in actions if action.action_type is ActionType.SPLIT]
    return additions[:2] + splits[:2]


def _update_texture_memory(
    *,
    net: Any,
    encoded: EncodedCrop,
    decoded_logits: torch.Tensor,
    prompt_iou_predictions: torch.Tensor,
    prompt_points: torch.Tensor,
    ori_shape: Any,
    cfgs: Any,
    memory_bank_list: list,
    device: torch.device,
) -> None:
    """Mirror first-pass texture-bank updates; actions never call this."""

    if not cfgs.texture:
        return
    # combine_mask performs prompt-level NMS and therefore requires one score
    # per prompt.  The scalar mean is only the memory-bank quality summary.
    inst_pred = combine_mask(ori_shape, prompt_points, decoded_logits, prompt_iou_predictions)
    mean_iou = prompt_iou_predictions.mean()
    high_res = torch.from_numpy(inst_pred.astype(float)).to(torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    mask_features, mask_positions = net._encode_new_memory(
        current_vision_feats=list(encoded.vision_feats),
        feat_sizes=[(64, 64), (32, 32), (16, 16)],
        pred_masks_high_res=high_res,
        is_mask_from_pts=True,
    )
    mask_features = mask_features.to(device=device, non_blocking=True)
    mask_position = mask_positions[0].to(device=device, non_blocking=True)
    if len(memory_bank_list) < cfgs.texture_memory_bank_size:
        for batch_index in range(mask_features.size(0)):
            memory_bank_list.append(
                [
                    mask_features[batch_index].unsqueeze(0),
                    mask_position[batch_index].unsqueeze(0),
                    mean_iou,
                    encoded.image_embed[batch_index].reshape(-1).detach(),
                ]
            )
        return
    # Same deterministic replacement policy as validation_on_epoch.
    for batch_index in range(mask_features.size(0)):
        flat = torch.stack([item[0].reshape(-1).to(device) for item in memory_bank_list])
        normalized = F.normalize(flat, p=2, dim=1)
        similarities = torch.mm(normalized, normalized.t())
        similarities.fill_diagonal_(float("-inf"))
        candidate = F.normalize(mask_features[batch_index].reshape(-1), p=2, dim=0).unsqueeze(1)
        scores = torch.mm(normalized, candidate).squeeze()
        least_similar = torch.argmin(scores)
        nearest_to_least = torch.argmax(similarities[least_similar])
        if scores[least_similar] < similarities[least_similar, nearest_to_least]:
            if mean_iou > memory_bank_list[int(nearest_to_least)][2] - 0.1:
                memory_bank_list.pop(int(nearest_to_least))
                memory_bank_list.append(
                    [
                        mask_features[batch_index].unsqueeze(0),
                        mask_position[batch_index].unsqueeze(0),
                        mean_iou,
                        encoded.image_embed[batch_index].reshape(-1).detach(),
                    ]
                )


def _base_prediction_with_cache(
    *,
    image_tensor: torch.Tensor,
    ori_shape: Any,
    cfgs: Any,
    args: Any,
    net: Any,
    point_net: Any,
    point_encoder: Any,
    texture_memory_bank_list: list,
    device: torch.device,
) -> tuple[np.ndarray, dict[tuple[int, int, int, int], EncodedCrop], dict[int, dict[str, Any]], dict[str, float | bool | None], dict[str, int]]:
    """Run the unchanged first pass once and retain every tile encoding."""

    height, width = _ori_hw(ori_shape)
    image_tensor = image_tensor.to(device)
    # validation_on_epoch mutates its texture bank across the evaluation
    # sequence.  Keep that exact behaviour so the frozen base map remains the
    # canonical StainPMS prediction; action decoding never updates this bank.
    memory_bank = texture_memory_bank_list
    context_bank: list = []
    all_masks: list[np.ndarray] = []
    all_boxes: list[list[float]] = []
    all_scores: list[float] = []
    all_inds: list[int] = []
    records: list[dict[str, Any]] = []
    all_points: list[np.ndarray] = []
    all_point_scores: list[np.ndarray] = []
    all_point_classes: list[np.ndarray] = []
    processed_boxes: list[tuple[int, int, int, int]] = []
    point_ids: dict[tuple[float, float], int] = {}
    next_id = 0
    caches: dict[tuple[int, int, int, int], EncodedCrop] = {}
    equivalence: dict[str, float | bool | None] = {"max_abs_logit_error": None, "max_abs_iou_error": None, "passed": None}
    counters = {"encoder_calls": 0, "base_decoder_actions": 0}

    crop_boxes = [tuple(int(item) for item in box) for box in crop_with_overlap(image_tensor[0], cfgs.crop_size, cfgs.crop_size, cfgs.overlap, cfgs.load).tolist()]
    print(f"[stage1-base] tiles={len(crop_boxes)} image_shape={height}x{width}", flush=True)
    for crop_index, crop_box in enumerate(crop_boxes, start=1):
        x1, y1, x2, y2 = crop_box
        crop = image_tensor[..., y1:y2, x1:x2]
        print(f"[stage1-base] tile={crop_index}/{len(crop_boxes)} box={crop_box}", flush=True)
        points, scores, classes, _, _, _, _ = predict(
            point_net,
            crop,
            ori_shape=np.array((y2 - y1, x2 - x1)),
            filtering=args.test.filtering,
            nms_thr=args.test.nms_thr,
        )
        if len(points) == 0:
            processed_boxes.append(crop_box)
            continue
        points[:, 0] += x1
        points[:, 1] += y1
        keep_new = np.ones(len(points), dtype=bool)
        for previous in processed_boxes:
            px1, py1, px2, py2 = previous
            keep_new &= ~(
                (points[:, 0] >= px1 + 1)
                & (points[:, 0] <= px2 - 1)
                & (points[:, 1] >= py1 + 1)
                & (points[:, 1] <= py2 - 1)
            )
        processed_boxes.append(crop_box)
        points, scores, classes = points[keep_new], scores[keep_new], classes[keep_new]
        if len(points) == 0:
            continue
        all_points.append(points)
        all_point_scores.append(scores)
        all_point_classes.append(classes)
        current_points, current_scores, current_classes = point_nms(
            np.vstack(all_points), np.concatenate(all_point_scores), np.concatenate(all_point_classes), args.test.nms_thr
        )
        current_indices = []
        for point in current_points:
            key = tuple(float(item) for item in point.tolist())
            if key not in point_ids:
                point_ids[key] = next_id
                next_id += 1
            current_indices.append(point_ids[key])
        current_indices_tensor = torch.tensor(current_indices, dtype=torch.long)
        global_prompts = torch.from_numpy(current_points).unsqueeze(1)
        keep = (
            (global_prompts[..., 0] >= x1)
            & (global_prompts[..., 0] < x2)
            & (global_prompts[..., 1] >= y1)
            & (global_prompts[..., 1] < y2)
        ).squeeze(1)
        if keep.sum() == 0:
            continue
        local_prompts = (global_prompts[keep] - torch.as_tensor([x1, y1])).to(device).float()
        labels = torch.ones(local_prompts.size(0), 1, dtype=torch.int, device=device)

        # One pre-run equivalence check on the first eligible crop only.
        if equivalence["passed"] is None:
            equivalence_context = list(context_bank)
            full_pred, full_values, _, _, _ = inference(
                net, point_encoder, crop, list(memory_bank), local_prompts, labels,
                [(64, 64), (32, 32), (16, 16)], equivalence_context, x1, y1, False, cfgs, device,
            )
            check_encoded = encode_crop(
                net, point_encoder, crop, list(memory_bank), list(context_bank), crop_box=crop_box,
                cfgs=cfgs, device=device,
            )
            cached_check = decode_prompts_from_features(
                net, check_encoded, local_prompts, labels, out_size=cfgs.out_size, device=device
            )
            logit_error = max_abs_logit_error(full_pred, cached_check.logits)
            iou_error = float((full_values.detach().float().cpu() - cached_check.predicted_iou.detach().float().cpu()).abs().max().item())
            equivalence = {
                "max_abs_logit_error": logit_error,
                "max_abs_iou_error": iou_error,
                "passed": bool(logit_error <= 1.0e-6 and iou_error <= 1.0e-6),
            }
            if not equivalence["passed"]:
                raise RuntimeError(f"Cached decode equivalence failed: {equivalence}")

        encoded = encode_crop(
            net, point_encoder, crop, memory_bank, context_bank, crop_box=crop_box,
            cfgs=cfgs, device=device,
        )
        counters["encoder_calls"] += 1
        decoded = decode_prompts_from_features(net, encoded, local_prompts, labels, out_size=cfgs.out_size, device=device)
        counters["base_decoder_actions"] += int(local_prompts.shape[0])
        caches[crop_box] = encoded
        _update_texture_memory(
            net=net,
            encoded=encoded,
            decoded_logits=decoded.logits,
            prompt_iou_predictions=decoded.predicted_iou,
            prompt_points=local_prompts,
            ori_shape=ori_shape,
            cfgs=cfgs,
            memory_bank_list=memory_bank,
            device=device,
        )
        masks = mask_process_eval(
            current_classes[keep.cpu().numpy()],
            current_indices_tensor[keep],
            crop_box,
            ori_shape,
            local_prompts,
            decoded.logits,
            decoded.predicted_iou,
        )
        for mask in masks:
            bx1, by1, bx2, by2 = mask["bbox"]
            margin = 7
            edge_penalized = bool(
                (bx1 > margin and abs(bx1 - x1) <= margin)
                or (abs(bx2 - height) > margin and abs(bx2 - x2) <= margin)
                or (by1 > margin and abs(by1 - y1) <= margin)
                or (abs(by2 - width) > margin and abs(by2 - y2) <= margin)
            )
            all_masks.append(mask["segmentation"][:height, :width])
            all_boxes.append(mask["bbox"])
            all_scores.append(float(mask["predicted_iou"]) * (0.3 if edge_penalized else 1.0))
            all_inds.append(mask["inds"])
            records.append(
                {
                    "bbox": mask["bbox"],
                    "crop_box": list(crop_box),
                    "predicted_iou": float(mask["predicted_iou"]),
                    "stability_score": float(mask["stability_score"]),
                    "edge_penalized": edge_penalized,
                }
            )

    prediction, selected_records = _assemble_instance_map(
        all_boxes, all_scores, all_masks, all_inds, (height, width), args.data.post.iou_threshold,
        all_records=records, return_records=True,
    )
    selected_by_id = {int(record["final_id"]): record for record in selected_records}
    return prediction.astype(np.int32), caches, selected_by_id, equivalence, counters


def _decode_grouped_actions(
    actions: list[ActionCandidate],
    caches: dict[tuple[int, int, int, int], EncodedCrop],
    *,
    ori_shape: Any,
    cfgs: Any,
    net: Any,
    device: torch.device,
) -> tuple[dict[str, DecodedAction], int]:
    decoded: dict[str, DecodedAction] = {}
    extra_cost = 0
    by_tile: dict[tuple[int, int, int, int] | None, list[ActionCandidate]] = {}
    for action in actions:
        by_tile.setdefault(action.tile_box, []).append(action)
    shape = _ori_hw(ori_shape)
    for tile_box, tile_actions in by_tile.items():
        if tile_box is None or tile_box not in caches:
            for action in tile_actions:
                decoded[action.action_id] = DecodedAction(action, decode_reason="no_common_cached_tile")
            continue
        encoded = caches[tile_box]
        additions = [action for action in tile_actions if action.action_type is ActionType.ADD]
        if additions:
            prompts = torch.tensor(
                [[[global_to_crop(action.positive_points[0], tile_box).x, global_to_crop(action.positive_points[0], tile_box).y]] for action in additions],
                dtype=torch.float32,
                device=device,
            )
            labels = torch.ones((len(additions), 1), dtype=torch.int, device=device)
            output = decode_prompts_from_features(net, encoded, prompts, labels, out_size=cfgs.out_size, device=device)
            extra_cost += len(additions)
            masks = mask_process_eval(
                np.ones(len(additions), dtype=np.int64), torch.arange(len(additions), device=device), tile_box,
                ori_shape, prompts, output.logits, output.predicted_iou,
            )
            records = {int(record["inds"]): record for record in masks}
            for index, action in enumerate(additions):
                record = records.get(index)
                if record is None:
                    decoded[action.action_id] = DecodedAction(action, decode_reason="mask_postprocess_rejection")
                    continue
                mask = component_containing_point(record["segmentation"][: shape[0], : shape[1]], action.positive_points[0].x, action.positive_points[0].y)
                feature_action = replace(
                    action,
                    support_box=_mask_box(mask),
                    decoded_features={
                        "decode_status": "decoded",
                        "decoded_predicted_iou": float(record["predicted_iou"]),
                        "decoded_stability_score": float(record["stability_score"]),
                        "decoded_mask_area": int(mask.sum()),
                    },
                )
                decoded[action.action_id] = DecodedAction(feature_action, add_mask=mask)
        splits = [action for action in tile_actions if action.action_type is ActionType.SPLIT]
        if splits:
            prompt_rows = []
            label_rows = []
            for action in splits:
                p1, p2 = (global_to_crop(point, tile_box) for point in action.positive_points)
                prompt_rows.extend([[[p1.x, p1.y], [p2.x, p2.y]], [[p2.x, p2.y], [p1.x, p1.y]]])
                label_rows.extend([[1, 0], [1, 0]])
            prompts = torch.tensor(prompt_rows, dtype=torch.float32, device=device)
            labels = torch.tensor(label_rows, dtype=torch.int, device=device)
            output = decode_prompts_from_features(net, encoded, prompts, labels, out_size=cfgs.out_size, device=device)
            extra_cost += 2 * len(splits)
            masks = mask_process_eval(
                np.ones(len(prompt_rows), dtype=np.int64), torch.arange(len(prompt_rows), device=device), tile_box,
                ori_shape, prompts, output.logits, output.predicted_iou,
            )
            records = {int(record["inds"]): record for record in masks}
            for index, action in enumerate(splits):
                first_index, second_index = 2 * index, 2 * index + 1
                first_record, second_record = records.get(first_index), records.get(second_index)
                if first_record is None or second_record is None:
                    decoded[action.action_id] = DecodedAction(action, decode_reason="mask_postprocess_rejection")
                    continue
                first_mask = component_containing_point(first_record["segmentation"][: shape[0], : shape[1]], action.positive_points[0].x, action.positive_points[0].y)
                second_mask = component_containing_point(second_record["segmentation"][: shape[0], : shape[1]], action.positive_points[1].x, action.positive_points[1].y)
                first_logits = _uncrop_logits(output.logits[first_index], tile_box, shape)
                second_logits = _uncrop_logits(output.logits[second_index], tile_box, shape)
                feature_action = replace(
                    action,
                    # A SPLIT affects the original parent support even when
                    # the decoded children cover only part of it.
                    support_box=action.support_box,
                    decoded_features={
                        "decode_status": "decoded",
                        "child_first_predicted_iou": float(first_record["predicted_iou"]),
                        "child_second_predicted_iou": float(second_record["predicted_iou"]),
                        "child_first_stability_score": float(first_record["stability_score"]),
                        "child_second_stability_score": float(second_record["stability_score"]),
                        "child_first_mask_area": int(first_mask.sum()),
                        "child_second_mask_area": int(second_mask.sum()),
                    },
                )
                decoded[action.action_id] = DecodedAction(
                    feature_action,
                    child_first=first_mask,
                    child_second=second_mask,
                    child_first_logits=first_logits,
                    child_second_logits=second_logits,
                )
    return decoded, extra_cost


def _candidate_error_diagnostics(gt: np.ndarray, base: np.ndarray, actions: Iterable[ActionCandidate]) -> dict[str, Any]:
    """GT-only error analysis, isolated from generation and action decoding."""

    evaluation = evaluate_pq(gt, base)
    paired_gt = {gt_id for gt_id, _, _ in evaluation.matched_pairs}
    paired_pred = {pred_id for _, pred_id, _ in evaluation.matched_pairs}
    missed_gt = {int(item) for item in np.unique(gt) if int(item) and int(item) not in paired_gt}
    false_positive_pred = {int(item) for item in np.unique(base) if int(item) and int(item) not in paired_pred}
    near_half = sum(0.5 <= iou < 0.6 for _, _, iou in evaluation.matched_pairs)
    stable = sum(iou >= 0.75 for _, _, iou in evaluation.matched_pairs)

    add_actions = [action for action in actions if action.action_type is ActionType.ADD]
    add_hits = {
        int(gt[action.positive_points[0].y, action.positive_points[0].x])
        for action in add_actions
        if int(gt[action.positive_points[0].y, action.positive_points[0].x]) in missed_gt
    }
    merge_parents: set[int] = set()
    for parent_id in (int(item) for item in np.unique(base) if int(item)):
        parent = base == parent_id
        overlapping = np.unique(gt[parent])
        count = 0
        for gt_id in overlapping:
            if gt_id == 0:
                continue
            fraction = float(np.count_nonzero(parent & (gt == gt_id)) / max(1, np.count_nonzero(parent)))
            count += int(fraction >= 0.1)
        if count >= 2:
            merge_parents.add(parent_id)
    split_like_gt: set[int] = set()
    for gt_id in (int(item) for item in np.unique(gt) if int(item)):
        gt_mask = gt == gt_id
        overlapping = [int(item) for item in np.unique(base[gt_mask]) if int(item)]
        substantial = sum(
            float(np.count_nonzero(gt_mask & (base == pred_id)) / max(1, np.count_nonzero(gt_mask))) >= 0.1
            for pred_id in overlapping
        )
        if substantial >= 2:
            split_like_gt.add(gt_id)
    split_parents = {action.affected_instance_ids[0] for action in actions if action.action_type is ActionType.SPLIT}
    return {
        "missed_gt_count": len(missed_gt),
        "false_positive_prediction_count": len(false_positive_pred),
        "merge_parent_count": len(merge_parents),
        "split_like_gt_count": len(split_like_gt),
        "matched_near_iou_0_5_to_0_6_count": int(near_half),
        "stable_matched_iou_ge_0_75_count": int(stable),
        "add_candidate_hit_missed_gt": len(add_hits),
        "add_missed_gt_recall": float(len(add_hits) / len(missed_gt)) if missed_gt else None,
        "split_candidate_hit_merge_parent": len(merge_parents & split_parents),
        "split_merge_parent_recall": float(len(merge_parents & split_parents) / len(merge_parents)) if merge_parents else None,
        # These IDs are kept only in this in-memory oracle-analysis object and
        # never enter candidates or action_features.csv.
        "_missed_gt_ids": missed_gt,
        "_merge_parent_ids": merge_parents,
    }


def _bootstrap(gains: list[float], samples: int, seed: int) -> dict[str, float | int | None]:
    if not gains:
        return {"n": 0, "mean": None, "median": None, "std": None, "ci95_low": None, "ci95_high": None}
    values = np.asarray(gains, dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = np.asarray([generator.choice(values, size=len(values), replace=True).mean() for _ in range(samples)])
    total = float(values.sum())
    largest_ratio = float(values.max() / total) if total > 0 else None
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=0)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "positive_image_fraction": float(np.mean(values > 0)),
        "negative_image_fraction": float(np.mean(values < 0)),
        "largest_single_image_contribution_ratio": largest_ratio,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _feature_row(action: ActionCandidate) -> dict[str, Any]:
    first = action.positive_points[0] if action.positive_points else None
    second = action.positive_points[1] if len(action.positive_points) > 1 else None
    return {
        "image": action.image_id,
        "action_id": action.action_id,
        "action_type": action.action_type.value,
        "action_cost": action.action_cost,
        "decoder_cost": action.action_cost,
        "parent_pred_id": action.affected_instance_ids[0] if action.affected_instance_ids else None,
        "affected_instance_ids": json.dumps(action.affected_instance_ids),
        "x1": first.x if first else None,
        "y1": first.y if first else None,
        "x2": second.x if second else None,
        "y2": second.y if second else None,
        "positive_points": json.dumps([point.as_dict() for point in action.positive_points]),
        "negative_points": json.dumps([point.as_dict() for point in action.negative_points]),
        "tile_box": json.dumps(action.tile_box),
        "support_box": json.dumps(action.support_box),
        "generation_features": json.dumps(action.generation_features, sort_keys=True),
        "decoded_features": json.dumps(action.decoded_features, sort_keys=True),
        "generator_version": action.generator_version,
        "config_hash": action.config_hash,
    }


def _gt_only_action_annotation(action: ActionCandidate, gt: np.ndarray, diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Oracle-only candidate-recall fields for action_labels.csv.

    This function is invoked after candidate generation and decoding.  Its
    return value must never be merged into generation_features or decoded
    features, which is why labels are persisted in a separate CSV.
    """

    if action.action_type is ActionType.ADD:
        point = action.positive_points[0]
        missed = set(diagnostics["_missed_gt_ids"])
        point_gt_id = int(gt[point.y, point.x])
        centroids = []
        for gt_id in missed:
            ys, xs = np.nonzero(gt == gt_id)
            if len(xs):
                centroids.append((float(xs.mean()), float(ys.mean())))
        nearest = (
            min(float(math.hypot(point.x - cx, point.y - cy)) for cx, cy in centroids)
            if centroids
            else None
        )
        return {
            "candidate_hits_missed_gt": bool(point_gt_id in missed),
            "nearest_missed_gt_centroid_distance": nearest,
        }
    parent_id = action.affected_instance_ids[0]
    return {"candidate_hits_merge_parent": bool(parent_id in set(diagnostics["_merge_parent_ids"]))}


def _oracle_rows_for_family(
    decoded: dict[str, DecodedAction],
    *,
    family: str,
    gt: np.ndarray,
    base: np.ndarray,
    split_config: SplitAssemblyConfig,
    min_added_area: int,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    if family == "ADD":
        selected = [item for item in decoded.values() if item.candidate.action_type is ActionType.ADD]
    elif family == "SPLIT":
        selected = [item for item in decoded.values() if item.candidate.action_type is ActionType.SPLIT]
    else:
        selected = list(decoded.values())
    actions = [item.candidate for item in selected]
    action_by_id = {item.candidate.action_id: item for item in selected}
    graph = build_conflict_graph(actions, support_iou_threshold=float(config["conflicts"]["support_iou_threshold"]))
    base_eval = evaluate_pq(gt, base)
    cache: dict[tuple[str, ...], PQEvaluation] = {}

    def evaluate_subset(ids: tuple[str, ...]) -> PQEvaluation:
        if ids in cache:
            return cache[ids]
        prediction = base.copy()
        for action_id in ids:
            prediction = action_by_id[action_id].apply(prediction, split_config, min_added_area).prediction
        cache[ids] = evaluate_pq(gt, prediction)
        return cache[ids]

    rows = []
    oracle_config = config["oracle"]
    for budget in oracle_config["budgets"]:
        print(f"[stage1-oracle] family={family} budget={budget} candidates={len(actions)}", flush=True)
        validation: dict[str, Any] = {
            "beam_validation_candidate_count": None,
            "beam_validation_pq_abs_error": None,
            "beam_validation_action_ids_match": None,
        }
        if len(actions) <= int(oracle_config["exact_max_candidates"]):
            result = exact_joint_oracle(actions, budget=int(budget), conflict_graph=graph, evaluate_subset=evaluate_subset)
            strategy = "exact"
        else:
            single_action_scores = {
                action.action_id: float(action.utility_fields.get("delta_matched_iou_sum", 0.0))
                for action in actions
            }
            result = utility_guided_beam_joint_oracle(
                actions,
                budget=int(budget),
                conflict_graph=graph,
                evaluate_subset=evaluate_subset,
                single_action_scores=single_action_scores,
                beam_width=int(oracle_config["beam_width"]),
                final_evaluation_limit=int(oracle_config["beam_final_evaluations"]),
            )
            strategy = "utility_guided_beam"
            validation_actions = sorted(actions, key=lambda action: action.action_id)[
                : int(oracle_config.get("beam_validation_max_candidates", 12))
            ]
            validation_graph = build_conflict_graph(
                validation_actions,
                support_iou_threshold=float(config["conflicts"]["support_iou_threshold"]),
            )
            exact_validation = exact_joint_oracle(
                validation_actions,
                budget=int(budget),
                conflict_graph=validation_graph,
                evaluate_subset=evaluate_subset,
            )
            beam_validation = utility_guided_beam_joint_oracle(
                validation_actions,
                budget=int(budget),
                conflict_graph=validation_graph,
                evaluate_subset=evaluate_subset,
                single_action_scores={
                    action.action_id: float(action.utility_fields.get("delta_matched_iou_sum", 0.0))
                    for action in validation_actions
                },
                beam_width=int(oracle_config["beam_width"]),
                final_evaluation_limit=int(oracle_config["beam_final_evaluations"]),
            )
            validation = {
                "beam_validation_candidate_count": len(validation_actions),
                "beam_validation_pq_abs_error": abs(
                    beam_validation.evaluation.pq - exact_validation.evaluation.pq
                ),
                "beam_validation_action_ids_match": beam_validation.action_ids == exact_validation.action_ids,
            }
        rows.append(
            {
                "family": family,
                "budget": int(budget),
                "search_strategy": strategy,
                "beam_width": int(oracle_config["beam_width"]) if strategy == "utility_guided_beam" else None,
                "beam_final_evaluations": result.full_evaluation_count if strategy == "utility_guided_beam" else None,
                "candidate_count": len(actions),
                "selected_action_ids": ";".join(result.action_ids),
                "selected_cost": result.cost,
                "base_pq": base_eval.pq,
                "oracle_pq": result.evaluation.pq,
                "delta_pq": result.evaluation.pq - base_eval.pq,
                "delta_dq": result.evaluation.dq - base_eval.dq,
                "delta_sq": result.evaluation.sq - base_eval.sq,
                "delta_matched_iou_sum": result.evaluation.matched_iou_sum - base_eval.matched_iou_sum,
                "recovery_ratio_perfect_pq_1": normalized_oracle_recovery(base_eval.pq, result.evaluation.pq),
                **validation,
            }
        )
    return rows


def _residual_evidence_score(action: ActionCandidate) -> float:
    """Fixed GT-free heuristic used only for the Stage 1 control baseline."""

    features = action.generation_features
    if action.action_type is ActionType.ADD:
        return float(features.get("h_evidence", 0.0)) / action.action_cost
    # SPLIT evidence combines its fixed, pre-decode topology diagnostics.  No
    # decoder output or oracle label is read here.
    topology = (
        float(features.get("peak_height_ratio", 0.0))
        * float(features.get("normalized_peak_distance", 0.0))
        * float(features.get("peak_valley_depth", 0.0))
        * float(features.get("distance_basin_area_ratio", 0.0))
    )
    return topology / action.action_cost


def _greedy_feasible_ids(
    actions: list[ActionCandidate],
    graph: dict[str, set[str]],
    budget: int,
    ordered_ids: Iterable[str],
) -> tuple[str, ...]:
    action_by_id = {action.action_id: action for action in actions}
    selected: list[str] = []
    cost = 0
    for action_id in ordered_ids:
        action = action_by_id[action_id]
        if cost + action.action_cost > budget:
            continue
        if any(action_id in graph.get(other, set()) for other in selected):
            continue
        selected.append(action_id)
        cost += action.action_cost
    return tuple(sorted(selected))


def _control_rows_for_family(
    decoded: dict[str, DecodedAction],
    *,
    family: str,
    gt: np.ndarray,
    base: np.ndarray,
    split_config: SplitAssemblyConfig,
    min_added_area: int,
    config: dict[str, Any],
    image_id: str,
) -> list[dict[str, Any]]:
    """Real-assembly execute-all, random@B, and residual-evidence@B controls."""

    if family == "ADD":
        selected = [item for item in decoded.values() if item.candidate.action_type is ActionType.ADD]
    elif family == "SPLIT":
        selected = [item for item in decoded.values() if item.candidate.action_type is ActionType.SPLIT]
    else:
        selected = list(decoded.values())
    action_by_id = {item.candidate.action_id: item for item in selected}
    actions = [item.candidate for item in selected]
    graph = build_conflict_graph(actions, support_iou_threshold=float(config["conflicts"]["support_iou_threshold"]))
    base_eval = evaluate_pq(gt, base)
    cache: dict[tuple[str, ...], PQEvaluation] = {}

    def evaluate_ids(ids: tuple[str, ...]) -> PQEvaluation:
        if ids not in cache:
            prediction = base.copy()
            for action_id in ids:
                prediction = action_by_id[action_id].apply(prediction, split_config, min_added_area).prediction
            cache[ids] = evaluate_pq(gt, prediction)
        return cache[ids]

    def row(method: str, budget: int | None, ids: tuple[str, ...], evaluation: PQEvaluation, **extra: Any) -> dict[str, Any]:
        return {
            "image": image_id,
            "family": family,
            "method": method,
            "budget": budget,
            "candidate_count": len(actions),
            "selected_action_ids": ";".join(ids),
            "selected_cost": sum(action_by_id[item].candidate.action_cost for item in ids),
            "base_pq": base_eval.pq,
            "result_pq": evaluation.pq,
            "delta_pq": evaluation.pq - base_eval.pq,
            "delta_dq": evaluation.dq - base_eval.dq,
            "delta_sq": evaluation.sq - base_eval.sq,
            "delta_matched_iou_sum": evaluation.matched_iou_sum - base_eval.matched_iou_sum,
            **extra,
        }

    rows: list[dict[str, Any]] = []
    execute_ids = tuple(sorted(action_by_id))
    rows.append(row("execute_all", None, execute_ids, evaluate_ids(execute_ids)))
    oracle_cfg = config["oracle"]
    trials = int(oracle_cfg.get("random_trials", 100))
    seed = int(oracle_cfg["bootstrap_seed"]) + sum(ord(char) for char in f"{image_id}:{family}")
    generator = np.random.default_rng(seed)
    for budget in oracle_cfg["budgets"]:
        ranked = sorted(
            actions,
            key=lambda action: (-_residual_evidence_score(action), action.action_id),
        )
        residual_ids = _greedy_feasible_ids(actions, graph, int(budget), (action.action_id for action in ranked))
        rows.append(row("residual_evidence", int(budget), residual_ids, evaluate_ids(residual_ids)))
        random_evaluations: list[PQEvaluation] = []
        random_costs: list[int] = []
        for _ in range(trials):
            random_ids = _greedy_feasible_ids(
                actions,
                graph,
                int(budget),
                (actions[index].action_id for index in generator.permutation(len(actions))),
            )
            random_evaluations.append(evaluate_ids(random_ids))
            random_costs.append(sum(action_by_id[item].candidate.action_cost for item in random_ids))
        if random_evaluations:
            rows.append(
                row(
                    "random_at_budget",
                    int(budget),
                    (),
                    PQEvaluation(
                        matched_iou_sum=float(np.mean([item.matched_iou_sum for item in random_evaluations])),
                        tp=0,
                        fp=0,
                        fn=0,
                        dq=float(np.mean([item.dq for item in random_evaluations])),
                        sq=float(np.mean([item.sq for item in random_evaluations])),
                        pq=float(np.mean([item.pq for item in random_evaluations])),
                        matched_pairs=(),
                    ),
                    random_trials=trials,
                    random_mean_selected_cost=float(np.mean(random_costs)),
                )
            )
    return rows


def _save_decoded_action_artifact(directory: Path, decoded_action: DecodedAction) -> None:
    """Persist replay masks/logits outside git for joint-oracle auditability."""

    candidate = decoded_action.candidate
    name = candidate.action_id.replace(":", "_").replace("/", "_").replace("\\", "_")
    destination = directory / f"{name}.npz"
    payload: dict[str, Any] = {
        "action_json": np.asarray(candidate.to_json()),
        "decode_reason": np.asarray(decoded_action.decode_reason),
    }
    if decoded_action.add_mask is not None:
        payload["add_mask"] = decoded_action.add_mask.astype(np.uint8)
    if decoded_action.child_first is not None:
        payload["child_first"] = decoded_action.child_first.astype(np.uint8)
    if decoded_action.child_second is not None:
        payload["child_second"] = decoded_action.child_second.astype(np.uint8)
    if decoded_action.child_first_logits is not None:
        payload["child_first_logits"] = decoded_action.child_first_logits.astype(np.float16)
    if decoded_action.child_second_logits is not None:
        payload["child_second_logits"] = decoded_action.child_second_logits.astype(np.float16)
    np.savez_compressed(destination, **payload)


def _attach_prediction_conflicts(
    decoded: dict[str, DecodedAction], support_iou_threshold: float
) -> dict[str, DecodedAction]:
    """Record GT-free parent/support conflicts on finalized decoded actions."""

    graph = build_conflict_graph(
        [item.candidate for item in decoded.values()],
        support_iou_threshold=support_iou_threshold,
    )
    return {
        action_id: replace(
            item,
            candidate=replace(item.candidate, conflict_ids=tuple(sorted(graph.get(action_id, set())))),
        )
        for action_id, item in decoded.items()
    }


def run_stage1_oracle(
    *,
    cfgs: Any,
    args: Any,
    test_dataset: Any,
    net: Any,
    point_net: Any,
    point_encoder: Any,
    texture_memory_bank_list: list,
    device: torch.device,
) -> None:
    """Execute the Stage 1 oracle on exactly one non-test split manifest."""

    if not cfgs.stainroute_split_manifest:
        raise ValueError("--stainroute_split_manifest is required for Stage 1")
    split_manifest_path = Path(cfgs.stainroute_split_manifest)
    config_path = Path(cfgs.stainroute_action_config)
    config = _json_config(config_path)
    max_images = int(cfgs.stainroute_max_images or 0)
    smoke_mode = max_images > 0
    baseline = _require_frozen_baseline(cfgs, split_manifest_path)
    for option, config_key in (
        ("stainroute_exact_max_candidates", "exact_max_candidates"),
        ("stainroute_beam_width", "beam_width"),
        ("stainroute_bootstrap_samples", "bootstrap_samples"),
    ):
        if int(getattr(cfgs, option)) != int(config["oracle"][config_key]):
            raise ValueError(
                f"--{option.replace('_', '-')} must equal the frozen Stage 1 config "
                f"({config['oracle'][config_key]}), not {getattr(cfgs, option)}"
            )
    loader, image_root, split_manifest = _subset_loader(cfgs, test_dataset, split_manifest_path, cfgs.stainroute_split)
    if smoke_mode:
        # Keep the configured formal action space intact on disk.  These
        # execution-only limits make a one-image plumbing smoke practical on
        # a high-resolution MoNuSeg slide.
        config = copy.deepcopy(config)
        config["oracle"].update(
            {
                "budgets": [1, 2],
                "exact_max_candidates": 4,
                "beam_width": 4,
                "bootstrap_samples": 20,
                "random_trials": 4,
            }
        )
    add_config = AddCandidateConfig(**config["add"])
    split_candidate_config = SplitCandidateConfig(**config["split"])
    split_assembly_config = SplitAssemblyConfig(
        **{
            key: value
            for key, value in config["assembly"].items()
            if key in {"min_child_area", "min_parent_coverage", "max_raw_child_iou"}
        }
    )
    out_dir = Path(cfgs.stainroute_out_dir or f"logs/stainroute/stage1/{Path(cfgs.data_path).name}_{cfgs.stainroute_split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decoded_actions").mkdir(exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    (out_dir / "resolved_config.yaml").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    environment = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "nvidia_smi": _command_output(
            "nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"
        ),
        "command": sys.argv,
    }
    (out_dir / "environment.txt").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    (out_dir / "data_manifest.json").write_text(
        json.dumps({"data": baseline.get("data", {}), "split": baseline.get("splits", {})}, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "checkpoint_manifest.json").write_text(
        json.dumps(baseline.get("checkpoints", {}), indent=2) + "\n", encoding="utf-8"
    )

    net.eval()
    point_net.eval()
    point_encoder.eval()
    feature_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    per_image_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    equivalence_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []

    for image_index, batch in enumerate(loader):
        if max_images > 0 and image_index >= max_images:
            break
        started = time.perf_counter()
        image_tensor, inst_maps, _, _, _, _, ori_shape, _, name = batch
        image_id = _safe_name(name)
        print(
            f"[stage1] image={image_index + 1}/{min(len(loader), max_images) if smoke_mode else len(loader)} "
            f"id={image_id} smoke={smoke_mode}",
            flush=True,
        )
        base, caches, selected_records, equivalence, counters = _base_prediction_with_cache(
            image_tensor=image_tensor,
            ori_shape=ori_shape,
            cfgs=cfgs,
            args=args,
            net=net,
            point_net=point_net,
            point_encoder=point_encoder,
            texture_memory_bank_list=texture_memory_bank_list,
            device=device,
        )
        raw_image = imread(_find_image(image_root, image_id))[..., :3]
        generated = generate_add_candidates(raw_image, base, image_id=image_id, config=add_config)
        generated += generate_split_candidates(raw_image, base, image_id=image_id, config=split_candidate_config)
        generated_add = sum(action.action_type is ActionType.ADD for action in generated)
        generated_split = sum(action.action_type is ActionType.SPLIT for action in generated)
        if smoke_mode:
            generated = _smoke_candidate_subset(generated)
        print(
            f"[stage1-actions] generated_add={generated_add} generated_split={generated_split} "
            f"decoded_candidates={len(generated)}",
            flush=True,
        )
        tiled = _assign_tiles(generated, caches)
        decoded, decoder_cost = _decode_grouped_actions(
            tiled, caches, ori_shape=ori_shape, cfgs=cfgs, net=net, device=device
        )
        print(f"[stage1-actions] decoded={len(decoded)} decoder_cost={decoder_cost}", flush=True)
        decoded = _attach_prediction_conflicts(
            decoded, float(config["conflicts"]["support_iou_threshold"])
        )
        # This is the first GT access in the image loop. Candidate generation,
        # cached action decoding, conflict construction, and assembly policy
        # above receive no GT-derived object or field.
        gt = np.asarray(inst_maps.numpy()[0]).astype(np.int32)
        base_eval = evaluate_pq(gt, base)
        diagnostics = _candidate_error_diagnostics(gt, base, [item.candidate for item in decoded.values()])
        finalized: dict[str, DecodedAction] = {}
        for action_id, decoded_action in decoded.items():
            assembly = decoded_action.apply(base, split_assembly_config, int(config["assembly"]["min_added_area"]))
            utility = compute_action_utility(gt, base, assembly.prediction)
            candidate = replace(decoded_action.candidate, utility_fields=utility.as_dict())
            finalized[action_id] = replace(decoded_action, candidate=candidate)
            feature_rows.append(_feature_row(candidate))
            label_rows.append(
                {
                    "image": image_id,
                    "action_id": action_id,
                    "action_type": candidate.action_type.value,
                    "action_cost": candidate.action_cost,
                    "decode_status": candidate.decoded_features.get("decode_status", decoded_action.decode_reason),
                    "assembly_applied": assembly.applied,
                    "assembly_reason": assembly.reason,
                    **utility.as_dict(),
                    **_gt_only_action_annotation(candidate, gt, diagnostics),
                }
            )
            _save_decoded_action_artifact(out_dir / "decoded_actions", finalized[action_id])

        public_diagnostics = {key: value for key, value in diagnostics.items() if not key.startswith("_")}
        error_rows.append({"image": image_id, **public_diagnostics})
        image_budget_rows = []
        for family in ("ADD", "SPLIT", "ADD+SPLIT"):
            rows = _oracle_rows_for_family(
                finalized,
                family=family,
                gt=gt,
                base=base,
                split_config=split_assembly_config,
                min_added_area=int(config["assembly"]["min_added_area"]),
                config=config,
            )
            for row in rows:
                row["image"] = image_id
                image_budget_rows.append(row)
        budget_rows.extend(image_budget_rows)
        for family in ("ADD", "SPLIT", "ADD+SPLIT"):
            control_rows.extend(
                _control_rows_for_family(
                    finalized,
                    family=family,
                    gt=gt,
                    base=base,
                    split_config=split_assembly_config,
                    min_added_area=int(config["assembly"]["min_added_area"]),
                    config=config,
                    image_id=image_id,
                )
            )
        per_image_rows.append(
            {
                "image": image_id,
                "base_pq": base_eval.pq,
                "base_dq": base_eval.dq,
                "base_sq": base_eval.sq,
                "base_matched_iou_sum": base_eval.matched_iou_sum,
                "generated_add_candidates": generated_add,
                "generated_split_candidates": generated_split,
                "add_candidates": sum(item.candidate.action_type is ActionType.ADD for item in finalized.values()),
                "split_candidates": sum(item.candidate.action_type is ActionType.SPLIT for item in finalized.values()),
                **public_diagnostics,
            }
        )
        runtime_rows.append(
            {
                "image": image_id,
                "encoder_calls": counters["encoder_calls"],
                "base_decoder_actions": counters["base_decoder_actions"],
                "extra_decoder_action_cost": decoder_cost,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        equivalence_rows.append({"image": image_id, **equivalence})

    summaries = []
    for family in ("ADD", "SPLIT", "ADD+SPLIT"):
        for budget in config["oracle"]["budgets"]:
            rows = [row for row in budget_rows if row["family"] == family and row["budget"] == budget]
            gains = [float(row["delta_pq"]) for row in rows]
            summaries.append(
                {
                    "split": cfgs.stainroute_split,
                    "family": family,
                    "budget": budget,
                    **_bootstrap(gains, int(config["oracle"]["bootstrap_samples"]), int(config["oracle"]["bootstrap_seed"])),
                }
            )
    control_summaries = []
    for family in ("ADD", "SPLIT", "ADD+SPLIT"):
        for method in ("execute_all", "random_at_budget", "residual_evidence"):
            method_rows = [row for row in control_rows if row["family"] == family and row["method"] == method]
            budgets = [None] if method == "execute_all" else list(config["oracle"]["budgets"])
            for budget in budgets:
                rows = [row for row in method_rows if row["budget"] == budget]
                control_summaries.append(
                    {
                        "split": cfgs.stainroute_split,
                        "family": family,
                        "method": method,
                        "budget": budget,
                        **_bootstrap(
                            [float(row["delta_pq"]) for row in rows],
                            int(config["oracle"]["bootstrap_samples"]),
                            int(config["oracle"]["bootstrap_seed"]),
                        ),
                    }
                )
    error_summary = {}
    if error_rows:
        for key in error_rows[0]:
            if key == "image":
                continue
            values = [row[key] for row in error_rows if row.get(key) is not None]
            error_summary[key] = {
                "sum": float(np.sum(values)) if values else None,
                "mean_per_image": float(np.mean(values)) if values else None,
            }
    baseline_summary: dict[str, Any] = {"split": cfgs.stainroute_split, "images": len(per_image_rows)}
    for source_key, summary_key in (
        ("base_pq", "pq"),
        ("base_dq", "dq"),
        ("base_sq", "sq"),
        ("base_matched_iou_sum", "matched_iou_sum"),
    ):
        values = np.asarray([float(row[source_key]) for row in per_image_rows], dtype=np.float64)
        baseline_summary[summary_key] = {
            "mean": float(values.mean()) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "std": float(values.std(ddof=0)) if len(values) else None,
        }
    action_summaries = []
    for action_type in ("ADD", "SPLIT"):
        rows = [row for row in label_rows if row["action_type"] == action_type]
        utilities = np.asarray([float(row["delta_pq"]) for row in rows], dtype=np.float64)
        decoded_ok = [row["decode_status"] == "decoded" for row in rows]
        applied = [bool(row["assembly_applied"]) for row in rows]
        action_summaries.append(
            {
                "split": cfgs.stainroute_split,
                "action_type": action_type,
                "candidates": len(rows),
                "positive": int(np.sum(utilities > 0)),
                "harmful": int(np.sum(utilities < 0)),
                "positive_rate": float(np.mean(utilities > 0)) if len(utilities) else None,
                "harm_rate": float(np.mean(utilities < 0)) if len(utilities) else None,
                "decoded_valid_rate": float(np.mean(decoded_ok)) if decoded_ok else None,
                "assembly_applied_rate": float(np.mean(applied)) if applied else None,
                "utility_mean": float(utilities.mean()) if len(utilities) else None,
                "utility_median": float(np.median(utilities)) if len(utilities) else None,
                "utility_std": float(utilities.std(ddof=0)) if len(utilities) else None,
            }
        )
    _write_csv(out_dir / "action_features.csv", feature_rows)
    _write_csv(out_dir / "action_labels.csv", label_rows)
    _write_csv(out_dir / "per_image_oracle.csv", per_image_rows)
    _write_csv(out_dir / "oracle_budget_per_image.csv", budget_rows)
    _write_csv(out_dir / "oracle_budget_summary.csv", summaries)
    _write_csv(out_dir / "control_per_image.csv", control_rows)
    _write_csv(out_dir / "control_summary.csv", control_summaries)
    _write_csv(out_dir / "action_summary.csv", action_summaries)
    _write_csv(out_dir / "error_type_summary.csv", error_rows)
    _write_csv(out_dir / "runtime_summary.csv", runtime_rows)
    (out_dir / "cached_decode_equivalence.json").write_text(json.dumps(equivalence_rows, indent=2) + "\n", encoding="utf-8")
    (out_dir / "error_type_aggregate.json").write_text(json.dumps(error_summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "baseline_summary.json").write_text(json.dumps(baseline_summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "oracle_summary.json").write_text(
        json.dumps(
            {
                "baseline": baseline_summary,
                "actions": action_summaries,
                "joint_oracle": summaries,
                "controls": control_summaries,
                "error_types": error_summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    run_manifest = {
        "schema_version": 1,
        "stage": "StainRoute Stage 1 oracle feasibility",
        "git_sha": os.popen("git rev-parse HEAD").read().strip(),
        "git_dirty_status": _command_output("git", "status", "--short"),
        "split": cfgs.stainroute_split,
        "split_manifest": str(split_manifest_path),
        "split_content_sha256": split_manifest.get("content_sha256"),
        "baseline_manifest": str(cfgs.stainroute_baseline_manifest),
        "baseline_manifest_sha256": sha256_file(cfgs.stainroute_baseline_manifest),
        "action_config": str(config_path),
        "action_config_sha256": sha256_file(config_path),
        "command": sys.argv,
        "dataset_root": cfgs.data_path,
        "image_count": len(per_image_rows),
        "is_smoke_run": bool(max_images > 0),
        "max_images": max_images,
        "smoke_profile": (
            {
                "candidate_limit_per_family": 2,
                "budgets": [1, 2],
                "bootstrap_samples": 20,
                "random_trials": 4,
                "purpose": "cache/assembly plumbing only; not a formal oracle measurement",
            }
            if smoke_mode
            else None
        ),
        "gt_policy": "GT read only after candidates are generated and decoded; utility/diagnostics only",
        "normalized_recovery_definition": "(PQ_oracle - PQ_StainPMS) / (1 - PQ_StainPMS)",
    }
    (out_dir / "manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "images": len(per_image_rows), "actions": len(feature_rows)}, indent=2))
