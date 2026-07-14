"""One-pass StainPMS candidate/mask cache and separate GT target store."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from skimage import io

from .data import authorized_images, crop_boxes, load_label, normalize_image
from .model import ModelBundle, decode_points, encode_crop, update_texture_memory
from .protocol import NMS_RADIUS, finite_arrays, inclusive_iou, json_dump, point_nms_indices, sha256_file, utility_target


def _inside(points: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return (points[:, 0] >= x1) & (points[:, 0] < x2) & (points[:, 1] >= y1) & (points[:, 1] < y2)


def _candidate_arrays(point_output: dict[str, torch.Tensor], crop_box: tuple[int, int, int, int], processed: list[tuple[int, int, int, int]], *, filtering: bool) -> dict[str, np.ndarray]:
    """Canonical predict() filtering plus validation overlap ownership, with source indices retained."""
    coords = point_output["pred_coords"][0].detach().cpu().numpy().astype(np.float32)
    logits = point_output["pred_logits"][0].detach().cpu().numpy().astype(np.float32)
    features = point_output["quality_roi_features"][0].detach().cpu().numpy().astype(np.float16)
    classes = logits.argmax(axis=-1).astype(np.int64)
    probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    scores = probabilities.max(axis=-1).astype(np.float32)
    semantic = point_output["pred_masks"][0, 0].detach().cpu().numpy() > 0
    height, width = semantic.shape
    coords[:, 0] = np.clip(coords[:, 0], 0, width - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, height - 1)
    keep = classes < (logits.shape[-1] - 1)
    if filtering:
        local = coords.astype(np.int64)
        keep &= semantic[local[:, 1], local[:, 0]]
    source = np.flatnonzero(keep)
    local_points = coords[source]
    global_points = local_points + np.asarray(crop_box[:2], dtype=np.float32)
    if processed and len(global_points):
        keep_new = np.ones(len(global_points), dtype=bool)
        for x1, y1, x2, y2 in processed:
            keep_new &= ~(
                (global_points[:, 0] >= x1 + 1) & (global_points[:, 0] <= x2 - 1)
                & (global_points[:, 1] >= y1 + 1) & (global_points[:, 1] <= y2 - 1)
            )
        source = source[keep_new]
        local_points = local_points[keep_new]
        global_points = global_points[keep_new]
    return {
        "raw_source_index": source.astype(np.int64),
        "local_point": local_points.astype(np.float32),
        "global_point": global_points.astype(np.float32),
        "objectness": scores[source].astype(np.float32),
        "class_id": classes[source].astype(np.int64),
        "feature": features[source],
    }


def _decode_partitioned(bundle: ModelBundle, image_embed: torch.Tensor, high_res: list[torch.Tensor], all_ids: np.ndarray, baseline_ids: np.ndarray, candidates: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Decode baseline prompts exactly as canonical, then cache all alternatives.

    The image encoding is deliberately shared.  A second prompt/decoder call is
    used only for candidates not selected by baseline NMS, so baseline logits
    have the canonical prompt-batch shape.
    """
    ordered = np.asarray(all_ids, dtype=np.int64)
    if not len(ordered):
        return ordered, np.empty((0, 256, 256), dtype=np.float32), np.empty(0, dtype=np.float32), 0
    base = np.asarray(baseline_ids, dtype=np.int64)
    extras = np.asarray([item for item in ordered.tolist() if item not in set(base.tolist())], dtype=np.int64)
    logits_by_id: dict[int, np.ndarray] = {}
    iou_by_id: dict[int, float] = {}
    calls = 0
    for ids in (base, extras):
        if not len(ids):
            continue
        coordinates = torch.as_tensor(np.stack([candidates[int(item)]["local_point_for_decode"] for item in ids]), dtype=torch.float32, device=bundle.device).unsqueeze(1)
        logits, predicted_iou = decode_points(bundle, image_embed, high_res, coordinates)
        calls += 1
        for index, source_id in enumerate(ids.tolist()):
            logits_by_id[int(source_id)] = logits[index].detach().cpu().numpy().astype(np.float32)
            iou_by_id[int(source_id)] = float(predicted_iou[index].detach().cpu())
    return ordered, np.stack([logits_by_id[int(item)] for item in ordered]), np.asarray([iou_by_id[int(item)] for item in ordered], dtype=np.float32), calls


def _cache_image(bundle: ModelBundle, image_id: str, patient: int, image_tensor: torch.Tensor, out_path: Path, *, texture: bool, context: bool, filtering: bool) -> dict:
    from .assembly import combine_mask

    candidates: list[dict[str, Any]] = []
    crop_records: list[dict[str, Any]] = []
    processed: list[tuple[int, int, int, int]] = []
    point_id: dict[tuple[float, float], int] = {}
    context_bank: list[Any] = []
    counts = {"point_model_forwards": 0, "sam2_image_encodes": 0, "prompt_encoder_calls": 0, "mask_decoder_calls": 0}
    original_shape = tuple(int(value) for value in image_tensor.shape[-2:])
    for crop_id, box in enumerate(crop_boxes(image_tensor)):
        x1, y1, x2, y2 = box
        crop = image_tensor[:, y1:y2, x1:x2].unsqueeze(0).to(bundle.device)
        with torch.no_grad():
            point_output, _, _, _ = bundle.point_net(crop)
        counts["point_model_forwards"] += 1
        new = _candidate_arrays(point_output, box, processed, filtering=filtering)
        for local_index in range(len(new["global_point"])):
            global_point = new["global_point"][local_index]
            key = tuple(float(value) for value in global_point.tolist())
            if key not in point_id:
                point_id[key] = len(point_id)
            candidates.append({
                "source_crop_id": crop_id,
                "raw_source_index": int(new["raw_source_index"][local_index]),
                "local_point_source": new["local_point"][local_index],
                "local_point_for_decode": new["local_point"][local_index],
                "global_point": global_point,
                "objectness": float(new["objectness"][local_index]),
                "class_id": int(new["class_id"][local_index]),
                "feature": new["feature"][local_index],
                "point_group": point_id[key],
            })
        processed.append(box)
        if not candidates:
            crop_records.append({"crop_id": crop_id, "box": box, "decode_source_ids": np.empty(0, dtype=np.int64), "logits": np.empty((0, 256, 256), dtype=np.float32), "decoded_iou": np.empty(0, dtype=np.float32)})
            continue
        points = np.stack([row["global_point"] for row in candidates]).astype(np.float32)
        base_indices = point_nms_indices(points, np.asarray([row["objectness"] for row in candidates]), NMS_RADIUS)
        visible = np.flatnonzero(_inside(points, box)).astype(np.int64)
        baseline_visible = base_indices[_inside(points[base_indices], box)]
        # Exact canonical validation behavior: no prompt in this crop means
        # no SAM2 image/prompt/decoder call and no context/texture update.
        if not len(visible):
            crop_records.append({"crop_id": crop_id, "box": box, "decode_source_ids": np.empty(0, dtype=np.int64), "logits": np.empty((0, 256, 256), dtype=np.float32), "decoded_iou": np.empty(0, dtype=np.float32)})
            continue
        # Point coordinates are local to this decoder crop even when they were
        # proposed by an earlier overlapping crop.
        for source_id in visible.tolist():
            candidates[source_id]["local_point_for_decode"] = candidates[source_id]["global_point"] - np.asarray([x1, y1], dtype=np.float32)
        image_embed, high_res, vision_feats, image_embed_for_memory = encode_crop(bundle, crop, context_bank, box, texture=texture, context=context)
        counts["sam2_image_encodes"] += 1
        decoded_ids, decoded_logits, decoded_iou, decoder_calls = _decode_partitioned(bundle, image_embed, high_res, visible, baseline_visible, candidates)
        counts["prompt_encoder_calls"] += decoder_calls
        counts["mask_decoder_calls"] += decoder_calls
        # Canonical texture memory update is driven by the baseline selected
        # prompts only.  Extra cached alternatives cannot change later crops.
        if len(baseline_visible):
            lookup = {int(source_id): index for index, source_id in enumerate(decoded_ids.tolist())}
            baseline_logits = torch.as_tensor(np.stack([decoded_logits[lookup[int(item)]] for item in baseline_visible]), device=bundle.device)
            baseline_iou = torch.as_tensor([decoded_iou[lookup[int(item)]] for item in baseline_visible], device=bundle.device)
            local_points = torch.as_tensor(np.stack([candidates[int(item)]["local_point_for_decode"] for item in baseline_visible]), device=bundle.device)
            baseline_mask = combine_mask(original_shape, local_points, baseline_logits, baseline_iou)
            update_texture_memory(bundle, vision_feats, baseline_mask, float(baseline_iou.mean().cpu()), image_embed_for_memory, texture=texture)
        crop_records.append({"crop_id": crop_id, "box": box, "decode_source_ids": decoded_ids, "logits": decoded_logits, "decoded_iou": decoded_iou})
    if not candidates:
        source_crop = np.empty(0, dtype=np.int64)
        raw_source = np.empty(0, dtype=np.int64)
        global_points = np.empty((0, 2), dtype=np.float32)
        objectness = np.empty(0, dtype=np.float32)
        classes = np.empty(0, dtype=np.int64)
        features = np.empty((0, 256), dtype=np.float16)
        groups = np.empty(0, dtype=np.int64)
    else:
        source_crop = np.asarray([row["source_crop_id"] for row in candidates], dtype=np.int64)
        raw_source = np.asarray([row["raw_source_index"] for row in candidates], dtype=np.int64)
        global_points = np.stack([row["global_point"] for row in candidates]).astype(np.float32)
        objectness = np.asarray([row["objectness"] for row in candidates], dtype=np.float32)
        classes = np.asarray([row["class_id"] for row in candidates], dtype=np.int64)
        features = np.stack([row["feature"] for row in candidates]).astype(np.float16)
        groups = np.asarray([row["point_group"] for row in candidates], dtype=np.int64)
    payload: dict[str, Any] = {
        "source_crop_id": source_crop, "raw_source_index": raw_source, "global_point": global_points,
        "objectness": objectness, "class_id": classes, "quality_feature": features, "point_group": groups,
        "image_shape": np.asarray(original_shape, dtype=np.int64), "crop_count": np.asarray(len(crop_records), dtype=np.int64),
    }
    for record in crop_records:
        prefix = f"crop_{record['crop_id']:03d}_"
        payload[prefix + "box"] = np.asarray(record["box"], dtype=np.int64)
        payload[prefix + "decode_source_ids"] = record["decode_source_ids"]
        payload[prefix + "decoded_logits"] = record["logits"]
        payload[prefix + "decoded_hard_masks"] = (record["logits"] > 0).astype(np.uint8)
        payload[prefix + "decoded_iou"] = record["decoded_iou"]
    if not finite_arrays(payload.values()):
        raise FloatingPointError(f"non-finite cache value for {image_id}")
    np.savez_compressed(out_path, **payload)
    return {"image_id": image_id, "patient": patient, "file": out_path.name, "sha256": sha256_file(out_path), "candidate_count": int(len(candidates)), "counts": counts, "image_shape": list(original_shape)}


def extract_role_cache(bundle: ModelBundle, data_root: Path, manifest_path: Path, role: str, out_dir: Path, *, texture: bool = True, context: bool = True, filtering: bool = True) -> dict:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite deployment cache: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    records = []
    started = time.perf_counter()
    for item in authorized_images(data_root, manifest_path, role):
        # Cache extraction opens image pixels only.  GT is materialized into a
        # separate label store by the caller and never appears in this NPZ.
        image = normalize_image(io.imread(item.image_path)[..., :3])
        records.append(_cache_image(bundle, item.image_id, item.patient, image, out_dir / f"{item.image_id}.npz", texture=texture, context=context, filtering=filtering))
    payload = {
        "schema_version": 2, "role": role, "contains_gt": False,
        "canonical_nms_radius": NMS_RADIUS, "tta": False, "batch_size": 1,
        "records": records, "elapsed_seconds": time.perf_counter() - started,
        "call_counts": {key: int(sum(record["counts"][key] for record in records)) for key in ("point_model_forwards", "sam2_image_encodes", "prompt_encoder_calls", "mask_decoder_calls")},
    }
    json_dump(out_dir / "manifest.json", payload)
    return payload


def _centroid(mask: np.ndarray) -> np.ndarray:
    coordinates = np.argwhere(mask)
    centre = np.rint(coordinates.mean(axis=0)).astype(np.int64)
    if not mask[tuple(centre)]:
        centre = coordinates[np.sum((coordinates - centre[None, :]) ** 2, axis=1).argmin()]
    return centre[[1, 0]].astype(np.float32)


def create_quality_targets(cache_dir: Path, label_dir: Path, out_dir: Path, *, role: str) -> dict:
    """Create the separate GT-only quality label store from cached masks."""
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite quality target store: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    cache_manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    records = []
    for item in cache_manifest["records"]:
        image_id = item["image_id"]
        with np.load(cache_dir / item["file"], allow_pickle=False) as cache:
            source_crop = np.asarray(cache["source_crop_id"], dtype=np.int64)
            points = np.asarray(cache["global_point"], dtype=np.float32)
            target = np.zeros(len(points), dtype=np.float32)
            matched = np.zeros(len(points), dtype=np.bool_)
            oracle_iou = np.zeros(len(points), dtype=np.float32)
            instance_map = load_label(label_dir, image_id)
            for crop_id in range(int(cache["crop_count"])):
                ids = np.flatnonzero(source_crop == crop_id)
                if not len(ids):
                    continue
                prefix = f"crop_{crop_id:03d}_"
                box = np.asarray(cache[prefix + "box"], dtype=np.int64)
                x1, y1, x2, y2 = box.tolist()
                crop_gt = instance_map[y1:y2, x1:x2]
                decode_ids = np.asarray(cache[prefix + "decode_source_ids"], dtype=np.int64)
                logits = np.asarray(cache[prefix + "decoded_logits"], dtype=np.float32)
                by_source = {int(source): index for index, source in enumerate(decode_ids.tolist())}
                local_points = points[ids] - np.asarray([x1, y1], dtype=np.float32)
                for instance_id in np.unique(crop_gt):
                    if instance_id == 0:
                        continue
                    gt = crop_gt == instance_id
                    source_index = int(ids[np.linalg.norm(local_points - _centroid(gt)[None, :], axis=1).argmin()])
                    if source_index not in by_source:
                        raise RuntimeError("quality target source was not decoded in its source crop")
                    hard_iou = inclusive_iou(logits[by_source[source_index]] > 0, gt)
                    value = float(utility_target(torch.tensor(hard_iou)).item())
                    target[source_index] = max(target[source_index], value)
                    oracle_iou[source_index] = max(oracle_iou[source_index], hard_iou)
                    matched[source_index] = True
        label_path = out_dir / f"{image_id}.npz"
        np.savez_compressed(label_path, utility_target=target, matched=matched, oracle_iou=oracle_iou)
        records.append({"image_id": image_id, "patient": item["patient"], "file": label_path.name, "sha256": sha256_file(label_path), "matched_count": int(matched.sum())})
    payload = {"schema_version": 1, "role": role, "contains_gt": True, "target": "decoded_hard_mask_iou * sigmoid((decoded_hard_mask_iou - 0.5) / 0.1)", "records": records}
    json_dump(out_dir / "manifest.json", payload)
    return payload
