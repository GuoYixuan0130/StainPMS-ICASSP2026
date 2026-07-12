"""NuPart Stage 0 cache-only ownership-conflict and partition-gradient audit.

This module is deliberately a fresh Stage-0 implementation.  It reads frozen
NuRank automatic-prompt cache files solely as an immutable baseline-output
format; it imports neither a ranker nor any multimask/fusion implementation.
No model is loaded, no parameter is mutable, and no cache file is written.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from nupart.core import (
    ConflictEdge,
    connected_components,
    distinct_gt_conflicts,
    foreground_dice,
    gt_ownership_oracle,
    logit_wta,
    nearest_prompt_wta,
)


SEED = 3407
TOKEN_INDEX = 0
CHECKPOINT_SHA256 = "44a3cb3e93051301d789e44f93769588abfa727d7a174b0270f55305ef023781"
TRAIN_PATIENTS = frozenset(range(1, 7))
DEVELOPMENT_PATIENTS = frozenset((7, 8))
REQUIRED_DEVELOPMENT_IMAGES = 7
TIME_CAP_SECONDS = 90 * 60
VISUAL_IMAGE_IDS = ("04_5", "07_3", "08_3")
METRICS = ("dice", "aji", "aji_plus", "dq", "sq", "pq", "tp", "fp", "fn", "matched_iou_sum", "instance_count")


class ProtocolInvalid(RuntimeError):
    """A frozen-baseline or closed-split contract did not hold."""


class TimeCap(RuntimeError):
    """The cache-only audit exceeded its preregistered wall-time cap."""


@dataclass
class Candidate:
    image_id: str
    crop_id: int
    prompt_id: int
    association: int
    point_xy: np.ndarray
    crop_box: tuple[int, int, int, int]
    ori_shape: tuple[int, int]
    cell_class: int
    predicted_iou: float
    edge_score: float
    mask: np.ndarray
    logits: np.ndarray
    bbox: Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _patient(image_id: str) -> int:
    try:
        return int(str(image_id).split("_", 1)[0])
    except (ValueError, IndexError) as error:
        raise ProtocolInvalid(f"invalid TNBC image id: {image_id!r}") from error


def _json(path: Path, value: Any) -> None:
    def convert(item: Any) -> Any:
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, dict):
            return {str(key): convert(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(value) for value in item]
        return item

    path.write_text(json.dumps(convert(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple, np.ndarray)) else value for key, value in row.items()})


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _checksums(out_dir: Path) -> None:
    lines = [f"{_sha256(path)}  {path.relative_to(out_dir).as_posix()}" for path in sorted(out_dir.rglob("*")) if path.is_file() and path.name != "SHA256SUMS"]
    (out_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _baseline_helpers():
    """Load the canonical assembly implementation without leaking this CLI argv."""
    before = sys.argv
    try:
        sys.argv = [sys.argv[0]]
        from run.run_on_epoch import _assemble_instance_map, mask_process_eval
    finally:
        sys.argv = before
    return _assemble_instance_map, mask_process_eval


def _metric_helpers():
    from sam2_train.modeling.stats_utils import get_dice_1, get_fast_aji, get_fast_aji_plus, get_fast_pq, remap_label
    return get_dice_1, get_fast_aji, get_fast_aji_plus, get_fast_pq, remap_label


def assembly_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    """Canonical inclusive-IoU>=0.5 instance metrics, including TP/FP/FN."""
    get_dice_1, get_fast_aji, get_fast_aji_plus, get_fast_pq, remap_label = _metric_helpers()
    gt, pred = remap_label(np.asarray(truth)), remap_label(np.asarray(prediction))
    gt_count, pred_count = int(len(np.unique(gt)) - 1), int(len(np.unique(pred)) - 1)
    if not pred_count:
        return {"dice": 0.0 if gt_count else 1.0, "aji": 0.0 if gt_count else 1.0, "aji_plus": 0.0 if gt_count else 1.0, "dq": 0.0 if gt_count else 1.0, "sq": 0.0 if gt_count else 1.0, "pq": 0.0 if gt_count else 1.0, "tp": 0, "fp": 0, "fn": gt_count, "matched_iou_sum": 0.0, "instance_count": 0}
    if not gt_count:
        return {"dice": 0.0, "aji": 0.0, "aji_plus": 0.0, "dq": 0.0, "sq": 0.0, "pq": 0.0, "tp": 0, "fp": pred_count, "fn": 0, "matched_iou_sum": 0.0, "instance_count": pred_count}
    (dq, sq, pq), paired = get_fast_pq(gt, pred, match_iou=0.5)
    return {"dice": float(get_dice_1(gt, pred)), "aji": float(get_fast_aji(gt, pred)), "aji_plus": float(get_fast_aji_plus(gt, pred)), "dq": float(dq), "sq": float(sq), "pq": float(pq), "tp": int(len(paired[0])), "fp": int(len(paired[3])), "fn": int(len(paired[2])), "matched_iou_sum": float(sq * len(paired[0])), "instance_count": pred_count}


def _load_manifest(cache_dir: Path, role: str) -> dict[str, Any]:
    path = cache_dir / "manifest.json"
    if not path.is_file():
        raise ProtocolInvalid(f"missing immutable cache manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "nurank_automatic_prompt_cache_v1" or manifest.get("token_count") != 4:
        raise ProtocolInvalid(f"unsupported formal cache format: {path}")
    if manifest.get("role") != role:
        raise ProtocolInvalid(f"cache {path} role is not {role}")
    allowed = TRAIN_PATIENTS if role == "train" else DEVELOPMENT_PATIENTS
    ids = list(manifest.get("image_ids", []))
    if not ids or any(_patient(image_id) not in allowed for image_id in ids):
        raise ProtocolInvalid(f"{role} cache crosses the authorized patient boundary")
    if role == "development" and len(ids) != REQUIRED_DEVELOPMENT_IMAGES:
        raise ProtocolInvalid("development cache must contain the preregistered seven images")
    checks = manifest.get("frozen_checksums", {})
    if not checks.get("before") or checks.get("before") != checks.get("after"):
        raise ProtocolInvalid("formal cache does not prove unchanged point/SAM2 checksums")
    if manifest.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ProtocolInvalid("formal cache checkpoint hash differs from frozen StainPMS baseline")
    return manifest


def _load_gt_maps(data_root: Path, image_ids: Iterable[str]) -> dict[str, np.ndarray]:
    """Resolve only cache-named train_12 labels; never enumerate TNBC test paths."""
    if data_root.name.lower() != "tnbc":
        raise ProtocolInvalid("NuPart authorizes TNBC only; MoNuSeg is prohibited")
    import scipy.io as sio

    label_root = data_root / "train_12" / "labels"
    if not label_root.is_dir():
        raise FileNotFoundError(f"missing frozen TNBC labels: {label_root}")
    maps: dict[str, np.ndarray] = {}
    for image_id in image_ids:
        path = label_root / f"{image_id}.mat"
        if not path.is_file():
            raise FileNotFoundError(f"missing cache-named GT label: {path}")
        maps[image_id] = np.asarray(sio.loadmat(path)["inst_map"], dtype=np.int64)
    return maps


def _upsample_identity(low_res: np.ndarray, upsampled: np.ndarray) -> tuple[float, bool]:
    import torch
    import torch.nn.functional as functional

    expected = functional.interpolate(torch.from_numpy(np.asarray(low_res[:, 0:1], dtype=np.float32)), size=tuple(upsampled.shape[-2:]), mode="bilinear", align_corners=False).numpy()[:, 0]
    observed = np.asarray(upsampled[:, 0], dtype=np.float32)
    return float(np.max(np.abs(expected - observed))) if expected.size else 0.0, bool(np.array_equal(expected > 0, observed > 0))


def _global_logits(local_logits: np.ndarray, crop_box: tuple[int, int, int, int], shape: tuple[int, int]) -> np.ndarray:
    yx = np.full(shape, -np.inf, dtype=np.float32)
    x1, y1, x2, y2 = crop_box
    height, width = y2 - y1, x2 - x1
    if local_logits.shape != (height, width):
        raise ProtocolInvalid("cached token-0 logits do not match their recorded crop box")
    yx[y1:y2, x1:x2] = local_logits
    return yx


def _edge_score(record: dict[str, Any], crop_box: tuple[int, int, int, int], shape: tuple[int, int]) -> float:
    bx1, by1, bx2, by2 = record["bbox"]
    sx1, sy1, sx2, sy2 = crop_box
    ori_h, ori_w = shape
    margin = 7
    edge = ((bx1 > margin and abs(bx1 - sx1) <= margin) or (abs(bx2 - ori_h) > margin and abs(bx2 - sx2) <= margin) or (by1 > margin and abs(by1 - sy1) <= margin) or (abs(by2 - ori_w) > margin and abs(by2 - sy2) <= margin))
    return float(record["predicted_iou"] * (0.3 if edge else 1.0))


def _candidate_records(group: dict[str, np.ndarray], entry: dict[str, Any], gt: np.ndarray) -> tuple[list[Candidate], float, bool]:
    """Build canonical token-0 candidate masks from one immutable cache group."""
    import torch

    _, mask_process_eval = _baseline_helpers()
    logits = np.asarray(group["mask_logits"], dtype=np.float32)
    if logits.ndim != 4 or logits.shape[1] != 4:
        raise ProtocolInvalid("cache group is not a four-token logit group")
    if "low_res_logits" in group:
        lowres = np.asarray(group["low_res_logits"], dtype=np.float32)
        if lowres.ndim != 4 or lowres.shape[:2] != logits.shape[:2]:
            raise ProtocolInvalid("cached low-resolution logits do not match the four-token layout")
        error, hard_equal = _upsample_identity(lowres, logits)
    else:
        # The formal NuRank cache predates low-resolution-logit persistence.
        # Its immutable per-group write-error proof is the only valid fallback;
        # no logits are regenerated and no non-token-0 data are read.
        cache_write_error = entry.get("cached_mask_logits_max_abs_error")
        if cache_write_error is None or float(cache_write_error) != 0.0:
            raise ProtocolInvalid("cache lacks low-resolution logits and exact upsampled-logit write proof")
        error, hard_equal = 0.0, True
    crop_box = tuple(int(value) for value in np.asarray(group["crop_box_xyxy"]).tolist())
    shape = tuple(int(value) for value in np.asarray(group["ori_shape_hw"]).tolist())
    count = logits.shape[0]
    fields = ("coordinates_local_xy", "classes", "prompt_ids", "target_instance_id", "original_predicted_iou")
    if any(len(np.asarray(group[field])) != count for field in fields) or shape != gt.shape:
        raise ProtocolInvalid("cache group fields do not agree with GT shape")
    local = np.asarray(group["coordinates_local_xy"], dtype=np.float32)
    prompt_ids = np.asarray(group["prompt_ids"], dtype=np.int64)
    if len(np.unique(prompt_ids)) != len(prompt_ids):
        raise ProtocolInvalid("cache group has duplicate automatic prompt IDs")
    points = torch.from_numpy(local).unsqueeze(1)
    records = mask_process_eval(np.asarray(group["classes"], dtype=np.int64), prompt_ids, crop_box, shape, points, torch.from_numpy(logits[:, TOKEN_INDEX]), torch.from_numpy(np.asarray(group["original_predicted_iou"], dtype=np.float32)[:, TOKEN_INDEX]))
    source = {int(prompt_ids[index]): index for index in range(count)}
    candidates: list[Candidate] = []
    x1, y1, _, _ = crop_box
    for record in records:
        prompt_id = int(record["inds"])
        if prompt_id not in source:
            raise ProtocolInvalid("standard baseline returned a mask without a cache prompt")
        index = source[prompt_id]
        point = local[index] + np.asarray((x1, y1), dtype=np.float32)
        px = int(np.clip(np.trunc(point[0]), 0, shape[1] - 1))
        py = int(np.clip(np.trunc(point[1]), 0, shape[0] - 1))
        association = int(gt[py, px])
        if association != int(np.asarray(group["target_instance_id"])[index]):
            raise ProtocolInvalid("cache association is not the fixed prompt-point GT association")
        candidates.append(Candidate(str(entry["image_id"]), int(entry["crop_id"]), prompt_id, association, point, crop_box, shape, int(np.asarray(group["classes"])[index]), float(np.asarray(group["original_predicted_iou"])[index, TOKEN_INDEX]), _edge_score(record, crop_box, shape), np.asarray(record["segmentation"], dtype=bool), _global_logits(logits[index, TOKEN_INDEX], crop_box, shape), record["bbox"]))
    return candidates, error, hard_equal


def _assemble(candidates: list[Candidate], masks: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Canonical baseline assembly; variants only substitute already-existing hard masks."""
    import torch

    assemble, mask_process_eval = _baseline_helpers()
    if not candidates:
        raise ProtocolInvalid("an image has no automatic token-0 candidates")
    active = np.stack([candidate.mask for candidate in candidates]) if masks is None else np.asarray(masks, dtype=bool)
    if active.shape != (len(candidates), *candidates[0].ori_shape):
        raise ValueError("variant masks do not align with canonical candidates")
    if masks is None:
        # These are the exact batch-wise canonical records emitted from the
        # formal cache replay.  Reusing them proves baseline equivalence before
        # any variant starts rebuilding changed hard masks.
        records = [{"bbox": candidate.bbox, "segmentation": candidate.mask, "predicted_iou": candidate.predicted_iou} for candidate in candidates]
        scores = [candidate.edge_score for candidate in candidates]
    else:
        records = []
        scores = []
        for candidate, hard in zip(candidates, active):
            x1, y1, x2, y2 = candidate.crop_box
            local = np.where(hard[y1:y2, x1:x2], 1.0, -1.0).astype(np.float32)
            rebuilt = mask_process_eval(np.asarray([candidate.cell_class]), np.asarray([candidate.prompt_id]), candidate.crop_box, candidate.ori_shape, torch.from_numpy((candidate.point_xy - np.asarray((x1, y1), dtype=np.float32))[None, None]), torch.from_numpy(local[None]), torch.tensor([candidate.predicted_iou], dtype=torch.float32))
            if len(rebuilt) != 1:
                raise ProtocolInvalid("resolver deleted a point or mask during canonical reconstruction")
            records.append(rebuilt[0])
            scores.append(float(_edge_score(rebuilt[0], candidate.crop_box, candidate.ori_shape)))
    prediction, selected = assemble([record["bbox"] for record in records], scores, [record["segmentation"][: candidates[0].ori_shape[0], : candidates[0].ori_shape[1]] for record in records], [candidate.prompt_id for candidate in candidates], candidates[0].ori_shape, 0.5, all_records=[{"source_candidate_index": index} for index in range(len(records))], return_records=True)
    owners = np.full(candidates[0].ori_shape, -1, dtype=np.int64)
    for record in selected:
        owners[prediction == int(record["final_id"])] = int(record["source_candidate_index"])
    return np.asarray(prediction, dtype=np.int64), owners


def _load_reference_maps(path: Path, image_ids: Iterable[str]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"missing formal baseline assembly reference: {path}")
    result: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        for image_id in image_ids:
            choices = (image_id, f"{image_id}__baseline_single", f"{image_id}__single", f"{image_id}:baseline_single")
            key = next((choice for choice in choices if choice in archive.files), None)
            if key is None:
                raise ProtocolInvalid(f"formal baseline assembly lacks image {image_id}")
            result[image_id] = np.asarray(archive[key], dtype=np.int64)
    return result


def _load_formal_development_metrics(development_cache: Path, image_ids: Iterable[str]) -> tuple[Path, dict[str, dict[str, float]]]:
    """Read immutable NuRank development token-0 metrics without a model call."""
    path = development_cache.parents[1] / "evaluation" / "per_image_metrics.csv"
    if not path.is_file():
        raise ProtocolInvalid(f"missing formal development baseline metrics: {path}")
    result: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("path") not in ("baseline_single", "single"):
                continue
            image_id = str(row.get("image_id"))
            result[image_id] = {metric: float(row[metric]) for metric in ("dice", "aji", "aji_plus", "dq", "sq", "pq")}
    missing = sorted(set(image_ids) - set(result))
    if missing:
        raise ProtocolInvalid(f"formal development metrics lack token-0 baseline rows: {missing}")
    return path, result


def _touching_pairs(gt: np.ndarray) -> set[tuple[int, int]]:
    from scipy.ndimage import binary_dilation

    ids = [int(value) for value in np.unique(gt) if value]
    masks = {value: gt == value for value in ids}
    pairs: set[tuple[int, int]] = set()
    for left, right in __import__("itertools").combinations(ids, 2):
        if np.logical_and(binary_dilation(masks[left], iterations=1), binary_dilation(masks[right], iterations=1)).any():
            pairs.add((left, right))
    return pairs


def _centroids_and_touching(gt: np.ndarray) -> tuple[dict[int, np.ndarray], set[int]]:
    ids = [int(value) for value in np.unique(gt) if value]
    centers: dict[int, np.ndarray] = {}
    masks = {value: gt == value for value in ids}
    for value, mask in masks.items():
        points = np.argwhere(mask)
        center = points.mean(axis=0)
        if not mask[tuple(np.rint(center).astype(int))]:
            center = points[np.argmin(((points - center) ** 2).sum(axis=1))]
        centers[value] = center[[1, 0]].astype(np.float32)
    touching = {item for pair in _touching_pairs(gt) for item in pair}
    return centers, touching


def _density_rows(role: str, image_id: str, gt: np.ndarray, associations: np.ndarray, edges: list[ConflictEdge]) -> list[dict[str, Any]]:
    centers, touching = _centroids_and_touching(gt)
    conflict_nodes = {node for edge in edges for node in (edge.left, edge.right)}
    rows = []
    for gt_id, center in centers.items():
        distances = [float(np.linalg.norm(center - other)) for other_id, other in centers.items() if other_id != gt_id]
        nearest = min(distances) if distances else float("inf")
        band = "<12" if nearest < 12 else "[12,24)" if nearest < 24 else ">=24"
        matching = np.flatnonzero(associations == gt_id)
        rows.append({"role": role, "image_id": image_id, "gt_instance_id": gt_id, "nearest_centroid_distance": nearest, "density_band": band, "touching": gt_id in touching, "prompt_count": int(len(matching)), "conflict_prompt_count": int(sum(index in conflict_nodes for index in matching))})
    return rows


def _foreign_rows(role: str, image_id: str, masks: np.ndarray, associations: np.ndarray, gt: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for index, (mask, association) in enumerate(zip(masks, associations)):
        area = int(mask.sum())
        foreign = int(np.logical_and(mask, (gt != 0) & (gt != association)).sum()) if association else int(np.logical_and(mask, gt != 0).sum())
        rows.append({"role": role, "image_id": image_id, "prompt_index": index, "association": int(association), "mask_pixels": area, "foreign_gt_pixels": foreign, "foreign_gt_occupancy": foreign / area if area else 0.0})
    for gt_id in (int(value) for value in np.unique(gt) if value):
        invaded = np.logical_and(gt == gt_id, np.any(masks[associations != gt_id], axis=0) if np.any(associations != gt_id) else np.zeros_like(gt, dtype=bool))
        area = int((gt == gt_id).sum())
        rows.append({"role": role, "image_id": image_id, "prompt_index": "GT", "association": gt_id, "mask_pixels": area, "foreign_gt_pixels": int(invaded.sum()), "foreign_gt_occupancy": float(invaded.sum() / area) if area else 0.0})
    return rows


def _boundary_inside(gt_mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt

    mask = np.asarray(gt_mask, dtype=bool)
    return mask & (distance_transform_edt(mask) <= 2)


def _ownership_and_gradients(components: list[list[int]], masks: np.ndarray, logits: np.ndarray, associations: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    """Detached-logit CE/BCE/focal autograd diagnostic; it never touches a model."""
    import torch
    import torch.nn.functional as functional
    from scipy.ndimage import binary_dilation

    started = time.perf_counter()
    records, all_cosines, partition_negative, partition_nonnegative, bce_negative, bce_nonnegative, focal_negative, focal_nonnegative = [], [], [], [], [], [], [], []
    wrong_total = correct_up = wrong_down = finite_components = nonzero_components = 0
    boundary_wrong = interior_wrong = 0
    bce_correct_up = bce_wrong_down = focal_correct_up = focal_wrong_down = 0
    peak_working_tensor_bytes = 0
    margin_values: list[float] = []
    for component_id, nodes in enumerate(components):
        gt_ids = sorted({int(associations[node]) for node in nodes if associations[node]})
        region = binary_dilation(np.isin(gt, gt_ids), iterations=2)
        coords = np.argwhere(region)
        if not len(coords):
            continue
        # A same-GT duplicate inside a distinct-GT component cannot have two
        # unique pixel labels.  The fixed canonical owner is the lowest node;
        # all logits remain present and therefore receive a partition gradient.
        owner = {gt_id: min(node for node in nodes if associations[node] == gt_id) for gt_id in gt_ids}
        node_to_class = {node: index + 1 for index, node in enumerate(nodes)}
        target = np.zeros(len(coords), dtype=np.int64)
        for index, (y, x) in enumerate(coords):
            target[index] = node_to_class[owner[int(gt[y, x])]] if int(gt[y, x]) in owner else 0
        values = np.stack([logits[node, coords[:, 0], coords[:, 1]] for node in nodes], axis=1)
        values = np.where(np.isfinite(values), values, -80.0).astype(np.float32)
        peak_working_tensor_bytes = max(peak_working_tensor_bytes, int(values.nbytes))
        detached = torch.tensor(values, dtype=torch.float32, requires_grad=True)
        partition_logits = torch.cat((torch.zeros((len(detached), 1), dtype=torch.float32), detached), dim=1)
        partition = functional.cross_entropy(partition_logits, torch.from_numpy(target), reduction="mean")
        partition.backward()
        partition_grad = detached.grad.detach().cpu().numpy()
        target_binary = np.zeros_like(values, dtype=np.float32)
        for column, node in enumerate(nodes):
            target_binary[:, column] = gt[coords[:, 0], coords[:, 1]] == associations[node]
        independent = torch.tensor(values, dtype=torch.float32, requires_grad=True)
        bce = functional.binary_cross_entropy_with_logits(independent, torch.from_numpy(target_binary), reduction="mean")
        bce.backward()
        bce_grad = independent.grad.detach().cpu().numpy()
        focal_input = torch.tensor(values, dtype=torch.float32, requires_grad=True)
        ce = functional.binary_cross_entropy_with_logits(focal_input, torch.from_numpy(target_binary), reduction="none")
        probability = torch.sigmoid(focal_input)
        pt = probability * torch.from_numpy(target_binary) + (1 - probability) * (1 - torch.from_numpy(target_binary))
        focal = ((1 - pt).pow(2) * ce).mean()
        focal.backward()
        focal_grad = focal_input.grad.detach().cpu().numpy()
        finite = bool(np.isfinite(partition_grad).all())
        nonzero = bool(np.any(np.abs(partition_grad) > 0))
        finite_components += int(finite)
        nonzero_components += int(nonzero)
        cosine_denominator = np.linalg.norm(partition_grad.ravel()) * np.linalg.norm(bce_grad.ravel())
        all_cosines.append(float(np.dot(partition_grad.ravel(), bce_grad.ravel()) / cosine_denominator) if cosine_denominator else 0.0)
        for node in nodes:
            gt_id = int(associations[node])
            if not gt_id or owner.get(gt_id) != node:
                continue
            member = np.asarray([item for item in nodes if item != node], dtype=np.int64)
            if not len(member):
                continue
            target_pixels = (gt == gt_id) & region
            other = logits[member].max(axis=0)
            margin = logits[node] - other
            wrong = target_pixels & (other > logits[node])
            boundary = _boundary_inside(gt == gt_id)
            margin_values.extend(margin[target_pixels].astype(float).tolist())
            owner_column = node_to_class[node] - 1
            wrong_columns = np.asarray([node_to_class[item] - 1 for item in member], dtype=np.int64)
            for y, x in np.argwhere(wrong):
                row_index = int(np.where((coords == (y, x)).all(axis=1))[0][0])
                wrong_column = int(wrong_columns[np.argmax(logits[member, y, x])])
                correct_up += int(partition_grad[row_index, owner_column] < 0)
                wrong_down += int(partition_grad[row_index, wrong_column] > 0)
                bce_correct_up += int(bce_grad[row_index, owner_column] < 0)
                bce_wrong_down += int(bce_grad[row_index, wrong_column] > 0)
                focal_correct_up += int(focal_grad[row_index, owner_column] < 0)
                focal_wrong_down += int(focal_grad[row_index, wrong_column] > 0)
            wrong_total += int(wrong.sum())
            boundary_wrong += int(np.logical_and(wrong, boundary).sum())
            interior_wrong += int(np.logical_and(wrong, ~boundary).sum())
            negative = target_pixels & (margin < 0)
            nonnegative = target_pixels & ~negative
            if negative.any():
                local = np.isin(coords[:, 0] * gt.shape[1] + coords[:, 1], np.flatnonzero(negative))
                partition_negative.extend(np.abs(partition_grad[local, owner_column]).tolist()); bce_negative.extend(np.abs(bce_grad[local, owner_column]).tolist()); focal_negative.extend(np.abs(focal_grad[local, owner_column]).tolist())
            if nonnegative.any():
                local = np.isin(coords[:, 0] * gt.shape[1] + coords[:, 1], np.flatnonzero(nonnegative))
                partition_nonnegative.extend(np.abs(partition_grad[local, owner_column]).tolist()); bce_nonnegative.extend(np.abs(bce_grad[local, owner_column]).tolist()); focal_nonnegative.extend(np.abs(focal_grad[local, owner_column]).tolist())
        records.append({"component_id": component_id, "node_count": len(nodes), "gt_id_count": len(gt_ids), "region_pixels": int(region.sum()), "partition_loss": float(partition.detach()), "finite": finite, "nonzero": nonzero, "bce_loss": float(bce.detach()), "focal_loss": float(focal.detach())})
    def mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0
    return {"component_count": len(records), "components": records, "background_logit": 0.0, "local_softmax_normalized": True, "finite_gradient_fraction": finite_components / len(records) if records else 1.0, "nonzero_gradient_fraction": nonzero_components / len(records) if records else 1.0, "wrong_winner_pixel_count": wrong_total, "boundary_band_wrong_winner_rate": boundary_wrong / (boundary_wrong + interior_wrong) if boundary_wrong + interior_wrong else 0.0, "interior_wrong_winner_rate": interior_wrong / (boundary_wrong + interior_wrong) if boundary_wrong + interior_wrong else 0.0, "wrong_winner_correct_owner_up_fraction": correct_up / wrong_total if wrong_total else 1.0, "wrong_winner_max_wrong_owner_down_fraction": wrong_down / wrong_total if wrong_total else 1.0, "independent_bce_correct_owner_up_fraction": bce_correct_up / wrong_total if wrong_total else 1.0, "independent_bce_max_wrong_owner_down_fraction": bce_wrong_down / wrong_total if wrong_total else 1.0, "independent_focal_correct_owner_up_fraction": focal_correct_up / wrong_total if wrong_total else 1.0, "independent_focal_max_wrong_owner_down_fraction": focal_wrong_down / wrong_total if wrong_total else 1.0, "partition_vs_independent_bce_gradient_cosine_mean": mean(all_cosines), "negative_margin_p10": float(np.quantile(margin_values, .10)) if margin_values else None, "negative_margin_p25": float(np.quantile(margin_values, .25)) if margin_values else None, "negative_margin_median": float(np.median(margin_values)) if margin_values else None, "partition_abs_gradient_negative_margin": mean(partition_negative), "partition_abs_gradient_nonnegative_margin": mean(partition_nonnegative), "bce_abs_gradient_negative_margin": mean(bce_negative), "bce_abs_gradient_nonnegative_margin": mean(bce_nonnegative), "focal_abs_gradient_negative_margin": mean(focal_negative), "focal_abs_gradient_nonnegative_margin": mean(focal_nonnegative), "peak_memory_bytes": peak_working_tensor_bytes, "compute_seconds": time.perf_counter() - started, "model_or_optimizer_touched": False}


def _aggregate(rows: list[dict[str, Any]], path: str) -> dict[str, float]:
    selected = [row for row in rows if row["path"] == path]
    return {metric: float(np.mean([row[metric] for row in selected])) for metric in METRICS}


def _bootstrap(rows: list[dict[str, Any]], path: str, seed: int = SEED, samples: int = 2000) -> dict[str, Any]:
    baseline = {row["image_id"]: row for row in rows if row["path"] == "standard"}
    target = {row["image_id"]: row for row in rows if row["path"] == path}
    ids = sorted(baseline)
    rng = np.random.default_rng(seed)
    result = {"seed": seed, "resamples": samples, "image_count": len(ids), "metrics": {}}
    for metric in ("dice", "aji", "aji_plus", "dq", "sq", "pq"):
        values = np.asarray([target[item][metric] - baseline[item][metric] for item in ids], dtype=np.float64)
        draws = np.asarray([values[rng.integers(0, len(values), len(values))].mean() for _ in range(samples)])
        result["metrics"][metric] = {"mean_delta": float(values.mean()), "ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]}
    return result


def _render_label(path: Path, array: np.ndarray, *, binary: bool = False) -> None:
    from PIL import Image

    value = np.asarray(array)
    if binary:
        rgb = np.zeros((*value.shape, 3), dtype=np.uint8); rgb[value.astype(bool)] = (255, 70, 60)
    else:
        rgb = np.zeros((*value.shape, 3), dtype=np.uint8)
        for label in np.unique(value):
            if label:
                seed = int(label) * 2654435761 % (2 ** 32)
                rng = np.random.default_rng(seed)
                rgb[value == label] = rng.integers(40, 256, size=3, dtype=np.uint8)
    Image.fromarray(rgb).save(path)


def _visualize(out_dir: Path, snapshots: dict[str, dict[str, np.ndarray]]) -> None:
    visual_dir = out_dir / "visuals"; visual_dir.mkdir()
    for image_id in VISUAL_IMAGE_IDS:
        if image_id not in snapshots:
            raise ProtocolInvalid(f"required fixed visualization image is missing: {image_id}")
        item = snapshots[image_id]
        _render_label(visual_dir / f"{image_id}_gt_instances.png", item["gt"])
        _render_label(visual_dir / f"{image_id}_standard_masks.png", item["standard"])
        _render_label(visual_dir / f"{image_id}_distinct_gt_conflict_pixels.png", item["conflict"], binary=True)
        _render_label(visual_dir / f"{image_id}_wrong_winner_pixels.png", item["wrong"], binary=True)
        _render_label(visual_dir / f"{image_id}_logit_wta.png", item["logit_wta"])
        _render_label(visual_dir / f"{image_id}_gt_ownership_oracle.png", item["oracle"])


def _verdict(development: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    conflict = development["conflict"]
    oracle = development["oracle"]
    fixed = development["fixed"]
    gradient = development["gradient"]
    ownership = conflict["images_with_conflict"] >= 5 and (conflict["conflict_prompt_fraction"] >= .05 or conflict["overlap_fraction_gt_foreground"] >= .005)
    b = oracle["delta"]["pq"] >= .008 and oracle["delta"]["aji"] >= .005 and oracle["delta"]["dq"] >= 0 and oracle["pq_non_decreasing_images"] >= 5 and oracle["largest_positive_image_contribution_fraction"] <= .50
    c = oracle["delta"]["pq"] >= .004 and oracle["delta"]["aji"] >= .003 and oracle["pq_non_decreasing_images"] >= 5 and oracle["largest_positive_image_contribution_fraction"] <= .60
    d = any(item["delta_pq"] >= .002 and item["delta_aji"] >= 0 and item["pq_non_decreasing_images"] >= 5 and item["delta_fp"] <= 0 for item in fixed)
    e = gradient["finite_gradient_fraction"] == 1.0 and gradient["nonzero_gradient_fraction"] >= .95 and gradient["wrong_winner_correct_owner_up_fraction"] >= .99 and gradient["wrong_winner_max_wrong_owner_down_fraction"] >= .99
    checks = {"OWNERSHIP_CONFLICT": ownership, "STRONG_ORACLE_HEADROOM": b, "MODERATE_ORACLE_HEADROOM": c, "FIXED_RESOLVER_SIGNAL": d, "PARTITION_GRADIENT_VALID": e}
    if ownership and b and e:
        return "STRONG GO", checks
    if ownership and c and d and e:
        return "CONDITIONAL GO", checks
    return "NO-GO", checks


def run_stage0(*, train_cache: Path, development_cache: Path, data_root: Path, checkpoint: Path, baseline_maps: Path | None, out_dir: Path) -> dict[str, Any]:
    """Run the one authorized NuPart Stage 0 audit and write a new artifact."""
    if out_dir.exists():
        raise FileExistsError(f"NuPart refuses to overwrite artifacts: {out_dir}")
    if _sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ProtocolInvalid("frozen StainPMS checkpoint SHA256 mismatch")
    random.seed(SEED); np.random.seed(SEED)
    manifests = {"train": _load_manifest(train_cache, "train"), "development": _load_manifest(development_cache, "development")}
    all_ids = [item for role in ("train", "development") for item in manifests[role]["image_ids"]]
    if len(set(all_ids)) != len(all_ids):
        raise ProtocolInvalid("train and development cache image IDs overlap")
    gt_maps = _load_gt_maps(data_root, all_ids)
    references = _load_reference_maps(baseline_maps, all_ids) if baseline_maps is not None else None
    formal_development_metrics_path, formal_development_metrics = _load_formal_development_metrics(development_cache, manifests["development"]["image_ids"])
    out_dir.mkdir(parents=True, exist_ok=False)
    _json(out_dir / "protocol.json", {"name": "NuPart Stage 0: Prompt-Mask Ownership Conflict and Local Partition Headroom Audit", "seed": SEED, "token": TOKEN_INDEX, "tta": False, "time_cap_seconds": TIME_CAP_SECONDS, "allowed_patients": {"train": sorted(TRAIN_PATIENTS), "development": sorted(DEVELOPMENT_PATIENTS)}, "prohibited": ["TNBC patients 9-11", "MoNuSeg", "training", "optimizer.step", "NuRank", "NuFuse", "multimask tokens 1-3"], "resolver_paths": ["standard", "logit_wta", "nearest_prompt_wta", "gt_ownership_oracle"]})
    (out_dir / "environment.txt").write_text(f"git_sha={_git_sha()}\npython={sys.version}\nplatform={platform.platform()}\nseed={SEED}\ntta=False\ncache_only=True\ncheckpoint_sha256={CHECKPOINT_SHA256}\n", encoding="utf-8")
    (out_dir / "tests.txt").write_text("Run: python -m unittest discover -s tests/nupart -v\n", encoding="utf-8")
    started = time.perf_counter()
    baseline_equivalence = {"token0_lowres_upsample_max_abs_error": 0.0, "token0_lowres_logits_available_for_all_groups": True, "token0_upsampled_cache_write_max_abs_error": 0.0, "token0_hard_masks_identical": True, "final_instance_map_identical": True, "metric_max_abs_error": 0.0, "formal_development_metrics_max_abs_error": 0.0, "formal_development_metrics_path": str(formal_development_metrics_path), "instance_map_reference": "external_formal_archive" if references is not None else "canonical_cache_reassembly_repeat", "inclusive_iou_threshold": 0.5, "frozen_checksums_unchanged": True, "passed": False}
    per_image, conflict_rows, component_rows, density_rows, foreign_rows = [], [], [], [], []
    role_results: dict[str, Any] = {}; snapshots: dict[str, dict[str, np.ndarray]] = {}
    for role, cache_dir in (("train", train_cache), ("development", development_cache)):
        manifest = manifests[role]
        candidates_by_image: dict[str, list[Candidate]] = {image_id: [] for image_id in manifest["image_ids"]}
        for entry in manifest["groups"]:
            if time.perf_counter() - started > TIME_CAP_SECONDS:
                raise TimeCap("NuPart cache-only audit exceeded 90 minutes")
            path = cache_dir / entry["path"]
            if not path.is_file() or _sha256(path) != entry.get("sha256"):
                raise ProtocolInvalid(f"formal cache group checksum mismatch: {path}")
            with np.load(path, allow_pickle=False) as archive:
                group = {name: archive[name] for name in archive.files}
            image_id = str(entry["image_id"])
            if image_id not in candidates_by_image:
                raise ProtocolInvalid("cache group image ID is absent from manifest")
            built, error, hard_equal = _candidate_records(group, entry, gt_maps[image_id])
            baseline_equivalence["token0_lowres_upsample_max_abs_error"] = max(baseline_equivalence["token0_lowres_upsample_max_abs_error"], error)
            baseline_equivalence["token0_lowres_logits_available_for_all_groups"] &= "low_res_logits" in group
            baseline_equivalence["token0_upsampled_cache_write_max_abs_error"] = max(baseline_equivalence["token0_upsampled_cache_write_max_abs_error"], float(entry.get("cached_mask_logits_max_abs_error", 0.0)))
            baseline_equivalence["token0_hard_masks_identical"] &= hard_equal
            candidates_by_image[image_id].extend(built)
        role_conflict_summaries, role_gradients, role_fixed, role_oracle = [], [], [], []
        for image_id, candidates in candidates_by_image.items():
            gt = gt_maps[image_id]
            standard, owners = _assemble(candidates)
            reference = references[image_id] if references is not None else _assemble(candidates)[0]
            same_map = bool(np.array_equal(standard, reference))
            standard_metrics = assembly_metrics(gt, standard)
            reference_metrics = assembly_metrics(gt, reference)
            metric_error = max(abs(standard_metrics[key] - reference_metrics[key]) for key in ("dice", "aji", "aji_plus", "dq", "sq", "pq"))
            baseline_equivalence["final_instance_map_identical"] &= same_map
            baseline_equivalence["metric_max_abs_error"] = max(baseline_equivalence["metric_max_abs_error"], float(metric_error))
            if role == "development":
                formal_error = max(abs(standard_metrics[key] - formal_development_metrics[image_id][key]) for key in ("dice", "aji", "aji_plus", "dq", "sq", "pq"))
                baseline_equivalence["formal_development_metrics_max_abs_error"] = max(baseline_equivalence["formal_development_metrics_max_abs_error"], float(formal_error))
            masks, logits = np.stack([candidate.mask for candidate in candidates]), np.stack([candidate.logits for candidate in candidates])
            associations, points = np.asarray([candidate.association for candidate in candidates]), np.stack([candidate.point_xy for candidate in candidates])
            edges = distinct_gt_conflicts(masks, associations); components = connected_components(len(candidates), edges)
            conflict_pixels = np.zeros(gt.shape, dtype=bool)
            for edge in edges: conflict_pixels |= masks[edge.left] & masks[edge.right]
            same_gt_overlap = np.zeros(gt.shape, dtype=bool)
            unmatched_overlap = np.zeros(gt.shape, dtype=bool)
            for left in range(len(candidates)):
                for right in range(left + 1, len(candidates)):
                    pair_overlap = masks[left] & masks[right]
                    if associations[left] and associations[left] == associations[right]:
                        same_gt_overlap |= pair_overlap
                    if associations[left] == 0 or associations[right] == 0:
                        unmatched_overlap |= pair_overlap
            matched = associations != 0
            duplicate_count = int(sum(max(0, int((associations == gt_id).sum()) - 1) for gt_id in np.unique(associations[matched])))
            overlap = int(conflict_pixels.sum())
            conflict_nodes = {node for edge in edges for node in (edge.left, edge.right)}
            touching_pairs = _touching_pairs(gt)
            touching_edge_count = sum(tuple(sorted((int(associations[edge.left]), int(associations[edge.right])))) in touching_pairs for edge in edges)
            gt_foreground = int((gt > 0).sum())
            summary = {"role": role, "image_id": image_id, "automatic_prompt_count": len(candidates), "matched_prompt_count": int(matched.sum()), "unmatched_prompt_count": int((~matched).sum()), "same_gt_duplicate_count": duplicate_count, "same_gt_duplicate_overlap_pixel_count": int(same_gt_overlap.sum()), "unmatched_overlap_pixel_count": int(unmatched_overlap.sum()), "distinct_gt_conflict_edge_count": len(edges), "distinct_gt_conflict_touching_edge_count": touching_edge_count, "conflict_component_count": len(components), "component_sizes": [len(component) for component in components], "conflict_matched_prompt_fraction": len(conflict_nodes) / int(matched.sum()) if matched.any() else 0.0, "overlap_pixel_count": overlap, "gt_foreground_pixel_count": gt_foreground, "overlap_over_gt_foreground": overlap / gt_foreground if gt_foreground else 0.0, "overlap_over_predicted_foreground": overlap / int(np.any(masks, axis=0).sum()) if np.any(masks) else 0.0}
            conflict_rows.append(summary); role_conflict_summaries.append(summary)
            density_rows.extend(_density_rows(role, image_id, gt, associations, edges)); foreign_rows.extend(_foreign_rows(role, image_id, masks, associations, gt))
            for component_id, component in enumerate(components):
                component_edges = [edge for edge in edges if edge.left in component and edge.right in component]
                component_rows.append({"role": role, "image_id": image_id, "component_id": component_id, "prompt_count": len(component), "prompt_indices": component, "gt_ids": sorted({int(associations[node]) for node in component}), "edge_count": len(component_edges), "touching_edge_count": int(sum(tuple(sorted((int(associations[edge.left]), int(associations[edge.right])))) in touching_pairs for edge in component_edges)), "overlap_pixels": int(sum(edge.overlap_pixels for edge in component_edges))})
            wta_masks, _ = logit_wta(masks, logits, owners); nearest_masks, _ = nearest_prompt_wta(masks, points, owners)
            oracle_masks, changed, authorized = gt_ownership_oracle(masks, associations, gt, owners)
            _, touching_ids = _centroids_and_touching(gt)
            touching_masks, _, _ = gt_ownership_oracle(masks, associations, gt, owners, allowed_gt_ids=touching_ids)
            non_touching_ids = {int(value) for value in np.unique(gt) if value and value not in touching_ids}
            non_touching_masks, _, _ = gt_ownership_oracle(masks, associations, gt, owners, allowed_gt_ids=non_touching_ids)
            variants = {"standard": (masks, standard), "logit_wta": (wta_masks, _assemble(candidates, wta_masks)[0]), "nearest_prompt_wta": (nearest_masks, _assemble(candidates, nearest_masks)[0]), "gt_ownership_oracle": (oracle_masks, _assemble(candidates, oracle_masks)[0]), "oracle_touching_only": (touching_masks, _assemble(candidates, touching_masks)[0]), "oracle_non_touching_only": (non_touching_masks, _assemble(candidates, non_touching_masks)[0])}
            for path_name, (_, prediction) in variants.items():
                per_image.append({"role": role, "image_id": image_id, "path": path_name, **assembly_metrics(gt, prediction)})
            gradient = _ownership_and_gradients(components, masks, logits, associations, gt); role_gradients.append(gradient)
            wrong = np.zeros(gt.shape, dtype=bool)
            for component in components:
                for node in component:
                    other = [item for item in component if item != node]
                    if other:
                        wrong |= (gt == associations[node]) & (np.max(logits[other], axis=0) > logits[node])
            snapshots[image_id] = {"gt": gt, "standard": standard, "conflict": conflict_pixels, "wrong": wrong, "logit_wta": variants["logit_wta"][1], "oracle": variants["gt_ownership_oracle"][1]}
            role_oracle.append({"image_id": image_id, "changed_pixels": int(changed.sum()), "authorized_conflict_pixels": int(authorized.sum()), "changed_pixels_touching_gt": int(np.logical_and(changed, np.isin(gt, list(touching_ids))).sum()), "changed_pixels_non_touching_gt": int(np.logical_and(changed, np.isin(gt, list(non_touching_ids))).sum())})
        role_rows = [row for row in per_image if row["role"] == role]
        standard_summary = _aggregate(role_rows, "standard")
        fixed_rows = []
        for path_name in ("logit_wta", "nearest_prompt_wta"):
            summary = _aggregate(role_rows, path_name); baseline = standard_summary
            deltas = {metric: summary[metric] - baseline[metric] for metric in METRICS}
            variant_rows = sorted([item for item in role_rows if item["path"] == path_name], key=lambda item: item["image_id"])
            baseline_rows = sorted([item for item in role_rows if item["path"] == "standard"], key=lambda item: item["image_id"])
            fixed_rows.append({
                "role": role,
                "resolver": path_name,
                **summary,
                **{f"delta_{metric}": value for metric, value in deltas.items()},
                "pq_non_decreasing_images": int(sum(row["pq"] >= base["pq"] for row, base in zip(variant_rows, baseline_rows))),
            })
        oracle_summary = _aggregate(role_rows, "gt_ownership_oracle")
        deltas = {metric: oracle_summary[metric] - standard_summary[metric] for metric in METRICS}
        image_deltas = {row["image_id"]: row["pq"] - next(base["pq"] for base in role_rows if base["path"] == "standard" and base["image_id"] == row["image_id"]) for row in role_rows if row["path"] == "gt_ownership_oracle"}
        positive = sum(max(0.0, value) for value in image_deltas.values())
        role_results[role] = {"conflict_rows": role_conflict_summaries, "gradient_records": role_gradients, "fixed": fixed_rows, "oracle": {"metrics": oracle_summary, "delta": deltas, "per_image_pq_delta": image_deltas, "pq_non_decreasing_images": int(sum(value >= 0 for value in image_deltas.values())), "largest_positive_image_contribution_fraction": max([max(0.0, value) for value in image_deltas.values()], default=0.0) / positive if positive else 0.0, "changed": role_oracle, "per_1000_changed_pixels_delta_pq": deltas["pq"] / sum(item["changed_pixels"] for item in role_oracle) * 1000 if sum(item["changed_pixels"] for item in role_oracle) else 0.0, "touching_only_delta": {metric: _aggregate(role_rows, "oracle_touching_only")[metric] - standard_summary[metric] for metric in ("dice", "aji", "aji_plus", "dq", "sq", "pq")}, "non_touching_only_delta": {metric: _aggregate(role_rows, "oracle_non_touching_only")[metric] - standard_summary[metric] for metric in ("dice", "aji", "aji_plus", "dq", "sq", "pq")}, "bootstrap": _bootstrap(role_rows, "gt_ownership_oracle")}}
    baseline_equivalence["frozen_checksums_unchanged"] = all(manifest["frozen_checksums"]["before"] == manifest["frozen_checksums"]["after"] for manifest in manifests.values())
    baseline_equivalence["passed"] = bool(baseline_equivalence["token0_lowres_upsample_max_abs_error"] == 0.0 and baseline_equivalence["token0_upsampled_cache_write_max_abs_error"] == 0.0 and baseline_equivalence["token0_hard_masks_identical"] and baseline_equivalence["final_instance_map_identical"] and baseline_equivalence["metric_max_abs_error"] <= 1e-7 and baseline_equivalence["formal_development_metrics_max_abs_error"] <= 1e-7 and baseline_equivalence["frozen_checksums_unchanged"])
    _json(out_dir / "baseline_equivalence.json", baseline_equivalence)
    if not baseline_equivalence["passed"]:
        report = {"verdict": "PROTOCOL INVALID", "reason": "baseline equivalence failed; no ownership/headroom conclusion is valid", "baseline_equivalence": baseline_equivalence}
        _json(out_dir / "report.json", report); _checksums(out_dir); return report
    _csv(out_dir / "conflict_summary.csv", conflict_rows); _csv(out_dir / "conflict_components.csv", component_rows); _csv(out_dir / "per_image_metrics.csv", per_image); _csv(out_dir / "density_stratification.csv", density_rows); _csv(out_dir / "foreign_gt_occupancy.csv", foreign_rows)
    gradient_summary = {}
    for role, result in role_results.items():
        records = result["gradient_records"]
        components = [component for record in records for component in record["components"]]
        wrong_pixels = sum(record["wrong_winner_pixel_count"] for record in records)
        finite = float(np.mean([component["finite"] for component in components])) if components else 1.0
        nonzero = float(np.mean([component["nonzero"] for component in components])) if components else 1.0
        def weighted(field: str) -> float:
            return sum(record[field] * record["wrong_winner_pixel_count"] for record in records) / wrong_pixels if wrong_pixels else 1.0
        def average(field: str) -> float | None:
            values = [record[field] for record in records if record.get(field) is not None]
            return float(np.mean(values)) if values else None
        gradient_summary[role] = {"component_count": len(components), "finite_gradient_fraction": finite, "nonzero_gradient_fraction": nonzero, "wrong_winner_pixel_count": wrong_pixels, "wrong_winner_correct_owner_up_fraction": weighted("wrong_winner_correct_owner_up_fraction"), "wrong_winner_max_wrong_owner_down_fraction": weighted("wrong_winner_max_wrong_owner_down_fraction"), "independent_bce_correct_owner_up_fraction": weighted("independent_bce_correct_owner_up_fraction"), "independent_bce_max_wrong_owner_down_fraction": weighted("independent_bce_max_wrong_owner_down_fraction"), "independent_focal_correct_owner_up_fraction": weighted("independent_focal_correct_owner_up_fraction"), "independent_focal_max_wrong_owner_down_fraction": weighted("independent_focal_max_wrong_owner_down_fraction"), "boundary_band_wrong_winner_rate": weighted("boundary_band_wrong_winner_rate"), "interior_wrong_winner_rate": weighted("interior_wrong_winner_rate"), "partition_vs_independent_bce_gradient_cosine_mean": average("partition_vs_independent_bce_gradient_cosine_mean"), "negative_margin_p10": average("negative_margin_p10"), "negative_margin_p25": average("negative_margin_p25"), "negative_margin_median": average("negative_margin_median"), "partition_abs_gradient_negative_margin": average("partition_abs_gradient_negative_margin"), "partition_abs_gradient_nonnegative_margin": average("partition_abs_gradient_nonnegative_margin"), "bce_abs_gradient_negative_margin": average("bce_abs_gradient_negative_margin"), "bce_abs_gradient_nonnegative_margin": average("bce_abs_gradient_nonnegative_margin"), "focal_abs_gradient_negative_margin": average("focal_abs_gradient_negative_margin"), "focal_abs_gradient_nonnegative_margin": average("focal_abs_gradient_nonnegative_margin"), "peak_memory_bytes": max([record["peak_memory_bytes"] for record in records], default=0), "compute_seconds": sum(record["compute_seconds"] for record in records), "components": components, "background_logit": 0.0, "local_softmax_normalized": True, "model_or_optimizer_touched": False}
    _json(out_dir / "gradient_diagnostic.json", gradient_summary)
    ownership_summary = {role: {"foreign_gt_occupancy_file": "foreign_gt_occupancy.csv", "margin_and_gradient": gradient_summary[role], "per_image_margin_records": role_results[role]["gradient_records"]} for role in role_results}
    _json(out_dir / "ownership_margin_summary.json", ownership_summary)
    resolver_rows = [item for result in role_results.values() for item in result["fixed"]]
    _csv(out_dir / "resolver_results.csv", resolver_rows)
    oracle_results = {role: result["oracle"] for role, result in role_results.items()}
    _json(out_dir / "oracle_results.json", oracle_results)
    development_conflicts = role_results["development"]["conflict_rows"]
    development_matched = sum(row["matched_prompt_count"] for row in development_conflicts)
    development_foreground = sum(row["gt_foreground_pixel_count"] for row in development_conflicts)
    development_conflict = {"images_with_conflict": int(sum(row["distinct_gt_conflict_edge_count"] > 0 for row in development_conflicts)), "conflict_prompt_fraction": sum(row["conflict_matched_prompt_fraction"] * row["matched_prompt_count"] for row in development_conflicts) / development_matched if development_matched else 0.0, "overlap_fraction_gt_foreground": sum(row["overlap_pixel_count"] for row in development_conflicts) / development_foreground if development_foreground else 0.0}
    development = {"conflict": development_conflict, "oracle": role_results["development"]["oracle"], "fixed": role_results["development"]["fixed"], "gradient": gradient_summary["development"]}
    verdict, checks = _verdict(development)
    _visualize(out_dir, snapshots)
    call_counts = {"frozen_model_calls_during_nupart": {"point_model": 0, "sam2": 0, "mask_decoder": 0}, "cache_source_call_counts": {role: manifests[role].get("call_counts", {}) for role in manifests}, "checkpoint_loaded": False, "optimizer_step": 0}
    _json(out_dir / "call_count_summary.json", call_counts)
    report = {"verdict": verdict, "decision_checks": checks, "baseline_equivalence": baseline_equivalence, "development": development, "train": {"conflict": role_results["train"]["conflict_rows"], "oracle": role_results["train"]["oracle"], "fixed": role_results["train"]["fixed"], "gradient": gradient_summary["train"]}, "frozen_parameter_checksums": {role: manifests[role]["frozen_checksums"] for role in manifests}, "call_counts": call_counts, "recommendation": "Stage 0 complete. Stop and await project-lead decision; do not enter NuPart Stage 1."}
    _json(out_dir / "report.json", report); _checksums(out_dir)
    return report
