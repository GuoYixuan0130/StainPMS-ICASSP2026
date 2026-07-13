"""Frozen standard-StainPMS deployment and fixed-prompt inference for Phase 0.

This module deliberately creates neither an optimizer nor a criterion.  It
uses the baseline deployment functions only (proposal, prompt/mask decoding,
and assembly) and keeps their standard overlap/NMS settings unchanged.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config

from .protocol import ProtocolError


@contextmanager
def _baseline_import_argv() -> Any:
    """The legacy utility module parses argv at import time; isolate that side effect."""
    original = sys.argv[:]
    sys.argv = [original[0]]
    try:
        yield
    finally:
        sys.argv = original


def _baseline_functions() -> Any:
    with _baseline_import_argv():
        return importlib.import_module("run.run_on_epoch")


def _crop_boxes(height: int, width: int, crop_size: int, overlap: int, load: str) -> list[list[int]]:
    stride = crop_size - overlap
    if stride <= 0:
        raise ProtocolError(f"invalid standard overlap={overlap} for crop_size={crop_size}")
    def starts(size: int) -> list[int]:
        result, index = [0], 1
        while True:
            point = stride * index
            if point + crop_size >= size:
                if crop_size != size:
                    result.append(size - crop_size)
                return result
            result.append(point); index += 1
    xs, ys = starts(width), starts(height)
    # This is the baseline unclockwise traversal (the default frozen evaluation order).
    boxes: list[list[int]] = []
    top, down, left, right = 0, len(ys) - 1, 0, len(xs) - 1
    while top <= down or left <= right:
        if top <= down:
            boxes.extend([[xs[top], ys[index], min(xs[top] + crop_size, width), min(ys[index] + crop_size, height)] for index in range(left, right + 1)])
            top += 1
        if left <= right:
            boxes.extend([[xs[index], ys[right], min(xs[index] + crop_size, width), min(ys[right] + crop_size, height)] for index in range(top, down + 1)])
            right -= 1
        if top <= down:
            boxes.extend([[xs[down], ys[index], min(xs[down] + crop_size, width), min(ys[index] + crop_size, height)] for index in range(right, left - 1, -1)])
            down -= 1
        if left <= right:
            boxes.extend([[xs[index], ys[left], min(xs[index] + crop_size, width), min(ys[left] + crop_size, height)] for index in range(down, top - 1, -1)])
            left += 1
    return boxes[::-1]


def _normalize(rgb: np.ndarray) -> torch.Tensor:
    value = np.asarray(rgb, dtype=np.float32) / 255.0
    value = (value - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(value.transpose(2, 0, 1))).unsqueeze(0)


def _clone_memory(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, list):
        return [_clone_memory(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_memory(item) for item in value)
    return copy.deepcopy(value)


def model_sha256(*models: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for model in models:
        for name, parameter in model.state_dict().items():
            digest.update(name.encode("utf-8"))
            value = parameter.detach().cpu().contiguous().numpy()
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
            digest.update(value.tobytes())
    return digest.hexdigest()


@contextmanager
def no_training_guard() -> Any:
    original_backward = torch.Tensor.backward
    original_autograd_backward = torch.autograd.backward
    original_optimizer_init = torch.optim.Optimizer.__init__
    def blocked(*_args: Any, **_kwargs: Any) -> None:
        raise ProtocolError("Phase 0 forbids backward/optimizer execution")
    torch.Tensor.backward = blocked  # type: ignore[method-assign]
    torch.autograd.backward = blocked  # type: ignore[assignment]
    torch.optim.Optimizer.__init__ = blocked  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.Tensor.backward = original_backward  # type: ignore[method-assign]
        torch.autograd.backward = original_autograd_backward  # type: ignore[assignment]
        torch.optim.Optimizer.__init__ = original_optimizer_init  # type: ignore[assignment]


@dataclass
class DeploymentResult:
    pred: np.ndarray
    points: np.ndarray
    point_scores: np.ndarray
    candidates: list[dict[str, Any]]
    selected: list[dict[str, Any]]
    prompt_observations: dict[int, list[dict[str, Any]]]


class FrozenStainPMS:
    """A pure-eval StainPMS loader with the canonical deployment parameters."""

    def __init__(self, stainpms_checkpoint: str | Path, sam2_checkpoint: str | Path, overlap: int, device: str = "cuda:0") -> None:
        self.stainpms_checkpoint = Path(stainpms_checkpoint).resolve()
        self.sam2_checkpoint = Path(sam2_checkpoint).resolve()
        if not self.stainpms_checkpoint.is_file() or not self.sam2_checkpoint.is_file():
            raise ProtocolError("StainPMS or official SAM2 checkpoint does not exist")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ProtocolError("CUDA requested but unavailable; Phase 0 does not silently change hardware")
        self.device = torch.device(device)
        self.cfg = SimpleNamespace(
            crop_size=256, out_size=256, overlap=int(overlap), load="unclockwise", texture=True, context=True,
            texture_memory_bank_size=64, context_memory_bank_size=100, context_atten_k=1,
            test=SimpleNamespace(nms_thr=12, filtering=True), tta=False,
        )
        self._functions = _baseline_functions()
        from sam2_train.build_sam import build_sam2
        from sam2_train.modeling.dpa_p2pnet import build_model
        base_args = Config.fromfile(str(Path(__file__).resolve().parents[1] / "args.py"))
        self.net = build_sam2("sam2_hiera_l", str(self.sam2_checkpoint), device=self.device).to(self.device)
        self.point_net, self.point_encoder = build_model(base_args)
        self.point_net, self.point_encoder = self.point_net.to(self.device), self.point_encoder.to(self.device)
        checkpoint = torch.load(self.stainpms_checkpoint, map_location="cpu", weights_only=False)
        if "model1" not in checkpoint:
            raise ProtocolError("StainPMS checkpoint has no model1 point-head state")
        self.point_net.load_state_dict(checkpoint["model1"], strict=True)
        self.memory_template = _clone_memory(checkpoint.get("texture_memory_bank_list", []) or [])
        self.net.eval(); self.point_net.eval(); self.point_encoder.eval()
        for model in (self.net, self.point_net, self.point_encoder):
            for parameter in model.parameters():
                parameter.requires_grad_(False)

    def parameter_checksum(self) -> str:
        return model_sha256(self.net, self.point_net, self.point_encoder)

    def _update_texture_memory(self, memory: list[Any], features: torch.Tensor, positions: torch.Tensor, mean_iou: torch.Tensor, image_embed: torch.Tensor) -> None:
        """Exact standard texture-bank insertion/replacement, without any update step."""
        if not self.cfg.texture:
            return
        if len(memory) < self.cfg.texture_memory_bank_size:
            memory.append([features[0].unsqueeze(0), positions[0].unsqueeze(0), mean_iou, image_embed[0].reshape(-1).detach()])
            return
        bank_flat = torch.stack([entry[0].reshape(-1).to(self.device) for entry in memory])
        bank_norm = F.normalize(bank_flat, p=2, dim=1)
        similarity = torch.mm(bank_norm, bank_norm.t())
        similarity_no_diag = similarity.clone()
        indices = torch.arange(similarity_no_diag.size(0), device=self.device)
        similarity_no_diag[indices, indices] = float("-inf")
        single = F.normalize(features[0].reshape(-1), p=2, dim=0).unsqueeze(1)
        scores = torch.mm(bank_norm, single).squeeze()
        least_similar = torch.argmin(scores)
        most_redundant = torch.argmax(similarity_no_diag[least_similar])
        if scores[least_similar] < similarity_no_diag[least_similar, most_redundant] and mean_iou > memory[int(most_redundant)][2] - 0.1:
            memory.pop(int(most_redundant))
            memory.append([features[0].unsqueeze(0), positions[0].unsqueeze(0), mean_iou, image_embed[0].reshape(-1).detach()])

    def _count_calls(self) -> tuple[dict[str, int], Callable[[], None]]:
        counts = {"image_encoder_calls": 0, "prompt_encoder_calls": 0, "mask_decoder_calls": 0}
        originals = (self.net.forward_image, self.net.sam_prompt_encoder.forward, self.net.sam_mask_decoder.forward)
        def image(*args: Any, **kwargs: Any) -> Any:
            counts["image_encoder_calls"] += 1
            return originals[0](*args, **kwargs)
        def prompt(*args: Any, **kwargs: Any) -> Any:
            counts["prompt_encoder_calls"] += 1
            return originals[1](*args, **kwargs)
        def mask(*args: Any, **kwargs: Any) -> Any:
            counts["mask_decoder_calls"] += 1
            return originals[2](*args, **kwargs)
        self.net.forward_image = image  # type: ignore[method-assign]
        self.net.sam_prompt_encoder.forward = prompt  # type: ignore[method-assign]
        self.net.sam_mask_decoder.forward = mask  # type: ignore[method-assign]
        def restore() -> None:
            self.net.forward_image, self.net.sam_prompt_encoder.forward, self.net.sam_mask_decoder.forward = originals  # type: ignore[method-assign]
        return counts, restore

    def deploy(self, rgb: np.ndarray, fixed_points: np.ndarray | None = None) -> tuple[DeploymentResult, dict[str, int]]:
        """Run canonical auto deployment, or the same decoder path with V0 fixed points."""
        image = _normalize(rgb).to(self.device)
        height, width = rgb.shape[:2]
        boxes = _crop_boxes(height, width, self.cfg.crop_size, self.cfg.overlap, self.cfg.load)
        all_masks: list[np.ndarray] = []; all_boxes: list[list[float]] = []; all_scores: list[float] = []; all_inds: list[int] = []
        all_points: list[np.ndarray] = []; all_point_scores: list[np.ndarray] = []; all_classes: list[np.ndarray] = []
        point_records: dict[int, tuple[np.ndarray, float]] = {}; point_ids: dict[tuple[float, float], int] = {}
        next_id, processed = 0, []; candidates: list[dict[str, Any]] = []; prompt_observations: dict[int, list[dict[str, Any]]] = {}
        memory = _clone_memory(self.memory_template); context_memory: list[Any] = []
        fixed = None if fixed_points is None else np.asarray(fixed_points, dtype=np.float32).reshape(-1, 2)
        if fixed is not None:
            for index, point in enumerate(fixed):
                point_ids[(float(point[0]), float(point[1]))] = index
                point_records[index] = (point, float("nan"))
        counts, restore = self._count_calls()
        try:
            with no_training_guard(), torch.inference_mode():
                for crop_box in boxes:
                    x1, y1, x2, y2 = crop_box
                    crop = image[..., y1:y2, x1:x2]
                    if fixed is None:
                        points, scores, classes, _, _, _, _ = self._functions.predict(
                            self.point_net, crop, ori_shape=np.asarray((y2 - y1, x2 - x1)),
                            filtering=self.cfg.test.filtering, nms_thr=self.cfg.test.nms_thr,
                        )
                        if len(points) == 0:
                            processed.append(crop_box); continue
                        points[:, 0] += x1; points[:, 1] += y1
                        keep_new = np.ones(len(points), dtype=bool)
                        for px1, py1, px2, py2 in processed:
                            keep_new &= ~((points[:, 0] >= px1 + 1) & (points[:, 0] <= px2 - 1) & (points[:, 1] >= py1 + 1) & (points[:, 1] <= py2 - 1))
                        processed.append(crop_box)
                        points, scores, classes = points[keep_new], scores[keep_new], classes[keep_new]
                        if len(points) == 0:
                            continue
                        all_points.append(points); all_point_scores.append(scores); all_classes.append(classes)
                        current_points = np.vstack(all_points); current_scores = np.concatenate(all_point_scores); current_classes = np.concatenate(all_classes)
                        current_points, current_scores, current_classes = self._functions.point_nms(current_points, current_scores, current_classes, self.cfg.test.nms_thr)
                    else:
                        keep_fixed = (fixed[:, 0] >= x1) & (fixed[:, 0] < x2) & (fixed[:, 1] >= y1) & (fixed[:, 1] < y2)
                        if not np.any(keep_fixed):
                            continue
                        current_points, current_scores = fixed[keep_fixed], np.full(int(keep_fixed.sum()), np.nan, dtype=np.float32)
                        current_classes = np.ones(len(current_points), dtype=np.int64)
                    current_inds: list[int] = []
                    for point, score in zip(current_points, current_scores):
                        key = (float(point[0]), float(point[1]))
                        if key not in point_ids:
                            point_ids[key] = next_id; next_id += 1
                        identifier = point_ids[key]
                        point_records[identifier] = (np.asarray(point, dtype=np.float32), float(score))
                        current_inds.append(identifier)
                    prompt_points = torch.from_numpy(current_points).unsqueeze(1)
                    keep = ((prompt_points[..., 0] >= x1) & (prompt_points[..., 0] < x2) & (prompt_points[..., 1] >= y1) & (prompt_points[..., 1] < y2)).squeeze(1)
                    if not int(keep.sum()):
                        continue
                    sub_points = (prompt_points[keep] - torch.as_tensor([x1, y1])).to(self.device).float()
                    sub_labels = torch.ones(sub_points.size(0), 1, dtype=torch.int, device=self.device)
                    keep_np = keep.cpu().numpy(); sub_inds = torch.as_tensor(np.asarray(current_inds, dtype=np.int64)[keep_np], device=self.device)
                    pred, values, mean_iou, vision_feats, image_embed = self._functions.inference(
                        self.net, self.point_encoder, crop, memory, sub_points, sub_labels, [(64, 64), (32, 32), (16, 16)],
                        context_memory, x1, y1, True, self.cfg, self.device,
                    )
                    inst_pred = self._functions.combine_mask(np.asarray((height, width)), sub_points, pred, values)
                    high = torch.from_numpy(inst_pred.astype(float)).to(torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
                    features, positions = self.net._encode_new_memory(current_vision_feats=vision_feats, feat_sizes=[(64, 64), (32, 32), (16, 16)], pred_masks_high_res=high, is_mask_from_pts=True)
                    features, positions = features.to(self.device), positions[0].to(self.device)
                    self._update_texture_memory(memory, features, positions, mean_iou, image_embed)
                    masks = self._functions.mask_process_eval(current_classes[keep_np], sub_inds, crop_box, np.asarray((height, width)), sub_points, pred, values)
                    raw_logits = pred.detach().cpu().numpy(); raw_values = values.detach().cpu().numpy(); sub_ids = sub_inds.detach().cpu().numpy()
                    for local_index, identifier in enumerate(sub_ids):
                        prompt_observations.setdefault(int(identifier), []).append({
                            "crop_box": [int(value) for value in crop_box], "logits": raw_logits[local_index].astype(np.float16),
                            "hard_mask": (raw_logits[local_index] > 0.0), "predicted_iou": float(raw_values[local_index]),
                        })
                    for mask_data in masks:
                        bx1, by1, bx2, by2 = mask_data["bbox"]
                        edge = ((bx1 > 7 and abs(bx1 - x1) <= 7) or (abs(bx2 - height) > 7 and abs(bx2 - x2) <= 7) or (by1 > 7 and abs(by1 - y1) <= 7) or (abs(by2 - width) > 7 and abs(by2 - y2) <= 7))
                        score = float(mask_data["predicted_iou"]) * (0.3 if edge else 1.0)
                        all_masks.append(mask_data["segmentation"][:height, :width]); all_boxes.append(mask_data["bbox"]); all_scores.append(score); all_inds.append(int(mask_data["inds"]))
                        candidates.append({"candidate_index": len(candidates), "bbox": mask_data["bbox"], "crop_box": [int(value) for value in crop_box], "predicted_iou": float(mask_data["predicted_iou"]), "assembly_score": score, "point": mask_data["point"], "point_id": int(mask_data["inds"]), "edge_penalized": bool(edge)})
                assembled, selected = self._functions._assemble_instance_map(all_boxes, all_scores, all_masks, all_inds, (height, width), 0.5, all_records=candidates, return_records=True)
        finally:
            restore()
        ordered = sorted(point_records.items())
        points = np.asarray([item[1][0] for item in ordered], dtype=np.float32).reshape(-1, 2)
        scores = np.asarray([item[1][1] for item in ordered], dtype=np.float32)
        return DeploymentResult(np.asarray(assembled, dtype=np.int32), points, scores, candidates, selected, prompt_observations), counts
