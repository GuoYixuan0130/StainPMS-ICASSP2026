"""One-pass detached PromptQ feature/utility cache extraction and loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterator

import numpy as np
import torch

from promptcredit.audit.guardrails import sha256_file
from promptcredit.audit.runner import _decode_standard, _prepare_decoder_features
from promptcredit.method import build_quality_targets, gather_nearest_coordinates
from promptcredit.promptq.data import PromptQCrop


def _json_dump(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _hard_iou(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    hard = logits > 0
    truth = masks.bool()
    intersection = (hard & truth).sum(dim=(1, 2)).float()
    union = (hard | truth).sum(dim=(1, 2)).float()
    return torch.where(union > 0, intersection / union, torch.ones_like(union))


def _metadata_for_sources(crop: PromptQCrop, source_indices: torch.Tensor, hard_iou: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One target per source: highest utility GT wins and duplicates are marked."""
    source_count = int(source_indices.max().item()) + 1 if source_indices.numel() else 0
    # The caller replaces this with the full proposal count after allocation.
    source_utility: dict[int, tuple[float, float, int, int, int]] = {}
    duplicate = set()
    for gt_index, source_index in enumerate(source_indices.detach().cpu().tolist()):
        utility = float(
            hard_iou[gt_index].detach().cpu()
            * torch.sigmoid((hard_iou[gt_index].detach().cpu() - 0.5) / 0.1)
        )
        candidate = (
            utility,
            float(hard_iou[gt_index].detach().cpu()),
            int(crop.gt_areas[gt_index]),
            int(crop.local_density[gt_index]),
            int(crop.gt_instance_ids[gt_index]),
        )
        if source_index in source_utility:
            duplicate.add(source_index)
        if source_index not in source_utility or candidate[0] > source_utility[source_index][0]:
            source_utility[source_index] = candidate
    utility = np.zeros(source_count, dtype=np.float32)
    hard = np.zeros(source_count, dtype=np.float32)
    area = np.zeros(source_count, dtype=np.int32)
    density = np.zeros(source_count, dtype=np.int32)
    duplicated = np.zeros(source_count, dtype=np.bool_)
    for source_index, (value, hard_value, area_value, density_value, _) in source_utility.items():
        utility[source_index] = value
        hard[source_index] = hard_value
        area[source_index] = area_value
        density[source_index] = density_value
        duplicated[source_index] = source_index in duplicate
    return utility, hard, area, density, duplicated


def extract_cache(
    *,
    bundle: Any,
    crops: Iterator[PromptQCrop],
    out_dir: Path,
    role: str,
    progress_callback=None,
) -> dict[str, Any]:
    """Extract once with frozen models; decode only nearest-matched GT prompts."""
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite PromptQ cache: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    bundle.point_net.eval()
    bundle.net.eval()
    records: list[dict[str, Any]] = []
    context_bank: list[Any] = []
    active_image: str | None = None
    started = time.perf_counter()
    for ordinal, crop in enumerate(crops):
        if crop.image_id != active_image:
            active_image = crop.image_id
            context_bank = []
        image = crop.image.unsqueeze(0).to(bundle.device)
        with torch.no_grad():
            output, _, _, _ = bundle.point_net(image)
            if "quality_roi_features" not in output:
                raise RuntimeError("PromptQ cache extraction requires exported detached quality_roi_features")
            features = output["quality_roi_features"][0].reshape(-1, output["quality_roi_features"].shape[-1])
            proposal_count = int(features.size(0))
            coords = output["pred_coords"][0].detach()
            point_probabilities = torch.softmax(output["pred_logits"][0], dim=-1)
            foreground = point_probabilities[:, 0].detach()
            foreground_decision = (torch.argmax(point_probabilities, dim=-1) < (point_probabilities.shape[-1] - 1)).detach()
            semantic = output["pred_masks"][0, 0].detach() > 0
            clipped = coords.round().long()
            clipped[:, 0].clamp_(0, semantic.shape[1] - 1)
            clipped[:, 1].clamp_(0, semantic.shape[0] - 1)
            semantic_valid = semantic[clipped[:, 1], clipped[:, 0]]
            utility = np.zeros(proposal_count, dtype=np.float32)
            hard_iou = np.zeros(proposal_count, dtype=np.float32)
            area = np.zeros(proposal_count, dtype=np.int32)
            density = np.zeros(proposal_count, dtype=np.int32)
            duplicate = np.zeros(proposal_count, dtype=np.bool_)
            matched = np.zeros(proposal_count, dtype=np.bool_)
            if len(crop.gt_centroids_xy):
                targets = [torch.as_tensor(crop.gt_centroids_xy, dtype=torch.float32, device=bundle.device)]
                selection = gather_nearest_coordinates(output["pred_coords"], targets)
                image_embed, high_res = _prepare_decoder_features(bundle, image, context_bank, crop)
                decoded_logits, _ = _decode_standard(bundle, image_embed, high_res, selection.coordinates.detach())
                decoded_hard_iou = _hard_iou(
                    decoded_logits,
                    torch.as_tensor(crop.gt_masks, dtype=torch.float32, device=bundle.device),
                )
                targets_for_sources = build_quality_targets(
                    output["pred_quality_logits"], selection.source_indices, decoded_hard_iou
                )
                utility = targets_for_sources.values[0].detach().cpu().numpy().astype(np.float32)
                matched = targets_for_sources.matched_proposals[0].detach().cpu().numpy().astype(np.bool_)
                partial_utility, partial_hard, partial_area, partial_density, partial_duplicate = _metadata_for_sources(
                    crop, selection.source_indices[0], decoded_hard_iou
                )
                source_ids = np.arange(len(partial_utility), dtype=np.int64)
                valid = source_ids < proposal_count
                hard_iou[source_ids[valid]] = partial_hard[valid]
                area[source_ids[valid]] = partial_area[valid]
                density[source_ids[valid]] = partial_density[valid]
                duplicate[source_ids[valid]] = partial_duplicate[valid]
            cache_path = out_dir / f"{ordinal:05d}_{crop.image_id}_{crop.crop_id:04d}.npz"
            np.savez_compressed(
                cache_path,
                features=features.detach().cpu().numpy().astype(np.float16),
                foreground_probability=foreground.cpu().numpy().astype(np.float32),
                foreground_decision=foreground_decision.cpu().numpy().astype(np.bool_),
                predicted_coordinate=coords.cpu().numpy().astype(np.float32),
                utility_target=utility,
                hard_mask_iou=hard_iou,
                matched=matched,
                nucleus_area=area,
                local_density=density,
                duplicate_source=duplicate,
                semantic_valid=semantic_valid.cpu().numpy().astype(np.bool_),
                crop_box_xyxy=np.asarray(crop.crop_box_xyxy, dtype=np.int32),
                image_id=np.asarray(crop.image_id),
                crop_id=np.asarray(crop.crop_id, dtype=np.int32),
            )
        records.append(
            {
                "file": cache_path.name,
                "sha256": sha256_file(cache_path),
                "image_id": crop.image_id,
                "crop_id": int(crop.crop_id),
                "crop_box_xyxy": list(crop.crop_box_xyxy),
                "proposal_count": proposal_count,
                "matched_positive_count": int(matched.sum()),
                "duplicate_source_count": int(duplicate.sum()),
                "decoded_prompt_count": int(len(crop.gt_centroids_xy)),
                "image_crop_sha256": crop.image_crop_sha256,
                "gt_crop_sha256": crop.gt_crop_sha256,
            }
        )
        if progress_callback is not None:
            progress_callback(ordinal + 1, time.perf_counter() - started)
    manifest = {
        "schema_version": 1,
        "role": role,
        "feature_dtype": "float16",
        "feature_contract": "PromptQ online quality head consumes detached FP16-to-FP32 ROI features",
        "utility_target": "hard_iou * sigmoid((hard_iou - 0.5) / 0.1)",
        "files": records,
        "total_crops": len(records),
        "total_proposals": int(sum(item["proposal_count"] for item in records)),
        "total_matched_positive_sources": int(sum(item["matched_positive_count"] for item in records)),
        "total_decoded_prompts": int(sum(item["decoded_prompt_count"] for item in records)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _json_dump(out_dir / "manifest.json", manifest)
    return manifest


def iter_cache_arrays(cache_manifest_path: Path) -> Iterator[dict[str, np.ndarray]]:
    manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    for record in manifest["files"]:
        path = cache_manifest_path.parent / record["file"]
        with np.load(path, allow_pickle=False) as payload:
            yield {key: payload[key] for key in payload.files}


def cache_sha256(manifest_path: Path) -> str:
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()
