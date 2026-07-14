"""Frozen canonical StainPMS/SAM2 loading and one-crop decoder helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .protocol import QUALITY_HEAD_PARAMETER_COUNT, state_sha256


@dataclass
class ModelBundle:
    point_net: Any
    point_encoder: Any
    sam2: Any
    texture_bank: list[Any]
    device: torch.device
    checkpoint_compatibility: dict[str, list[str]]


def load_frozen_models(checkpoint: Path, sam_config: str, device: torch.device) -> ModelBundle:
    from mmengine.config import Config
    from sam2_train.build_sam import build_sam2
    from sam2_train.modeling.dpa_p2pnet import build_model

    config = Config.fromfile("args.py")
    point_net, point_encoder = build_model(
        config,
        enable_quality_head=True,
        detach_quality_features=True,
        quantize_quality_features_fp16=True,
        export_quality_features=True,
    )
    payload = torch.load(checkpoint, map_location="cpu")
    if "model1" not in payload or "model" not in payload:
        raise ValueError("e156 checkpoint must contain model1 and model")
    missing, unexpected = point_net.load_state_dict(payload["model1"], strict=False)
    allowed_missing = {"quality_head.fc1.weight", "quality_head.fc1.bias", "quality_head.fc2.weight", "quality_head.fc2.bias"}
    if set(missing) != allowed_missing or unexpected:
        raise RuntimeError(f"checkpoint/model compatibility mismatch: missing={missing}, unexpected={unexpected}")
    sam2 = build_sam2(sam_config, str(checkpoint), device=device, mode="eval")
    point_net.to(device).eval()
    point_encoder.to(device).eval()
    sam2.eval()
    return ModelBundle(
        point_net=point_net,
        point_encoder=point_encoder,
        sam2=sam2,
        texture_bank=list(payload.get("texture_memory_bank_list", []) or []),
        device=device,
        checkpoint_compatibility={"missing_keys": sorted(missing), "unexpected_keys": sorted(unexpected)},
    )


def configure_quality_only(bundle: ModelBundle) -> dict:
    for parameter in bundle.point_net.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    for parameter in bundle.sam2.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    for parameter in bundle.point_net.quality_head.parameters():
        parameter.requires_grad_(True)
    if any(module.__class__.__name__ == "Dropout" for module in bundle.point_net.quality_head.modules()):
        raise RuntimeError("PromptQ-v2 quality head must not have dropout")
    count = sum(parameter.numel() for parameter in bundle.point_net.quality_head.parameters())
    if count != QUALITY_HEAD_PARAMETER_COUNT:
        raise RuntimeError(f"quality-head parameter count {count} != {QUALITY_HEAD_PARAMETER_COUNT}")
    return {
        "quality_head_parameter_count": count,
        "trainable_parameter_names": [name for name, parameter in bundle.point_net.named_parameters() if parameter.requires_grad],
        "frozen_point_parameter_count": sum(parameter.numel() for parameter in bundle.point_net.parameters() if not parameter.requires_grad),
        "frozen_sam2_parameter_count": sum(parameter.numel() for parameter in bundle.sam2.parameters()),
    }


def frozen_checksums(bundle: ModelBundle) -> dict[str, str]:
    return {
        "inherited_point": state_sha256(bundle.point_net, exclude_prefixes=("quality_head.",)),
        "sam2": state_sha256(bundle.sam2),
    }


def assert_frozen_without_grads(bundle: ModelBundle) -> None:
    if any(parameter.grad is not None for name, parameter in bundle.point_net.named_parameters() if not name.startswith("quality_head.")):
        raise RuntimeError("quality loss reached inherited point-model parameter")
    if any(parameter.grad is not None for parameter in bundle.sam2.parameters()):
        raise RuntimeError("quality loss reached SAM2")


def _apply_context(context_bank: list[Any], feats: list[torch.Tensor], positions: list[torch.Tensor], bundle: ModelBundle, x: int, y: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    from .assembly import context_memory_attention

    # Canonical validation uses the supplied context bank and same 64/32/16 geometry.
    return context_memory_attention(context_bank, feats, positions, [x], [y], bundle.sam2, 1)


def _apply_texture(texture_bank: list[Any], feats: list[torch.Tensor], positions: list[torch.Tensor], bundle: ModelBundle) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    batch_size = feats[-1].size(1)
    device = bundle.device
    if not texture_bank:
        zero = torch.zeros(1, batch_size, bundle.sam2.hidden_dim, device=device)
        feats[-1] = feats[-1] + zero
        positions[-1] = positions[-1] + zero
        return feats, positions
    memory_stack = torch.stack([entry[0].to(device, non_blocking=True).flatten(2).permute(2, 0, 1) for entry in texture_bank])
    position_stack = torch.stack([entry[1].to(device, non_blocking=True).flatten(2).permute(2, 0, 1) for entry in texture_bank])
    embedding_stack = torch.stack([entry[3].to(device, non_blocking=True) for entry in texture_bank])
    query = feats[-1].permute(1, 0, 2).reshape(batch_size, -1)
    embedding_stack = F.normalize(embedding_stack, p=2, dim=1)
    query = F.normalize(query, p=2, dim=1)
    sampled = torch.topk(F.softmax(torch.mm(embedding_stack, query.t()).t(), dim=1), batch_size, dim=1).indices.squeeze(1)
    selected_memory = memory_stack[sampled].squeeze(3).permute(1, 2, 0, 3)
    selected_position = position_stack[sampled].squeeze(3).permute(1, 2, 0, 3)
    memory = selected_memory.reshape(-1, selected_memory.size(2), selected_memory.size(3))
    memory_pos = selected_position.reshape(-1, selected_position.size(2), selected_position.size(3))
    feats[-1], positions[-1] = bundle.sam2.memory_attention(
        state="texture", curr=[feats[-1]], curr_pos=[positions[-1]], memory=memory, memory_pos=memory_pos, num_obj_ptr_tokens=0
    )
    return feats, positions


@torch.no_grad()
def encode_crop(bundle: ModelBundle, image: torch.Tensor, context_bank: list[Any], crop_box: tuple[int, int, int, int], *, texture: bool, context: bool) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """One canonical SAM2 image encoding, reused by all candidate decodes."""
    point_features, _ = bundle.point_encoder(image)
    backbone_out, _ = bundle.sam2.forward_image(image, point_features)
    _, vision_feats, vision_positions, _ = bundle.sam2._prepare_backbone_features(backbone_out)
    raw_feats, raw_positions = vision_feats, vision_positions
    if context:
        vision_feats, vision_positions = _apply_context(context_bank, vision_feats, vision_positions, bundle, crop_box[0], crop_box[1])
    if texture:
        vision_feats, vision_positions = _apply_texture(bundle.texture_bank, vision_feats, vision_positions, bundle)
    batch_size = vision_feats[-1].size(1)
    shaped = [
        feature.permute(1, 2, 0).reshape(batch_size, -1, *size)
        for feature, size in zip(vision_feats[::-1], [(64, 64), (32, 32), (16, 16)][::-1])
    ][::-1]
    if context and len(context_bank) < 100:
        context_bank.append([raw_feats[-1].detach(), raw_positions[-1].detach(), crop_box[0], crop_box[1]])
    return shaped[-1], shaped[:-1], vision_feats, shaped[-1]


@torch.no_grad()
def decode_points(bundle: ModelBundle, image_embed: torch.Tensor, high_res: list[torch.Tensor], coordinates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not len(coordinates):
        return torch.empty(0, 256, 256, device=bundle.device), torch.empty(0, device=bundle.device)
    labels = torch.ones(coordinates.size(0), 1, dtype=torch.int, device=bundle.device)
    sparse, dense = bundle.sam2.sam_prompt_encoder(points=(coordinates, labels), boxes=None, masks=None, batch_size=1)
    low_res, iou_predictions, _, _ = bundle.sam2.sam_mask_decoder(
        image_embeddings=image_embed,
        image_pe=bundle.sam2.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
        repeat_image=False,
        cell_nums=torch.as_tensor([coordinates.size(0)], device=bundle.device),
        high_res_features=high_res,
    )
    return F.interpolate(low_res, size=(256, 256), mode="bilinear", align_corners=False)[:, 0], torch.max(iou_predictions, dim=1).values


@torch.no_grad()
def update_texture_memory(bundle: ModelBundle, vision_feats: list[torch.Tensor], baseline_mask: np.ndarray, mean_iou: float, image_embed: torch.Tensor, *, texture: bool) -> None:
    """Canonical validation's post-crop texture-bank update, baseline arm only."""
    if not texture:
        return
    high_res = torch.from_numpy(np.asarray(baseline_mask, dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(bundle.device)
    features, positions = bundle.sam2._encode_new_memory(
        current_vision_feats=vision_feats, feat_sizes=[(64, 64), (32, 32), (16, 16)], pred_masks_high_res=high_res, is_mask_from_pts=True
    )
    features = features.to(bundle.device, non_blocking=True)
    positions = positions[0].to(bundle.device, non_blocking=True)
    if len(bundle.texture_bank) < 64:
        for index in range(features.size(0)):
            bundle.texture_bank.append([features[index].unsqueeze(0).detach(), positions[index].unsqueeze(0).detach(), float(mean_iou), image_embed[index].reshape(-1).detach()])
        return
    # This is the same replacement rule as validation_on_epoch.
    for index in range(features.size(0)):
        # The e156 checkpoint is loaded on CPU to protect the immutable
        # artifact.  The canonical replacement comparison is GPU math, so
        # transfer every persisted bank entry explicitly before stacking.
        bank_flat = torch.stack([entry[0].to(bundle.device, non_blocking=True).reshape(-1) for entry in bundle.texture_bank])
        bank_norm = F.normalize(bank_flat, p=2, dim=1)
        similar = torch.mm(bank_norm, bank_norm.t())
        no_diag = similar.clone()
        diag = torch.arange(no_diag.size(0), device=no_diag.device)
        no_diag[diag, diag] = float("-inf")
        current = F.normalize(features[index].reshape(-1), p=2, dim=0).unsqueeze(1)
        scores = torch.mm(bank_norm, current).squeeze()
        low = torch.argmin(scores)
        closest = torch.argmax(no_diag[low])
        old_iou = float(torch.as_tensor(bundle.texture_bank[int(closest)][2]).detach().cpu())
        if scores[low] < no_diag[low][closest] and float(mean_iou) > old_iou - 0.1:
            bundle.texture_bank.pop(int(closest))
            bundle.texture_bank.append([features[index].unsqueeze(0).detach(), positions[index].unsqueeze(0).detach(), float(mean_iou), image_embed[index].reshape(-1).detach()])
