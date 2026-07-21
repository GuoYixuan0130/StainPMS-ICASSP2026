"""Read-only all-native-token decoder helpers for the Phase 1 audit.

The regular StainPMS evaluation route remains untouched. This module mirrors
the feature preparation in ``run.run_on_epoch.inference`` but uses the mask
decoder's internal ``predict_masks`` method to expose token 0 and its three
ambiguity tokens in a diagnostic-only path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from run.run_on_epoch import context_memory_attention


@dataclass
class PreparedImage:
    image_embed: torch.Tensor
    high_res_feats: list[torch.Tensor]
    vision_feats: list[torch.Tensor]
    image_embed_for_texture: torch.Tensor
    context_entry: list[Any] | None


@torch.no_grad()
def prepare_image_for_all_token_decode(
    *,
    net,
    point_encoder,
    image: torch.Tensor,
    texture_memory_bank: list,
    context_memory_bank: list,
    x1: int,
    y1: int,
    texture: bool,
    context: bool,
    context_atten_k: int,
    device: torch.device,
) -> PreparedImage:
    """Mirror the regular inference feature path without updating state."""

    feat_sizes = [(64, 64), (32, 32), (16, 16)]
    feats, _ = point_encoder(image)
    backbone_out, _ = net.forward_image(image, feats)
    _, vision_feats, vision_pos_embeds, _ = net._prepare_backbone_features(backbone_out)
    batch_size = vision_feats[-1].size(1)
    memfeatures = vision_feats
    memfeatures_pos = vision_pos_embeds

    if context:
        vision_feats, vision_pos_embeds = context_memory_attention(
            context_memory_bank,
            vision_feats,
            vision_pos_embeds,
            [x1],
            [y1],
            net,
            feat_sizes,
            context_atten_k,
        )

    if texture:
        if len(texture_memory_bank) == 0:
            zero = torch.zeros(1, batch_size, net.hidden_dim, device=device)
            vision_feats[-1] = vision_feats[-1] + zero
            vision_pos_embeds[-1] = vision_pos_embeds[-1] + zero
        else:
            # This is the original validation inference's retrieval key: the
            # image embedding saved in entry 3, rather than the mask-memory
            # tensor in entry 0.  Those tensors have different dimensions.
            # Using entry 0 here both changed the baseline route and fails as
            # soon as the first texture-bank item is consulted.
            image_embed_bank = torch.stack(
                [item[3].to(device, non_blocking=True) for item in texture_memory_bank]
            )
            current = vision_feats[-1].permute(1, 0, 2).reshape(batch_size, -1, 64, 64)
            current = current.reshape(batch_size, -1)
            image_embed_bank = F.normalize(image_embed_bank, p=2, dim=1)
            current = F.normalize(current, p=2, dim=1)
            similarities = torch.mm(image_embed_bank, current.t()).t()
            indices = torch.topk(F.softmax(similarities, dim=1), batch_size, dim=1).indices.squeeze(1)
            stacks = torch.stack(
                [item[0].to(device, non_blocking=True).flatten(2).permute(2, 0, 1) for item in texture_memory_bank],
                dim=0,
            )
            pos_stacks = torch.stack(
                [item[1].to(device, non_blocking=True).flatten(2).permute(2, 0, 1) for item in texture_memory_bank],
                dim=0,
            )
            memory_new = stacks[indices].squeeze(3).permute(1, 2, 0, 3)
            memory_pos_new = pos_stacks[indices].squeeze(3).permute(1, 2, 0, 3)
            memory = memory_new.reshape(-1, memory_new.size(2), memory_new.size(3))
            memory_pos = memory_pos_new.reshape(-1, memory_new.size(2), memory_new.size(3))
            vision_feats[-1], vision_pos_embeds[-1] = net.memory_attention(
                state="texture",
                curr=[vision_feats[-1]],
                curr_pos=[vision_pos_embeds[-1]],
                memory=memory,
                memory_pos=memory_pos,
                num_obj_ptr_tokens=0,
            )

    feature_maps = [
        feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
        for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])
    ][::-1]
    context_entry = None
    if context:
        context_entry = [memfeatures[-1].detach(), memfeatures_pos[-1].detach(), x1, y1]
    return PreparedImage(
        image_embed=feature_maps[-1],
        high_res_feats=feature_maps[:-1],
        vision_feats=vision_feats,
        image_embed_for_texture=feature_maps[-1],
        context_entry=context_entry,
    )


@torch.no_grad()
def decode_all_native_mask_tokens(
    *,
    net,
    prepared: PreparedImage,
    prompt_points: torch.Tensor,
    prompt_labels: torch.Tensor,
    out_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return low/high logits and quality predictions with shape [prompts, 4, ...]."""

    if prompt_points.ndim != 3 or prompt_points.shape[1:] != (1, 2):
        raise ValueError(f"expected [P,1,2] prompts, received {tuple(prompt_points.shape)}")
    sparse, dense = net.sam_prompt_encoder(
        points=(prompt_points, prompt_labels),
        boxes=None,
        masks=None,
        batch_size=prepared.image_embed.shape[0],
    )
    low_res_masks, quality_predictions, _, _ = net.sam_mask_decoder.predict_masks(
        image_embeddings=prepared.image_embed,
        image_pe=net.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        repeat_image=False,
        cell_nums=torch.as_tensor([prompt_points.shape[0]], device=device),
        high_res_features=prepared.high_res_feats,
    )
    if low_res_masks.shape[1] != 4 or quality_predictions.shape[1] != 4:
        raise RuntimeError(
            "Phase 1 requires all four native mask tokens; decoder returned "
            f"masks={tuple(low_res_masks.shape)}, quality={tuple(quality_predictions.shape)}"
        )
    high_res_masks = F.interpolate(
        low_res_masks,
        size=(int(out_size), int(out_size)),
        mode="bilinear",
        align_corners=False,
    )
    return low_res_masks, high_res_masks, quality_predictions


@torch.no_grad()
def select_standard_single_mask(
    *,
    net,
    low_res_logits: torch.Tensor,
    high_res_logits: torch.Tensor,
    quality_predictions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mirror ``multimask_output=False`` including its optional dynamic fallback."""

    decoder = net.sam_mask_decoder
    token_indices = torch.zeros(low_res_logits.shape[0], dtype=torch.long, device=low_res_logits.device)
    if bool(getattr(decoder, "dynamic_multimask_via_stability", False)) and not decoder.training:
        stability = decoder._get_stability_scores(low_res_logits[:, 0:1])[:, 0]
        fallback = stability < float(decoder.dynamic_multimask_stability_thresh)
        best_ambiguity = torch.argmax(quality_predictions[:, 1:], dim=1) + 1
        token_indices = torch.where(fallback, best_ambiguity, token_indices)
    batch_indices = torch.arange(low_res_logits.shape[0], device=low_res_logits.device)
    return (
        high_res_logits[batch_indices, token_indices],
        quality_predictions[batch_indices, token_indices],
        token_indices,
    )


@torch.no_grad()
def update_validation_texture_memory(
    *,
    net,
    prepared: PreparedImage,
    default_mask_logits: torch.Tensor,
    default_quality: torch.Tensor,
    texture_memory_bank: list,
    texture_memory_bank_size: int,
    device: torch.device,
) -> None:
    """Apply the existing validation texture-bank policy using token 0 only."""

    if texture_memory_bank_size <= 0:
        return
    maskmem_features, maskmem_pos_enc = net._encode_new_memory(
        current_vision_feats=prepared.vision_feats,
        feat_sizes=[(64, 64), (32, 32), (16, 16)],
        pred_masks_high_res=default_mask_logits,
        is_mask_from_pts=True,
    )
    maskmem_features = maskmem_features.to(device=device, non_blocking=True)
    maskmem_pos_enc = maskmem_pos_enc[0].to(device=device, non_blocking=True)
    score = default_quality.mean()
    image_embed = prepared.image_embed_for_texture
    if len(texture_memory_bank) < texture_memory_bank_size:
        for batch_idx in range(maskmem_features.size(0)):
            texture_memory_bank.append(
                [
                    maskmem_features[batch_idx].unsqueeze(0),
                    maskmem_pos_enc[batch_idx].unsqueeze(0),
                    score,
                    image_embed[batch_idx].reshape(-1).detach(),
                ]
            )
        return
    for batch_idx in range(maskmem_features.size(0)):
        bank_flat = torch.stack([entry[0].reshape(-1).to(device) for entry in texture_memory_bank])
        bank_norm = F.normalize(bank_flat, p=2, dim=1)
        similarity = torch.mm(bank_norm, bank_norm.t())
        similarity_no_diag = similarity.clone()
        diag = torch.arange(similarity_no_diag.size(0), device=similarity_no_diag.device)
        similarity_no_diag[diag, diag] = float("-inf")
        key = F.normalize(maskmem_features[batch_idx].reshape(-1), p=2, dim=0).unsqueeze(1)
        scores = torch.mm(bank_norm, key).squeeze()
        minimum = torch.argmin(scores)
        replace = torch.argmax(similarity_no_diag[minimum])
        # Preserve the regular validation replacement gate: only consider a
        # redundant bank entry and then require the same quality margin.
        if scores[minimum] < similarity_no_diag[minimum][replace] and score > texture_memory_bank[int(replace)][2] - 0.1:
            texture_memory_bank.pop(int(replace))
            texture_memory_bank.append(
                [
                    maskmem_features[batch_idx].unsqueeze(0),
                    maskmem_pos_enc[batch_idx].unsqueeze(0),
                    score,
                    image_embed[batch_idx].reshape(-1).detach(),
                ]
            )
