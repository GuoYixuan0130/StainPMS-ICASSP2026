"""Cached crop encoding and prompt-only action decoding.

The split mirrors the existing inference function exactly: ``encode_crop`` may
run the image encoder once per crop; ``decode_prompts_from_features`` must not
call the image encoder or point head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label as connected_components


@dataclass
class EncodedCrop:
    image_embed: torch.Tensor
    high_res_feats: tuple[torch.Tensor, ...]
    vision_feats: tuple[torch.Tensor, ...]
    crop_box: tuple[int, int, int, int]


@dataclass
class DecodedPrompts:
    logits: torch.Tensor
    predicted_iou: torch.Tensor
    mean_predicted_iou: torch.Tensor


@torch.no_grad()
def encode_crop(
    net: Any,
    point_encoder: Any,
    image: torch.Tensor,
    memory_bank_list: list,
    context_memory_bank_list: list,
    *,
    crop_box: tuple[int, int, int, int],
    cfgs: Any,
    feat_sizes: tuple[tuple[int, int], ...] = ((64, 64), (32, 32), (16, 16)),
    device: torch.device | str,
) -> EncodedCrop:
    """Encode one crop once, including frozen context/texture conditioning."""

    # Import lazily to avoid a hard model dependency for pure action tests.
    from run.run_on_epoch import context_memory_attention

    x1, y1, _, _ = crop_box
    point_feats, _ = point_encoder(image)
    backbone_out, _ = net.forward_image(image, point_feats)
    _, vision_feats, vision_pos_embeds, _ = net._prepare_backbone_features(backbone_out)
    batch_size = vision_feats[-1].size(1)
    memory_features = vision_feats
    memory_positions = vision_pos_embeds

    if cfgs.context:
        vision_feats, vision_pos_embeds = context_memory_attention(
            context_memory_bank_list,
            vision_feats,
            vision_pos_embeds,
            [x1],
            [y1],
            net,
            list(feat_sizes),
            cfgs.context_atten_k,
        )
    if cfgs.texture:
        if len(memory_bank_list) == 0:
            zero = torch.zeros(1, batch_size, net.hidden_dim, device=device)
            vision_feats[-1] = vision_feats[-1] + zero
            vision_pos_embeds[-1] = vision_pos_embeds[-1] + zero
        else:
            memories = []
            memory_positions_list = []
            image_embeds = []
            for item in memory_bank_list:
                memories.append(item[0].to(device, non_blocking=True).flatten(2).permute(2, 0, 1))
                memory_positions_list.append(item[1].to(device, non_blocking=True).flatten(2).permute(2, 0, 1))
                image_embeds.append(item[3].to(device, non_blocking=True))
            memory_stack = torch.stack(memories, dim=0)
            position_stack = torch.stack(memory_positions_list, dim=0)
            image_embed_stack = F.normalize(torch.stack(image_embeds, dim=0), p=2, dim=1)
            current = vision_feats[-1].permute(1, 0, 2).reshape(batch_size, -1, 64, 64).reshape(batch_size, -1)
            current = F.normalize(current, p=2, dim=1)
            indices = torch.topk(F.softmax(torch.mm(image_embed_stack, current.t()).t(), dim=1), batch_size, dim=1).indices.squeeze(1)
            selected_memory = memory_stack[indices].squeeze(3).permute(1, 2, 0, 3)
            selected_position = position_stack[indices].squeeze(3).permute(1, 2, 0, 3)
            memory = selected_memory.reshape(-1, selected_memory.size(2), selected_memory.size(3))
            position = selected_position.reshape(-1, selected_position.size(2), selected_position.size(3))
            vision_feats[-1], vision_pos_embeds[-1] = net.memory_attention(
                state="texture",
                curr=[vision_feats[-1]],
                curr_pos=[vision_pos_embeds[-1]],
                memory=memory,
                memory_pos=position,
                num_obj_ptr_tokens=0,
            )

    decoded_feats = tuple(
        feat.permute(1, 2, 0).view(batch_size, -1, *size)
        for feat, size in zip(vision_feats[::-1], feat_sizes[::-1])
    )[::-1]
    if cfgs.context and len(context_memory_bank_list) < cfgs.context_memory_bank_size:
        context_memory_bank_list.append([memory_features[-1].detach(), memory_positions[-1].detach(), x1, y1])
    return EncodedCrop(
        image_embed=decoded_feats[-1],
        high_res_feats=tuple(decoded_feats[:-1]),
        vision_feats=tuple(vision_feats),
        crop_box=crop_box,
    )


@torch.no_grad()
def decode_prompts_from_features(
    net: Any,
    encoded: EncodedCrop,
    prompt_points: torch.Tensor,
    prompt_labels: torch.Tensor,
    *,
    out_size: int,
    device: torch.device | str,
) -> DecodedPrompts:
    """Prompt-encode and mask-decode from ``EncodedCrop`` without re-encoding."""

    sparse, dense = net.sam_prompt_encoder(
        points=(prompt_points, prompt_labels),
        boxes=None,
        masks=None,
        batch_size=encoded.image_embed.shape[0],
    )
    low_res, iou_predictions, _, _ = net.sam_mask_decoder(
        image_embeddings=encoded.image_embed,
        image_pe=net.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
        repeat_image=False,
        cell_nums=torch.as_tensor([prompt_points.shape[0]], device=device),
        high_res_features=list(encoded.high_res_feats),
    )
    logits = F.interpolate(low_res, size=(out_size, out_size), mode="bilinear", align_corners=False)[:, 0]
    values, _ = torch.max(iou_predictions, dim=1)
    return DecodedPrompts(logits=logits, predicted_iou=values, mean_predicted_iou=values.mean())


def component_containing_point(mask: np.ndarray, point_x: int, point_y: int) -> np.ndarray:
    """Retain only the connected component containing a positive prompt point."""

    mask = np.asarray(mask, dtype=bool)
    if not 0 <= point_y < mask.shape[0] or not 0 <= point_x < mask.shape[1]:
        raise ValueError("Point lies outside decoded mask")
    labels, _ = connected_components(mask, structure=np.ones((3, 3), dtype=np.int8))
    component_id = int(labels[point_y, point_x])
    return labels == component_id if component_id else np.zeros_like(mask, dtype=bool)


def max_abs_logit_error(first: torch.Tensor, second: torch.Tensor) -> float:
    if first.shape != second.shape:
        raise ValueError(f"Logit shape mismatch: {first.shape} != {second.shape}")
    return float((first.detach().float().cpu() - second.detach().float().cpu()).abs().max().item())
