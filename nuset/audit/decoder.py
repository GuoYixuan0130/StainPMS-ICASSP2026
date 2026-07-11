"""One-call SAM2 four-token extraction; no model mutation or training."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AllTokenMasks:
    """All four outputs yielded by exactly one ``predict_masks`` invocation."""

    low_res_logits: torch.Tensor  # [prompts, 4, low_h, low_w]
    upsampled_logits: torch.Tensor  # [prompts, 4, 256, 256]
    predicted_iou: torch.Tensor  # [prompts, 4]
    mask_tokens: torch.Tensor
    object_score_logits: torch.Tensor
    predict_masks_seconds: float
    all_token_upsample_seconds: float


def assert_four_token_decoder(mask_decoder: Any) -> None:
    if int(mask_decoder.num_multimask_outputs) != 3 or int(mask_decoder.num_mask_tokens) != 4:
        raise RuntimeError(
            "NuSet requires SAM2 num_multimask_outputs=3 and num_mask_tokens=4; "
            f"got {mask_decoder.num_multimask_outputs}/{mask_decoder.num_mask_tokens}"
        )


@torch.no_grad()
def extract_all_tokens_once(
    *,
    mask_decoder: Any,
    prompt_encoder: Any,
    image_embeddings: torch.Tensor,
    high_res_features: list[torch.Tensor],
    coordinates: torch.Tensor,
    out_size: int = 256,
) -> AllTokenMasks:
    """Encode prompts once and expose all four masks from one decoder call."""
    assert_four_token_decoder(mask_decoder)
    if coordinates.ndim != 3 or coordinates.shape[1:] != (1, 2):
        raise ValueError("NuSet coordinates must be [prompts, 1, 2]")
    labels = torch.ones(coordinates.size(0), 1, dtype=torch.int, device=coordinates.device)
    sparse, dense = prompt_encoder(points=(coordinates, labels), boxes=None, masks=None, batch_size=1)
    if image_embeddings.is_cuda:
        torch.cuda.synchronize(image_embeddings.device)
    predict_started = time.perf_counter()
    low_res, predicted_iou, tokens, object_score_logits = mask_decoder.predict_masks(
        image_embeddings=image_embeddings,
        image_pe=prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        repeat_image=False,
        cell_nums=torch.as_tensor([coordinates.size(0)], device=coordinates.device),
        high_res_features=high_res_features,
    )
    if image_embeddings.is_cuda:
        torch.cuda.synchronize(image_embeddings.device)
    predict_seconds = time.perf_counter() - predict_started
    if low_res.ndim != 4 or low_res.size(1) != 4 or predicted_iou.shape != low_res.shape[:2]:
        raise RuntimeError("NuSet all-token extractor received an unexpected SAM2 token layout")
    if low_res.is_cuda:
        torch.cuda.synchronize(low_res.device)
    upsample_started = time.perf_counter()
    upsampled = F.interpolate(low_res, size=(out_size, out_size), mode="bilinear", align_corners=False)
    if low_res.is_cuda:
        torch.cuda.synchronize(low_res.device)
    upsample_seconds = time.perf_counter() - upsample_started
    return AllTokenMasks(
        low_res_logits=low_res,
        upsampled_logits=upsampled,
        predicted_iou=predicted_iou,
        mask_tokens=tokens,
        object_score_logits=object_score_logits,
        predict_masks_seconds=predict_seconds,
        all_token_upsample_seconds=upsample_seconds,
    )


def select_token_logits(tokens: AllTokenMasks, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure gather; it cannot trigger a prompt/image/decoder operation."""
    if indices.ndim != 1 or len(indices) != tokens.upsampled_logits.size(0):
        raise ValueError("NuSet token indices must have one entry per prompt")
    batch = torch.arange(len(indices), device=indices.device)
    return tokens.upsampled_logits[batch, indices], tokens.predicted_iou[batch, indices]


def token0_view(tokens: AllTokenMasks) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The exact selector used by MaskDecoder.forward(multimask_output=False)."""
    count = tokens.upsampled_logits.size(0)
    indices = torch.zeros(count, dtype=torch.long, device=tokens.upsampled_logits.device)
    logits, predicted_iou = select_token_logits(tokens, indices)
    return logits, predicted_iou, indices
