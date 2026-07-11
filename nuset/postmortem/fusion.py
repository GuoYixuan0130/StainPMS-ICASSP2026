"""Preregistered, parameter-free four-token fusion operators."""

from __future__ import annotations

from itertools import product

import torch
import torch.nn.functional as F


FIXED_FUSIONS = (
    "equal_logit_mean",
    "equal_probability_mean",
    "logit_median",
    "hard_majority",
    "logit_max",
    "logit_min",
)


def upsample_logits(low_res_logits: torch.Tensor, size: int = 256) -> torch.Tensor:
    """Use the repository's fixed bilinear SAM2 upsampling contract."""
    if low_res_logits.ndim != 4:
        raise ValueError("Expected [prompts, tokens, height, width] low-resolution logits")
    return F.interpolate(low_res_logits, size=(size, size), mode="bilinear", align_corners=False)


def fixed_lowres_fusions(low_res_logits: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return all fixed logit-domain operators except final-resolution majority vote."""
    if low_res_logits.ndim != 4 or low_res_logits.size(1) != 4:
        raise ValueError("NuSet Postmortem-A requires exactly four mask token logits")
    probability = torch.sigmoid(low_res_logits).mean(dim=1).clamp(1e-6, 1 - 1e-6)
    return {
        "equal_logit_mean": low_res_logits.mean(dim=1),
        "equal_probability_mean": torch.logit(probability),
        "logit_median": low_res_logits.median(dim=1).values,
        "logit_max": low_res_logits.max(dim=1).values,
        "logit_min": low_res_logits.min(dim=1).values,
    }


def hard_majority_logits(upsampled_token_logits: torch.Tensor) -> torch.Tensor:
    """Vote on final hard masks; 2:2 ties take the token-0 pixel by preregistration."""
    if upsampled_token_logits.ndim != 4 or upsampled_token_logits.size(1) != 4:
        raise ValueError("NuSet Postmortem-A requires [prompts,4,H,W] token logits")
    hard = upsampled_token_logits > 0
    votes = hard.sum(dim=1)
    selected = torch.where(votes == 2, hard[:, 0], votes >= 3)
    # ±1 fixes the final threshold without adding a tunable threshold or score.
    return torch.where(selected, torch.ones_like(upsampled_token_logits[:, 0]), -torch.ones_like(upsampled_token_logits[:, 0]))


def fixed_fusions(low_res_logits: torch.Tensor, *, size: int = 256) -> dict[str, torch.Tensor]:
    """Return final-resolution logits for the complete fixed operator library."""
    upsampled = upsample_logits(low_res_logits, size=size)
    fused = {name: upsample_logits(value[:, None], size=size)[:, 0] for name, value in fixed_lowres_fusions(low_res_logits).items()}
    fused["hard_majority"] = hard_majority_logits(upsampled)
    return {name: fused[name] for name in FIXED_FUSIONS}


def simplex_weights() -> torch.Tensor:
    """The fixed 35 four-simplex weights on {0,.25,.5,.75,1}."""
    values = tuple(range(5))
    weights = [candidate for candidate in product(values, repeat=4) if sum(candidate) == 4]
    result = torch.tensor(weights, dtype=torch.float32) / 4.0
    if result.shape != (35, 4):
        raise RuntimeError("Expected exactly 35 preregistered convex fusion weights")
    return result


def is_one_hot(weight: torch.Tensor) -> bool:
    return bool((weight == 1).sum().item() == 1 and (weight == 0).sum().item() == 3)
