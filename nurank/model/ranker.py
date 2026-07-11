"""The token-shared, token-ID-free NuRank quality ranker."""

from __future__ import annotations

import torch
from torch import nn


MASK_TOKEN_DIM = 256
SCALAR_DIM = 8
FEATURE_DIM = MASK_TOKEN_DIM + SCALAR_DIM
RANKER_PARAMETER_LIMIT = 100_000


class NuRankSharedRanker(nn.Module):
    """Apply exactly the same ranker to every candidate in a prompt group."""

    def __init__(self, scalar_mean: torch.Tensor | None = None, scalar_std: torch.Tensor | None = None) -> None:
        super().__init__()
        mean = torch.zeros(SCALAR_DIM) if scalar_mean is None else scalar_mean.detach().float().reshape(SCALAR_DIM)
        std = torch.ones(SCALAR_DIM) if scalar_std is None else scalar_std.detach().float().reshape(SCALAR_DIM).clamp_min(1e-6)
        self.token_norm = nn.LayerNorm(MASK_TOKEN_DIM)
        self.register_buffer("scalar_mean", mean)
        self.register_buffer("scalar_std", std)
        self.head = nn.Sequential(nn.Linear(FEATURE_DIM, 128), nn.GELU(), nn.Linear(128, 1), nn.Sigmoid())
        if self.parameter_count() >= RANKER_PARAMETER_LIMIT:
            raise RuntimeError("NuRank ranker exceeds the preregistered 0.1M parameter limit")

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Score [batch, four tokens, 264] features without token index features."""
        if features.ndim != 3 or features.shape[-1] != FEATURE_DIM:
            raise ValueError(f"NuRank requires [groups,4,{FEATURE_DIM}] features")
        token, scalar = features[..., :MASK_TOKEN_DIM], features[..., MASK_TOKEN_DIM:]
        normalized = (scalar - self.scalar_mean) / self.scalar_std
        return self.head(torch.cat((self.token_norm(token), normalized), dim=-1)).squeeze(-1)


def build_ranker(*, scalar_mean: torch.Tensor, scalar_std: torch.Tensor, seed: int = 3407) -> NuRankSharedRanker:
    torch.manual_seed(seed)
    return NuRankSharedRanker(scalar_mean=scalar_mean, scalar_std=scalar_std)
