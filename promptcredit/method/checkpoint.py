"""Strict old-checkpoint compatibility for the optional quality head."""

from __future__ import annotations

from typing import Any

import torch


QUALITY_HEAD_PREFIX = "quality_head."


def load_point_checkpoint_compat(model: torch.nn.Module, state_dict: dict[str, Any]) -> dict[str, list[str]]:
    """Allow only deterministic quality-head missing keys from a legacy checkpoint."""
    result = model.load_state_dict(state_dict, strict=False)
    missing = sorted(result.missing_keys)
    unexpected = sorted(result.unexpected_keys)
    allowed_missing = sorted(name for name in model.state_dict() if name.startswith(QUALITY_HEAD_PREFIX))
    if missing not in ([], allowed_missing) or unexpected:
        raise RuntimeError(
            "Checkpoint incompatibility: "
            f"missing={missing}, allowed_quality_head_missing={allowed_missing}, unexpected={unexpected}"
        )
    return {"missing_keys": missing, "unexpected_keys": unexpected}
