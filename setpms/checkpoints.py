"""Space-efficient archival checkpoints for authorised SetPMS continuation nodes."""

from __future__ import annotations

from typing import Any

import torch


def _archive_tensor(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().cpu()
    if value.is_floating_point():
        return value.to(dtype=torch.float16).contiguous()
    return value.contiguous()


def _archive_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return _archive_tensor(value)
    if isinstance(value, dict):
        return {key: _archive_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_archive_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_archive_value(item) for item in value)
    return value


def compact_continuation_checkpoint(payload: dict[str, Any]) -> dict[str, Any]:
    """Retain every learned weight while dropping redundant optimizer moments.

    These files are archival epoch nodes, not live recovery snapshots.  Model
    and point-head weights are stored as CPU FP16, which ``load_state_dict``
    safely copies back into the model's FP32 parameters when needed.
    """

    required = {"model", "model1", "epoch"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Continuation checkpoint missing required keys: {missing}")
    return {
        "model": _archive_value(payload["model"]),
        "model1": _archive_value(payload["model1"]),
        "epoch": int(payload["epoch"]),
        "texture_memory_bank_list": _archive_value(
            payload.get("texture_memory_bank_list", [])
        ),
        "checkpoint_kind": "continuation_model_weights_fp16_archive",
        "optimizer_state_included": False,
    }
