"""Frozen baseline-v1 loading and integrity checks for NuSet Stage 0."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from nuset.audit.data import BASELINE_V1_TNBC_SHA256, sha256_file


@dataclass
class FrozenNuSetBundle:
    point_net: Any
    point_encoder: Any
    net: Any
    texture_memory_bank: list[Any]
    device: torch.device


def module_state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_frozen_bundle(config_path: Path, sam_config: str, checkpoint: Path, device: torch.device) -> FrozenNuSetBundle:
    if sha256_file(checkpoint) != BASELINE_V1_TNBC_SHA256:
        raise ValueError("NuSet requires the frozen TNBC StainPMS baseline-v1 checkpoint SHA256")
    from mmengine.config import Config
    from sam2_train.build_sam import build_sam2
    from sam2_train.modeling.dpa_p2pnet import build_model

    config = Config.fromfile(str(config_path))
    point_net, point_encoder = build_model(config, enable_quality_head=False)
    payload = torch.load(checkpoint, map_location="cpu")
    if "model1" not in payload or "model" not in payload:
        raise ValueError("NuSet checkpoint must contain model1 and model states")
    point_net.load_state_dict(payload["model1"], strict=True)
    net = build_sam2(sam_config, str(checkpoint), device=device, mode="eval")
    point_net.to(device).eval()
    point_encoder.to(device).eval()
    net.eval()
    for parameter in list(point_net.parameters()) + list(point_encoder.parameters()) + list(net.parameters()):
        parameter.requires_grad_(False)
        parameter.grad = None
    return FrozenNuSetBundle(
        point_net=point_net,
        point_encoder=point_encoder,
        net=net,
        texture_memory_bank=list(payload.get("texture_memory_bank_list", []) or []),
        device=device,
    )
