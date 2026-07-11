"""PromptCredit v1 freezing, optimizer, and checksum guards."""

from __future__ import annotations

import hashlib
from typing import Any

import torch


TRAINABLE_PREFIXES = ("conv.", "deform_layer.", "reg_head.", "cls_head.", "quality_head.")
PROMPTQ_TRAINABLE_PREFIXES = ("quality_head.",)


def configure_promptcredit_v1_trainable(point_net: torch.nn.Module, sam2_net: torch.nn.Module) -> dict[str, Any]:
    """Freeze every non-authorized module and return the exact trainable manifest."""
    for parameter in point_net.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    for parameter in sam2_net.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    trainable: list[str] = []
    for name, parameter in point_net.named_parameters():
        if name.startswith(TRAINABLE_PREFIXES):
            parameter.requires_grad_(True)
            trainable.append(name)
    if not any(name.startswith("quality_head.") for name in trainable):
        raise ValueError("PromptCredit v1 requires an enabled quality_head")
    return {
        "trainable_parameter_names": trainable,
        "trainable_parameter_count": int(sum(parameter.numel() for parameter in point_net.parameters() if parameter.requires_grad)),
        "quality_head_parameter_count": int(sum(parameter.numel() for name, parameter in point_net.named_parameters() if name.startswith("quality_head."))),
        "frozen_sam2_parameter_count": int(sum(parameter.numel() for parameter in sam2_net.parameters())),
    }


def configure_promptq_trainable(point_net: torch.nn.Module, sam2_net: torch.nn.Module) -> dict[str, Any]:
    """Freeze every inherited model parameter; PromptQ trains only quality_head."""
    for parameter in point_net.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    for parameter in sam2_net.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    trainable: list[str] = []
    for name, parameter in point_net.named_parameters():
        if name.startswith(PROMPTQ_TRAINABLE_PREFIXES):
            parameter.requires_grad_(True)
            trainable.append(name)
    if not trainable:
        raise ValueError("PromptQ requires an enabled quality_head")
    if any(module.__class__.__name__ == "Dropout" for module in point_net.quality_head.modules()):
        raise ValueError("PromptQ quality_head must not contain dropout")
    return {
        "trainable_parameter_names": trainable,
        "trainable_parameter_count": int(sum(parameter.numel() for parameter in point_net.parameters() if parameter.requires_grad)),
        "quality_head_parameter_count": int(sum(parameter.numel() for parameter in point_net.quality_head.parameters())),
        "frozen_point_parameter_count": int(sum(parameter.numel() for parameter in point_net.parameters() if not parameter.requires_grad)),
        "frozen_sam2_parameter_count": int(sum(parameter.numel() for parameter in sam2_net.parameters())),
    }


def module_state_sha256_excluding(module: torch.nn.Module, excluded_prefixes: tuple[str, ...]) -> str:
    """Hash model state except an explicit trainable-prefix allowlist."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        if name.startswith(excluded_prefixes):
            continue
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def module_state_sha256(module: torch.nn.Module) -> str:
    """Hash state tensors deterministically for frozen-parameter invariance checks."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        # Support scalar state buffers (for example BatchNorm counters) too.
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def optimizer_excludes_frozen(optimizer: torch.optim.Optimizer) -> bool:
    return all(parameter.requires_grad for group in optimizer.param_groups for parameter in group["params"])


def frozen_parameters_have_no_grad(module: torch.nn.Module) -> bool:
    return all(parameter.grad is None for parameter in module.parameters())
