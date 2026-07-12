"""Frozen-parameter, checksum, and paired-state guards for SafePMS."""

from __future__ import annotations

import hashlib
from typing import Iterable

import torch


def tensor_state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def freeze_decoder_only(net: torch.nn.Module, point_net: torch.nn.Module, point_encoder: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    for module in (net, point_net, point_encoder):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    decoder = net.sam_mask_decoder
    for parameter in decoder.parameters():
        parameter.requires_grad_(True)
    named = list(decoder.named_parameters())
    if not named:
        raise RuntimeError("SafePMS could not locate trainable net.sam_mask_decoder parameters")
    return named


def frozen_checksums(net: torch.nn.Module, point_net: torch.nn.Module, point_encoder: torch.nn.Module) -> dict[str, str]:
    decoder = net.sam_mask_decoder
    checksums = {
        "point_net": tensor_state_sha256(point_net),
        "point_encoder": tensor_state_sha256(point_encoder),
        "sam2_without_decoder": _net_without_decoder_sha256(net, decoder),
    }
    for name in ("image_encoder", "sam_prompt_encoder", "memory_encoder", "memory_attention"):
        module = getattr(net, name, None)
        if module is not None:
            checksums[f"sam2_{name}"] = tensor_state_sha256(module)
    return checksums


def _net_without_decoder_sha256(net: torch.nn.Module, decoder: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(net.state_dict().items()):
        if name.startswith("sam_mask_decoder."):
            continue
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def only_decoder_changed(before: dict[str, str], after: dict[str, str]) -> bool:
    return before == after


def state_equal(left: torch.nn.Module, right: torch.nn.Module) -> bool:
    for first, second in zip(left.state_dict().values(), right.state_dict().values(), strict=True):
        if not torch.equal(first.detach().cpu(), second.detach().cpu()):
            return False
    return True


def optimizer_state_sha256(optimizer: torch.optim.Optimizer) -> str:
    digest = hashlib.sha256()
    state = optimizer.state_dict()
    digest.update(repr(state["param_groups"]).encode("utf-8"))
    for key, values in sorted(state["state"].items(), key=lambda item: str(item[0])):
        digest.update(str(key).encode("utf-8"))
        for name, value in sorted(values.items()):
            digest.update(name.encode("utf-8"))
            if torch.is_tensor(value):
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            else:
                digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()
