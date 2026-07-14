"""L2-SP anchoring against a fixed continuation checkpoint."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


class L2SPAnchor:
    """Keep trainable parameters close to their fixed initial values.

    The references are copied exactly once, immediately after the warm-start
    checkpoint has been loaded.  They are intentionally kept on the model
    device so the per-step penalty does not introduce CPU/GPU transfers.
    """

    def __init__(self, modules: Iterable[nn.Module], eps: float = 1.0e-8):
        self.eps = float(eps)
        self._pairs: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
        with torch.no_grad():
            for module in modules:
                for parameter in module.parameters():
                    if parameter.requires_grad:
                        self._pairs.append((parameter, parameter.detach().clone()))

    def loss(self) -> torch.Tensor:
        if not self._pairs:
            return torch.zeros((), dtype=torch.float32)

        numerator = None
        denominator = None
        for parameter, reference in self._pairs:
            squared_delta = (parameter - reference).square().sum()
            squared_reference = reference.square().sum()
            numerator = squared_delta if numerator is None else numerator + squared_delta
            denominator = (
                squared_reference if denominator is None else denominator + squared_reference
            )
        assert numerator is not None and denominator is not None
        # Dividing both sums by the common parameter count gives the requested
        # mean(delta^2) / mean(theta0^2) exactly.
        return numerator / (denominator + self.eps)
