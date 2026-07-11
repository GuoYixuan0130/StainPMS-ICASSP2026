"""Deterministic, non-mutating evaluation snapshots for paired smoke tests."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import random
from typing import Iterator

import numpy as np
import torch


@dataclass(frozen=True)
class RngSnapshot:
    """All RNG streams which an evaluation is required to leave untouched."""

    python: object
    numpy: tuple[object, ...]
    torch_cpu: torch.Tensor
    torch_cuda: tuple[torch.Tensor, ...]


def capture_rng_snapshot() -> RngSnapshot:
    cuda_states: tuple[torch.Tensor, ...] = ()
    if torch.cuda.is_available():
        cuda_states = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    return RngSnapshot(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch_cpu=torch.get_rng_state().clone(),
        torch_cuda=cuda_states,
    )


def restore_rng_snapshot(snapshot: RngSnapshot) -> None:
    random.setstate(snapshot.python)
    np.random.set_state(snapshot.numpy)
    torch.set_rng_state(snapshot.torch_cpu)
    if snapshot.torch_cuda:
        torch.cuda.set_rng_state_all(list(snapshot.torch_cuda))


def rng_snapshots_equal(first: RngSnapshot, second: RngSnapshot) -> bool:
    return bool(
        first.python == second.python
        and first.numpy[0] == second.numpy[0]
        and np.array_equal(first.numpy[1], second.numpy[1])
        and first.numpy[2:] == second.numpy[2:]
        and torch.equal(first.torch_cpu, second.torch_cpu)
        and len(first.torch_cuda) == len(second.torch_cuda)
        and all(torch.equal(left, right) for left, right in zip(first.torch_cuda, second.torch_cuda, strict=True))
    )


def _module_training_states(modules: tuple[torch.nn.Module, ...]) -> list[tuple[torch.nn.Module, bool]]:
    states: list[tuple[torch.nn.Module, bool]] = []
    seen: set[int] = set()
    for root in modules:
        for module in root.modules():
            if id(module) not in seen:
                seen.add(id(module))
                states.append((module, bool(module.training)))
    return states


@contextmanager
def model_state_snapshot(*modules: torch.nn.Module) -> Iterator[None]:
    """Temporarily set complete modules to eval and restore modes plus RNG."""
    states = _module_training_states(tuple(modules))
    rng = capture_rng_snapshot()
    try:
        for module in modules:
            module.eval()
        yield
    finally:
        for module, was_training in states:
            module.training = was_training
        restore_rng_snapshot(rng)


@contextmanager
def evaluation_snapshot(*modules: torch.nn.Module) -> Iterator[None]:
    """Run a no-grad eval section and restore modes plus every RNG stream.

    Directly restoring each module's ``training`` flag preserves heterogeneous
    submodule states exactly; using ``root.train(...)`` would not.
    """
    with model_state_snapshot(*modules):
        with torch.no_grad():
            yield
