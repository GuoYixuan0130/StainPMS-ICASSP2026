"""Integrity-preserving enrichment for formal TNBC checkpoint declarations."""

from __future__ import annotations

from pathlib import Path
from typing import Any


REQUIRED_FORMAL_DECLARATION_FIELDS = (
    "phase",
    "protocol",
    "dataset",
    "arm",
    "epoch",
)


def enrich_declaration_from_state(
    declaration: dict[str, Any],
    state: dict[str, Any],
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
) -> tuple[dict[str, Any], list[str]]:
    """Add missing provenance fields after validating a local formal state.

    The operation never changes a field that already has a conflicting value.
    It is therefore safe for repairing declarations written by an earlier
    formal-screen implementation that omitted these fields.
    """

    if declaration.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("declaration SHA256 does not match checkpoint bytes")
    if declaration.get("dataset") != "tnbc":
        raise ValueError("only TNBC formal declarations may be enriched")
    if state.get("dataset") != "tnbc":
        raise ValueError("checkpoint state is not a TNBC formal state")

    updated = dict(declaration)
    changed: list[str] = []
    expected = {key: state.get(key) for key in REQUIRED_FORMAL_DECLARATION_FIELDS}
    for key, value in expected.items():
        if value is None:
            raise ValueError(f"checkpoint state omits required field: {key}")
        observed = updated.get(key)
        if observed is not None and observed != value:
            raise ValueError(f"declaration conflicts with checkpoint state for {key}")
        if observed is None:
            updated[key] = value
            changed.append(key)

    state_path = str(checkpoint_path.resolve())
    observed_path = updated.get("checkpoint_path")
    if observed_path is not None and str(Path(observed_path).resolve()) != state_path:
        raise ValueError("declaration checkpoint path conflicts with supplied state")
    updated["checkpoint_path"] = state_path
    if observed_path is None:
        changed.append("checkpoint_path")
    return updated, changed
