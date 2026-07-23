#!/usr/bin/env python3
"""Repair missing provenance fields in locally produced formal TNBC states.

The script validates SHA256 before loading each trusted full state.  By default
it is read-only; ``--apply`` atomically writes only the adjacent JSON
declarations and never modifies checkpoint bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stainpms.formal_checkpoint_declaration import enrich_declaration_from_state


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c2-root", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    root = args.c2_root.resolve()
    checkpoints = sorted((root / "c2_ar" / "checkpoints").glob("epoch_*.pth"))
    declarations_dir = root / "c2_ar" / "checkpoint_declarations"
    if len(checkpoints) != 5:
        raise ValueError(f"expected exactly five full epoch states under {root}")

    results: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        declaration_path = declarations_dir / f"{checkpoint.stem}.json"
        if not declaration_path.is_file():
            raise ValueError(f"missing declaration: {declaration_path}")
        observed_sha = sha256_file(checkpoint)
        declaration = read_json(declaration_path)
        if declaration.get("checkpoint_sha256") != observed_sha:
            raise ValueError(f"SHA256 mismatch: {checkpoint}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(state, dict):
            raise ValueError(f"checkpoint payload must be an object: {checkpoint}")
        updated, changed = enrich_declaration_from_state(
            declaration,
            state,
            checkpoint_path=checkpoint,
            checkpoint_sha256=observed_sha,
        )
        if args.apply and changed:
            write_json_atomic(declaration_path, updated)
        results.append({
            "checkpoint": str(checkpoint),
            "declaration": str(declaration_path),
            "sha256": observed_sha,
            "changed_fields": changed,
        })
    print(json.dumps({"status": "applied" if args.apply else "validated_read_only", "states": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
