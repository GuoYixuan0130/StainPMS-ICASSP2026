"""Read-only metadata audit for trusted StainPMS warm-start checkpoints.

The tool never loads a dataset or evaluates a model. It hashes checkpoint
bytes, summarizes only top-level metadata/state-dict structure, and optionally
inventories names of possible provenance files without reading their content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"expected NAME=PATH, received {value!r}")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name or not raw_path.strip():
        raise ValueError(f"expected non-empty NAME=PATH, received {value!r}")
    return name, Path(raw_path).expanduser().resolve()


def _small_metadata(value: Any, *, depth: int = 0) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if depth >= 2:
        return None
    if isinstance(value, dict) and len(value) <= 64:
        result = {}
        for key, item in value.items():
            converted = _small_metadata(item, depth=depth + 1)
            if converted is not None:
                result[str(key)] = converted
        return result or None
    if isinstance(value, (list, tuple)) and len(value) <= 32:
        converted = [_small_metadata(item, depth=depth + 1) for item in value]
        return converted if all(item is not None for item in converted) else None
    return None


def _state_dict_summary(value: Any, torch_module) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    tensors = [(str(key), item) for key, item in value.items() if torch_module.is_tensor(item)]
    if not tensors:
        return None
    dtype_counts = Counter(str(tensor.dtype) for _, tensor in tensors)
    return {
        "tensor_count": len(tensors),
        "parameter_or_buffer_elements": int(sum(tensor.numel() for _, tensor in tensors)),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "first_keys": [key for key, _ in tensors[:12]],
    }


def audit_checkpoint(
    name: str,
    path: Path,
    *,
    trusted_pickle: bool,
    expected_sha256: str | None,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha256 = sha256_file(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(f"{name} checkpoint SHA256 mismatch")
    import torch

    load_mode = "weights_only"
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
        load_mode = "legacy_torch_without_weights_only_argument"
    except Exception:
        if not trusted_pickle:
            raise
        if expected_sha256 is None:
            raise ValueError(
                f"{name}: trusted pickle fallback requires a predeclared SHA256"
            )
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        load_mode = "trusted_pickle_fallback"
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint {path} is {type(checkpoint).__name__}, expected dict")

    large_keys = {
        "model",
        "model1",
        "optimizer",
        "scheduler",
        "rng_state",
        "parameter",
        "texture_memory_bank_list",
    }
    embedded_metadata = {}
    for key, value in checkpoint.items():
        if str(key) in large_keys:
            continue
        converted = _small_metadata(value)
        if converted is not None:
            embedded_metadata[str(key)] = converted
    state_summaries = {}
    for key in ("model", "model1", "optimizer", "scheduler"):
        summary = _state_dict_summary(checkpoint.get(key), torch)
        if summary is not None:
            state_summaries[key] = summary

    return {
        "name": name,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": actual_sha256,
        "expected_sha256": expected_sha256,
        "sha256_matches_expected": expected_sha256 is None or actual_sha256 == expected_sha256,
        "load_mode": load_mode,
        "top_level_keys": sorted(str(key) for key in checkpoint),
        "embedded_epoch": (
            int(checkpoint["epoch"])
            if isinstance(checkpoint.get("epoch"), (int, float))
            else None
        ),
        "embedded_metadata": embedded_metadata,
        "state_summaries": state_summaries,
        "texture_memory_bank_count": (
            len(checkpoint.get("texture_memory_bank_list") or [])
            if isinstance(checkpoint.get("texture_memory_bank_list"), (list, tuple))
            else None
        ),
        "has_optimizer_state": "optimizer" in checkpoint,
        "has_scheduler_state": "scheduler" in checkpoint,
        "has_rng_state": "rng_state" in checkpoint,
        "embedded_manifest_evidence": any(
            "manifest" in str(key).lower() for key in checkpoint
        ),
        "embedded_command_or_config_evidence": any(
            token in str(key).lower()
            for key in checkpoint
            for token in ("command", "config", "args")
        ),
    }


def inventory_evidence_names(root: Path, *, limit: int = 500) -> dict[str, Any]:
    if not root.is_dir():
        return {"root": str(root), "status": "missing", "content_read": False, "files": []}
    excluded_dirs = {
        ".git",
        "data",
        "checkpoints",
        "deliver_ckpts",
        "predictions",
        "preds",
        "baseline_masks",
    }
    name_tokens = ("command", "config", "args", "manifest", "stdout", "train")
    suffixes = {".log", ".yaml", ".yml"}
    files = []
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name.lower() not in excluded_dirs)
        for filename in sorted(filenames):
            lower = filename.lower()
            path = Path(directory) / filename
            if not (any(token in lower for token in name_tokens) or path.suffix.lower() in suffixes):
                continue
            stat = path.stat()
            files.append(
                {
                    "path": str(path.resolve()),
                    "size_bytes": stat.st_size,
                    "modified_at_utc": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(),
                }
            )
            if len(files) >= limit:
                return {
                    "root": str(root),
                    "status": "truncated",
                    "content_read": False,
                    "limit": limit,
                    "files": files,
                }
    return {
        "root": str(root),
        "status": "complete",
        "content_read": False,
        "limit": limit,
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--expected-sha256", action="append", default=[], help="NAME=SHA256")
    parser.add_argument("--evidence-root", action="append", default=[])
    parser.add_argument("--trusted-pickle", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    expected_hashes: dict[str, str] = {}
    for value in args.expected_sha256:
        if "=" not in value:
            raise ValueError(f"expected NAME=SHA256, received {value!r}")
        name, digest = value.split("=", 1)
        expected_hashes[name] = digest.strip().lower()

    records = []
    for value in args.checkpoint:
        name, path = parse_named_path(value)
        expected = expected_hashes.get(name)
        record = audit_checkpoint(
            name,
            path,
            trusted_pickle=args.trusted_pickle,
            expected_sha256=expected,
        )
        records.append(record)

    report = {
        "schema_version": 1,
        "phase": "2A-warmstart-feasibility",
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "checkpoint metadata and provenance-filename inventory only; no dataset/model evaluation",
        "checkpoints": records,
        "evidence_inventories": [
            inventory_evidence_names(Path(value).expanduser().resolve())
            for value in args.evidence_root
        ],
    }
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite existing report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
