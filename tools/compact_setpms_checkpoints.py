"""Compact active SetPMS archival checkpoints without losing epoch weights.

The tool writes each compacted candidate to the system disk first, validates it,
then replaces only the corresponding generated continuation checkpoint on the
data disk.  Fixed input checkpoints and all datasets are never touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from setpms.checkpoints import compact_continuation_checkpoint


class InvalidCheckpointError(RuntimeError):
    """The source file cannot be decoded as a completed PyTorch checkpoint."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _verify_archival_payload(payload: dict, expected_epoch: int) -> None:
    if int(payload.get("epoch", -1)) != int(expected_epoch):
        raise RuntimeError("Compacted checkpoint epoch changed")
    if payload.get("checkpoint_kind") != "continuation_model_weights_fp16_archive":
        raise RuntimeError("Compacted checkpoint kind is incorrect")
    if payload.get("optimizer_state_included") is not False:
        raise RuntimeError("Compacted checkpoint retained optimizer state")
    for key in ("model", "model1"):
        if not isinstance(payload.get(key), dict) or not payload[key]:
            raise RuntimeError(f"Compacted checkpoint lacks {key}")


def _compact_one(path: Path, temp_dir: Path) -> dict:
    original_bytes = path.stat().st_size
    original_sha256 = _sha256(path)
    try:
        original = torch.load(path, map_location="cpu")
    except Exception as error:
        raise InvalidCheckpointError(f"Cannot load completed checkpoint: {error}") from error
    compact = compact_continuation_checkpoint(original)
    temporary = temp_dir / (path.name + ".compact")
    torch.save(compact, temporary)
    verified = torch.load(temporary, map_location="cpu")
    _verify_archival_payload(verified, int(compact["epoch"]))

    # The validated copy stays on the system disk until the old data-disk file
    # is removed.  That ordering frees enough data-disk space for the smaller
    # replacement and leaves a recoverable copy if cross-device copying fails.
    path.unlink()
    copied = False
    try:
        shutil.copy2(temporary, path)
        copied = True
    except Exception:
        raise RuntimeError(
            f"Replacement failed; validated emergency copy remains at {temporary}"
        )
    finally:
        if copied:
            temporary.unlink(missing_ok=True)

    return {
        "path": str(path),
        "original_bytes": original_bytes,
        "original_sha256": original_sha256,
        "compacted_bytes": path.stat().st_size,
        "compacted_sha256": _sha256(path),
        "epoch": int(compact["epoch"]),
        "format": compact["checkpoint_kind"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--temp-dir", default="/tmp/setpms_checkpoint_compact")
    parser.add_argument("--drop-invalid", action="store_true")
    options = parser.parse_args()

    artifact_root = Path(options.artifact_root).resolve()
    if not artifact_root.is_dir():
        raise FileNotFoundError(f"Artifact root is absent: {artifact_root}")
    temp_dir = Path(options.temp_dir).resolve()
    temp_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_root / "checkpoint_compaction.json"
    rows = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    completed = {row["path"] for row in rows if row.get("status") == "compacted"}

    for checkpoint in sorted(artifact_root.glob("*/Model/*.pth")):
        relative = str(checkpoint.relative_to(artifact_root))
        if relative in completed:
            continue
        try:
            row = _compact_one(checkpoint, temp_dir)
            row["status"] = "compacted"
        except InvalidCheckpointError as error:
            if not options.drop_invalid:
                raise
            checkpoint.unlink(missing_ok=True)
            row = {
                "path": relative,
                "status": "removed_invalid_partial",
                "reason": str(error),
            }
        rows.append(row)
        _write_manifest(manifest_path, rows)
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
