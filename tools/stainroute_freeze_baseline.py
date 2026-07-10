"""Write the immutable StainRoute Development Baseline v1 manifest.

This is intentionally CPU-only: it reads environment metadata, hashes the
declared checkpoint and data files, and never runs a model.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stainroute.utils import canonical_json_sha256, sha256_file


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
LABEL_EXTENSIONS = {".mat", ".npy", ".npz"}


def _run(*command: str) -> str | None:
    try:
        return subprocess.check_output(command, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _gpu_metadata() -> dict[str, Any]:
    result: dict[str, Any] = {"nvidia_smi": None, "torch_cuda": None}
    query = _run(
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader",
    )
    if query:
        result["nvidia_smi"] = [line.strip() for line in query.splitlines() if line.strip()]
    try:
        import torch

        result["torch_cuda"] = {
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
            "device_memory_bytes": [torch.cuda.get_device_properties(index).total_memory for index in range(torch.cuda.device_count())],
        }
    except Exception as exc:
        result["torch_import_error"] = repr(exc)
    return result


def _relative_file_manifest(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {root}")
    files: list[dict[str, str]] = []
    image_count = 0
    label_count = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        suffix = path.suffix.lower()
        if suffix not in IMAGE_EXTENSIONS | LABEL_EXTENSIONS:
            continue
        relative = path.relative_to(root).as_posix()
        files.append({"path": relative, "sha256": sha256_file(path)})
        image_count += int(suffix in IMAGE_EXTENSIONS)
        label_count += int(suffix in LABEL_EXTENSIONS)
    payload = {"root": str(root), "files": files, "image_file_count": image_count, "instance_label_file_count": label_count}
    payload["manifest_sha256"] = canonical_json_sha256(payload)
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Baseline config must be JSON-compatible YAML: {path}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--monuseg-root", required=True, type=Path)
    parser.add_argument("--tnbc-root", required=True, type=Path)
    parser.add_argument("--monuseg-split", required=True, type=Path)
    parser.add_argument("--tnbc-split", required=True, type=Path)
    parser.add_argument("--out", default=Path("logs/stainroute/stage1/baseline_v1_manifest.json"), type=Path)
    args = parser.parse_args()

    config = _load_json(args.config)
    checkpoints = config.get("checkpoints", {})
    if not checkpoints:
        raise ValueError("No checkpoints declared in baseline config")
    checkpoint_manifest = {}
    for key, item in checkpoints.items():
        path = Path(item["path"])
        observed = sha256_file(path)
        expected = str(item["sha256"]).lower()
        if observed.lower() != expected:
            raise RuntimeError(f"Checkpoint SHA256 mismatch for {key}: {observed} != {expected}")
        checkpoint_manifest[key] = {"path": str(path), "sha256": observed}

    split_paths = {"monuseg": args.monuseg_split, "tnbc": args.tnbc_split}
    split_manifest = {}
    for name, path in split_paths.items():
        payload = _load_json(path)
        split_manifest[name] = {"path": str(path), "sha256": sha256_file(path), "content_sha256": payload.get("content_sha256")}

    manifest = {
        "schema_version": 1,
        "baseline_name": config.get("baseline_name"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _run("git", "rev-parse", "HEAD"),
        "git_dirty_status": _run("git", "status", "--short"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "package_versions": {
            "torch": _package_version("torch"),
            "torchvision": _package_version("torchvision"),
            "numpy": _package_version("numpy"),
            "scipy": _package_version("scipy"),
            "scikit-image": _package_version("scikit-image"),
        },
        "gpu": _gpu_metadata(),
        "config": {"path": str(args.config), "sha256": sha256_file(args.config), "resolved": config},
        "checkpoints": checkpoint_manifest,
        "data": {"monuseg": _relative_file_manifest(args.monuseg_root), "tnbc": _relative_file_manifest(args.tnbc_root)},
        "splits": split_manifest,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out} (manifest_sha256={manifest['manifest_sha256']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
