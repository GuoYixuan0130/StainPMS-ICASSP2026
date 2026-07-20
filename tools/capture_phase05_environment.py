"""Capture the frozen Phase 0.5 runtime and dependency environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _nvidia_smi() -> list[dict[str, str]]:
    query = "name,driver_version,memory.total,memory.free"
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    keys = ["name", "driver_version", "memory_total_mib", "memory_free_mib"]
    return [
        dict(zip(keys, (value.strip() for value in line.split(","))))
        for line in output.splitlines()
        if line.strip()
    ]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def capture(pip_freeze_output: Path) -> dict[str, Any]:
    import numpy
    import scipy
    import skimage
    import torch

    freeze = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"],
        stderr=subprocess.STDOUT,
    )
    pip_freeze_output.parent.mkdir(parents=True, exist_ok=True)
    pip_freeze_output.write_bytes(freeze)
    return {
        "schema_version": 1,
        "phase": "0.5",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "skimage": skimage.__version__,
        },
        "cuda": {
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        },
        "determinism": {
            "torch_deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        },
        "nvidia_smi": _nvidia_smi(),
        "repository": {
            "branch": _git_value("branch", "--show-current"),
            "commit": _git_value("rev-parse", "HEAD"),
            "status_short": _git_value("status", "--short"),
        },
        "pip_freeze": {
            "path": str(pip_freeze_output.resolve()),
            "size_bytes": len(freeze),
            "sha256": _sha256_bytes(freeze),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pip-freeze-output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = capture(args.pip_freeze_output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        print(json.dumps({"status": "issues_found", "error": str(exc)}))
        return 2
    print(json.dumps({"status": "complete", "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
