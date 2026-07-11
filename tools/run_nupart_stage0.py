"""CLI for the only authorized NuPart Stage 0 cache-only audit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
from nupart.stage0 import run_stage0


def main() -> None:
    parser = argparse.ArgumentParser(description="NuPart Stage 0: frozen cache-only ownership audit")
    parser.add_argument("--train-cache", required=True, type=Path)
    parser.add_argument("--development-cache", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path, help="TNBC root only; train_12 labels are read by cache image ID")
    parser.add_argument("--checkpoint", required=True, type=Path, help="checked by SHA256 only; never loaded")
    parser.add_argument("--baseline-maps", required=True, type=Path, help="immutable formal token-0 baseline assembly .npz")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    report = run_stage0(train_cache=args.train_cache, development_cache=args.development_cache, data_root=args.data_root, checkpoint=args.checkpoint, baseline_maps=args.baseline_maps, out_dir=args.out_dir)
    print(f"NuPart Stage 0: {report['verdict']}")


if __name__ == "__main__":
    main()
