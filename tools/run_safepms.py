"""Run the only authorized SafePMS Stage 0→1 compact validation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main() -> None:
    parser = argparse.ArgumentParser(description="SafePMS anchor-constrained validation")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--coverage-dir", required=True, type=Path, help="immutable PMS coverage maps for train patients 1-6")
    parser.add_argument("--train-manifest", required=True, type=Path, help="formal cache manifest naming only patients 1-6")
    parser.add_argument("--development-manifest", required=True, type=Path, help="formal cache manifest naming only patients 7-8")
    parser.add_argument("--continuation-config", "--pms-config", dest="continuation_config", required=True, type=Path, help="immutable recovered e156 PMS and runtime settings JSON")
    parser.add_argument("--args-config", type=Path, default=Path("args.py"))
    parser.add_argument("--stage0-out", required=True, type=Path)
    parser.add_argument("--stage1-out", required=True, type=Path)
    parser.add_argument("--num-workers", type=int, default=0, help="fixed to zero for paired augmentation determinism")
    parser.add_argument("--lr", type=float, help="only use when the continuation LR is uniquely recovered")
    args = parser.parse_args()
    if args.num_workers != 0:
        parser.error("SafePMS fixes num-workers=0 to preserve the paired augmentation stream")
    if args.stage0_out.exists() or args.stage1_out.exists():
        parser.error("SafePMS refuses to overwrite either requested artifact directory")
    from safepms.runner import run_stage0, run_stage1

    stage0 = run_stage0(args_config=args.args_config, pms_config=args.continuation_config, data_root=args.data_root, checkpoint=args.checkpoint, coverage_dir=args.coverage_dir, train_manifest=args.train_manifest, development_manifest=args.development_manifest, out_dir=args.stage0_out, b=None, num_workers=args.num_workers)
    print(f"SafePMS Stage 0: {stage0['verdict']}")
    if stage0["verdict"] != "GO":
        return
    stage1 = run_stage1(args_config=args.args_config, pms_config=args.continuation_config, data_root=args.data_root, checkpoint=args.checkpoint, coverage_dir=args.coverage_dir, train_manifest=args.train_manifest, development_manifest=args.development_manifest, out_dir=args.stage1_out, b=None, num_workers=args.num_workers, lr=args.lr)
    print(f"SafePMS Stage 1: {stage1['verdict']}")


if __name__ == "__main__":
    main()
