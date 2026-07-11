"""Run the one authorized read-only NuSet Postmortem-A fusion feasibility audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("./data/tnbc"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--nurank-run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=Path("configs/splits/stainroute_tnbc.json"))
    parser.add_argument("--model-config", type=Path, default=Path("args.py"))
    parser.add_argument("--sam-config", default="sam2_hiera_l")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    from nuset.postmortem.runner import run_postmortem_fusion_audit

    report = run_postmortem_fusion_audit(
        data_root=args.data_root,
        checkpoint=args.checkpoint,
        nurank_run_dir=args.nurank_run_dir,
        out_dir=args.out_dir,
        split_manifest_path=args.split_manifest,
        config_path=args.model_config,
        sam_config=args.sam_config,
        device_name=args.device,
    )
    print(json.dumps({"verdicts": report["verdicts"], "report": str(args.out_dir / "report.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
