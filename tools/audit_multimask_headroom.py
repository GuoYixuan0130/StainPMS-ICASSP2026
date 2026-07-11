"""Run NuSet Stage 0: no-training SAM2 four-token multimask headroom audit."""

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
    parser.add_argument("--checkpoint", type=Path, default=Path("../CA-SAM2-HRC/deliver_ckpts/tnbc_pms_best_e156.pth"))
    parser.add_argument("--model-config", type=Path, default=Path("args.py"))
    parser.add_argument("--sam-config", default="sam2_hiera_l")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    from nuset.audit.runner import run_stage0

    report = run_stage0(
        data_root=args.data_root,
        checkpoint=args.checkpoint,
        config_path=args.model_config,
        sam_config=args.sam_config,
        out_dir=args.out_dir,
        device_name=args.device,
    )
    print(json.dumps({"verdict": report["verdict"], "report": str(args.out_dir / "report.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
