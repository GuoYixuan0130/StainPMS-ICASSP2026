"""Run the one authorized PromptQ TNBC scalar-only development protocol."""

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
    parser.add_argument("--split-manifest", type=Path, default=Path("configs/splits/stainroute_tnbc.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("../CA-SAM2-HRC/deliver_ckpts/tnbc_pms_best_e156.pth"))
    parser.add_argument("--model-config", type=Path, default=Path("args.py"))
    parser.add_argument("--sam-config", default="sam2_hiera_l")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    from promptcredit.promptq.runner import run_promptq_development

    report = run_promptq_development(
        data_root=args.data_root,
        split_manifest_path=args.split_manifest,
        checkpoint=args.checkpoint,
        config_path=args.model_config,
        sam_config=args.sam_config,
        out_dir=args.out_dir,
        device_name=args.device,
    )
    print(json.dumps({"recommendation": report["recommendation"], "report": str(args.out_dir / "report.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
