"""Run the authorized one-shot PromptQ-v2 Primary-Metric Audit on AutoDL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/tnbc"))
    parser.add_argument("--checkpoint", type=Path, default=Path("deliver_ckpts/tnbc_pms_best_e156.pth"))
    parser.add_argument("--manifest", type=Path, default=Path("configs/promptq_v2/tnbc_authorized_manifest.json"))
    parser.add_argument("--sam-config", default="sam2_hiera_l")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    from promptq_v2.runner import run_primary_metric_audit
    report = run_primary_metric_audit(data_root=args.data_root, checkpoint=args.checkpoint, manifest_path=args.manifest, out_dir=args.out_dir, sam_config=args.sam_config)
    print(json.dumps({"recommendation": report["recommendation"], "report": str(args.out_dir / "PROJECT_LEAD_REPORT.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
