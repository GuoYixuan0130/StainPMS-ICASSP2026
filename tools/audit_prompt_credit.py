"""PromptCredit PC-Stage 0 read-only mechanism audit entry point."""

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
    parser.add_argument("--split-manifest", type=Path, default=Path("configs/splits/stainroute_tnbc.json"))
    parser.add_argument("--selection", type=Path, default=Path("configs/promptcredit/pc_stage0_tnbc_router_train_six.json"))
    parser.add_argument("--write-selection", action="store_true", help="Create/validate the six-image manifest only; opens no data, GT, model, or checkpoint.")
    parser.add_argument("--data-root", type=Path, default=Path("./data/tnbc"))
    parser.add_argument("--checkpoint", type=Path, default=Path("../CA-SAM2-HRC/deliver_ckpts/tnbc_pms_best_e156.pth"))
    parser.add_argument("--model-config", type=Path, default=Path("args.py"))
    parser.add_argument("--sam-config", default="sam2_hiera_l")
    parser.add_argument("--out-dir", type=Path, default=Path("logs/promptcredit/stage0"))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    if args.write_selection:
        from promptcredit.utils.selection import build_selection_payload

        with args.split_manifest.open(encoding="utf-8") as handle:
            payload = build_selection_payload(json.load(handle))
        if args.selection.exists():
            with args.selection.open(encoding="utf-8") as handle:
                if json.load(handle) != payload:
                    raise FileExistsError(f"Refusing to overwrite differing selection manifest: {args.selection}")
        else:
            args.selection.parent.mkdir(parents=True, exist_ok=True)
            args.selection.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"selection": str(args.selection), "image_ids": payload["image_ids"], "content_sha256": payload["content_sha256"]}, indent=2))
        return 0
    from promptcredit.audit.runner import run_stage0

    report = run_stage0(
        data_root=args.data_root,
        split_manifest_path=args.split_manifest,
        selection_path=args.selection,
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
