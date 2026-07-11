"""Build one immutable NuRank automatic-prompt cache; no model training."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("./data/tnbc")); parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True); parser.add_argument("--role", choices=("train", "development"), required=True)
    parser.add_argument("--split-manifest", type=Path, default=Path("configs/splits/stainroute_tnbc.json")); parser.add_argument("--model-config", type=Path, default=Path("args.py")); parser.add_argument("--sam-config", default="sam2_hiera_l"); parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    from nuset.audit.models import load_frozen_bundle
    from nurank.cache.builder import build_automatic_prompt_cache
    from nurank.stage1 import write_environment
    if args.stage_dir.exists() and (args.stage_dir / "cache" / args.role).exists():
        raise FileExistsError(f"NuRank immutable cache already exists: {args.stage_dir / 'cache' / args.role}")
    args.stage_dir.mkdir(parents=True, exist_ok=True)
    if not (args.stage_dir / "environment.txt").exists(): write_environment(args.stage_dir, __import__("torch").device(args.device))
    clock = args.stage_dir / "stage_clock.json"
    if clock.exists():
        started_at = float(json.loads(clock.read_text(encoding="utf-8"))["started_at_unix"])
    else:
        started_at = time.time(); clock.write_text(json.dumps({"started_at_unix": started_at, "time_cap_seconds": 21600}, indent=2) + "\n", encoding="utf-8")
    prior_stage_seconds = time.time() - started_at
    if prior_stage_seconds >= 21600: raise RuntimeError("NuRank Stage 1 fixed six GPU-hour cap is already exhausted")
    bundle = load_frozen_bundle(args.model_config, args.sam_config, args.checkpoint, __import__("torch").device(args.device))
    result = build_automatic_prompt_cache(bundle=bundle, data_root=args.data_root, split_manifest_path=args.split_manifest, role=args.role, cache_dir=args.stage_dir / "cache" / args.role, prior_stage_seconds=prior_stage_seconds)
    message = {"role": args.role, "cache_manifest": str(result.manifest_path), "elapsed_seconds": result.elapsed_seconds, "estimated_total_seconds": result.estimated_total_seconds}
    with (args.stage_dir / "stdout.log").open("a", encoding="utf-8") as handle: handle.write(json.dumps(message, sort_keys=True) + "\n")
    print(json.dumps(message, indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())
