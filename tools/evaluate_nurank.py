"""Replay NuRank development selectors from immutable cache and finalize Stage 1."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--stage-dir", type=Path, required=True); parser.add_argument("--data-root", type=Path, default=Path("./data/tnbc")); parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--split-manifest", type=Path, default=Path("configs/splits/stainroute_tnbc.json")); parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    import torch
    from nurank.inference.replay import evaluate_cached_development
    from nurank.stage1 import finalize_stage, load_ranker_checkpoint, validate_cache_isolation
    ranker, _ = load_ranker_checkpoint(args.stage_dir / "training" / "nurank_epoch_030.pt", torch.device(args.device))
    isolation = validate_cache_isolation(args.stage_dir / "cache" / "train", args.stage_dir / "cache" / "development")
    evaluation = evaluate_cached_development(development_cache_dir=args.stage_dir / "cache" / "development", ranker=ranker, data_root=args.data_root, split_manifest_path=args.split_manifest, out_dir=args.stage_dir / "evaluation", device=torch.device(args.device))
    with (args.stage_dir / "tests.txt").open("w", encoding="utf-8") as handle:
        checked = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests/nurank", "-v"], cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if checked.returncode: raise RuntimeError("NuRank unit tests failed; evaluation artifact is not finalized")
    with (args.stage_dir / "stdout.log").open("a", encoding="utf-8") as handle: handle.write(json.dumps({"evaluation": "completed", "tests": "passed"}, sort_keys=True) + "\n")
    report = finalize_stage(stage_dir=args.stage_dir, checkpoint=args.checkpoint, cache_isolation=isolation, evaluation=evaluation, ranker_checkpoint=args.stage_dir / "training" / "nurank_epoch_030.pt")
    print(json.dumps({"verdict": report["verdict"]["verdict"], "report": str(args.stage_dir / "report.json")}, indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())
