"""Train only NuRank's shared frozen-feature ranker for the fixed 30 epochs."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--stage-dir", type=Path, required=True); parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda"); parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    from nurank.model.training import train_nurank
    result = train_nurank(train_cache_dir=args.stage_dir / "cache" / "train", development_cache_dir=args.stage_dir / "cache" / "development", out_dir=args.stage_dir / "training", device=__import__("torch").device(args.device), batch_size=args.batch_size)
    message = {"checkpoint": str(result.checkpoint_path), "curves": str(result.curves_path)}
    with (args.stage_dir / "stdout.log").open("a", encoding="utf-8") as handle: handle.write(json.dumps(message, sort_keys=True) + "\n")
    print(json.dumps(message, indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())
