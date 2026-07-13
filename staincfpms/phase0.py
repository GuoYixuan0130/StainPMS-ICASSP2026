"""Command line entry points for the frozen StainCF-PMS Phase 0 protocol."""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

from .prepare import freeze_audit
from .protocol import BASE_SHA, ProtocolError, baseline_selection_payload, write_json


def _default_out() -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("logs") / "staincfpms" / "phase0" / f"{stamp}_{BASE_SHA[:7]}"


def _prepare(args: argparse.Namespace) -> None:
    out = Path(args.out).resolve() if args.out else _default_out().resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "baseline_selection.json", baseline_selection_payload())
    try:
        manifest = freeze_audit(
            out, args.tnbc_audit_manifest, args.tnbc_calibration_manifest,
            args.monuseg_train_images, args.monuseg_train_labels, args.monuseg_organ_map,
            args.monuseg_calibration_manifest,
        )
    except Exception:
        # Preserve the failed transform-quality artifact rather than delete evidence.
        raise
    print(manifest)


def _run(args: argparse.Namespace) -> None:
    from .audit import run_audit
    out = Path(args.out).resolve()
    if not (out / "fixed_audit_manifest.json").is_file():
        raise ProtocolError(f"fixed manifest missing; run prepare first: {out}")
    repo_root = Path(__file__).resolve().parents[1]
    tests = subprocess.run(
        [sys.executable, "-m", "unittest", "tests.test_staincfpms_phase0", "-v"],
        cwd=repo_root, capture_output=True, text=True, check=False,
    )
    (out / "tests.txt").write_text(tests.stdout + tests.stderr, encoding="utf-8")
    if tests.returncode:
        raise ProtocolError("Phase 0 tests failed; model inference is forbidden (see tests.txt)")
    run_audit(out, args.tnbc_checkpoint, args.monuseg_checkpoint, args.sam2_checkpoint, args.device)
    print(out / "report.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="freeze samples, calibration references, transforms, and quality gate")
    prepare.add_argument("--out", default="", help="optional artifact directory; default is logs/staincfpms/phase0/<timestamp>_2a1348c")
    prepare.add_argument("--tnbc-audit-manifest", required=True)
    prepare.add_argument("--tnbc-calibration-manifest", required=True)
    prepare.add_argument("--monuseg-train-images", required=True)
    prepare.add_argument("--monuseg-train-labels", required=True)
    prepare.add_argument("--monuseg-organ-map", required=True)
    prepare.add_argument("--monuseg-calibration-manifest", required=True)
    prepare.set_defaults(func=_prepare)
    run = commands.add_parser("run", help="run the one frozen, eval-only audit after prepare succeeds")
    run.add_argument("--out", required=True)
    run.add_argument("--tnbc-checkpoint", required=True)
    run.add_argument("--monuseg-checkpoint", required=True)
    run.add_argument("--sam2-checkpoint", required=True)
    run.add_argument("--device", default="cuda:0")
    run.set_defaults(func=_run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
