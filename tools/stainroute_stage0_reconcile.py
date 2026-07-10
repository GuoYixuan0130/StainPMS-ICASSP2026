"""Reconcile main-evaluation, artifact-analysis, and factorized PQ metrics.

Run this only after each frozen baseline evaluation has written its artifact
directory.  The input specification intentionally records every selection
made before reading test metrics: checkpoint, split, overlap, NMS threshold,
seed, and full evaluation command.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stainroute.oracle import pq_factorized
from tools.analyze_eval_artifacts import analyze_pair, summarize


METRIC_KEYS = ("dice1", "dice2", "aji", "aji_p", "dq", "sq", "pq")
REQUIRED_RUN_KEYS = (
    "dataset",
    "method",
    "artifact_dir",
    "checkpoint_path",
    "split",
    "overlap",
    "nms_threshold",
    "seed",
    "command",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return None


def _parse_main_stdout(path: Path | None) -> dict[str, float] | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"main_stdout_path does not exist: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    number = r"([-+]?\d+(?:\.\d+)?)"
    pattern = re.compile(
        r"split:\s+\S+\s+epoch:.*?"
        + rf"dice1:\s*{number}\s+dice2:\s*{number}\s+"
        + rf"aji:\s*{number}\s+aji_p:\s*{number}\s+"
        + rf"dq:\s*{number}\s+sq:\s*{number}\s+pq:\s*{number}",
        flags=re.DOTALL,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        raise ValueError(f"No final main.py metric line found in {path}")
    values = [float(value) / 100.0 for value in matches[-1].groups()]
    return dict(zip(METRIC_KEYS, values, strict=True))


def _load_main_metrics(artifact_dir: Path, stdout_path: Path | None) -> tuple[dict[str, float] | None, str | None]:
    """Load exact metrics written by main.py, with stdout parsing as fallback."""

    summary_path = artifact_dir / "main_eval_metrics.json"
    if summary_path.is_file():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"Invalid metrics payload in {summary_path}")
        missing = [key for key in METRIC_KEYS if key not in metrics]
        if missing:
            raise ValueError(f"Missing metrics in {summary_path}: {', '.join(missing)}")
        return ({key: float(metrics[key]) for key in METRIC_KEYS}, str(summary_path))
    metrics = _parse_main_stdout(stdout_path)
    return metrics, str(stdout_path) if metrics is not None and stdout_path is not None else None


def _artifact_rows(artifact_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, float]]:
    gt_paths = sorted(artifact_dir.glob("*_gt.npy"))
    if not gt_paths:
        raise FileNotFoundError(f"No '*_gt.npy' files under {artifact_dir}")

    rows: list[dict[str, Any]] = []
    factorized_values: list[float] = []
    for gt_path in gt_paths:
        stem = gt_path.name[: -len("_gt.npy")]
        pred_path = artifact_dir / f"{stem}_pred.npy"
        if not pred_path.is_file():
            raise FileNotFoundError(f"Missing prediction for {gt_path.name}: {pred_path}")
        gt = np.load(gt_path)
        pred = np.load(pred_path)
        rows.append(
            analyze_pair(
                stem,
                gt,
                pred,
                match_iou=0.5,
                near_low=0.3,
                weak_high=0.6,
                overlap_frac=0.1,
            )
        )
        factorized_values.append(pq_factorized(gt, pred, match_iou=0.5))

    analysis = summarize(rows)
    factorized = {"pq": float(np.mean(factorized_values))}
    return rows, analysis, factorized


def _as_run_record(spec: dict[str, Any], tolerance: float) -> dict[str, Any]:
    missing = [key for key in REQUIRED_RUN_KEYS if key not in spec]
    if missing:
        raise ValueError(f"Run spec is missing required keys: {', '.join(missing)}")

    artifact_dir = Path(spec["artifact_dir"])
    checkpoint_path = Path(spec["checkpoint_path"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint_path does not exist: {checkpoint_path}")
    rows, artifact_summary, factorized = _artifact_rows(artifact_dir)
    stdout_path = Path(spec["main_stdout_path"]) if spec.get("main_stdout_path") else None
    main_metrics, main_metrics_source = _load_main_metrics(artifact_dir, stdout_path)
    artifact_metrics = artifact_summary["mean_metrics"]

    differences: dict[str, float] = {}
    if main_metrics is not None:
        for key in METRIC_KEYS:
            differences[f"main_vs_artifact_{key}"] = float(
                main_metrics[key] - artifact_metrics[key]
            )
        differences["main_vs_factorized_pq"] = float(main_metrics["pq"] - factorized["pq"])
    differences["artifact_vs_factorized_pq"] = float(
        artifact_metrics["pq"] - factorized["pq"]
    )

    compatible = all(abs(value) <= tolerance for value in differences.values())
    return {
        "git_sha": _git_sha(),
        "dataset": str(spec["dataset"]),
        "method": str(spec["method"]),
        "command": str(spec["command"]),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "split": str(spec["split"]),
        "overlap": int(spec["overlap"]),
        "nms_threshold": int(spec["nms_threshold"]),
        "seed": int(spec["seed"]),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_images": len(rows),
        "main_metrics_source": main_metrics_source,
        "metrics": {
            "main_evaluation": main_metrics,
            "artifact_analysis": artifact_metrics,
            "pq_factorized": factorized["pq"],
        },
        "differences": differences,
        "metric_consistent": compatible,
        "expected_metrics": spec.get("expected_metrics"),
        "canonical": bool(spec.get("canonical", False)),
    }


def _write_csv(records: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "dataset",
        "method",
        "split",
        "checkpoint_path",
        "checkpoint_sha256",
        "overlap",
        "nms_threshold",
        "seed",
        "num_images",
        "canonical",
        "metric_consistent",
    ]
    for source in ("main_evaluation", "artifact_analysis"):
        fieldnames.extend(f"{source}_{key}" for key in METRIC_KEYS)
    fieldnames.extend(("pq_factorized", "main_vs_artifact_pq", "main_vs_factorized_pq", "artifact_vs_factorized_pq"))

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {key: record[key] for key in fieldnames if key in record}
            metrics = record["metrics"]
            for source in ("main_evaluation", "artifact_analysis"):
                source_metrics = metrics[source] or {}
                for key in METRIC_KEYS:
                    row[f"{source}_{key}"] = source_metrics.get(key)
            row["pq_factorized"] = metrics["pq_factorized"]
            for key in (
                "main_vs_artifact_pq",
                "main_vs_factorized_pq",
                "artifact_vs_factorized_pq",
            ):
                row[key] = record["differences"].get(key)
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path, help="JSON run specification")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--tolerance", default=2.0e-6, type=float)
    args = parser.parse_args()

    payload = json.loads(args.spec.read_text(encoding="utf-8"))
    run_specs = payload.get("runs")
    if not isinstance(run_specs, list) or not run_specs:
        raise ValueError("Spec must contain a non-empty 'runs' list")

    records = [_as_run_record(run_spec, float(args.tolerance)) for run_spec in run_specs]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "baseline_manifest.json"
    metrics_path = args.out_dir / "baseline_metrics.csv"
    manifest = {
        "git_sha": _git_sha(),
        "tolerance": float(args.tolerance),
        "runs": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_csv(records, metrics_path)

    print(json.dumps({"manifest": str(manifest_path), "metrics": str(metrics_path), "runs": records}, indent=2))
    return 0 if all(record["metric_consistent"] for record in records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
