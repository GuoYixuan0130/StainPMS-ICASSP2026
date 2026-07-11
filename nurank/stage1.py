"""Stage-level validation, immutable manifest and preregistered NuRank verdict."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
import csv
from pathlib import Path
from typing import Any

import torch

from nuset.audit.data import BASELINE_V1_TNBC_SHA256, sha256_file
from nurank.cache.io import cache_patient_ids, load_manifest
from nurank.model.ranker import build_ranker


def _git_sha() -> str | None:
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception: return None


def sha256_tree(root: Path) -> list[str]:
    records = []
    for path in sorted(path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"):
        records.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    return records


def write_environment(stage_dir: Path, device: torch.device) -> None:
    (stage_dir / "environment.txt").write_text("\n".join((f"git_sha={_git_sha()}", f"python={sys.version}", f"platform={platform.platform()}", f"torch={torch.__version__}", f"device={device}", "seed=3407", "nms=12", "tta=False", "texture=True", "context=True", "directional_credit=disabled", "promptcredit=terminated", "promptq=terminated", "stainroute=terminated")) + "\n", encoding="utf-8")


def validate_cache_isolation(train_cache: Path, development_cache: Path) -> dict[str, Any]:
    train, development = load_manifest(train_cache), load_manifest(development_cache)
    if train["role"] != "train" or development["role"] != "development": raise RuntimeError("NuRank cache role mismatch")
    train_patients, dev_patients = cache_patient_ids(train), cache_patient_ids(development)
    if train_patients != set(range(1, 7)) or dev_patients != {7, 8}: raise RuntimeError("NuRank fixed patient isolation violation")
    if train_patients & dev_patients or (train_patients | dev_patients) & set(range(9, 12)): raise RuntimeError("NuRank accessed a closed patient")
    if train["checkpoint_sha256"] != BASELINE_V1_TNBC_SHA256 or development["checkpoint_sha256"] != BASELINE_V1_TNBC_SHA256: raise RuntimeError("NuRank checkpoint checksum violation")
    return {"train_patients": sorted(train_patients), "development_patients": sorted(dev_patients), "test_patients_closed": True, "train_cache": str(train_cache), "development_cache": str(development_cache)}


def load_ranker_checkpoint(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "nurank_ranker_v1" or payload.get("epoch") != 30 or payload.get("seed") != 3407: raise RuntimeError("NuRank requires fixed epoch-30 ranker checkpoint")
    stats = payload["normalization"]
    ranker = build_ranker(scalar_mean=torch.as_tensor(stats["mean"]), scalar_std=torch.as_tensor(stats["std"])).to(device)
    ranker.load_state_dict(payload["model_state"], strict=True); ranker.eval()
    return ranker, payload


def stage_verdict(evaluation: dict[str, Any]) -> dict[str, Any]:
    paths = evaluation["segmentation"]["paths"]; comp = evaluation["segmentation"]["comparisons"]; ranking = evaluation["ranking"]
    nu, existing, single, oracle = paths["nurank"], paths["existing_all_pred"], paths["baseline_single"], paths["oracle_all"]
    nu_delta, existing_delta = nu["pq"] - single["pq"], existing["pq"] - single["pq"]
    oracle_delta = oracle["pq"] - single["pq"]
    recovery = comp["nurank_recovery_ratio"]
    top1 = ranking["top1_accuracy_improvement_points"]
    regret = ranking["mean_regret_reduction_fraction"]
    nondecreasing = comp["nurank"]["pq_non_decreasing_images"]
    largest = comp["nurank"]["largest_positive_image_contribution_fraction"]
    runtime = evaluation["runtime"]["full_path_runtime_overhead_ratio_vs_cached_baseline"]
    fp_proxy = ranking["unmatched_false_positive_proxy"]
    no_fp_increase = fp_proxy["nurank_selected_iou_below_0_5"] <= fp_proxy["existing_selected_iou_below_0_5"]
    stage_time = evaluation["runtime"].get("stage_wall_clock_seconds")
    conditions = {"oracle_pq_headroom_ge_0_003": oracle_delta >= .003, "nurank_delta_pq_ge_0_005": nu_delta >= .005, "nurank_vs_existing_ge_0_0015": nu["pq"] - existing["pq"] >= .0015, "recovery_ge_60pct": recovery is not None and recovery >= .60, "top1_improvement_ge_15pp": top1 >= 15.0, "regret_reduction_ge_35pct": regret is not None and regret >= .35, "aji_not_decreased": nu["aji"] >= single["aji"], "five_of_seven_pq_non_decreasing": nondecreasing >= 5, "largest_image_contribution_le_60pct": largest <= .60, "unmatched_fp_not_increased": no_fp_increase, "runtime_overhead_le_5pct": runtime is not None and runtime <= 1.05, "stage_wall_clock_le_6h": stage_time is not None and stage_time <= 21600}
    no_go = (not conditions["oracle_pq_headroom_ge_0_003"] or nu_delta < .003 or nu["pq"] <= existing["pq"] or nu["aji"] < single["aji"] or not no_fp_increase or runtime is None or runtime > 1.05 or not conditions["stage_wall_clock_le_6h"])
    strong = all(conditions.values())
    conditional = not no_go and not strong
    verdict = "STRONG GO" if strong else "CONDITIONAL" if conditional else "NO-GO"
    return {"verdict": verdict, "conditions": conditions, "observed": {"single_pq": single["pq"], "existing_all_pred_pq": existing["pq"], "nurank_pq": nu["pq"], "oracle_all_pq": oracle["pq"], "nurank_delta_pq": nu_delta, "existing_delta_pq": existing_delta, "nurank_minus_existing_pq": nu["pq"] - existing["pq"], "oracle_delta_pq": oracle_delta, "nurank_recovery_ratio": recovery, "top1_accuracy_improvement_points": top1, "mean_regret_reduction_fraction": regret, "aji_delta": nu["aji"] - single["aji"], "runtime_overhead_ratio": runtime}}


def finalize_stage(*, stage_dir: Path, checkpoint: Path, cache_isolation: dict[str, Any], evaluation: dict[str, Any], ranker_checkpoint: Path) -> dict[str, Any]:
    before = load_manifest(stage_dir / "cache" / "train")["frozen_checksums"]
    development_checks = load_manifest(stage_dir / "cache" / "development")["frozen_checksums"]
    if before["before"] != before["after"] or development_checks["before"] != development_checks["after"]: raise RuntimeError("NuRank frozen parameter checksum changed")
    clock = stage_dir / "stage_clock.json"
    if not clock.exists(): raise RuntimeError("NuRank Stage 1 stage clock is missing")
    evaluation["runtime"]["stage_wall_clock_seconds"] = time.time() - float(json.loads(clock.read_text(encoding="utf-8"))["started_at_unix"])
    runtime_path = stage_dir / "evaluation" / "runtime_summary.json"
    runtime_path.write_text(json.dumps(evaluation["runtime"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (stage_dir / "evaluation" / "runtime_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(evaluation["runtime"])); writer.writeheader(); writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value for key, value in evaluation["runtime"].items()})
    reproducibility = {"git_sha": _git_sha(), "checkpoint_sha256": sha256_file(checkpoint), "ranker_checkpoint": str(ranker_checkpoint), "ranker_checkpoint_sha256": sha256_file(ranker_checkpoint), "seed": 3407}
    report = {"title": "REPORT FOR PROJECT LEAD — NURANK STAGE 1", "verdict": stage_verdict(evaluation), "reproducibility": reproducibility, "artifact_paths": {"root": str(stage_dir), "checksums": str(stage_dir / "SHA256SUMS"), "training_curves": str(stage_dir / "training" / "training_curves.csv"), "per_image_metrics": str(stage_dir / "evaluation" / "per_image_metrics.csv"), "per_prompt_ranking": str(stage_dir / "evaluation" / "per_prompt_ranking.csv")}, "cache_and_leakage_checks": cache_isolation, "frozen_model_checksums": {"train_cache": before, "development_cache": development_checks}, "development_ranking": evaluation["ranking"], "segmentation": evaluation["segmentation"], "bootstrap": evaluation["bootstrap"], "runtime": evaluation["runtime"], "status": "Completed Stage 1 only; no TNBC test or MoNuSeg was run."}
    (stage_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (stage_dir / "checkpoint_manifest.json").write_text(json.dumps({"baseline_v1_sha256": BASELINE_V1_TNBC_SHA256, **reproducibility}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (stage_dir / "data_split_manifest.json").write_text(json.dumps(cache_isolation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (stage_dir / "manifest.json").write_text(json.dumps({"schema": "nurank_stage1_tnbc_development_v1", "report": "report.json", "cache": {"train": "cache/train/manifest.json", "development": "cache/development/manifest.json"}, "training": {"checkpoint": "training/nurank_epoch_030.pt", "curves": "training/training_curves.csv"}, "evaluation": {"per_image": "evaluation/per_image_metrics.csv", "per_prompt": "evaluation/per_prompt_ranking.csv", "confusion": "evaluation/token_confusion_matrices.csv", "bootstrap": "evaluation/bootstrap_summary.json"}, "test_command": "python -m unittest discover -s tests/nurank -v", "status": "terminal: await project-lead decision"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (stage_dir / "SHA256SUMS").write_text("\n".join(sha256_tree(stage_dir)) + "\n", encoding="utf-8")
    return report
