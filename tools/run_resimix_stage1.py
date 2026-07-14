"""Fail-closed AutoDL driver for the single authorized ResiMix-PMS Stage-1 run."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resimixpms.experiment import require_sha256, sha256_file, write_json  # noqa: E402
from resimixpms.manifests import (  # noqa: E402
    load_allowed_image_names,
    validate_manifest_patient_isolation,
)
from resimixpms.protocol import ProtocolError, derive_monuseg_lite_protocol  # noqa: E402


CANONICAL_SHA = "2a1348cb7a1158a6f77aae2f92c168f9552d8068"
FORMAL_BRANCH = "research/resimix_pms"
FORMAL_OVERLAP = {"tnbc": 32, "monuseg_lite": 92}
FROZEN_MONUSEG_BUNDLE = Path(
    "/root/autodl-tmp/projects/StainPMS-ICASSP2026/.setpms/logs/setpms/"
    "stage1_dual_dev/20260714_172429_6a40ce194788"
).resolve()
FORMAL_SAM_CONFIG = "sam2_hiera_l"
FORMAL_SEED = 3407
FORMAL_COMMON = {
    "seed": FORMAL_SEED,
    "tta": False,
    "batch_size": 1,
    "nms": 12,
    "texture": True,
    "context": True,
    "crop_size": 256,
    "load": "unclockwise",
    "epochs": 10,
    "lr": 1e-5,
    "lr_min": 1e-6,
    "optimizer": "AdamW",
    "weight_decay": 1e-4,
    "schedule": "cosine",
    "pms_loss_coef": 0.5,
    "pms_residual_mask_weight": 0.3,
    "pms_preserve_loss_coef": 1.0,
    "pms_object_weight": 1.0,
    "overlap_by_dataset": FORMAL_OVERLAP,
}
EVALUATION_SCHEDULES = {"tnbc": (0, 2, 4, 6, 8, 10), "monuseg_lite": (0, 5, 10)}
EXPECTED_EVALUATION_ITEMS = {"tnbc": 7, "monuseg_lite": 12}
FORMAL_MONUSEG_SELECTORS = {
    "train_images": "monuseg_lite_manifest.json#/train_files",
    "development_images": "monuseg_lite_manifest.json#/holdout_files",
    "train_crops": "monuseg_lite_manifest.json#/crop_indices",
    "evaluation_patches": "monuseg_lite_patches.json#/patches",
}


def _run(command: list[str], *, stdout=None) -> None:
    subprocess.run(
        [str(item) for item in command],
        cwd=ROOT,
        check=True,
        stdout=stdout,
        stderr=subprocess.STDOUT if stdout is not None else None,
    )


def _git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def _require_clean_formal_checkout() -> str:
    branch = _git("branch", "--show-current")
    if branch != FORMAL_BRANCH:
        raise RuntimeError(f"formal ResiMix must run on {FORMAL_BRANCH}, not {branch!r}")
    status = _git("status", "--porcelain")
    if status:
        raise RuntimeError("formal ResiMix checkout is dirty; commit code before running")
    subprocess.run(["git", "merge-base", "--is-ancestor", CANONICAL_SHA, "HEAD"], cwd=ROOT, check=True)
    introduced = [item for item in _git("rev-list", "--reverse", f"{CANONICAL_SHA}..HEAD").splitlines() if item]
    if not introduced:
        raise RuntimeError("formal ResiMix branch must contain its committed implementation after the canonical SHA")
    first_parents = _git("rev-list", "--parents", "-n", "1", introduced[0]).split()
    if len(first_parents) != 2 or first_parents[1] != CANONICAL_SHA:
        raise RuntimeError("research/resimix_pms must begin directly from the approved canonical SHA")
    if _git("rev-list", "--merges", f"{CANONICAL_SHA}..HEAD"):
        raise RuntimeError("formal ResiMix history may not merge another route after the canonical SHA")
    return _git("rev-parse", "HEAD")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _assert_root_fields(name: str, data: Mapping[str, Any]) -> None:
    required = {
        "data_path", "checkpoint_path", "checkpoint_sha256",
        "train_image_root", "train_label_root", "test_image_root", "test_label_root",
    }
    missing = sorted(key for key in required if not data.get(key))
    if missing:
        raise ValueError(f"{name} is missing explicit isolated roots/inputs: {missing}")
    if int(data.get("overlap", FORMAL_OVERLAP[name])) != FORMAL_OVERLAP[name]:
        raise ValueError(f"{name} overlap is fixed to canonical {FORMAL_OVERLAP[name]}")
    if str(data.get("sam_config", FORMAL_SAM_CONFIG)) != FORMAL_SAM_CONFIG:
        raise ValueError(f"{name} sam_config is fixed to {FORMAL_SAM_CONFIG}")


def _read_spec(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("stage specification must be a JSON object")
    if payload.get("canonical_sha") != CANONICAL_SHA:
        raise ValueError(f"spec canonical_sha must be exactly {CANONICAL_SHA}")
    if int(payload.get("seed", FORMAL_SEED)) != FORMAL_SEED:
        raise ValueError("formal ResiMix seed is fixed at 3407")
    datasets = payload.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != {"tnbc", "monuseg_lite"}:
        raise ValueError("spec must contain exactly tnbc and monuseg_lite datasets")
    for name, raw_data in datasets.items():
        if not isinstance(raw_data, dict):
            raise ValueError(f"{name} dataset definition must be an object")
        _assert_root_fields(name, raw_data)
    tnbc = datasets["tnbc"]
    if not tnbc.get("train_manifest") or not tnbc.get("test_manifest"):
        raise ValueError("TNBC requires explicit source train/test manifests")
    if any(tnbc.get(key) for key in ("train_crop_manifest", "eval_crop_manifest", "frozen_bundle", "frozen_protocol")):
        raise ValueError("TNBC formal protocol does not admit crop or frozen-MoNuSeg overrides")
    mono = datasets["monuseg_lite"]
    if not mono.get("frozen_bundle") or not isinstance(mono.get("frozen_protocol"), dict):
        raise ValueError("MoNuSeg-Lite requires frozen_bundle and explicit frozen_protocol selectors")
    if Path(mono["frozen_bundle"]).resolve() != FROZEN_MONUSEG_BUNDLE:
        raise ValueError(f"MoNuSeg-Lite frozen_bundle is fixed to {FROZEN_MONUSEG_BUNDLE}")
    if any(mono.get(key) for key in ("train_manifest", "test_manifest", "train_crop_manifest", "eval_crop_manifest")):
        raise ValueError("MoNuSeg-Lite run manifests must be derived only from the validated frozen bundle")
    if mono["frozen_protocol"] != FORMAL_MONUSEG_SELECTORS:
        raise ValueError("MoNuSeg-Lite frozen selectors differ from the sealed canonical manifest/patch protocol")
    if Path(mono["train_image_root"]).resolve() != Path(mono["test_image_root"]).resolve() or Path(mono["train_label_root"]).resolve() != Path(mono["test_label_root"]).resolve():
        raise ValueError("MoNuSeg-Lite development must use the admitted train roots, never an official-test root")
    return payload


def _copy_file(source: str | Path, destination: Path, label: str) -> dict[str, str]:
    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(f"{label} is missing: {source_path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite copied frozen input: {destination}")
    digest = sha256_file(source_path)
    shutil.copy2(source_path, destination)
    copied = sha256_file(destination)
    if copied != digest:
        raise RuntimeError(f"copy checksum mismatch for {label}")
    return {"source": str(source_path), "copy": str(destination), "sha256": digest}


def _prepare_tnbc_inputs(data: dict[str, Any], artifact: Path) -> dict[str, Any]:
    train_source, test_source = Path(data["train_manifest"]), Path(data["test_manifest"])
    train_rows = validate_manifest_patient_isolation(train_source, range(1, 7), {9, 10, 11})
    test_rows = validate_manifest_patient_isolation(test_source, {7, 8}, {9, 10, 11})
    train_patients = {int(row["patient_id"]) for row in train_rows}
    development_patients = {int(row["patient_id"]) for row in test_rows}
    if train_patients != set(range(1, 7)):
        raise ValueError(f"TNBC train manifest must include exactly patients 1--6, got {sorted(train_patients)}")
    if development_patients != {7, 8}:
        raise ValueError(f"TNBC development manifest must include exactly patients 7--8, got {sorted(development_patients)}")
    if len(test_rows) != EXPECTED_EVALUATION_ITEMS["tnbc"]:
        raise ValueError(f"TNBC Full-Dev must contain exactly 7 images, got {len(test_rows)}")
    inputs = artifact / "frozen_tnbc"
    train_copy = _copy_file(train_source, inputs / "train_manifest.json", "TNBC train manifest")
    test_copy = _copy_file(test_source, inputs / "development_manifest.json", "TNBC development manifest")
    data["train_manifest"], data["test_manifest"] = train_copy["copy"], test_copy["copy"]
    manifest = {
        "train": train_copy,
        "development": test_copy,
        "train_patients": sorted(train_patients),
        "development_patients": sorted(development_patients),
        "forbidden_patients": [9, 10, 11],
        "development_image_count": len(test_rows),
        "development_records": test_rows,
    }
    write_json(artifact / "tnbc_data_manifest.json", manifest)
    return manifest


def _prepare_monuseg_inputs(data: dict[str, Any], artifact: Path) -> dict[str, Any]:
    try:
        derived = derive_monuseg_lite_protocol(
            data["frozen_bundle"], data["frozen_protocol"], artifact / "frozen_monuseg_lite"
        )
    except (ProtocolError, ValueError) as exc:
        raise RuntimeError("frozen MoNuSeg-Lite protocol cannot be derived; stop without reselecting patches") from exc
    for key in ("train_manifest", "test_manifest", "train_crop_manifest", "eval_crop_manifest"):
        data[key] = derived[key]
    return derived["provenance"]


def _dataset_args(dataset_name: str, data: Mapping[str, Any]) -> list[str]:
    args = [
        "--dataset", "monuseg", "--data_path", str(data["data_path"]),
        "--sam_ckpt", str(data["checkpoint_path"]), "--sam_config", FORMAL_SAM_CONFIG,
        "--seed", str(FORMAL_SEED), "--b", "1", "--num_workers", "0",
        "--crop_size", "256", "--load", "unclockwise", "--texture", "--context", "--test_nms_thr", "12",
        "--overlap", str(FORMAL_OVERLAP[dataset_name]), "--data_identity", dataset_name,
        "--train_manifest", str(data["train_manifest"]), "--test_manifest", str(data["test_manifest"]),
        "--train_image_root", str(data["train_image_root"]), "--train_label_root", str(data["train_label_root"]),
        "--test_image_root", str(data["test_image_root"]), "--test_label_root", str(data["test_label_root"]),
    ]
    if data.get("train_crop_manifest"):
        args.extend(["--train_crop_manifest", str(data["train_crop_manifest"])])
    if data.get("eval_crop_manifest"):
        args.extend(["--eval_crop_manifest", str(data["eval_crop_manifest"])])
    if dataset_name == "tnbc":
        args.extend([
            "--train_allowed_patient_ids", "1,2,3,4,5,6",
            "--test_allowed_patient_ids", "7,8", "--forbidden_patient_ids", "9,10,11",
        ])
    return args


def _run_unit_tests(artifact: Path) -> None:
    command = [
        sys.executable, "-m", "unittest", "-v",
        "tests.test_resimix_transplant", "tests.test_resimix_donor", "tests.test_resimix_coverage",
        "tests.test_resimix_runtime", "tests.test_resimix_dataset", "tests.test_resimix_metrics",
        "tests.test_resimix_protocol", "tests.test_resimix_report",
    ]
    output = artifact / "unit_tests.txt"
    with output.open("w", encoding="utf-8") as handle:
        _run(command, stdout=handle)
    text = output.read_text(encoding="utf-8").lower()
    if "skipped" in text:
        raise RuntimeError("a formal ResiMix unit test was skipped; the AutoDL environment is incomplete")


def _training_command(
    dataset_name: str,
    data: Mapping[str, Any],
    coverage_manifest: Path,
    run_dir: Path,
    evaluation_epochs: tuple[int, ...],
    *,
    resimix_config: Path | None = None,
) -> list[str]:
    command = [sys.executable, "main.py", *_dataset_args(dataset_name, data)]
    command.extend([
        "--epochs", "10", "--lr", "1e-5", "--lr_min", "1e-6", "--weight_decay", "1e-4",
        "--use_pms", "--pms_loss_coef", "0.5", "--pms_residual_mask_weight", "0.3",
        "--pms_preserve_loss_coef", "1.0", "--pms_object_weight", "1.0",
        "--baseline_masks_dir", str(coverage_manifest.parent), "--coverage_manifest", str(coverage_manifest),
        "--evaluation_epochs", ",".join(map(str, evaluation_epochs)), "--save_eval_checkpoints",
        "--artifact_dir", str(run_dir), "--per_image_metrics_path", str(run_dir / "per_image.csv"),
    ])
    if resimix_config is not None:
        command.extend(["--resimix_enabled", "--resimix_config", str(resimix_config)])
    if "--tta" in command:
        raise AssertionError("formal command must never enable TTA")
    return command


def _command_fingerprint(command: list[str]) -> str:
    omitted_values = {"--artifact_dir", "--per_image_metrics_path", "--resimix_config"}
    normalized: list[str] = []
    index = 0
    while index < len(command):
        item = command[index]
        if item == "--resimix_enabled":
            index += 1
            continue
        if item in omitted_values:
            index += 2
            continue
        normalized.append(item)
        index += 1
    encoded = json.dumps(normalized, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    import hashlib
    return hashlib.sha256(encoded).hexdigest()


def _combine_csv(paths: list[tuple[str, Path]], destination: Path) -> None:
    rows: list[dict[str, str]] = []
    fields: set[str] = set()
    for label, path in paths:
        if not path.is_file():
            continue
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                materialized = {"dataset": label, **row}
                rows.append(materialized)
                fields.update(materialized)
    if not rows:
        return
    ordered = ["dataset", *sorted(field for field in fields if field != "dataset")]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def _verify_run(run_dir: Path, dataset: str) -> None:
    expected_epochs = EVALUATION_SCHEDULES[dataset]
    expected_items = EXPECTED_EVALUATION_ITEMS[dataset]
    for epoch in expected_epochs:
        metric_path = run_dir / f"evaluation_epoch_{epoch:02d}.json"
        item_path = run_dir / f"per_image_epoch_{epoch:02d}.csv"
        checkpoint_path = run_dir / "Model" / f"epoch_{epoch:02d}.pth"
        if not metric_path.is_file() or not item_path.is_file() or not checkpoint_path.is_file():
            raise RuntimeError(f"{dataset} run lacks registered artifacts for epoch {epoch}")
        with item_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        names = [row.get("image", "") for row in rows]
        if len(rows) != expected_items or len(set(names)) != expected_items or any(not name for name in names):
            raise RuntimeError(f"{dataset} epoch {epoch} must contain exactly {expected_items} unique evaluation items")


def _collect_runtime_memory() -> dict[str, Any]:
    payload: dict[str, Any] = {"platform": platform.platform(), "python": sys.version}
    try:
        payload["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip().splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        payload["nvidia_smi_error"] = str(exc)
    try:
        import torch
        payload["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            payload["torch_max_memory_allocated"] = int(torch.cuda.max_memory_allocated())
            payload["torch_max_memory_reserved"] = int(torch.cuda.max_memory_reserved())
    except ModuleNotFoundError:
        payload["torch_cuda_available"] = False
    return payload


def _checkpoint_checksums(artifact: Path) -> dict[str, Any]:
    entries = []
    for dataset, epochs in EVALUATION_SCHEDULES.items():
        for method in ("static_control", "resimix"):
            for epoch in epochs:
                path = artifact / dataset / method / "Model" / f"epoch_{epoch:02d}.pth"
                entries.append({
                    "dataset": dataset, "method": method, "completed_epochs": epoch,
                    "path": str(path), "sha256": sha256_file(path),
                })
    return {"checkpoints": entries}


def _sha256sums(root: Path) -> None:
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            entries.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--artifact-root", default=ROOT / "logs" / "resimixpms" / "stage1_dual_dev", type=Path)
    options = parser.parse_args()
    spec = _read_spec(options.spec)
    head = _require_clean_formal_checkout()
    artifact = options.artifact_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{head[:12]}"
    artifact.mkdir(parents=True, exist_ok=False)

    datasets = spec["datasets"]
    write_json(artifact / "git_manifest.json", {
        "canonical_sha": CANONICAL_SHA, "head": head, "branch": FORMAL_BRANCH,
        "clean_before_run": True, "formal_common": FORMAL_COMMON,
    })
    write_json(artifact / "resimix_config.json", {"formal_common": FORMAL_COMMON, "resolved_spec": spec})
    (artifact / "environment.txt").write_text(
        subprocess.check_output([sys.executable, "--version"], text=True)
        + subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True),
        encoding="utf-8",
    )
    _run_unit_tests(artifact)
    _prepare_tnbc_inputs(datasets["tnbc"], artifact)
    mono_manifest = _prepare_monuseg_inputs(datasets["monuseg_lite"], artifact)
    checkpoint_entries = []
    for dataset, data in datasets.items():
        actual = require_sha256(data["checkpoint_path"], data["checkpoint_sha256"], f"{dataset} frozen StainPMS checkpoint")
        checkpoint_entries.append({"dataset": dataset, "path": str(data["checkpoint_path"]), "sha256": actual})
    write_json(artifact / "checkpoint_manifest.json", {"checkpoints": checkpoint_entries})
    # Root-level copies satisfy the required artifact naming while the raw
    # bundle and its original SHA256SUMS remain together under frozen inputs.
    raw_mono = artifact / "frozen_monuseg_lite" / "raw"
    shutil.copy2(raw_mono / "monuseg_lite_manifest.json", artifact / "monuseg_lite_manifest.json")
    write_json(artifact / "monuseg_lite_protocol.json", mono_manifest)
    memory = {"before": _collect_runtime_memory()}

    resolved_spec = artifact / "resolved_stage_spec.json"
    write_json(resolved_spec, spec)
    coverage_manifests: dict[str, Path] = {}
    coverage_records: dict[str, Any] = {}
    for dataset, data in datasets.items():
        build_dir = artifact / dataset / "coverage_build"
        _run([sys.executable, "tools/build_resimix_coverage.py", "--spec", str(resolved_spec), "--dataset", dataset, "--artifact-dir", str(build_dir)])
        manifest = build_dir / "static_coverage" / "coverage_manifest.json"
        if not manifest.is_file():
            raise RuntimeError(f"{dataset} static coverage builder did not seal its manifest")
        coverage_manifests[dataset] = manifest
        coverage_records[dataset] = {
            "manifest": str(manifest), "sha256": sha256_file(manifest),
            "build_record": str(build_dir / "coverage_manifest.json"),
            "build_record_sha256": sha256_file(build_dir / "coverage_manifest.json"),
        }
    write_json(artifact / "coverage_manifest.json", coverage_records)

    configs: dict[str, Path] = {}
    for dataset, data in datasets.items():
        donor_dir = artifact / dataset / "donor_bank"
        donor_command = [
            sys.executable, "tools/build_resimix_donor_bank.py", "--dataset", dataset,
            "--data-path", str(data["data_path"]), "--train-image-root", str(data["train_image_root"]),
            "--train-label-root", str(data["train_label_root"]), "--train-manifest", str(data["train_manifest"]),
            "--coverage-manifest", str(coverage_manifests[dataset]), "--output-dir", str(donor_dir),
            "--overlap", str(FORMAL_OVERLAP[dataset]),
        ]
        if data.get("train_crop_manifest"):
            donor_command.extend(["--train-crop-manifest", str(data["train_crop_manifest"])])
        _run(donor_command)
        donor_csv, host_stats = donor_dir / "donor_bank_manifest.csv", donor_dir / "host_context_statistics.json"
        config = {
            "seed": FORMAL_SEED, "augmentation_probability": 0.5,
            "active_start_epoch": 2, "active_end_epoch": 9, "dataset": dataset,
            "donor_bank_manifest": str(donor_csv), "donor_bank_manifest_sha256": sha256_file(donor_csv),
            "donor_payload_dir": str(donor_dir / "donor_payloads"),
            "host_context_statistics": str(host_stats), "host_context_statistics_sha256": sha256_file(host_stats),
            "static_coverage_manifest": str(coverage_manifests[dataset]),
            "static_coverage_manifest_sha256": sha256_file(coverage_manifests[dataset]),
            "train_manifest": str(data["train_manifest"]), "train_manifest_sha256": sha256_file(data["train_manifest"]),
            "train_crop_manifest": str(data.get("train_crop_manifest", "") or ""),
            "train_crop_manifest_sha256": sha256_file(data["train_crop_manifest"]) if data.get("train_crop_manifest") else "",
        }
        config_path = artifact / dataset / "resimix_config.json"
        write_json(config_path, config)
        configs[dataset] = config_path
        smoke_dir = artifact / dataset / "smoke"
        smoke_command = [
            sys.executable, "tools/resimix_smoke.py", "--dataset", dataset,
            "--data-path", str(data["data_path"]), "--train-manifest", str(data["train_manifest"]),
            "--test-manifest", str(data["test_manifest"]), "--coverage-manifest", str(coverage_manifests[dataset]),
            "--resimix-config", str(config_path), "--output-dir", str(smoke_dir), "--overlap", str(FORMAL_OVERLAP[dataset]),
            "--load", "unclockwise",
            "--train-image-root", str(data["train_image_root"]), "--train-label-root", str(data["train_label_root"]),
            "--test-image-root", str(data["test_image_root"]), "--test-label-root", str(data["test_label_root"]),
        ]
        if data.get("train_crop_manifest"):
            smoke_command.extend(["--train-crop-manifest", str(data["train_crop_manifest"])])
        if data.get("eval_crop_manifest"):
            smoke_command.extend(["--eval-crop-manifest", str(data["eval_crop_manifest"])])
        _run(smoke_command)

    matched_contracts = {}
    for dataset, data in datasets.items():
        control_dir, resimix_dir = artifact / dataset / "static_control", artifact / dataset / "resimix"
        control = _training_command(dataset, data, coverage_manifests[dataset], control_dir, EVALUATION_SCHEDULES[dataset])
        resimix = _training_command(dataset, data, coverage_manifests[dataset], resimix_dir, EVALUATION_SCHEDULES[dataset], resimix_config=configs[dataset])
        control_fp, resimix_fp = _command_fingerprint(control), _command_fingerprint(resimix)
        if control_fp != resimix_fp:
            raise AssertionError(f"{dataset} Control/ResiMix commands differ outside the authorized augmentation switch")
        matched_contracts[dataset] = {"control_fingerprint": control_fp, "resimix_fingerprint": resimix_fp, "matched": True}
        _run(control)
        _run(resimix)
        _verify_run(control_dir, dataset)
        _verify_run(resimix_dir, dataset)
    write_json(artifact / "matched_control_contract.json", matched_contracts)

    _run([sys.executable, "tools/summarize_resimix_stage1.py", "--artifact-dir", str(artifact)])
    _combine_csv(
        [(f"{dataset}/control", artifact / dataset / "static_control" / "training_curves.csv") for dataset in datasets]
        + [(f"{dataset}/resimix", artifact / dataset / "resimix" / "training_curves.csv") for dataset in datasets],
        artifact / "training_curves.csv",
    )
    _combine_csv([(dataset, artifact / dataset / "resimix" / "synthetic_acceptance.csv") for dataset in datasets], artifact / "synthetic_acceptance.csv")
    _combine_csv([(dataset, artifact / dataset / "donor_bank" / "donor_bank_manifest.csv") for dataset in datasets], artifact / "donor_bank_manifest.csv")
    write_json(artifact / "donor_bank_summary.json", {dataset: _read_json(artifact / dataset / "donor_bank" / "donor_bank_summary.json") for dataset in datasets})
    write_json(artifact / "host_context_statistics.json", {dataset: _read_json(artifact / dataset / "donor_bank" / "host_context_statistics.json") for dataset in datasets})
    write_json(artifact / "smoke_report.json", {dataset: _read_json(artifact / dataset / "smoke" / "smoke_report.json") for dataset in datasets})
    write_json(artifact / "baseline_equivalence.json", {dataset: _read_json(artifact / dataset / "smoke" / "baseline_equivalence.json") for dataset in datasets})
    shutil.copy2(artifact / "tnbc" / "smoke" / "synthetic_montage.png", artifact / "synthetic_montage.png")
    shutil.copy2(artifact / "tnbc" / "smoke" / "synthetic_montage.png", artifact / "tnbc_synthetic_montage.png")
    shutil.copy2(artifact / "monuseg_lite" / "smoke" / "synthetic_montage.png", artifact / "monuseg_lite_synthetic_montage.png")
    write_json(artifact / "checkpoint_checksums.json", _checkpoint_checksums(artifact))
    memory["after"] = _collect_runtime_memory()
    write_json(artifact / "runtime_memory.json", memory)
    _sha256sums(artifact)
    print(artifact)


if __name__ == "__main__":
    main()
