"""Recover only the interrupted MoNuSeg-Lite tail of a formal ResiMix run.

This tool never rebuilds coverage/donors/smoke and never opens TNBC images.
It starts a fresh, deterministic MoNuSeg-Lite Static-PMS arm only because the
original arm was interrupted by a full filesystem before epoch 5 could be
serialized.  The original partial directory remains untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resimixpms.experiment import require_sha256, sha256_file, write_json  # noqa: E402
from tools.run_resimix_stage1 import (  # noqa: E402
    CANONICAL_SHA,
    EVALUATION_SCHEDULES,
    EXPECTED_EVALUATION_ITEMS,
    FORMAL_BRANCH,
    _checkpoint_checksums,
    _collect_runtime_memory,
    _combine_csv,
    _command_fingerprint,
    _require_clean_formal_checkout,
    _run,
    _sha256sums,
    _training_command,
    _verify_run,
)


RECOVERY_CONTROL = "static_control_recovery"
RECOVERY_MANIFEST = "recovery_manifest.json"
FINAL_OUTPUTS = (
    "report.json", "tnbc_metrics.csv", "monuseg_lite_metrics.csv", "tnbc_per_image.csv",
    "tnbc_per_patient.csv", "monuseg_lite_per_patch.csv", "resimix_training_augmentation.csv",
    "training_curves.csv", "synthetic_acceptance.csv", "donor_bank_manifest.csv",
    "donor_bank_summary.json", "host_context_statistics.json", "smoke_report.json",
    "baseline_equivalence.json", "checkpoint_checksums.json", "runtime_memory.json", "SHA256SUMS",
)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _inside(root: Path, path: Path, label: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"{label} must be a frozen file inside this artifact: {path}") from exc


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"required {label} is missing: {path}")
    return path


def _require_json_equal(left: Path, right: Path, label: str) -> None:
    if _read_json(left) != _read_json(right):
        raise RuntimeError(f"recovered {label} differs from the interrupted deterministic arm")


def _unit_tests(artifact: Path) -> None:
    command = [
        sys.executable, "-m", "unittest", "-v",
        "tests.test_resimix_transplant", "tests.test_resimix_donor", "tests.test_resimix_coverage",
        "tests.test_resimix_runtime", "tests.test_resimix_dataset", "tests.test_resimix_metrics",
        "tests.test_resimix_protocol", "tests.test_resimix_report", "tests.test_resimix_offline",
        "tests.test_resimix_driver_contract",
    ]
    output = artifact / "unit_tests_recovery.txt"
    if output.exists():
        raise FileExistsError(f"recovery unit-test record already exists: {output}")
    with output.open("w", encoding="utf-8") as handle:
        _run(command, stdout=handle)
    if "skipped" in output.read_text(encoding="utf-8").lower():
        raise RuntimeError("a recovery unit test was skipped")


def _load_inputs(artifact: Path) -> tuple[dict[str, Any], Path, Path]:
    source = _read_json(_require_file(artifact / "git_manifest.json", "source git manifest"))
    if source.get("canonical_sha") != CANONICAL_SHA or source.get("branch") != FORMAL_BRANCH:
        raise RuntimeError("source artifact is not a formal ResiMix artifact from the approved baseline")
    spec = _read_json(_require_file(artifact / "resolved_stage_spec.json", "resolved stage spec"))
    try:
        data = dict(spec["datasets"]["monuseg_lite"])
    except (KeyError, TypeError) as exc:
        raise RuntimeError("resolved artifact lacks frozen MoNuSeg-Lite inputs") from exc
    for field in ("train_manifest", "test_manifest", "train_crop_manifest", "eval_crop_manifest"):
        frozen = _require_file(Path(data[field]), f"frozen MoNuSeg-Lite {field}")
        _inside(artifact, frozen, field)
    require_sha256(data["checkpoint_path"], data["checkpoint_sha256"], "MoNuSeg-Lite initialization checkpoint")
    coverage = _require_file(
        artifact / "monuseg_lite" / "coverage_build" / "static_coverage" / "coverage_manifest.json",
        "sealed MoNuSeg-Lite coverage manifest",
    )
    recorded_coverage = _read_json(_require_file(artifact / "coverage_manifest.json", "coverage record"))["monuseg_lite"]
    if sha256_file(coverage) != recorded_coverage["sha256"]:
        raise RuntimeError("sealed MoNuSeg-Lite coverage manifest checksum changed")
    config = _require_file(artifact / "monuseg_lite" / "resimix_config.json", "sealed ResiMix config")
    smoke = _read_json(_require_file(artifact / "monuseg_lite" / "smoke" / "smoke_report.json", "MoNuSeg-Lite smoke report"))
    if smoke.get("passed") is not True:
        raise RuntimeError("MoNuSeg-Lite smoke did not pass; recovery is forbidden")
    return data, coverage, config


def _verify_existing(artifact: Path) -> None:
    _verify_run(artifact / "tnbc" / "static_control", "tnbc")
    _verify_run(artifact / "tnbc" / "resimix", "tnbc")
    partial = artifact / "monuseg_lite" / "static_control"
    for epoch in (0, 5):
        _require_file(partial / f"evaluation_epoch_{epoch:02d}.json", f"interrupted Static-PMS epoch-{epoch} metrics")
        _require_file(partial / f"per_image_epoch_{epoch:02d}.csv", f"interrupted Static-PMS epoch-{epoch} per-item metrics")


def _commands(artifact: Path, data: Mapping[str, Any], coverage: Path, config: Path) -> tuple[list[str], list[str]]:
    control = _training_command(
        "monuseg_lite", data, coverage, artifact / "monuseg_lite" / RECOVERY_CONTROL,
        EVALUATION_SCHEDULES["monuseg_lite"],
    )
    resimix = _training_command(
        "monuseg_lite", data, coverage, artifact / "monuseg_lite" / "resimix",
        EVALUATION_SCHEDULES["monuseg_lite"], resimix_config=config,
    )
    if "--save_eval_checkpoints" in control or "--save_eval_checkpoints" in resimix:
        raise AssertionError("recovery must not retain full evaluation checkpoints")
    if _command_fingerprint(control) != _command_fingerprint(resimix):
        raise AssertionError("recovery Control/ResiMix commands differ outside the augmentation switch")
    return control, resimix


def _write_recovery_manifest(artifact: Path, head: str, control: list[str], resimix: list[str], status: str) -> None:
    write_json(artifact / RECOVERY_MANIFEST, {
        "reason": "filesystem full while saving interrupted MoNuSeg-Lite Static-PMS epoch-5 checkpoint",
        "source_static_control": str(artifact / "monuseg_lite" / "static_control"),
        "recovery_static_control": str(artifact / "monuseg_lite" / RECOVERY_CONTROL),
        "recovery_resimix": str(artifact / "monuseg_lite" / "resimix"),
        "recovery_code_head": head,
        "status": status,
        "control_command": control,
        "resimix_command": resimix,
        "control_fingerprint": _command_fingerprint(control),
        "resimix_fingerprint": _command_fingerprint(resimix),
        "full_evaluation_checkpoints": False,
    })


def _assert_recovered_prefix(artifact: Path) -> None:
    old, recovered = artifact / "monuseg_lite" / "static_control", artifact / "monuseg_lite" / RECOVERY_CONTROL
    _verify_run(recovered, "monuseg_lite")
    for epoch in (0, 5):
        _require_json_equal(old / f"evaluation_epoch_{epoch:02d}.json", recovered / f"evaluation_epoch_{epoch:02d}.json", f"Static-PMS epoch-{epoch} evaluation")


def _assert_new_final_outputs(artifact: Path) -> None:
    existing = [name for name in FINAL_OUTPUTS if (artifact / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing final artifact files: {existing}")


def _finalize(artifact: Path, control_dir: Path) -> None:
    _assert_new_final_outputs(artifact)
    _run([
        sys.executable, "tools/summarize_resimix_stage1.py", "--artifact-dir", str(artifact),
        "--monuseg-control-dir", str(control_dir),
    ])
    _combine_csv(
        [
            ("tnbc/control", artifact / "tnbc" / "static_control" / "training_curves.csv"),
            ("tnbc/resimix", artifact / "tnbc" / "resimix" / "training_curves.csv"),
            ("monuseg_lite/control", control_dir / "training_curves.csv"),
            ("monuseg_lite/resimix", artifact / "monuseg_lite" / "resimix" / "training_curves.csv"),
        ],
        artifact / "training_curves.csv",
    )
    _combine_csv(
        [(dataset, artifact / dataset / "resimix" / "synthetic_acceptance.csv") for dataset in ("tnbc", "monuseg_lite")],
        artifact / "synthetic_acceptance.csv",
    )
    _combine_csv(
        [(dataset, artifact / dataset / "donor_bank" / "donor_bank_manifest.csv") for dataset in ("tnbc", "monuseg_lite")],
        artifact / "donor_bank_manifest.csv",
    )
    for name, filename in (
        ("donor_bank_summary.json", "donor_bank_summary.json"),
        ("host_context_statistics.json", "host_context_statistics.json"),
        ("smoke_report.json", "smoke_report.json"),
        ("baseline_equivalence.json", "baseline_equivalence.json"),
    ):
        write_json(artifact / name, {dataset: _read_json(artifact / dataset / ("smoke" if "smoke" in filename or "equivalence" in filename else "donor_bank") / filename) for dataset in ("tnbc", "monuseg_lite")})
    for source, destination in (
        (artifact / "tnbc" / "smoke" / "synthetic_montage.png", artifact / "synthetic_montage.png"),
        (artifact / "tnbc" / "smoke" / "synthetic_montage.png", artifact / "tnbc_synthetic_montage.png"),
        (artifact / "monuseg_lite" / "smoke" / "synthetic_montage.png", artifact / "monuseg_lite_synthetic_montage.png"),
    ):
        shutil.copy2(source, destination)
    write_json(artifact / "checkpoint_checksums.json", _checkpoint_checksums(artifact))
    write_json(artifact / "runtime_memory.json", {"recovery_after": _collect_runtime_memory()})
    _sha256sums(artifact)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("control", "resimix-finalize"))
    options = parser.parse_args()
    artifact = options.artifact_dir.resolve()
    head = _require_clean_formal_checkout()
    data, coverage, config = _load_inputs(artifact)
    _verify_existing(artifact)
    control, resimix = _commands(artifact, data, coverage, config)
    recovery_control, recovery_resimix = artifact / "monuseg_lite" / RECOVERY_CONTROL, artifact / "monuseg_lite" / "resimix"

    if options.phase == "control":
        if recovery_control.exists() or recovery_resimix.exists() or (artifact / RECOVERY_MANIFEST).exists():
            raise FileExistsError("recovery was already started; refusing to rerun an authorized arm")
        _unit_tests(artifact)
        _write_recovery_manifest(artifact, head, control, resimix, "control_started")
        _run(control)
        _assert_recovered_prefix(artifact)
        _write_recovery_manifest(artifact, head, control, resimix, "control_completed")
        print(recovery_control)
        return

    recovery = _read_json(_require_file(artifact / RECOVERY_MANIFEST, "recovery manifest"))
    if recovery.get("status") != "control_completed" or recovery.get("recovery_code_head") != head:
        raise RuntimeError("recovery Control is incomplete or the checked-out code changed")
    if recovery_resimix.exists():
        raise FileExistsError("recovery ResiMix arm already exists; refusing to rerun it")
    _assert_recovered_prefix(artifact)
    _write_recovery_manifest(artifact, head, control, resimix, "resimix_started")
    _run(resimix)
    _verify_run(recovery_resimix, "monuseg_lite")
    _finalize(artifact, recovery_control)
    _write_recovery_manifest(artifact, head, control, resimix, "completed")
    _sha256sums(artifact)
    print(artifact)


if __name__ == "__main__":
    main()
