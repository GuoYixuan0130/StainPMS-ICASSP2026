"""One-time anchored reconstruction for SemiPMS Stage 1B.

This module reconstructs the missing 720-step supervised checkpoint from the
same Stage-1 code path and six-image split.  It deliberately has no unlabeled
loader, EMA teacher, pseudo-label code, or mask pseudo loss.  The output is the
sole candidate anchor for the separately authorised Stage 1B comparison.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from semipms.guards import ImageRecord, inspect_clean_initialization, sha256_file, validate_clean_checkpoint_name, write_json
from semipms.phase0 import CANONICAL_BASELINE, StepBudgetReached, _assert_baseline, _build_models, _environment, _git, _legacy_helpers, _read_image, _run_tests, _runtime_config
from semipms.stage1 import Stage1Optimizer, _aggregate_development, _evaluate_development, _load_checkpoint, _make_labeled_loader, _new_optimizer
from semipms.stage1_guards import DEVELOPMENT_PATIENTS, Stage1AccessGuard


SOURCE_STAGE1_SHA = "aab0a3de2d01c765e7464b6f025e17f154c1d770"
SEED = 3407
CHECKPOINT_STEPS = (0, 240, 480, 720)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _state_hash(*modules: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for module in modules:
        for name, value in sorted(module.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(np.asarray(value.detach().cpu()).tobytes())
    return digest.hexdigest()


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _tensor_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, Mapping):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _checkpoint_budget_bytes(point_net: torch.nn.Module, net: torch.nn.Module) -> int:
    model_bytes = _tensor_bytes(point_net.state_dict()) + _tensor_bytes(net.state_dict())
    trainable_bytes = sum(parameter.numel() * parameter.element_size() for module in (point_net, net) for parameter in module.parameters() if parameter.requires_grad)
    # AdamW stores first and second moments for every trainable tensor. Add a
    # conservative metadata margin and reserve the temporary atomic save file.
    per_training_checkpoint = model_bytes + 2 * trainable_bytes
    return model_bytes + 3 * per_training_checkpoint + 2 * 1024**3


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    required = _tensor_bytes(payload) + 512 * 1024**2
    free = shutil.disk_usage(path.parent).free
    if free < required:
        raise RuntimeError(
            f"Insufficient free space for {path.name}: need at least {required / 1024**3:.2f} GiB for an atomic save, "
            f"have {free / 1024**3:.2f} GiB. No checkpoint was replaced."
        )
    temporary = path.with_suffix(path.suffix + ".partial")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _records_from_manifest(path: Path) -> tuple[Path, list[ImageRecord], list[ImageRecord], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    def records(name: str) -> list[ImageRecord]:
        return [ImageRecord(**item) for item in payload[name]]
    labeled, development = records("labeled"), records("development")
    if len(labeled) != 6 or {record.patient for record in labeled} != set(range(1, 7)):
        raise PermissionError("Anchor reconstruction requires exactly the original six labelled patients 1--6.")
    if {record.patient for record in development} != DEVELOPMENT_PATIENTS:
        raise PermissionError("Anchor reconstruction requires only development patients 7--8.")
    return Path(payload["data_root"]), labeled, development, payload


def _trainable_manifest(point_net: torch.nn.Module, net: torch.nn.Module) -> dict[str, Any]:
    groups = []
    for owner, module in (("point_head", point_net), ("sam2", net)):
        entries = []
        for name, parameter in module.named_parameters():
            if parameter.requires_grad:
                entries.append({"name": name, "numel": parameter.numel(), "shape": list(parameter.shape), "dtype": str(parameter.dtype)})
        groups.append({"module": owner, "trainable_parameter_count": sum(item["numel"] for item in entries), "parameters": entries})
    return {
        "trainable_groups": groups,
        "frozen_contract_for_future_stage1b": [
            "SAM2 image encoder", "SAM2 prompt encoder", "SAM2 mask decoder",
            "mask-quality/multimask modules", "static teacher", "static pseudo-cache",
        ],
        "anchor_reconstruction_note": "This supervised reconstruction reproduces original Stage-1 training; its trainable groups are recorded rather than altered.",
    }


def _checkpoint_payload(point_net, net, optimizer, texture_memory, *, step: int, official_provenance: Mapping[str, Any], command: Sequence[str]) -> dict[str, Any]:
    return {
        "model": net.state_dict(),
        "model1": point_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler_state": None,
        "grad_scaler_state": None,
        "amp": {"enabled": False, "reason": "original Stage-1 path does not instantiate autocast or GradScaler"},
        "rng_state": _rng_state(),
        "texture_memory_bank_list": list(texture_memory),
        "semipms_stage1b_anchor": {
            "step": int(step), "seed": SEED, "source_stage1_sha": SOURCE_STAGE1_SHA,
            "canonical_baseline": CANONICAL_BASELINE, "official_checkpoint_sha256": official_provenance["sha256"],
            "command": list(command), "selection_rule": "one fixed 720-step supervised reconstruction; no development-based checkpoint selection",
        },
        # Compatibility metadata lets the original Stage-1 loader rebuild the
        # exact model architecture at the fixed 240-step branch point.
        "semipms_stage1": {
            "optimizer_steps": int(step), "role": "stage1b_anchor_reconstruction",
            "initial_state_sha256": "recorded_in_checkpoint_manifest",
            "selection_rule": "one fixed 720-step supervised reconstruction",
        },
    }


def _raw_signature(record: ImageRecord, point_net, point_encoder, net, cfg, device) -> dict[str, Any]:
    """Hash coords/logits from a fixed crop and the full deployment mask map."""
    from semipms.phase0 import _infer_standard

    raw, image = _read_image(record)
    del raw
    crop = image[..., :cfg.crop_size, :cfg.crop_size].to(device)
    point_net.eval(); point_encoder.eval(); net.eval()
    with torch.inference_mode():
        outputs, _, _, _ = point_net(crop)
        mask_map = _infer_standard(image.to(device), point_net, point_encoder, net, [], cfg, device)
    def array_hash(value: np.ndarray) -> str:
        return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
    return {
        "coords_sha256": array_hash(outputs["pred_coords"].detach().cpu().numpy()),
        "logits_sha256": array_hash(outputs["pred_logits"].detach().cpu().numpy()),
        "mask_sha256": array_hash(mask_map.astype(np.int32)),
        "point_count_raw": int(outputs["pred_coords"].shape[1]),
        "assembled_instances": int(mask_map.max()),
    }


def _compare_240(
    artifact: Path,
    reference_checkpoint: Path,
    reference_artifact: Path,
    anchor_checkpoint: Path,
    args: argparse.Namespace,
    data_root: Path,
    labeled: Sequence[ImageRecord],
    development: Sequence[ImageRecord],
    cfg,
    device,
) -> dict[str, Any]:
    _, ref_point, ref_encoder, ref_net, _, ref_texture, _ = _load_checkpoint(reference_checkpoint, args, device, require_optimizer=False)
    payload = torch.load(anchor_checkpoint, map_location="cpu")
    _, anchor_point, anchor_encoder, anchor_net, _, anchor_texture, _ = _load_checkpoint(anchor_checkpoint, args, device, require_optimizer=False)
    guard = Stage1AccessGuard()
    ref_metrics = _evaluate_development(development, guard, ref_point, ref_encoder, ref_net, ref_texture, cfg, device, method="reference_240", step=240)
    anchor_metrics = _evaluate_development(development, guard, anchor_point, anchor_encoder, anchor_net, anchor_texture, cfg, device, method="anchor_240", step=240)
    ref_curve_path = reference_artifact / "training_curve.partial.csv"
    reference_curve = []
    if ref_curve_path.is_file():
        import csv
        reference_curve = [row for row in csv.DictReader(ref_curve_path.open(encoding="utf-8")) if row.get("method") == "Shared-Warmup"]
    anchor_curve = json.loads((artifact / "training_curve.json").read_text(encoding="utf-8"))
    return {
        "reference_checkpoint": str(reference_checkpoint), "reference_checkpoint_sha256": sha256_file(reference_checkpoint),
        "anchor_checkpoint": str(anchor_checkpoint), "anchor_checkpoint_sha256": sha256_file(anchor_checkpoint),
        "reference_model_state_sha256": _state_hash(ref_point, ref_net),
        "anchor_model_state_sha256": _state_hash(anchor_point, anchor_net),
        "checkpoint_byte_identical": sha256_file(reference_checkpoint) == sha256_file(anchor_checkpoint),
        "standard_inference_signatures": {"reference": _raw_signature(labeled[0], ref_point, ref_encoder, ref_net, cfg, device), "anchor": _raw_signature(labeled[0], anchor_point, anchor_encoder, anchor_net, cfg, device)},
        "development_reference": _aggregate_development(ref_metrics),
        "development_anchor": _aggregate_development(anchor_metrics),
        "training_curve_reference_shared_warmup": reference_curve,
        "training_curve_anchor": anchor_curve,
        "anchor_checkpoint_rng_state_present": all(key in payload for key in ("rng_state", "optimizer", "scheduler_state", "grad_scaler_state")),
        "possible_nondeterminism_sources": ["CUDA kernel implementation/version", "DataLoader worker scheduling", "library version differences"],
    }


def run_anchor(args: argparse.Namespace) -> Path:
    started = time.monotonic()
    repo = Path(__file__).resolve().parents[1]
    _assert_baseline(repo)
    if _git(repo, "merge-base", "HEAD", SOURCE_STAGE1_SHA) != SOURCE_STAGE1_SHA:
        raise PermissionError("Anchor runner must descend from the original Stage-1 training code SHA.")
    if args.steps != 720:
        raise PermissionError("Anchor reconstruction is fixed at exactly 720 optimizer steps.")
    source_manifest = Path(args.stage1_manifest).resolve()
    data_root, labeled, development, source_manifest_payload = _records_from_manifest(source_manifest)
    init_checkpoint = Path(args.init_checkpoint).resolve(); validate_clean_checkpoint_name(init_checkpoint)
    official_payload = torch.load(init_checkpoint, map_location="cpu")
    provenance = inspect_clean_initialization(init_checkpoint, official_payload)
    run_id = args.run_id or f"semipms_anchor_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{_git(repo, 'rev-parse', '--short', 'HEAD')}"
    artifact = Path(args.output_root).resolve() / run_id
    if artifact.exists():
        raise FileExistsError(f"Refusing to overwrite {artifact}")
    checkpoints = artifact / "checkpoints"; checkpoints.mkdir(parents=True)
    _seed_everything(SEED)
    device = torch.device(f"cuda:{args.gpu_device}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    args_cfg, point_net, point_encoder, net = _build_models(args, official_payload, device)
    del official_payload
    budget = _checkpoint_budget_bytes(point_net, net)
    free = shutil.disk_usage(artifact).free
    if free < budget:
        raise RuntimeError(f"Anchor protocol needs about {budget / 1024**3:.1f} GiB free for four required full checkpoints; only {free / 1024**3:.1f} GiB available.")
    cfg = _runtime_config(args)
    helpers = _legacy_helpers()
    loader = _make_labeled_loader(cfg, args_cfg, data_root, labeled, args.num_workers)
    criterion, _ = helpers.build_criterion(args_cfg, device)
    optimizer = _new_optimizer(point_net, net)
    training_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    latest_texture: list[Any] = []
    command = list(sys.argv)

    def save_step(step: int, texture_memory: Sequence[Any], role: str) -> None:
        path = checkpoints / f"supervised_stainpms20_step_{step:04d}.pth"
        _atomic_torch_save(_checkpoint_payload(point_net, net, optimizer, texture_memory, step=step, official_provenance=provenance, command=command), path)
        checkpoint_rows.append({"step": step, "role": role, "path": str(path), "sha256": sha256_file(path), "model_state_sha256": _state_hash(point_net, net)})

    save_step(0, [], "official_initialization_plus_random_point_head")
    phase_endpoints = {240, 480, 720}
    global_step = 0
    epoch = 0
    while global_step < args.steps:
        current_phase_end = 240 if global_step < 240 else (480 if global_step < 480 else 720)
        def after_step(step: int, texture: Sequence[Any]) -> None:
            if step in phase_endpoints:
                save_step(step, texture, "fixed_step")
        phase_optimizer = Stage1Optimizer(optimizer, current_phase_end, start_steps=global_step, after_step=after_step)
        texture_memory: list[Any] = []
        phase_optimizer.texture_memory_bank = texture_memory
        started_epoch = time.monotonic(); completed = True
        try:
            log_info = helpers.train(cfg, point_net, point_encoder, net, loader, criterion, phase_optimizer, epoch, texture_memory, device)
        except StepBudgetReached:
            completed = False; log_info = {"partial_epoch": True}
        global_step = phase_optimizer.steps
        latest_texture = list(texture_memory)
        training_rows.append({"epoch": epoch, "optimizer_steps_completed": global_step, "epoch_completed": completed, "seconds": time.monotonic() - started_epoch, **{key: float(value) for key, value in log_info.items()}})
        (artifact / "training_curve.json").write_text(json.dumps(training_rows, indent=2), encoding="utf-8")
        if global_step == 240:
            # The historical Stage-1 schedule reset Adam at the model-only
            # 240-step branch point. Recreate that exact continuation contract.
            _, point_net, point_encoder, net, _, _, _ = _load_checkpoint(checkpoints / "supervised_stainpms20_step_0240.pth", args, device, require_optimizer=False)
            optimizer = _new_optimizer(point_net, net)
        epoch += 1
    if global_step != 720:
        raise AssertionError(f"Anchor ended at {global_step}, not 720.")
    write_json(artifact / "training_config.json", {
        "source_stage1_sha": SOURCE_STAGE1_SHA, "anchor_runner_sha": _git(repo, "rev-parse", "HEAD"), "canonical_baseline": CANONICAL_BASELINE,
        "seed": SEED, "steps": 720, "batch_size": 1, "num_workers": args.num_workers,
        "optimizer": {"type": "AdamW", "lr": 1e-4, "weight_decay": 1e-4, "scheduler": None},
        "amp": {"enabled": False, "grad_scaler": None}, "command": command,
        "labeled_stems": [record.stem for record in labeled], "unlabeled_train_access": "forbidden", "ema": "forbidden", "pseudo_labels": "forbidden",
    })
    write_json(artifact / "checkpoint_provenance.json", provenance)
    write_json(artifact / "source_stage1_manifest.json", source_manifest_payload)
    write_json(artifact / "trainable_parameter_manifest.json", _trainable_manifest(point_net, net))
    write_json(artifact / "checkpoint_manifest.json", {"checkpoints": checkpoint_rows, "scheduler_state": None, "grad_scaler_state": None, "rng_state": "embedded in every checkpoint"})
    (artifact / "environment.txt").write_text(json.dumps(_environment(), indent=2) + "\n", encoding="utf-8")
    _run_tests(artifact)
    comparison = _compare_240(artifact, Path(args.reference_240_checkpoint).resolve(), Path(args.reference_stage1_artifact).resolve(), checkpoints / "supervised_stainpms20_step_0240.pth", args, data_root, labeled, development, cfg, device)
    write_json(artifact / "comparison_step240.json", comparison)
    guard = Stage1AccessGuard()
    dev_rows = _evaluate_development(development, guard, point_net, point_encoder, net, latest_texture, cfg, device, method="anchor_supervised_stainpms20", step=720)
    from semipms.phase0 import _csv
    _csv(artifact / "training_curve.csv", training_rows); _csv(artifact / "per_image_metrics.csv", dev_rows)
    dev_aggregate = _aggregate_development(dev_rows); _csv(artifact / "per_patient_metrics.csv", dev_aggregate)
    final = [row for row in dev_aggregate if row["level"] == "all"][0]
    finite = all(math.isfinite(float(value)) for row in training_rows for value in row.values() if isinstance(value, (int, float)))
    recovery = _load_checkpoint(checkpoints / "supervised_stainpms20_step_0720.pth", args, device, require_optimizer=True)
    recoverable = all(module is not None for module in recovery[1:4])
    eligible = bool(finite and recoverable and float(final["pq"]) > 0.0)
    report = {
        "phase": "Anchored SemiPMS Stage 1B anchor reconstruction", "git_sha": _git(repo, "rev-parse", "HEAD"), "source_stage1_sha": SOURCE_STAGE1_SHA,
        "canonical_baseline": CANONICAL_BASELINE, "checkpoint_provenance": provenance, "checkpoint_manifest": "checkpoint_manifest.json",
        "step240_comparison": comparison, "development_step720": final, "anchor_eligible": eligible,
        "eligibility_conditions": {"finite_training": finite, "recoverable_checkpoint": recoverable, "pq_nonzero": float(final["pq"]) > 0.0},
        "access": {"labeled_patients": [1, 2, 3, 4, 5, 6], "development_patients": [7, 8], "unlabeled_train": "not opened", "patients_9_to_11": "forbidden", "monuseg": "forbidden"},
        "stop_condition": "Anchor reconstruction complete. If anchor_eligible is true, it is the sole permitted initialization for the authorised Stage 1B paths; otherwise stop without Stage 1B.",
        "runtime_seconds": time.monotonic() - started,
    }
    write_json(artifact / "report.json", report)
    with (artifact / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in sorted(item for item in artifact.rglob("*") if item.is_file() and item.name != "SHA256SUMS"):
            handle.write(f"{sha256_file(path)}  {path.relative_to(artifact).as_posix()}\n")
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One-time supervised 720-step anchor reconstruction for Anchored SemiPMS Stage 1B")
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--stage1-manifest", required=True)
    parser.add_argument("--reference-240-checkpoint", required=True)
    parser.add_argument("--reference-stage1-artifact", required=True)
    parser.add_argument("--output-root", default="logs/semipms/stage1b_anchor_reconstruction")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--sam-config", default="sam2_hiera_l")
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=720)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    artifact = run_anchor(build_parser().parse_args(argv))
    print(f"SemiPMS Stage 1B anchor reconstruction complete: {artifact}")
    return 0
