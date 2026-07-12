"""SafePMS Stage 0/1 runner; no inference-path or data-split expansion."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from safepms.data import DEVELOPMENT_PATIENTS, TRAIN_PATIENTS, PatientBalancedSampler, load_cache_manifest_ids, patient_of
from safepms.gradient import GradientController
from safepms.guards import freeze_decoder_only, frozen_checksums, optimizer_state_sha256, state_equal, tensor_state_sha256


SEED = 3407
CHECKPOINT_SHA256 = "44a3cb3e93051301d789e44f93769588abfa727d7a174b0270f55305ef023781"
STAGE0_VALID_TARGET = 36
STAGE0_MIN_VALID = 24
CONTINUATION_KEYS = (
    "pms_loss_coef", "pms_focal_weight", "pms_dice_weight", "pms_iou_weight",
    "pms_object_weight", "pms_residual_mask_weight", "pms_preserve_loss_coef",
    "pms_gt_match_radius", "pms_baseline_prompts", "pms_preserve_max_prompts",
    "texture", "context", "overlap", "test_nms_thr", "test_filtering", "load",
    "crop_batch", "clip_grad", "texture_memory_bank_size",
    "context_memory_bank_size", "context_atten_k",
)


class ProtocolInvalid(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _coverage_sha256(coverage_dir: Path, image_ids: list[str]) -> str:
    """Hash exactly the named train coverage artifacts; never enumerate data."""
    digest = hashlib.sha256()
    for image_id in image_ids:
        path = coverage_dir / f"{image_id}.npy"
        if not path.is_file():
            raise ProtocolInvalid(f"Missing frozen coverage artifact for {image_id}: {path}")
        digest.update(image_id.encode("utf-8"))
        digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    def convert(item):
        if isinstance(item, Path): return str(item)
        if isinstance(item, np.ndarray): return item.tolist()
        if isinstance(item, np.generic): return item.item()
        if torch.is_tensor(item): return item.detach().cpu().tolist()
        if isinstance(item, dict): return {str(key): convert(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)): return [convert(value) for value in item]
        return item
    path.write_text(json.dumps(convert(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})


def _checksums(out_dir: Path) -> None:
    lines = [f"{_sha256(path)}  {path.relative_to(out_dir).as_posix()}" for path in sorted(out_dir.rglob("*")) if path.is_file() and path.name != "SHA256SUMS"]
    (out_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_test_suite(out_dir: Path) -> None:
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests/safepms", "-v"]
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
    transcript = "$ " + " ".join(command) + "\n\n" + result.stdout + result.stderr
    (out_dir / "tests.txt").write_text(transcript, encoding="utf-8")
    if result.returncode:
        raise ProtocolInvalid("SafePMS unit tests failed; GPU execution is not authorized")


def _git_sha() -> str | None:
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception: return None


def _import_baseline():
    """Baseline modules parse argv on import; isolate the SafePMS CLI first."""
    old = sys.argv
    try:
        sys.argv = [sys.argv[0]]
        import cfg
        from main import load_ca_sam2_point_head_checkpoint, load_ca_sam2_texture_bank
        from mmengine import Config
        from run.dataset.monuseg import MONUSEG
        from run.run_on_epoch import train_on_epoch, validation_on_epoch
        from run.utils import get_network
        from sam2_train.modeling.criterion import build_criterion
        from sam2_train.modeling.dpa_p2pnet import build_model
        from sam2_train.modeling.utils import collate_fn, set_seed
    finally:
        sys.argv = old
    return cfg, Config, MONUSEG, train_on_epoch, validation_on_epoch, get_network, build_criterion, build_model, collate_fn, set_seed, load_ca_sam2_point_head_checkpoint, load_ca_sam2_texture_bank


def _configure(*, args_config: Path, pms_config: Path, data_root: Path, checkpoint: Path, coverage_dir: Path, b: int | None, num_workers: int):
    cfg, Config, *_ = _import_baseline()
    old = sys.argv
    try:
        sys.argv = [sys.argv[0]]
        cfgs = cfg.parse_args()
    finally:
        sys.argv = old
    args = Config.fromfile(str(args_config))
    payload = json.loads(pms_config.read_text(encoding="utf-8"))
    missing = [key for key in CONTINUATION_KEYS if key not in payload]
    if missing: raise ProtocolInvalid(f"immutable PMS config lacks fields: {missing}")
    for key in CONTINUATION_KEYS:
        if key.startswith("pms_"):
            setattr(args.criterion, key, payload[key])
    for key in ("texture", "context", "overlap", "test_nms_thr", "load", "crop_batch", "clip_grad", "texture_memory_bank_size", "context_memory_bank_size", "context_atten_k"):
        setattr(cfgs, "b" if key == "crop_batch" else key, payload[key])
    args.test.nms_thr = int(payload["test_nms_thr"])
    args.test.filtering = bool(payload["test_filtering"])
    if b is not None and int(b) != int(payload["crop_batch"]):
        raise ProtocolInvalid("--crop-batch disagrees with the immutable continuation configuration")
    if float(args.criterion.pms_loss_coef) <= 0:
        raise ProtocolInvalid("SafePMS requires the frozen nonzero PMS loss coefficient")
    cfgs.data_path, cfgs.sam_ckpt = str(data_root), str(checkpoint)
    cfgs.dataset, cfgs.distributed, cfgs.tta = "monuseg", "none", False
    cfgs.use_pms, cfgs.pms_self_bootstrap = True, False
    cfgs.baseline_masks_dir, cfgs.iterative_baseline_refresh_every = str(coverage_dir), 0
    cfgs.pms_loss_coef = float(args.criterion.pms_loss_coef)
    cfgs.pms_object_weight = float(args.criterion.pms_object_weight)
    cfgs.pms_residual_mask_weight = float(args.criterion.pms_residual_mask_weight)
    cfgs.pms_preserve_loss_coef = float(args.criterion.pms_preserve_loss_coef)
    cfgs.pms_gt_match_radius = int(args.criterion.pms_gt_match_radius)
    cfgs.pms_baseline_prompts = bool(args.criterion.pms_baseline_prompts)
    cfgs.pms_preserve_max_prompts = int(args.criterion.pms_preserve_max_prompts)
    cfgs.num_workers = int(num_workers)
    return args, cfgs, payload


def _build_bundle(args, cfgs, device):
    _, _, _, _, _, get_network, build_criterion, build_model, _, _, load_point, load_bank = _import_baseline()
    net = get_network(cfgs, cfgs.net, use_gpu=cfgs.gpu, gpu_device=device, distribution="none")
    point_net, point_encoder = build_model(args)
    point_net.to(device); point_encoder.to(device)
    load_point(cfgs, point_net)
    texture_template = load_bank(cfgs)
    named_decoder = freeze_decoder_only(net, point_net, point_encoder)
    criterion, _ = build_criterion(args, device)
    return {"net": net, "point_net": point_net, "point_encoder": point_encoder, "criterion": criterion, "texture_template": texture_template, "named_decoder": named_decoder}


def _make_dataset(args, cfgs, image_ids: list[str], *, training: bool):
    _, _, MONUSEG, *_ = _import_baseline()
    # MONUSEG's historical transform builder pops ``type`` from config
    # dictionaries.  Each paired loader therefore receives an independent
    # immutable-equivalent copy rather than inheriting a mutated transform.
    return MONUSEG(cfgs, copy.deepcopy(args), cfgs.data_path, cfgs.load, mode="train" if training else "test", image_ids=image_ids, source_split="train")


def _loader(dataset, sampler=None):
    collate_fn = _import_baseline()[8]
    return DataLoader(dataset, batch_size=1, sampler=sampler, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)


def _evaluate(bundle, cfgs, args, dev_loader, device) -> tuple[dict[str, float], list[dict[str, Any]], float, int]:
    validation_on_epoch = _import_baseline()[4]
    rows: list[dict[str, Any]] = []
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats(device)
    metrics = validation_on_epoch(cfgs, args, dev_loader, 0, bundle["point_net"], bundle["point_encoder"], bundle["net"], cfgs.load, args.data.post.iou_threshold, copy.deepcopy(bundle["texture_template"]), device, per_image_records=rows)
    names = ("dice", "dice2", "aji", "aji_plus", "dq", "sq", "pq")
    summary = {name: float(value) for name, value in zip(names, metrics, strict=True)}
    summary["matched_mean_iou"] = summary["sq"]
    return summary, rows, time.perf_counter() - started, int(torch.cuda.max_memory_allocated(device))


def _bootstrap(rows: list[dict[str, Any]], left: str, right: str, *, seed: int = SEED) -> dict[str, Any]:
    by_path = {path: {row["image_id"]: row for row in rows if row["path"] == path} for path in (left, right)}
    ids = sorted(by_path[left]); rng = np.random.default_rng(seed); result = {"seed": seed, "resamples": 2000, "image_count": len(ids), "metrics": {}}
    for metric in ("dice", "aji", "aji_plus", "dq", "sq", "pq"):
        delta = np.asarray([by_path[right][item][metric] - by_path[left][item][metric] for item in ids], dtype=np.float64)
        draws = np.asarray([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(2000)])
        result["metrics"][metric] = {"mean_delta": float(delta.mean()), "ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]}
    return result


def _distribution(values: list[float]) -> dict[str, float | None]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if not len(finite):
        return {"count": 0, "mean": None, "p10": None, "p25": None, "median": None, "p75": None, "p90": None}
    return {"count": int(len(finite)), "mean": float(finite.mean()), "p10": float(np.quantile(finite, .10)), "p25": float(np.quantile(finite, .25)), "median": float(np.median(finite)), "p75": float(np.quantile(finite, .75)), "p90": float(np.quantile(finite, .90))}


def _stage0_gate(records: list[dict[str, Any]], *, nonfinite_batches: int, other_decoder_dependency_batches: int) -> tuple[bool, dict[str, bool]]:
    valid = [row for row in records if row["valid"]]
    conflicts = [row for row in valid if row["conflict"]]
    retained = [row["retained_expand_norm_ratio"] for row in conflicts]
    checks = {
        "valid_batches_ge_24": len(valid) >= STAGE0_MIN_VALID,
        "global_conflict_rate_ge_25pct": len(conflicts) / len(valid) >= .25 if valid else False,
        "conflict_patients_ge_4": len({row["patient"] for row in conflicts}) >= 4,
        "conflict_median_cosine_le_neg_005": float(np.median([row["cosine"] for row in conflicts if row["cosine"] is not None])) <= -.05 if conflicts else False,
        "retained_expand_norm_ge_20pct_for_90pct": float(np.mean(np.asarray(retained) >= .20)) >= .90 if retained else False,
        "all_gradients_finite": nonfinite_batches == 0 and all(row["finite"] for row in valid),
        "projection_dot_ge_neg_1e7": all(row["projection_dot"] >= -1e-7 for row in valid),
        "anchor_safety_contract": all(row["anchor_final_margin"] >= -1e-7 for row in valid),
        "no_other_decoder_dependency": other_decoder_dependency_batches == 0 and not any(row["other_decoder_dependency"] for row in valid),
    }
    return all(checks.values()), checks


def run_stage0(*, args_config: Path, pms_config: Path, data_root: Path, checkpoint: Path, coverage_dir: Path, train_manifest: Path, development_manifest: Path, out_dir: Path, b: int | None, num_workers: int) -> dict[str, Any]:
    if out_dir.exists(): raise FileExistsError(f"SafePMS refuses to overwrite {out_dir}")
    if _sha256(checkpoint) != CHECKPOINT_SHA256: raise ProtocolInvalid("checkpoint SHA256 mismatch")
    train_ids, dev_ids = load_cache_manifest_ids(train_manifest, role="train"), load_cache_manifest_ids(development_manifest, role="development")
    coverage_before = _coverage_sha256(coverage_dir, train_ids)
    args, cfgs, pms_payload = _configure(args_config=args_config, pms_config=pms_config, data_root=data_root, checkpoint=checkpoint, coverage_dir=coverage_dir, b=b, num_workers=num_workers)
    if not torch.cuda.is_available(): raise RuntimeError("SafePMS Stage 0 requires CUDA")
    out_dir.mkdir(parents=True); device = torch.device("cuda")
    _run_test_suite(out_dir)
    _, _, _, train_on_epoch, _, *rest = _import_baseline(); set_seed = rest[4]
    set_seed(cfgs); bundle = _build_bundle(args, cfgs, device)
    before = frozen_checksums(bundle["net"], bundle["point_net"], bundle["point_encoder"])
    decoder_before = tensor_state_sha256(bundle["net"].sam_mask_decoder)
    sampler = PatientBalancedSampler(train_ids, rounds_per_patient=6, seed=SEED)
    dataset = _make_dataset(args, cfgs, train_ids, training=True)
    loader = _loader(dataset, sampler)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    controller = GradientController(bundle["named_decoder"], mode="audit", patient_order=sampler.image_order, target_valid=STAGE0_VALID_TARGET, deadline_monotonic=time.monotonic() + 60 * 60)
    audit_optimizer = torch.optim.AdamW([p for _, p in bundle["named_decoder"]], lr=1e-5)
    train_on_epoch(cfgs, bundle["point_net"], bundle["point_encoder"], bundle["net"], loader, bundle["criterion"], audit_optimizer, 0, [], device, gradient_controller=controller, decoder_only_training=True)
    elapsed = time.perf_counter() - started
    peak_memory = int(torch.cuda.max_memory_allocated(device))
    coverage_after = _coverage_sha256(coverage_dir, train_ids)
    after = frozen_checksums(bundle["net"], bundle["point_net"], bundle["point_encoder"])
    decoder_after = tensor_state_sha256(bundle["net"].sam_mask_decoder)
    passed, checks = _stage0_gate(controller.records, nonfinite_batches=controller.nonfinite_batches, other_decoder_dependency_batches=controller.other_decoder_dependency_batches)
    checks["stage0_wall_time_le_1_gpu_hour"] = elapsed <= 60 * 60 and not controller.time_cap_exceeded
    checks["frozen_modules_unchanged"] = before == after
    checks["coverage_artifacts_unchanged"] = coverage_before == coverage_after
    checks["decoder_unchanged"] = decoder_before == decoder_after
    checks["optimizer_step_count_zero"] = int(getattr(audit_optimizer, "_safepms_step_count", 0)) == 0
    passed = passed and checks["stage0_wall_time_le_1_gpu_hour"] and checks["frozen_modules_unchanged"] and checks["coverage_artifacts_unchanged"] and checks["decoder_unchanged"] and checks["optimizer_step_count_zero"]
    _json(out_dir / "gradient_batches_manifest.json", {"seed": SEED, "train_image_ids": train_ids, "development_image_ids": dev_ids, "sampler_order": sampler.image_order, "sampler_checksum": sampler.checksum, "valid_target": STAGE0_VALID_TARGET})
    flat_rows = [{key: value for key, value in row.items() if key != "layerwise"} for row in controller.records]
    _csv(out_dir / "gradient_conflicts.csv", flat_rows)
    layer_rows = [{"batch": index, "patient": row["patient"], "layer": layer, **value} for index, row in enumerate(controller.records) for layer, value in row["layerwise"].items()]
    _csv(out_dir / "layerwise_conflicts.csv", layer_rows)
    projection = {"max_negative_projection_dot": min([row["projection_dot"] for row in controller.records], default=0.0), "max_negative_anchor_margin": min([row["anchor_final_margin"] for row in controller.records], default=0.0), "all_constraints_pass": all(row["projection_dot"] >= -1e-7 and row["anchor_final_margin"] >= -1e-7 for row in controller.records)}
    _json(out_dir / "projection_validation.json", projection)
    (out_dir / "environment.txt").write_text(f"git_sha={_git_sha()}\npython={sys.version}\nplatform={platform.platform()}\ntorch={torch.__version__}\nseed={SEED}\ncheckpoint_sha256={CHECKPOINT_SHA256}\nwall_seconds={elapsed}\npeak_memory_bytes={peak_memory}\n", encoding="utf-8")
    report = {"verdict": "GO" if passed else "NO-GO", "stage": "SafePMS Stage 0", "checks": checks, "valid_batch_count": len(controller.records), "invalid_batch_count": controller.invalid_batches, "nonfinite_batch_count": controller.nonfinite_batches, "other_decoder_dependency_batch_count": controller.other_decoder_dependency_batches, "time_cap_exceeded": controller.time_cap_exceeded, "frozen_checksums": {"before": before, "after": after, "unchanged": before == after}, "decoder_sha256": {"before": decoder_before, "after": decoder_after, "unchanged": decoder_before == decoder_after}, "coverage_sha256": {"before": coverage_before, "after": coverage_after, "unchanged": coverage_before == coverage_after}, "optimizer_step_count": int(getattr(audit_optimizer, "_safepms_step_count", 0)), "projection_validation": projection, "runtime_seconds": elapsed, "peak_memory_bytes": peak_memory, "pms_config": pms_payload, "recommendation": "Proceed to the preregistered Stage 1 only." if passed else "Stop; do not train or enter Stage 1."}
    _json(out_dir / "report.json", report); _checksums(out_dir)
    return report


def _optimizer(bundle, lr: float, weight_decay: float):
    return torch.optim.AdamW([parameter for _, parameter in bundle["named_decoder"]], lr=lr, weight_decay=weight_decay)


def _aggregate(rows: list[dict[str, Any]], path: str) -> dict[str, float]:
    selected = [row for row in rows if row["path"] == path]
    mean_keys = ("dice", "aji", "aji_plus", "dq", "sq", "pq", "matched_mean_iou")
    total_keys = ("tp", "fp", "fn")
    return {
        **{key: float(np.mean([row[key] for row in selected])) for key in mean_keys},
        **{key: int(sum(row[key] for row in selected)) for key in total_keys},
    }


def _stage1_verdict(rows: list[dict[str, Any]], safe_stats: dict[str, Any], control_seconds: float, safe_seconds: float) -> tuple[str, dict[str, bool]]:
    starting, control, safe = (_aggregate(rows, name) for name in ("starting", "control_sum", "safepms"))
    by_path = {name: {row["image_id"]: row for row in rows if row["path"] == name} for name in ("starting", "safepms")}
    deltas = {key: safe[key] - starting[key] for key in ("aji", "dq", "pq")}
    safe_control_pq = safe["pq"] - control["pq"]
    image_delta = [by_path["safepms"][key]["pq"] - by_path["starting"][key]["pq"] for key in sorted(by_path["starting"])]
    positive = sum(max(0.0, value) for value in image_delta)
    contribution = max([max(0.0, value) for value in image_delta], default=0.0) / positive if positive else 0.0
    common = {
        "aji_not_down": deltas["aji"] >= 0,
        "dq_not_down": deltas["dq"] >= 0,
        "pq_non_decreasing_images_ge_5": sum(value >= 0 for value in image_delta) >= 5,
        "contribution_le_60pct": contribution <= .60,
        "no_contract_violations": safe_stats["contract_violation_count"] == 0,
    }
    strong = {**common, "safe_vs_starting_pq_ge_003": deltas["pq"] >= .003, "safe_vs_control_pq_ge_002": safe_control_pq >= .002, "runtime_le_22x_control": safe_seconds <= 2.2 * control_seconds}
    conditional = {**common, "safe_vs_starting_pq_ge_0015": deltas["pq"] >= .0015, "safe_vs_control_pq_ge_0015": safe_control_pq >= .0015}
    details = {"strong": strong, "conditional": conditional, "delta_vs_starting": deltas, "delta_pq_vs_control": safe_control_pq, "pq_non_decreasing_images": sum(value >= 0 for value in image_delta), "largest_positive_image_contribution_fraction": contribution}
    return "STRONG GO" if all(strong.values()) else "CONDITIONAL GO" if all(conditional.values()) else "NO-GO", details


def run_stage1(*, args_config: Path, pms_config: Path, data_root: Path, checkpoint: Path, coverage_dir: Path, train_manifest: Path, development_manifest: Path, out_dir: Path, b: int | None, num_workers: int, lr: float | None = None) -> dict[str, Any]:
    """Strict paired five-epoch decoder-only Control-Sum versus SafePMS run."""
    if out_dir.exists(): raise FileExistsError(f"SafePMS refuses to overwrite {out_dir}")
    if _sha256(checkpoint) != CHECKPOINT_SHA256: raise ProtocolInvalid("checkpoint SHA256 mismatch")
    train_ids, dev_ids = load_cache_manifest_ids(train_manifest, role="train"), load_cache_manifest_ids(development_manifest, role="development")
    coverage_before = _coverage_sha256(coverage_dir, train_ids)
    args, cfgs, pms_payload = _configure(args_config=args_config, pms_config=pms_config, data_root=data_root, checkpoint=checkpoint, coverage_dir=coverage_dir, b=b, num_workers=num_workers)
    if not torch.cuda.is_available(): raise RuntimeError("SafePMS Stage 1 requires CUDA")
    out_dir.mkdir(parents=True); device = torch.device("cuda")
    _run_test_suite(out_dir)
    train_on_epoch, set_seed = _import_baseline()[3], _import_baseline()[9]
    actual_lr = float(lr) if lr is not None else 1e-5
    weight_decay = float(args.optimizer.weight_decay)
    rounds = max(sum(patient_of(item) == patient for item in train_ids) for patient in TRAIN_PATIENTS)
    sampler = PatientBalancedSampler(train_ids, rounds_per_patient=rounds, seed=SEED)
    batch_checksums = []
    for epoch in range(5): sampler.set_epoch(epoch); batch_checksums.append(sampler.checksum)
    dev_dataset, dev_loader = _make_dataset(args, cfgs, dev_ids, training=False), None
    dev_loader = _loader(dev_dataset)

    set_seed(cfgs); control = _build_bundle(args, cfgs, device); control_optimizer = _optimizer(control, actual_lr, weight_decay)
    initial_frozen = frozen_checksums(control["net"], control["point_net"], control["point_encoder"])
    control_decoder_before = tensor_state_sha256(control["net"].sam_mask_decoder)
    set_seed(cfgs)
    starting_metrics, starting_rows, starting_seconds, starting_memory = _evaluate(control, cfgs, args, dev_loader, device)
    control_start_state = {"net": tensor_state_sha256(control["net"]), "point_net": tensor_state_sha256(control["point_net"]), "point_encoder": tensor_state_sha256(control["point_encoder"]), "optimizer": optimizer_state_sha256(control_optimizer)}

    set_seed(cfgs); safe = _build_bundle(args, cfgs, device); safe_optimizer = _optimizer(safe, actual_lr, weight_decay)
    safe_decoder_before = tensor_state_sha256(safe["net"].sam_mask_decoder)
    safe_start_optimizer = optimizer_state_sha256(safe_optimizer)
    paired_init = state_equal(control["net"], safe["net"]) and state_equal(control["point_net"], safe["point_net"]) and state_equal(control["point_encoder"], safe["point_encoder"]) and optimizer_state_sha256(control_optimizer) == safe_start_optimizer
    set_seed(cfgs)
    safe_start_metrics, safe_start_rows, safe_start_seconds, safe_start_memory = _evaluate(safe, cfgs, args, dev_loader, device)
    step0_equal = starting_metrics == safe_start_metrics and starting_rows == safe_start_rows
    if not paired_init or not step0_equal: raise ProtocolInvalid("paired initialization or step-0 development equivalence failed")

    curves: list[dict[str, Any]] = []
    control_started = time.perf_counter(); set_seed(cfgs); torch.cuda.reset_peak_memory_stats(device)
    control_dataset = _make_dataset(args, cfgs, train_ids, training=True)
    control_sampler = PatientBalancedSampler(train_ids, rounds_per_patient=rounds, seed=SEED)
    control_loader = _loader(control_dataset, control_sampler)
    control_batch_checksums = []
    for epoch in range(5):
        control_sampler.set_epoch(epoch)
        control_batch_checksums.append(control_sampler.checksum)
        log = train_on_epoch(cfgs, control["point_net"], control["point_encoder"], control["net"], control_loader, control["criterion"], control_optimizer, epoch, [], device, decoder_only_training=True)
        curves.append({"path": "control_sum", "epoch": epoch + 1, **log})
    control_seconds = time.perf_counter() - control_started
    control_train_memory = int(torch.cuda.max_memory_allocated(device))
    set_seed(cfgs)
    control_metrics, control_rows, control_eval_seconds, control_memory = _evaluate(control, cfgs, args, dev_loader, device)
    control_frozen_after = frozen_checksums(control["net"], control["point_net"], control["point_encoder"])
    control_decoder_after = tensor_state_sha256(control["net"].sam_mask_decoder)

    safe_started = time.perf_counter(); set_seed(cfgs); torch.cuda.reset_peak_memory_stats(device)
    safe_dataset = _make_dataset(args, cfgs, train_ids, training=True)
    safe_sampler = PatientBalancedSampler(train_ids, rounds_per_patient=rounds, seed=SEED)
    safe_loader = _loader(safe_dataset, safe_sampler)
    safe_controllers = []
    safe_batch_checksums = []
    for epoch in range(5):
        safe_sampler.set_epoch(epoch)
        safe_batch_checksums.append(safe_sampler.checksum)
        controller = GradientController(safe["named_decoder"], mode="safe", patient_order=safe_sampler.image_order)
        log = train_on_epoch(cfgs, safe["point_net"], safe["point_encoder"], safe["net"], safe_loader, safe["criterion"], safe_optimizer, epoch, [], device, gradient_controller=controller, decoder_only_training=True)
        safe_controllers.append(controller)
        for record in controller.records:
            record["epoch"] = epoch + 1
        curves.append({"path": "safepms", "epoch": epoch + 1, **log})
    safe_seconds = time.perf_counter() - safe_started
    safe_train_memory = int(torch.cuda.max_memory_allocated(device))
    set_seed(cfgs)
    safe_metrics, safe_rows, safe_eval_seconds, safe_memory = _evaluate(safe, cfgs, args, dev_loader, device)
    safe_frozen_after = frozen_checksums(safe["net"], safe["point_net"], safe["point_encoder"])
    safe_decoder_after = tensor_state_sha256(safe["net"].sam_mask_decoder)
    coverage_after = _coverage_sha256(coverage_dir, train_ids)

    per_image = [{"path": "starting", **row} for row in starting_rows] + [{"path": "control_sum", **row} for row in control_rows] + [{"path": "safepms", **row} for row in safe_rows]
    starting_summary, control_summary, safe_summary = (_aggregate(per_image, path) for path in ("starting", "control_sum", "safepms"))
    safe_records = [record for controller in safe_controllers for record in controller.records]
    violations = sum(record["projection_dot"] < -1e-7 or record["anchor_final_margin"] < -1e-7 for record in safe_records)
    safe_stats = {
        "projection_activation_rate": float(np.mean([record["projected"] for record in safe_records])) if safe_records else 0.0,
        "trust_ratio_clipping_rate": float(np.mean([record["trust_clipped"] for record in safe_records])) if safe_records else 0.0,
        "epoch_conflict_rate": {
            str(epoch + 1): float(np.mean([record["conflict"] for record in controller.records])) if controller.records else 0.0
            for epoch, controller in enumerate(safe_controllers)
        },
        "contract_violation_count": int(violations),
        "invalid_batch_count": int(sum(controller.invalid_batches for controller in safe_controllers)),
        "nonfinite_batch_count": int(sum(controller.nonfinite_batches for controller in safe_controllers)),
        "other_decoder_dependency_batch_count": int(sum(controller.other_decoder_dependency_batches for controller in safe_controllers)),
        "cosine_distribution": _distribution([record["cosine"] for record in safe_records if record["cosine"] is not None]),
        "norm_ratio_distribution": _distribution([record["norm_expand"] / record["norm_anchor"] for record in safe_records if record["norm_anchor"] > 0]),
        "anchor_loss_change": float(np.mean([record["anchor_loss"] for record in safe_controllers[-1].records]) - np.mean([record["anchor_loss"] for record in safe_controllers[0].records])) if safe_controllers and safe_controllers[0].records and safe_controllers[-1].records else None,
        "expansion_loss_change": float(np.mean([record["expansion_loss"] for record in safe_controllers[-1].records]) - np.mean([record["expansion_loss"] for record in safe_controllers[0].records])) if safe_controllers and safe_controllers[0].records and safe_controllers[-1].records else None,
    }
    verdict, decision = _stage1_verdict(per_image, safe_stats, control_seconds, safe_seconds)
    paired_manifest = {"seed": SEED, "train_image_ids": train_ids, "development_image_ids": dev_ids, "train_manifest_sha256": _sha256(train_manifest), "development_manifest_sha256": _sha256(development_manifest), "expected_batch_order_checksums": batch_checksums, "control_batch_order_checksums": control_batch_checksums, "safepms_batch_order_checksums": safe_batch_checksums, "paired_initialization": paired_init, "step0_metrics_identical": step0_equal, "optimizer_state_identical": control_start_state["optimizer"] == safe_start_optimizer, "augmentation_seed_stream": {"seed": SEED, "num_workers": 0}, "lr": actual_lr, "lr_source": "explicit_recovered_continuation" if lr is not None else "preregistered_fallback_1e-5", "weight_decay": weight_decay, "epochs": 5, "tta": False, "checkpoint_sha256": CHECKPOINT_SHA256, "pms_config_sha256": _sha256(pms_config), "trainable_parameter_names": [name for name, _ in control["named_decoder"]]}
    frozen = {"before": initial_frozen, "control_after": control_frozen_after, "safepms_after": safe_frozen_after, "control_unchanged": initial_frozen == control_frozen_after, "safepms_unchanged": initial_frozen == safe_frozen_after, "control_decoder_sha256": {"before": control_decoder_before, "after": control_decoder_after}, "safepms_decoder_sha256": {"before": safe_decoder_before, "after": safe_decoder_after}, "control_steps": int(getattr(control_optimizer, "_safepms_step_count", 0)), "safepms_steps": int(getattr(safe_optimizer, "_safepms_step_count", 0))}
    total_stage1_seconds = control_seconds + safe_seconds + starting_seconds + safe_start_seconds + control_eval_seconds + safe_eval_seconds
    decision["paired_step_counts_equal"] = frozen["control_steps"] == frozen["safepms_steps"]
    decision["batch_order_checksums_identical"] = control_batch_checksums == safe_batch_checksums == batch_checksums
    decision["frozen_modules_unchanged"] = frozen["control_unchanged"] and frozen["safepms_unchanged"]
    decision["coverage_artifacts_unchanged"] = coverage_before == coverage_after
    decision["all_safepms_gradients_finite"] = safe_stats["nonfinite_batch_count"] == 0
    decision["no_other_decoder_dependency"] = safe_stats["other_decoder_dependency_batch_count"] == 0
    decision["total_wall_time_le_10_gpu_hours"] = total_stage1_seconds <= 10 * 60 * 60
    if not all((decision["paired_step_counts_equal"], decision["batch_order_checksums_identical"], decision["frozen_modules_unchanged"], decision["coverage_artifacts_unchanged"], decision["all_safepms_gradients_finite"], decision["no_other_decoder_dependency"], decision["total_wall_time_le_10_gpu_hours"])):
        verdict = "NO-GO"
    _json(out_dir / "protocol.json", {"method": "SafePMS full only", "control": "standard L_anchor + L_expand", "anchor_keys": ["loss_focal", "loss_dice", "loss_iou", "loss_pms_preserve_focal", "loss_pms_preserve_dice", "loss_pms_preserve_iou"], "expansion_keys": ["loss_pms_focal", "loss_pms_dice", "loss_pms_iou", "loss_pms_object"], "trust_ratio": 1.0, "checkpoint_sha256": CHECKPOINT_SHA256, "pms_config": pms_payload, "inclusive_match_iou": 0.5, "prohibited": ["TNBC patients 9-11", "MoNuSeg", "scheduler", "early stopping", "coverage refresh", "intermediate development evaluation", "inference changes"], "train_patients": sorted(TRAIN_PATIENTS), "development_patients": sorted(DEVELOPMENT_PATIENTS), "development_evaluation_events": ["step_0", "epoch_5"]})
    _json(out_dir / "paired_manifest.json", paired_manifest); _csv(out_dir / "train_curves.csv", curves)
    _csv(out_dir / "gradient_statistics.csv", [{key: value for key, value in row.items() if key != "layerwise"} for row in safe_records]); _csv(out_dir / "per_image_metrics.csv", per_image)
    _json(out_dir / "bootstrap.json", {"safepms_vs_starting": _bootstrap(per_image, "starting", "safepms"), "safepms_vs_control": _bootstrap(per_image, "control_sum", "safepms")})
    runtime = {"control_train_seconds": control_seconds, "safepms_train_seconds": safe_seconds, "control_evaluation_seconds": control_eval_seconds, "safepms_evaluation_seconds": safe_eval_seconds, "starting_control_evaluation_seconds": starting_seconds, "starting_safepms_evaluation_seconds": safe_start_seconds, "total_stage1_seconds": total_stage1_seconds, "control_train_peak_memory_bytes": control_train_memory, "safepms_train_peak_memory_bytes": safe_train_memory, "control_evaluation_peak_memory_bytes": control_memory, "safepms_evaluation_peak_memory_bytes": safe_memory, "starting_control_evaluation_peak_memory_bytes": starting_memory, "starting_safepms_evaluation_peak_memory_bytes": safe_start_memory, "control_peak_memory_bytes": max(control_train_memory, control_memory), "safepms_peak_memory_bytes": max(safe_train_memory, safe_memory)}
    _json(out_dir / "runtime.json", runtime); _json(out_dir / "frozen_checksums.json", {**frozen, "coverage_sha256": {"before": coverage_before, "after": coverage_after, "unchanged": coverage_before == coverage_after}})
    (out_dir / "environment.txt").write_text(f"git_sha={_git_sha()}\npython={sys.version}\nplatform={platform.platform()}\ntorch={torch.__version__}\nseed={SEED}\ncheckpoint_sha256={CHECKPOINT_SHA256}\n", encoding="utf-8")
    report = {"verdict": verdict, "stage": "SafePMS Stage 1", "decision": decision, "starting": starting_summary, "control_sum": control_summary, "safepms": safe_summary, "safe_statistics": safe_stats, "runtime": runtime, "frozen": frozen, "paired_manifest": paired_manifest, "recommendation": "Stop and await project-lead decision; do not enter any further stage."}
    _json(out_dir / "report.json", report); _checksums(out_dir)
    return report
