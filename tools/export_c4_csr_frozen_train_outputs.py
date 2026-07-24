#!/usr/bin/env python3
"""Export frozen C1 selected predictions for C4-CSR training on TNBC p1--6.

This is inference-only.  It deliberately reuses the audited native C1
decoder/token-selection path and writes compact RLE artifacts, not model
copies.  The exported objects are the only C1-derived inputs accepted by the
C4 ranker trainer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run.dataset.manifest import load_dataset_manifest
from run.dataset.tnbc import TNBC
from sam2_train.build_sam import build_sam2
from sam2_train.modeling.dpa_p2pnet import build_model
from tools.run_zero_training_oracle_diagnosis import (
    diagnose_image,
    git_value,
    json_sha256,
    read_gzip_json,
    read_json,
    runtime_cfg,
    set_determinism,
    sha256_file,
    write_gzip_json_atomic,
    write_json_atomic,
)


SEEDS = (2027, 1337)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-declaration", required=True)
    parser.add_argument("--lineage-contract", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--model-config", default="args.py")
    parser.add_argument("--sam-config", default="sam2_hiera_l")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--out-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--load", default="unclockwise", choices=("sequence", "unsequence", "clockwise", "unclockwise"))
    parser.add_argument("--point-nms-thr", type=int, default=12)
    parser.add_argument("--instance-nms-iou", type=float, default=0.5)
    parser.add_argument("--prompt-chunk-size", type=int, default=64)
    parser.add_argument("--texture-memory-bank-size", type=int, default=64)
    parser.add_argument("--context-atten-k", type=int, default=1)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--point-filtering", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--texture", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--context", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def validate_train_scope(manifest: dict, records: list[dict]) -> None:
    allowed = {int(value) for value in manifest.get("allowed_patients", [])}
    observed = {int(row.get("patient", -1)) for row in records}
    if allowed != {1, 2, 3, 4, 5, 6} or observed != allowed or len(records) != 30:
        raise ValueError("C4 frozen output export requires exactly the 30-image TNBC p1-p6 train manifest")
    if observed & {7, 8, 9, 10, 11}:
        raise ValueError("C4 frozen output exporter rejects all development and sealed TNBC patients before dataset construction")


def require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a complete lowercase SHA256")
    return value


def validate_checkpoint(declaration_path: Path, checkpoint: Path, lineage_contract_path: Path, seed: int) -> dict:
    declaration = read_json(declaration_path, "C1 epoch-5 checkpoint declaration")
    observed_sha = sha256_file(checkpoint)
    if declaration.get("checkpoint_sha256") != observed_sha:
        raise ValueError("C1 checkpoint SHA256 does not match its declaration")
    contract = read_json(lineage_contract_path, "C4 C1 lineage contract")
    contract_sha = sha256_file(lineage_contract_path)
    if declaration.get("dataset") != "tnbc" or declaration.get("arm") != "c1" or int(declaration.get("epoch", -1)) != 5:
        raise ValueError("C4 requires a TNBC C1 epoch-5 declaration")
    if seed == 2027:
        recovered = contract.get("best_pq")
        if (
            contract.get("protocol") != "tnbc_c1_epoch5_recovery_audit_v1"
            or int(contract.get("seed", -1)) != 2027
            or contract.get("status") != "recovered_epoch5_weights_only"
            or not isinstance(recovered, dict)
            or Path(str(recovered.get("path", ""))).resolve() != checkpoint
            or require_sha256(recovered.get("sha256"), "seed-2027 recovered weights SHA256") != observed_sha
            or recovered.get("load_mode") != "weights_only"
            or declaration.get("classification") != "historical_exploratory"
            or declaration.get("protocol") != "tnbc_c0_c1_second_seed_2027_v1"
            or contract.get("c3_consistency_evidence", {}).get("status") != "pass"
            or not all(bool(value) for value in contract.get("configuration_identity_conditions", {}).values())
        ):
            raise ValueError("seed-2027 C4 source must be the recovery-audited original epoch-5 weights-only lineage")
        canonical = require_sha256(recovered.get("canonical_model_model1_tensor_sha256"), "seed-2027 canonical model/model1 tensor SHA256")
        source = {
            "kind": "recovery_audited_epoch5_weights_only",
            "recovery_audit_path": str(lineage_contract_path),
            "recovery_audit_sha256": contract_sha,
            "weights_only_sha256": observed_sha,
            "canonical_model_model1_tensor_sha256": canonical,
            "source_last_state_sha256": require_sha256(recovered.get("embedded_provenance", {}).get("source_last_state_sha256"), "seed-2027 source last-state SHA256"),
        }
    else:
        complete = contract.get("complete_state")
        weights = contract.get("weights_only")
        if (
            contract.get("protocol") != "tnbc_c1_seed1337_reconstructed_epoch5_freeze_v1"
            or contract.get("status") != "frozen_before_development_access"
            or contract.get("lineage") != "reconstructed C1 seed-1337 lineage"
            or not isinstance(complete, dict)
            or not isinstance(weights, dict)
            or Path(str(complete.get("path", ""))).resolve() != checkpoint
            or require_sha256(complete.get("sha256"), "seed-1337 reconstructed complete-state SHA256") != observed_sha
            or declaration.get("classification") != "historical_exploratory_reconstructed"
            or declaration.get("protocol") != "tnbc_c1_seed1337_reconstructed_epoch5_v1"
        ):
            raise ValueError("seed-1337 C4 source must be the frozen reconstructed epoch-5 full-state lineage")
        weights_path = Path(str(weights.get("path", ""))).resolve()
        if not weights_path.is_file() or sha256_file(weights_path) != require_sha256(weights.get("sha256"), "seed-1337 reconstructed weights-only SHA256"):
            raise ValueError("seed-1337 reconstructed weights-only artifact fails the frozen-manifest SHA256")
        source = {
            "kind": "reconstructed_epoch5_full_state",
            "frozen_epoch5_manifest_path": str(lineage_contract_path),
            "frozen_epoch5_manifest_sha256": contract_sha,
            "complete_state_sha256": observed_sha,
            "weights_only_path": str(weights_path),
            "weights_only_sha256": require_sha256(weights.get("sha256"), "seed-1337 reconstructed weights-only SHA256"),
            "canonical_model_model1_tensor_sha256": require_sha256(weights.get("canonical_model_model1_tensor_sha256"), "seed-1337 canonical model/model1 tensor SHA256"),
        }
    return {**declaration, "checkpoint_path": str(checkpoint), "checkpoint_sha256": observed_sha, "lineage_contract": source}


def compact_artifact(artifact: dict) -> dict:
    """Drop pools irrelevant to C4 ranker training, retaining selected masks/GT."""

    return {
        "schema_version": 1,
        "sample_id": artifact["sample_id"],
        "patient": int(artifact["patient"]),
        "image_shape": artifact["image_shape"],
        "gt_instances": artifact["gt_instances"],
        "native_selected_before_assembly": artifact["native_selected_before_assembly"],
        "source": "frozen_c1_selected_predictions_only",
    }


def main() -> int:
    args = parse_args()
    if (args.crop_size, args.out_size, args.overlap, args.load) != (256, 256, 32, "unclockwise"):
        raise ValueError("C4 freezes C1 inference geometry at crop=256, output=256, overlap=32, unclockwise")
    if (args.point_nms_thr, args.instance_nms_iou, args.prompt_chunk_size) != (12, 0.5, 64):
        raise ValueError("C4 freezes C1 point-NMS, instance-NMS and chunking")
    if not args.point_filtering or not args.texture or not args.context or args.context_atten_k != 1:
        raise ValueError("C4 requires the frozen C1 filtering/texture/context inference path")
    if not torch.cuda.is_available():
        raise RuntimeError("C4 frozen C1 export requires CUDA inference")

    manifest_path = Path(args.manifest).resolve()
    manifest, records = load_dataset_manifest(manifest_path, expected_dataset="tnbc", require_labels=True, verify_hashes=True)
    validate_train_scope(manifest, records)
    checkpoint = Path(args.checkpoint).resolve()
    declaration = validate_checkpoint(Path(args.checkpoint_declaration).resolve(), checkpoint, Path(args.lineage_contract).resolve(), args.seed)
    output = Path(args.output_dir).resolve()
    completed_dir = output / "completed_images"
    progress_path = output / "progress.json"
    texture_path = output / "texture_memory_bank.pt"
    fingerprint = {
        "protocol": "tnbc_c4_csr_frozen_c1_train_outputs_v1",
        "seed": args.seed,
        "arm": "c1",
        "manifest_sha256": manifest["manifest_sha256"],
        "checkpoint_sha256": declaration["checkpoint_sha256"],
        "lineage_contract": declaration["lineage_contract"],
        "frozen_inference": {
            "crop_size": args.crop_size, "out_size": args.out_size, "overlap": args.overlap,
            "load": args.load, "point_nms_thr": args.point_nms_thr,
            "instance_nms_iou": args.instance_nms_iou, "prompt_chunk_size": args.prompt_chunk_size,
            "point_filtering": True, "texture": True, "context": True, "context_atten_k": 1,
        },
    }
    fingerprint_sha = json_sha256(fingerprint)
    completed: list[int] = []
    texture_memory_bank: list = []
    if args.resume:
        progress = read_json(progress_path, "C4 frozen-output progress")
        if progress.get("fingerprint_sha256") != fingerprint_sha:
            raise ValueError("C4 resume fingerprint differs from frozen C1 source")
        completed = [int(value) for value in progress.get("completed_indices", [])]
        if completed != list(range(len(completed))):
            raise ValueError("C4 frozen-output resume must have a contiguous prefix")
        if completed:
            texture_memory_bank = list(torch.load(texture_path, map_location="cpu", weights_only=False))
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("C4 frozen-output directory already contains data; use --resume or a new path")
        output.mkdir(parents=True, exist_ok=True)
        write_json_atomic(progress_path, {"schema_version": 1, "status": "in_progress", "fingerprint": fingerprint, "fingerprint_sha256": fingerprint_sha, "completed_indices": []})

    set_determinism(args.seed)
    torch.cuda.set_device(args.gpu_device)
    device = torch.device("cuda", int(torch.cuda.current_device()))
    model_config = Config.fromfile(str(Path(args.model_config).resolve()))
    net = build_sam2(args.sam_config, str(checkpoint), device=device, checkpoint_has_training_state=declaration["lineage_contract"]["kind"] == "reconstructed_epoch5_full_state")
    point_net, point_encoder = build_model(model_config)
    point_net.to(device).eval()
    point_encoder.to(device).eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=declaration["lineage_contract"]["kind"] == "recovery_audited_epoch5_weights_only")
    if list(state.get("texture_memory_bank_list", []) or []):
        raise ValueError("C4 requires the approved empty embedded C1 texture bank")
    missing, unexpected = point_net.load_state_dict(state["model1"], strict=False)
    if missing or unexpected:
        raise ValueError(f"C1 point-head state mismatch: missing={len(missing)} unexpected={len(unexpected)}")
    dataset = TNBC(runtime_cfg(args), model_config, args.data_path, args.load, mode="test", manifest_path=str(manifest_path), data_split="train", verify_manifest_hashes=True)

    started = time.perf_counter()
    for index in range(len(completed), len(dataset)):
        image_started = time.perf_counter()
        image, inst_map, *_unused, sample_id = dataset[index]
        inst_np = np.asarray(inst_map.cpu().numpy() if torch.is_tensor(inst_map) else inst_map, dtype=np.int32)
        image_record, artifact = diagnose_image(
            image=image, inst_map=inst_np, sample_id=str(sample_id), patient=int(records[index]["patient"]),
            point_net=point_net, point_encoder=point_encoder, net=net,
            texture_memory_bank=texture_memory_bank, args=args, device=device,
        )
        image_record["wall_seconds"] = time.perf_counter() - image_started
        payload = {
            "schema_version": 1, "record_index": index, "sample_id": str(sample_id),
            "manifest_sha256": manifest["manifest_sha256"], "image_record": image_record,
            "artifact": compact_artifact(artifact),
        }
        write_gzip_json_atomic(completed_dir / f"{index:05d}.json.gz", payload)
        torch.save(texture_memory_bank, texture_path)
        completed.append(index)
        write_json_atomic(progress_path, {"schema_version": 1, "status": "in_progress", "fingerprint": fingerprint, "fingerprint_sha256": fingerprint_sha, "completed_indices": completed})
        print(f"[c4-frozen-c1] seed={args.seed} {index + 1}/{len(dataset)} {sample_id} wall_s={image_record['wall_seconds']:.2f}", flush=True)

    summary = {
        "schema_version": 1, "protocol": "tnbc_c4_csr_frozen_c1_train_outputs_v1", "status": "complete",
        "scope": "TNBC p1-p6 only; no-grad C1 frozen selected-mask export; no development or sealed data accessed",
        "seed": args.seed, "arm": "c1", "checkpoint": declaration,
        "manifest": {"path": str(manifest_path), "sha256": manifest["manifest_sha256"], "record_count": len(records)},
        "fingerprint": fingerprint, "fingerprint_sha256": fingerprint_sha,
        "completed_record_count": len(completed), "wall_seconds": time.perf_counter() - started,
        "repository": {"branch": git_value("branch", "--show-current"), "commit": git_value("rev-parse", "HEAD"), "dirty": bool(git_value("status", "--porcelain"))},
    }
    write_json_atomic(output / "summary.json", summary)
    write_json_atomic(progress_path, {"schema_version": 1, "status": "complete", "fingerprint": fingerprint, "fingerprint_sha256": fingerprint_sha, "completed_indices": completed})
    print(json.dumps({"status": "complete", "output_dir": str(output), "summary": str(output / "summary.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
