#!/usr/bin/env bash
set -euo pipefail

repo_root="/root/autodl-tmp/projects/StainPMS-ICASSP2026-f3c"
smoke_root="${1:?Usage: run_phase2a_warmstart_timing.sh SMOKE_ROOT [TIMING_ROOT]}"
timing_root="${2:-/root/autodl-tmp/f3c_phase2a_warmstart_timing_$(git -C "$repo_root" rev-parse --short=12 HEAD)}"

if [[ -e "$timing_root" ]]; then
  echo "Refusing to reuse existing timing root: $timing_root" >&2
  exit 2
fi
mkdir -p "$timing_root/reports"
cd "$repo_root"

python - "$smoke_root" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for dataset in ("tnbc", "monuseg"):
    gate = json.loads((root / dataset / "smoke_gate.json").read_text())
    if gate.get("status") != "pass":
        raise SystemExit(f"{dataset} smoke gate did not pass")
print("Both smoke gates passed; timing is allowed by this script.")
PY

run_dataset() {
  local dataset="$1"
  local manifest="$2"
  local checkpoint="$3"
  local checkpoint_sha="$4"
  local data_path="$5"
  local overlap="$6"
  local coverage_manifest="$smoke_root/$dataset/coverage_manifest.json"
  local dataset_root="$timing_root/$dataset"
  mkdir -p "$dataset_root"

  local common=(
    --dataset "$dataset"
    --data_path "$data_path"
    --train_manifest "$manifest"
    --verify_manifest_hashes
    --sam_ckpt "$checkpoint"
    --warmstart_checkpoint_sha256 "$checkpoint_sha"
    --sam_config sam2_hiera_l
    --seed 3407
    --epochs 10
    --lr 1e-5
    --weight_decay 1e-4
    --lr_milestones 80 140 200
    --clip-grad 0.1
    --crop_size 256
    --out_size 256
    --overlap "$overlap"
    --load unclockwise
    --b 1
    --num_workers 0
    --texture
    --context
    --use_pms
    --pms_self_bootstrap
    --coverage_accumulate
    --pms_start_epoch 0
    --iterative_baseline_refresh_every 20
    --pms_loss_coef 0.5
    --pms_object_weight 1.0
    --pms_residual_mask_weight 0.3
    --pms_preserve_loss_coef 1.0
    --pms_gt_match_radius 8
    --pms_preserve_covered
    --pms_preserve_max_prompts 20
    --stain_min_distance 12
    --stain_top_k 20
    --test_nms_thr 12
    --test_filtering true
    --evaluator_mode strict
    --candidate_coverage_tau 0.1
    --candidate_coverage_coefficient 1.0
    --candidate_quality_coefficient 1.0
    --val_start_epoch -1
    --warmstart_stage timing
    --warmstart_coverage_manifest "$coverage_manifest"
    --phase2a_warmup_updates 10
    --phase2a_timed_updates 100
  )

  local arm
  for arm in c0 c1; do
    conda run -n agentseg python main.py \
      "${common[@]}" \
      --warmstart_candidate_arm "$arm" \
      --warmstart_output "$dataset_root/${arm}_timing.json" \
      --exp_name "f3c_phase2a_${dataset}_${arm}_timing" \
      2>&1 | tee "$timing_root/reports/${dataset}_${arm}_timing.log"
  done
}

run_dataset \
  tnbc \
  /root/autodl-tmp/f3c_phase1/manifests/tnbc_p1_6_phase1.json \
  /root/autodl-tmp/projects/CA-SAM2-HRC/deliver_ckpts/tnbc_pms_best_e156.pth \
  44a3cb3e93051301d789e44f93769588abfa727d7a174b0270f55305ef023781 \
  /root/autodl-tmp/projects/AgentSeg-CA-SAM2/data/tnbc \
  32

run_dataset \
  monuseg \
  /root/autodl-tmp/f3c_phase1/manifests/monuseg_train37_phase1.json \
  /root/autodl-tmp/projects/CA-SAM2-HRC/deliver_ckpts/monuseg_pms_best_pq.pth \
  6616c24626a162580d79aa7035d0b6c1a4ae6240a6c2f4599960dd4efac95db1 \
  /root/autodl-tmp/projects/AgentSeg-CA-SAM2/data/monuseg \
  92

conda run -n agentseg python tools/estimate_warmstart_budget.py \
  --tnbc-c0 "$timing_root/tnbc/c0_timing.json" \
  --tnbc-c1 "$timing_root/tnbc/c1_timing.json" \
  --tnbc-coverage "$smoke_root/tnbc/coverage_manifest.json" \
  --monuseg-c0 "$timing_root/monuseg/c0_timing.json" \
  --monuseg-c1 "$timing_root/monuseg/c1_timing.json" \
  --monuseg-coverage "$smoke_root/monuseg/coverage_manifest.json" \
  --output "$timing_root/warmstart_budget_estimate.json" \
  2>&1 | tee "$timing_root/reports/warmstart_budget_estimate.log"

echo "Timing complete: $timing_root"
