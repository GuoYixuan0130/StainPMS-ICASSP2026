#!/usr/bin/env bash
set -euo pipefail

repo_root="/root/autodl-tmp/projects/StainPMS-ICASSP2026-f3c"
run_root="${1:-/root/autodl-tmp/f3c_phase2a_warmstart_smoke_$(git -C "$repo_root" rev-parse --short=12 HEAD)}"

if [[ -e "$run_root" ]]; then
  echo "Refusing to reuse existing run root: $run_root" >&2
  exit 2
fi
mkdir -p "$run_root/reports"
cd "$repo_root"

conda run -n agentseg python -m unittest discover \
  -s tests -p 'test_candidate_coverage.py' -v \
  2>&1 | tee "$run_root/reports/test_candidate_coverage.txt"
conda run -n agentseg python -m unittest discover \
  -s tests -p 'test_warmstart_protocol.py' -v \
  2>&1 | tee "$run_root/reports/test_warmstart_protocol.txt"
conda run -n agentseg python -m unittest discover \
  -s tests -p 'test_warmstart_equivalence.py' -v \
  2>&1 | tee "$run_root/reports/test_warmstart_equivalence.txt"

run_dataset() {
  local dataset="$1"
  local manifest="$2"
  local checkpoint="$3"
  local checkpoint_sha="$4"
  local data_path="$5"
  local overlap="$6"
  local dataset_root="$run_root/$dataset"
  local coverage_cache="$dataset_root/coverage_cache"
  local coverage_manifest="$dataset_root/coverage_manifest.json"

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
  )

  conda run -n agentseg python main.py \
    "${common[@]}" \
    --warmstart_stage prepare_coverage \
    --warmstart_candidate_arm c0 \
    --baseline_masks_dir "$coverage_cache" \
    --warmstart_output "$coverage_manifest" \
    --exp_name "f3c_phase2a_${dataset}_coverage_smoke" \
    2>&1 | tee "$run_root/reports/${dataset}_prepare_coverage.log"

  local arm
  for arm in legacy c0 c1; do
    conda run -n agentseg python main.py \
      "${common[@]}" \
      --warmstart_stage smoke \
      --warmstart_candidate_arm "$arm" \
      --warmstart_coverage_manifest "$coverage_manifest" \
      --warmstart_smoke_updates 1 \
      --warmstart_output "$dataset_root/${arm}_smoke_1update.json" \
      --exp_name "f3c_phase2a_${dataset}_${arm}_smoke" \
      2>&1 | tee "$run_root/reports/${dataset}_${arm}_smoke.log"
  done

  conda run -n agentseg python tools/compare_warmstart_smokes.py \
    --legacy "$dataset_root/legacy_smoke_1update.json" \
    --c0 "$dataset_root/c0_smoke_1update.json" \
    --c1 "$dataset_root/c1_smoke_1update.json" \
    --output "$dataset_root/smoke_gate.json" \
    2>&1 | tee "$run_root/reports/${dataset}_smoke_gate.log"
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

python - "$run_root" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
summary = {name: json.loads((root / name / "smoke_gate.json").read_text())["status"] for name in ("tnbc", "monuseg")}
print(json.dumps({"status": "complete", "run_root": str(root), "smoke_gates": summary}, indent=2))
PY
