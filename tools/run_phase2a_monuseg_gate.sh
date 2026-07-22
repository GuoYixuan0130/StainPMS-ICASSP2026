#!/usr/bin/env bash
set -euo pipefail

repo_root=/root/autodl-tmp/projects/StainPMS-ICASSP2026-f3c
phase1_root=/root/autodl-tmp/f3c_phase1
phase2a_root=/root/autodl-tmp/f3c_phase2a
generic_sam2=/root/autodl-tmp/projects/CA-SAM2-HRC/checkpoints/sam2_hiera_large.pt
monuseg_data=/root/autodl-tmp/projects/AgentSeg-CA-SAM2/data/monuseg
train_manifest="$phase1_root/manifests/monuseg_train37_phase1.json"

cd "$repo_root"
mkdir -p "$phase2a_root/reports"

conda run -n agentseg python main.py \
  --dataset monuseg \
  --data_path "$monuseg_data" \
  --train_manifest "$train_manifest" \
  --verify_manifest_hashes \
  --phase2a_timing_profile base \
  --phase2a_timing_output "$phase2a_root/reports/monuseg_timing_base.json" \
  --phase2a_warmup_updates 10 \
  --phase2a_timed_updates 100 \
  --sam_ckpt "$generic_sam2" \
  --sam_config sam2_hiera_l \
  --seed 3407 \
  --epochs 200 \
  --lr 1e-4 \
  --lr_min 1e-6 \
  --lr_cosine_t_max 200 \
  --weight_decay 1e-4 \
  --crop_size 256 \
  --out_size 256 \
  --overlap 92 \
  --load unclockwise \
  --test_nms_thr 12 \
  --test_filtering true \
  --b 1 \
  --texture \
  --context \
  --evaluator_mode strict \
  --exp_name f3c_phase2a_monuseg_timing_base

conda run -n agentseg python main.py \
  --dataset monuseg \
  --data_path "$monuseg_data" \
  --train_manifest "$train_manifest" \
  --verify_manifest_hashes \
  --phase2a_timing_profile pms_active \
  --phase2a_timing_output "$phase2a_root/reports/monuseg_timing_pms_active.json" \
  --phase2a_warmup_updates 10 \
  --phase2a_timed_updates 100 \
  --sam_ckpt "$generic_sam2" \
  --sam_config sam2_hiera_l \
  --seed 3407 \
  --epochs 200 \
  --lr 1e-4 \
  --lr_min 1e-6 \
  --lr_cosine_t_max 200 \
  --weight_decay 1e-4 \
  --crop_size 256 \
  --out_size 256 \
  --overlap 92 \
  --load unclockwise \
  --test_nms_thr 12 \
  --test_filtering true \
  --b 1 \
  --texture \
  --context \
  --use_pms \
  --pms_self_bootstrap \
  --coverage_accumulate \
  --pms_start_epoch 0 \
  --iterative_baseline_refresh_every 20 \
  --pms_loss_coef 0.5 \
  --pms_object_weight 1.0 \
  --pms_residual_mask_weight 0.3 \
  --pms_preserve_loss_coef 1.0 \
  --pms_gt_match_radius 8 \
  --pms_preserve_covered \
  --pms_preserve_max_prompts 20 \
  --stain_min_distance 12 \
  --stain_top_k 20 \
  --evaluator_mode strict \
  --exp_name f3c_phase2a_monuseg_timing_pms_active

set +e
conda run -n agentseg python tools/estimate_phase2a_baseline_budget.py \
  --recipe configs/phase2a/baseline_recipe_v1.json \
  --dataset monuseg \
  --base-timing "$phase2a_root/reports/monuseg_timing_base.json" \
  --active-timing "$phase2a_root/reports/monuseg_timing_pms_active.json" \
  --output "$phase2a_root/reports/monuseg_budget_gate.json"
monuseg_status=$?

conda run -n agentseg python tools/assess_phase2a_combined_budget.py \
  --recipe configs/phase2a/baseline_recipe_v1.json \
  --dataset-report "$phase2a_root/reports/tnbc_budget_gate.json" \
  --dataset-report "$phase2a_root/reports/monuseg_budget_gate.json" \
  --output "$phase2a_root/reports/combined_budget_gate.json"
combined_status=$?
set -e

if [[ "$monuseg_status" -ne 0 && "$monuseg_status" -ne 2 ]]; then
  exit "$monuseg_status"
fi
if [[ "$combined_status" -ne 0 && "$combined_status" -ne 2 ]]; then
  exit "$combined_status"
fi

python -c 'import json,sys; m=json.load(open(sys.argv[1],encoding="utf-8")); c=json.load(open(sys.argv[2],encoding="utf-8")); print(json.dumps({"monuseg": {"status":m["status"],"estimated_gpu_hours":m["estimated_total_gpu_hours"],"components_seconds":m["estimated_components_seconds"]},"combined":c},indent=2))' \
  "$phase2a_root/reports/monuseg_budget_gate.json" \
  "$phase2a_root/reports/combined_budget_gate.json"

if [[ "$monuseg_status" -eq 2 || "$combined_status" -eq 2 ]]; then
  printf 'Phase 2A budget gate says STOP. Do not start either formal baseline; return the gate JSON files.\n'
  exit 2
fi

printf 'Phase 2A dataset and combined budget gates pass. Return the gate JSON files before formal training.\n'
