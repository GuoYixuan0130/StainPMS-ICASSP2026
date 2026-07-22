#!/usr/bin/env bash
set -euo pipefail

repo_root=/root/autodl-tmp/projects/StainPMS-ICASSP2026-f3c
phase1_root=/root/autodl-tmp/f3c_phase1
phase2a_root=/root/autodl-tmp/f3c_phase2a
generic_sam2=/root/autodl-tmp/projects/CA-SAM2-HRC/checkpoints/sam2_hiera_large.pt
tnbc_data=/root/autodl-tmp/projects/AgentSeg-CA-SAM2/data/tnbc
train_manifest="$phase1_root/manifests/tnbc_p1_6_phase1.json"

cd "$repo_root"
mkdir -p "$phase2a_root/reports" "$phase2a_root/phase1_tables"

conda run -n agentseg python -m unittest discover \
  -s tests -p 'test_export_phase1_tables.py' -v
conda run -n agentseg python -m unittest discover \
  -s tests -p 'test_phase2a_*.py' -v

monuseg_phase1_dir=$(python tools/resolve_phase1_output.py \
  --root "$phase1_root/diagnostics" \
  --dataset monuseg \
  --processed-records 37 \
  --require-file gt_instances.csv \
  --require-file images.json)
printf 'Resolved MoNuSeg Phase 1 output: %s\n' "$monuseg_phase1_dir"

conda run -n agentseg python tools/export_phase1_tables.py \
  --input-dir "$phase1_root/diagnostics/tnbc_p1_6_full" \
  --input-dir "$phase1_root/diagnostics/tnbc_p7_8_full" \
  --input-dir "$monuseg_phase1_dir" \
  --output-dir "$phase2a_root/phase1_tables"

conda run -n agentseg python main.py \
  --dataset tnbc \
  --data_path "$tnbc_data" \
  --train_manifest "$train_manifest" \
  --verify_manifest_hashes \
  --phase2a_timing_profile base \
  --phase2a_timing_output "$phase2a_root/reports/tnbc_timing_base.json" \
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
  --overlap 32 \
  --load unclockwise \
  --test_nms_thr 12 \
  --b 1 \
  --texture \
  --context \
  --evaluator_mode strict \
  --exp_name f3c_phase2a_tnbc_timing_base

conda run -n agentseg python main.py \
  --dataset tnbc \
  --data_path "$tnbc_data" \
  --train_manifest "$train_manifest" \
  --verify_manifest_hashes \
  --phase2a_timing_profile pms_active \
  --phase2a_timing_output "$phase2a_root/reports/tnbc_timing_pms_active.json" \
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
  --overlap 32 \
  --load unclockwise \
  --test_nms_thr 12 \
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
  --exp_name f3c_phase2a_tnbc_timing_pms_active

set +e
conda run -n agentseg python tools/estimate_phase2a_baseline_budget.py \
  --recipe configs/phase2a/baseline_recipe_v1.json \
  --dataset tnbc \
  --base-timing "$phase2a_root/reports/tnbc_timing_base.json" \
  --active-timing "$phase2a_root/reports/tnbc_timing_pms_active.json" \
  --output "$phase2a_root/reports/tnbc_budget_gate.json"
gate_status=$?
set -e

if [[ "$gate_status" -ne 0 && "$gate_status" -ne 2 ]]; then
  exit "$gate_status"
fi

python -c 'import json,sys; r=json.load(open(sys.argv[1],encoding="utf-8")); print(json.dumps({"status":r["status"],"estimated_total_gpu_hours":r["estimated_total_gpu_hours"],"components_seconds":r["estimated_components_seconds"]},indent=2))' \
  "$phase2a_root/reports/tnbc_budget_gate.json"

printf 'Phase 2A TNBC gate complete. Return files under %s/phase1_tables and %s/reports.\n' \
  "$phase2a_root" "$phase2a_root"
exit "$gate_status"
