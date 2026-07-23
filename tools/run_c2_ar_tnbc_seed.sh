#!/usr/bin/env bash
set -euo pipefail

# Run exactly one approved C2-AR five-epoch, p1-p6-only training arm.
# The formal runner itself refuses to begin without enough space to retain all
# five full states. It does not construct a p7/p8 loader.
seed="${1:?usage: run_c2_ar_tnbc_seed.sh SEED [repo_root] [coverage_manifest] [output_root]}"
repo_root="${2:-/root/autodl-tmp/projects/StainPMS-ICASSP2026-f3c}"
coverage_manifest="${3:-/root/autodl-tmp/f3c_phase2a_warmstart_smoke_5aacd74711d0/tnbc/coverage_manifest.json}"
output_root="${4:-/root/autodl-tmp/f3c_c2_ar_tnbc_seed${seed}_$(git -C "$repo_root" rev-parse --short=12 HEAD)}"

if [[ "$seed" != "2027" && "$seed" != "1337" ]]; then
  echo "Only pre-registered C2-AR seeds 2027 and 1337 are allowed." >&2
  exit 2
fi
if [[ -e "$output_root" ]]; then
  echo "Refusing to reuse output root: $output_root" >&2
  exit 2
fi
if [[ ! -f "$coverage_manifest" ]]; then
  echo "Missing frozen p1-p6 coverage manifest: $coverage_manifest" >&2
  exit 2
fi

mkdir -p "$output_root/reports"
cd "$repo_root"
df -h /root/autodl-tmp | tee "$output_root/reports/free_space_before_training.txt"

conda run -n agentseg python main.py \
  --dataset tnbc --data_path /root/autodl-tmp/projects/AgentSeg-CA-SAM2/data/tnbc \
  --train_manifest /root/autodl-tmp/f3c_phase1/manifests/tnbc_p1_6_phase1.json --verify_manifest_hashes \
  --sam_ckpt /root/autodl-tmp/projects/CA-SAM2-HRC/deliver_ckpts/tnbc_pms_best_e156.pth \
  --warmstart_checkpoint_sha256 44a3cb3e93051301d789e44f93769588abfa727d7a174b0270f55305ef023781 \
  --sam_config sam2_hiera_l --seed "$seed" --epochs 5 --lr 1e-5 --weight_decay 1e-4 \
  --lr_milestones 80 140 200 --clip-grad 0.1 --crop_size 256 --out_size 256 --overlap 32 \
  --load unclockwise --b 1 --num_workers 0 --texture --context --use_pms --pms_self_bootstrap \
  --coverage_accumulate --pms_start_epoch 0 --iterative_baseline_refresh_every 20 \
  --pms_loss_coef 0.5 --pms_object_weight 1.0 --pms_residual_mask_weight 0.3 \
  --pms_preserve_loss_coef 1.0 --pms_gt_match_radius 8 --pms_preserve_covered \
  --pms_preserve_max_prompts 20 --stain_min_distance 12 --stain_top_k 20 \
  --test_nms_thr 12 --test_filtering true --evaluator_mode strict \
  --candidate_coverage_tau 0.1 --candidate_coverage_coefficient 1.0 --candidate_quality_coefficient 1.0 \
  --c2_ar_exclusivity_coefficient 0.25 --c2_ar_utility_coefficient 0.25 \
  --c2_ar_neighbor_radius 2 --c2_ar_match_iou 0.5 --c2_ar_merge_risk_overlap_fraction 0.1 \
  --phase2a_warmup_updates 10 --phase2a_timed_updates 100 \
  --warmstart_stage formal_tnbc_c2_ar_5epoch --warmstart_candidate_arm c2_ar \
  --warmstart_coverage_manifest "$coverage_manifest" \
  --warmstart_screen_config configs/phase2a/tnbc_c2_ar_two_seed_v1.json \
  --warmstart_required_free_gib 40 \
  --warmstart_output "$output_root/c2_ar/training_summary.json" \
  --exp_name "f3c_c2_ar_tnbc_seed${seed}_formal5" \
  2>&1 | tee "$output_root/reports/c2_ar_train.log"

df -h /root/autodl-tmp | tee "$output_root/reports/free_space_after_training.txt"
echo "C2-AR train-only run complete. Do not compact its five epoch states before the read-only epoch-5 diagnosis."
