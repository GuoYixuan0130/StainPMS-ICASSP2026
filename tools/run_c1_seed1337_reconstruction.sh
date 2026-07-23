#!/usr/bin/env bash
# Train-only, fail-closed reconstruction of the missing C1 seed-1337 epoch-5
# lineage.  p7/p8 is intentionally absent from this launcher.
set -euo pipefail

if [[ $# -ne 8 ]]; then
  echo "usage: $0 REPO_ROOT ORIGINAL_SEED1337_ROOT COVERAGE_MANIFEST OUTPUT_ROOT TRAIN_MANIFEST INIT_CHECKPOINT TNBC_DATA REQUIRED_FREE_GIB" >&2
  exit 2
fi

repo_root=$1
original_root=$2
coverage_manifest=$3
output_root=$4
train_manifest=$5
init_checkpoint=$6
tnbc_data=$7
required_free_gib=$8

git -C "$repo_root" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
  || { echo "invalid repository worktree: $repo_root" >&2; exit 2; }
[[ -f "$original_root/c1/training_summary.json" ]] || { echo "missing original seed-1337 C1 training summary" >&2; exit 2; }
[[ -f "$coverage_manifest" ]] || { echo "missing frozen train-only coverage manifest" >&2; exit 2; }
[[ -f "$train_manifest" ]] || { echo "missing p1-p6 train manifest" >&2; exit 2; }
[[ -f "$init_checkpoint" ]] || { echo "missing C1 initialization checkpoint" >&2; exit 2; }
[[ ! -e "$output_root" ]] || { echo "refusing to overwrite output root: $output_root" >&2; exit 2; }
case "$required_free_gib" in 14|14.0|14.00) ;; *) echo "frozen reconstruction storage gate is 14 GiB" >&2; exit 2;; esac

mkdir -p "$output_root/provenance"
# Retain an in-run copy even if the caller's terminal disconnects or directs
# nohup output elsewhere.  The outer launcher may still keep its own log.
exec > >(tee -a "$output_root/provenance/reconstruction_training.log") 2>&1
cd "$repo_root"

conda run -n agentseg python tools/audit_c1_seed1337_reconstruction_inputs.py \
  --source-training-summary "$original_root/c1/training_summary.json" \
  --screen-config configs/phase2a/tnbc_c1_seed1337_reconstruction_v1.json \
  --train-manifest "$train_manifest" \
  --coverage-manifest "$coverage_manifest" \
  --initialization-checkpoint "$init_checkpoint" \
  --output "$output_root/provenance/reconstruction_input_audit.json"

conda run -n agentseg python main.py \
  --dataset tnbc \
  --data_path "$tnbc_data" \
  --train_manifest "$train_manifest" \
  --verify_manifest_hashes \
  --sam_ckpt "$init_checkpoint" \
  --warmstart_checkpoint_sha256 44a3cb3e93051301d789e44f93769588abfa727d7a174b0270f55305ef023781 \
  --sam_config sam2_hiera_l \
  --seed 1337 \
  --epochs 5 \
  --lr 1e-5 \
  --weight_decay 1e-4 \
  --lr_milestones 80 140 200 \
  --clip-grad 0.1 \
  --crop_size 256 --out_size 256 --overlap 32 --load unclockwise --b 1 --num_workers 0 \
  --texture --context --use_pms --pms_self_bootstrap --coverage_accumulate --pms_start_epoch 0 \
  --iterative_baseline_refresh_every 20 --pms_loss_coef 0.5 --pms_object_weight 1.0 \
  --pms_residual_mask_weight 0.3 --pms_preserve_loss_coef 1.0 --pms_gt_match_radius 8 \
  --pms_preserve_covered --pms_preserve_max_prompts 20 --stain_min_distance 12 --stain_top_k 20 \
  --test_nms_thr 12 --test_filtering true --evaluator_mode strict \
  --candidate_coverage_tau 0.1 --candidate_coverage_coefficient 1.0 --candidate_quality_coefficient 1.0 \
  --c2_ar_exclusivity_coefficient 0.0 --c2_ar_utility_coefficient 0.0 \
  --phase2a_warmup_updates 10 --phase2a_timed_updates 100 \
  --warmstart_stage formal_tnbc_c1_seed1337_reconstruction_5epoch \
  --warmstart_candidate_arm c1 \
  --warmstart_coverage_manifest "$coverage_manifest" \
  --warmstart_screen_config configs/phase2a/tnbc_c1_seed1337_reconstruction_v1.json \
  --warmstart_required_free_gib "$required_free_gib" \
  --warmstart_output "$output_root/c1_reconstructed/training_summary.json" \
  --exp_name f3c_reconstructed_c1_seed1337_epoch5

shopt -s nullglob
states=("$output_root/c1_reconstructed/checkpoints"/epoch_0005_*.pth)
if [[ ${#states[@]} -ne 1 ]]; then
  echo "expected exactly one retained epoch-5 full state, found ${#states[@]}" >&2
  exit 2
fi
state=${states[0]}
declaration="$output_root/c1_reconstructed/checkpoint_declarations/$(basename "${state%.pth}").json"
[[ -f "$declaration" ]] || { echo "missing epoch-5 declaration: $declaration" >&2; exit 2; }

conda run -n agentseg python tools/freeze_reconstructed_c1_epoch5.py \
  --repo-root "$repo_root" \
  --run-root "$output_root/c1_reconstructed" \
  --training-summary "$output_root/c1_reconstructed/training_summary.json" \
  --full-state "$state" \
  --full-declaration "$declaration" \
  --screen-config configs/phase2a/tnbc_c1_seed1337_reconstruction_v1.json \
  --train-manifest "$train_manifest" \
  --coverage-manifest "$coverage_manifest" \
  --initialization-checkpoint "$init_checkpoint" \
  --input-audit "$output_root/provenance/reconstruction_input_audit.json"

sha256sum "$output_root/provenance/reconstruction_training.log" \
  > "$output_root/provenance/reconstruction_training.log.sha256"
echo "C1 seed-1337 reconstructed epoch-5 frozen before development access: $output_root"
