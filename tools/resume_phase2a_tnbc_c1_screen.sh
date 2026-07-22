#!/usr/bin/env bash
# Recover only an interrupted C1 arm of the owner-approved TNBC C0/C1 screen.
# This never retrains C0 and never constructs a development loader during C1 updates.
set -euo pipefail

repo_root="/root/autodl-tmp/projects/StainPMS-ICASSP2026-f3c"
screen_root="${1:?Usage: resume_phase2a_tnbc_c1_screen.sh SCREEN_ROOT SMOKE_ROOT}"
smoke_root="${2:?Usage: resume_phase2a_tnbc_c1_screen.sh SCREEN_ROOT SMOKE_ROOT}"

cd "$repo_root"
if [[ -n "$(git status --short)" ]]; then
  echo "Refusing recovery from a dirty worktree." >&2
  git status --short >&2
  exit 2
fi

c0_summary="$screen_root/c0/training_summary.json"
c1_summary="$screen_root/c1/training_summary.json"
if [[ ! -f "$c0_summary" ]]; then
  echo "C0 training_summary.json is required before C1-only recovery." >&2
  exit 2
fi
if [[ ! -f "$screen_root/diagnostics/epoch_000_shared/summary.json" ]]; then
  echo "Shared epoch-0 diagnosis is required and will not be recomputed." >&2
  exit 2
fi

recover_c1=true
if [[ -f "$c1_summary" ]]; then
  python - "$c1_summary" <<'PY'
import json
import pathlib
import sys

summary = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if summary.get("status") != "complete" or len(summary.get("epochs", [])) != 5:
    raise SystemExit("existing C1 summary is not a complete five-epoch formal screen")
print("C1 is already complete; recovery driver will run only fairness, diagnosis, and summary.")
PY
  recover_c1=false
else
  resume_checkpoint="$(find "$screen_root/c1/checkpoints" -maxdepth 1 -type f -name 'epoch_*.pth' -printf '%f\n' | sort | tail -n 1)"
  if [[ -z "$resume_checkpoint" ]]; then
    echo "No C1 recovery checkpoint found." >&2
    exit 2
  fi
  resume_checkpoint="$screen_root/c1/checkpoints/$resume_checkpoint"
fi

mkdir -p "$screen_root/reports"
conda run -n agentseg python -m unittest discover \
  -s tests -p 'test_phase2a_tnbc_fairness.py' -v \
  2>&1 | tee "$screen_root/reports/test_phase2a_tnbc_recovery_preflight.txt"

train_manifest="/root/autodl-tmp/f3c_phase1/manifests/tnbc_p1_6_phase1.json"
dev_manifest="/root/autodl-tmp/f3c_phase1/manifests/tnbc_p7_8_phase1.json"
data_path="/root/autodl-tmp/projects/AgentSeg-CA-SAM2/data/tnbc"
initial_checkpoint="/root/autodl-tmp/projects/CA-SAM2-HRC/deliver_ckpts/tnbc_pms_best_e156.pth"
initial_sha="44a3cb3e93051301d789e44f93769588abfa727d7a174b0270f55305ef023781"
coverage_manifest="$smoke_root/tnbc/coverage_manifest.json"
screen_config="configs/phase2a/tnbc_warmstart_screen_v1.json"

if [[ "$recover_c1" == true ]]; then
conda run -n agentseg python main.py \
  --dataset tnbc \
  --data_path "$data_path" \
  --train_manifest "$train_manifest" \
  --verify_manifest_hashes \
  --sam_ckpt "$initial_checkpoint" \
  --warmstart_checkpoint_sha256 "$initial_sha" \
  --sam_config sam2_hiera_l \
  --seed 3407 --epochs 5 \
  --lr 1e-5 --weight_decay 1e-4 --lr_milestones 80 140 200 --clip-grad 0.1 \
  --crop_size 256 --out_size 256 --overlap 32 --load unclockwise --b 1 --num_workers 0 \
  --texture --context --use_pms --pms_self_bootstrap --coverage_accumulate \
  --pms_start_epoch 0 --iterative_baseline_refresh_every 20 \
  --pms_loss_coef 0.5 --pms_object_weight 1.0 --pms_residual_mask_weight 0.3 \
  --pms_preserve_loss_coef 1.0 --pms_gt_match_radius 8 --pms_preserve_covered \
  --pms_preserve_max_prompts 20 --stain_min_distance 12 --stain_top_k 20 \
  --test_nms_thr 12 --test_filtering true --evaluator_mode strict \
  --candidate_coverage_tau 0.1 --candidate_coverage_coefficient 1.0 --candidate_quality_coefficient 1.0 \
  --phase2a_warmup_updates 10 --phase2a_timed_updates 100 \
  --warmstart_stage formal_tnbc_5epoch --warmstart_candidate_arm c1 \
  --warmstart_coverage_manifest "$coverage_manifest" \
  --warmstart_screen_config "$screen_config" \
  --warmstart_resume_checkpoint "$resume_checkpoint" \
  --warmstart_output "$c1_summary" \
  --exp_name f3c_phase2a_tnbc_c1_formal5_recovery \
  2>&1 | tee "$screen_root/reports/c1_recovery_train.log"
fi

conda run -n agentseg python tools/verify_phase2a_tnbc_fairness.py \
  --c0-summary "$c0_summary" \
  --c1-summary "$c1_summary" \
  --output "$screen_root/reports/c0_c1_fairness_gate.json" \
  2>&1 | tee "$screen_root/reports/c0_c1_fairness_gate.log"

diagnose_checkpoint() {
  local checkpoint="$1"
  local declaration="$2"
  local output="$3"
  local scope="$4"
  local load_args=()
  if [[ "$checkpoint" == "$screen_root/c0/checkpoints/"* || "$checkpoint" == "$screen_root/c1/checkpoints/"* ]]; then
    load_args+=(--checkpoint-has-training-state)
  fi
  conda run -n agentseg python tools/run_phase1_candidate_diagnosis.py \
    --dataset tnbc \
    --manifest "$dev_manifest" \
    --checkpoint "$checkpoint" \
    --checkpoint-declaration "$declaration" \
    --data-path "$data_path" \
    --output-dir "$output" \
    --scope-label "$scope" \
    --crop-size 256 --out-size 256 --overlap 32 --load unclockwise \
    --point-nms-thr 12 --instance-nms-iou 0.5 --prompt-chunk-size 64 \
    --texture --context --discard-checkpoint-texture-bank --include-final-task-metrics \
    --drop-completed-resume-state \
    "${load_args[@]}" \
    2>&1 | tee "$screen_root/reports/$(basename "$output").log"
}

epoch_checkpoint_paths() {
  local arm="$1"
  local epoch="$2"
  python - "$screen_root/$arm/training_summary.json" "$epoch" <<'PY'
import json
import pathlib
import sys

summary = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
epoch = int(sys.argv[2])
matches = [record for record in summary.get("epochs", []) if int(record["epoch"]) == epoch]
if len(matches) != 1:
    raise SystemExit(f"expected one epoch record for epoch {epoch}")
print(matches[0]["checkpoint_path"])
print(matches[0]["checkpoint_declaration"])
PY
}

evaluate_arm() {
  local arm="$1"
  local epoch checkpoint declaration
  for epoch in 1 2 3 4 5; do
    readarray -t paths < <(epoch_checkpoint_paths "$arm" "$epoch")
    checkpoint="${paths[0]}"
    declaration="${paths[1]}"
    diagnose_checkpoint "$checkpoint" "$declaration" "$screen_root/diagnostics/${arm}_epoch_$(printf '%04d' "$epoch")" "tnbc_p7_8_${arm}_epoch_${epoch}"
  done
}

evaluate_arm c0
evaluate_arm c1

summary_args=(
  --epoch0-dir "$screen_root/diagnostics/epoch_000_shared"
  --output-dir "$screen_root/summary"
)
for epoch in 1 2 3 4 5; do
  summary_args+=(--c0-epoch-dir "$epoch=$screen_root/diagnostics/c0_epoch_$(printf '%04d' "$epoch")")
  summary_args+=(--c1-epoch-dir "$epoch=$screen_root/diagnostics/c1_epoch_$(printf '%04d' "$epoch")")
done
conda run -n agentseg python tools/summarize_phase2a_tnbc_warmstart_screen.py \
  "${summary_args[@]}" \
  2>&1 | tee "$screen_root/reports/summarize_screen.log"

git branch --show-current > "$screen_root/reports/git_branch.txt"
git rev-parse HEAD > "$screen_root/reports/git_commit.txt"
git status --short > "$screen_root/reports/git_status_final.txt"

echo "Recovered TNBC C1 and completed the fixed five-epoch screen: $screen_root"
