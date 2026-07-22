#!/usr/bin/env bash
# Owner-approved formal TNBC-only C0/C1 five-epoch exploratory screen.
set -euo pipefail

repo_root="/root/autodl-tmp/projects/StainPMS-ICASSP2026-f3c"
smoke_root="${1:?Usage: run_phase2a_tnbc_c0c1_screen.sh SMOKE_ROOT [SCREEN_ROOT]}"
screen_root="${2:-/root/autodl-tmp/f3c_phase2a_tnbc_c0c1_screen_$(git -C "$repo_root" rev-parse --short=12 HEAD)}"

if [[ -e "$screen_root" ]]; then
  echo "Refusing to reuse existing screen root: $screen_root" >&2
  exit 2
fi
mkdir -p "$screen_root/reports"
cd "$repo_root"

if [[ -n "$(git status --short)" ]]; then
  echo "Refusing formal run from a dirty worktree. Commit or remove only code changes first." >&2
  git status --short >&2
  exit 2
fi

conda run -n agentseg python -m unittest discover \
  -s tests -p 'test_candidate_coverage.py' -v \
  2>&1 | tee "$screen_root/reports/test_candidate_coverage_preflight.txt"
conda run -n agentseg python -m unittest discover \
  -s tests -p 'test_warmstart_protocol.py' -v \
  2>&1 | tee "$screen_root/reports/test_warmstart_protocol_preflight.txt"
conda run -n agentseg python -m unittest discover \
  -s tests -p 'test_strict_evaluator.py' -v \
  2>&1 | tee "$screen_root/reports/test_strict_evaluator_preflight.txt"
conda run -n agentseg python -m unittest discover \
  -s tests -p 'test_phase2a_tnbc_screen.py' -v \
  2>&1 | tee "$screen_root/reports/test_phase2a_tnbc_screen_preflight.txt"

python - "$smoke_root" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
gate = json.loads((root / "tnbc" / "smoke_gate.json").read_text(encoding="utf-8"))
if gate.get("status") != "pass":
    raise SystemExit("TNBC C0/C1 smoke gate did not pass")
coverage = root / "tnbc" / "coverage_manifest.json"
if not coverage.is_file():
    raise SystemExit("TNBC shared train-only coverage manifest is missing")
print(json.dumps({"smoke_gate": "pass", "coverage_manifest": str(coverage)}))
PY

train_manifest="/root/autodl-tmp/f3c_phase1/manifests/tnbc_p1_6_phase1.json"
dev_manifest="/root/autodl-tmp/f3c_phase1/manifests/tnbc_p7_8_phase1.json"
data_path="/root/autodl-tmp/projects/AgentSeg-CA-SAM2/data/tnbc"
initial_checkpoint="/root/autodl-tmp/projects/CA-SAM2-HRC/deliver_ckpts/tnbc_pms_best_e156.pth"
initial_declaration="configs/phase1/checkpoints/tnbc_pms_e156_historical_exploratory.json"
initial_sha="44a3cb3e93051301d789e44f93769588abfa727d7a174b0270f55305ef023781"
coverage_manifest="$smoke_root/tnbc/coverage_manifest.json"
screen_config="configs/phase2a/tnbc_warmstart_screen_v1.json"

diagnose_checkpoint() {
  local checkpoint="$1"
  local declaration="$2"
  local output="$3"
  local scope="$4"
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
    2>&1 | tee "$screen_root/reports/$(basename "$output").log"
}

# Shared epoch zero is computed once before either independent optimizer run.
diagnose_checkpoint \
  "$initial_checkpoint" \
  "$initial_declaration" \
  "$screen_root/diagnostics/epoch_000_shared" \
  tnbc_p7_8_shared_epoch0

run_arm() {
  local arm="$1"
  local arm_root="$screen_root/$arm"
  mkdir -p "$arm_root"
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
    --warmstart_stage formal_tnbc_5epoch --warmstart_candidate_arm "$arm" \
    --warmstart_coverage_manifest "$coverage_manifest" \
    --warmstart_screen_config "$screen_config" \
    --warmstart_output "$arm_root/training_summary.json" \
    --exp_name "f3c_phase2a_tnbc_${arm}_formal5" \
    2>&1 | tee "$screen_root/reports/${arm}_train.log"

  local epoch checkpoint declaration
  for epoch in 1 2 3 4 5; do
    checkpoint="$arm_root/checkpoints/epoch_$(printf '%04d' "$epoch")_update_$(printf '%06d' "$((epoch * 270))").pth"
    declaration="$arm_root/checkpoint_declarations/epoch_$(printf '%04d' "$epoch")_update_$(printf '%06d' "$((epoch * 270))").json"
    diagnose_checkpoint "$checkpoint" "$declaration" "$screen_root/diagnostics/${arm}_epoch_$(printf '%04d' "$epoch")" "tnbc_p7_8_${arm}_epoch_${epoch}"
  done
}

# Separate Python processes make C0/C1 initialization, optimizer, scheduler,
# RNG, crop ordering, and coverage consumption independent by construction.
run_arm c0
run_arm c1

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

echo "TNBC C0/C1 five-epoch screen complete: $screen_root"
