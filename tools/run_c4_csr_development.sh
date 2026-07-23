#!/usr/bin/env bash
# Fixed C4-CSR development run.  This script is intentionally fail-closed:
# preparation freezes p1-p6-only feature stats before any p7/p8 replay, the
# zero-residual C1 gate must pass before ranker training, and no C1 checkpoint
# is copied into the output tree.
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 REPO_ROOT C1_STATE_2027_ROOT C1_STATE_1337_ROOT C0C1_P7P8_ORACLE_ROOT C3_AUDIT_ROOT OUTPUT_ROOT" >&2
  exit 2
fi

repo_root=$1
c1_state_2027=$2
c1_state_1337=$3
c0c1_dev_root=$4
c3_root=$5
output_root=$6

[[ ! -e "$output_root" ]] || { echo "refusing existing C4 output root: $output_root" >&2; exit 2; }
[[ -f "$c3_root/results/c3_score_control_audit.json" ]] || { echo "missing frozen C3 audit" >&2; exit 2; }
mkdir -p "$output_root/reports"
cd "$repo_root"

for seed in 2027 1337; do
  if [[ "$seed" == 2027 ]]; then state_root=$c1_state_2027; else state_root=$c1_state_1337; fi
  checkpoint="$state_root/c1/checkpoints/last_complete_state.pth"
  declaration="$state_root/c1/checkpoints/last_complete_state.json"
  [[ -f "$checkpoint" && -f "$declaration" ]] || { echo "missing retained C1 epoch-5 full state/declaration for seed $seed" >&2; exit 2; }
  conda run -n agentseg python tools/export_c4_csr_frozen_train_outputs.py \
    --manifest /root/autodl-tmp/f3c_phase1/manifests/tnbc_p1_6_phase1.json \
    --checkpoint "$checkpoint" --checkpoint-declaration "$declaration" \
    --data-path /root/autodl-tmp/projects/AgentSeg-CA-SAM2/data/tnbc \
    --output-dir "$output_root/frozen_train/seed${seed}" --seed "$seed" \
    --crop-size 256 --out-size 256 --overlap 32 --load unclockwise \
    --point-nms-thr 12 --instance-nms-iou 0.5 --prompt-chunk-size 64 \
    --point-filtering --texture --context \
    2>&1 | tee "$output_root/reports/frozen_c1_seed${seed}.log"
done

conda run -n agentseg python tools/run_c4_csr.py prepare \
  --config configs/phase2a/tnbc_c4_csr_v1.json \
  --train-source "2027=$output_root/frozen_train/seed2027" \
  --train-source "1337=$output_root/frozen_train/seed1337" \
  --output-dir "$output_root/prepared" \
  2>&1 | tee "$output_root/reports/prepare.log"

# The gate has no learned weights: a zero-initialized residual must reproduce
# C1 exactly before C4 sees a single train update.
conda run -n agentseg python tools/run_c4_csr.py preflight \
  --prepared-dir "$output_root/prepared" \
  --c1-source "2027=$c0c1_dev_root/seed2027_c1" --c1-source "1337=$c0c1_dev_root/seed1337_c1" \
  --c0-source "2027=$c0c1_dev_root/seed2027_c0" --c0-source "1337=$c0c1_dev_root/seed1337_c0" \
  --c3-audit "$c3_root/results/c3_score_control_audit.json" \
  --output-dir "$output_root/preflight" \
  2>&1 | tee "$output_root/reports/preflight.log"

for seed in 2027 1337; do
  conda run -n agentseg python tools/run_c4_csr.py train \
    --prepared-dir "$output_root/prepared" --train-source "$output_root/frozen_train/seed${seed}" \
    --seed "$seed" --output-dir "$output_root/ranker/seed${seed}" \
    2>&1 | tee "$output_root/reports/train_seed${seed}.log"
done

conda run -n agentseg python tools/run_c4_csr.py evaluate \
  --prepared-dir "$output_root/prepared" \
  --c1-source "2027=$c0c1_dev_root/seed2027_c1" --c1-source "1337=$c0c1_dev_root/seed1337_c1" \
  --c0-source "2027=$c0c1_dev_root/seed2027_c0" --c0-source "1337=$c0c1_dev_root/seed1337_c0" \
  --ranker-weights "2027=$output_root/ranker/seed2027/ranker_epoch20_weights.pth" \
  --ranker-weights "1337=$output_root/ranker/seed1337/ranker_epoch20_weights.pth" \
  --c3-audit "$c3_root/results/c3_score_control_audit.json" \
  --output-dir "$output_root/results" \
  2>&1 | tee "$output_root/reports/evaluate.log"

conda run -n agentseg python tools/render_c4_csr_cases.py \
  --evaluation "$output_root/results/c4_csr_results.json" --prepared-dir "$output_root/prepared" \
  --c1-source "2027=$c0c1_dev_root/seed2027_c1" --c1-source "1337=$c0c1_dev_root/seed1337_c1" \
  --ranker-weights "2027=$output_root/ranker/seed2027/ranker_epoch20_weights.pth" \
  --ranker-weights "1337=$output_root/ranker/seed1337/ranker_epoch20_weights.pth" \
  --output-dir "$output_root/cases" \
  2>&1 | tee "$output_root/reports/cases.log"

cp "$output_root/prepared/c4_csr_design.md" "$output_root/results/c4_csr_design.md"
cp "$output_root/prepared/c4_csr_preregistered_config.json" "$output_root/results/c4_csr_preregistered_config.json"
cp "$output_root/prepared/c4_csr_feature_schema.json" "$output_root/results/c4_csr_feature_schema.json"
cp "$output_root/cases/c4_csr_case_index.json" "$output_root/results/c4_csr_case_index.json"
sha256sum "$output_root/ranker/seed2027/ranker_epoch20_weights.pth" "$output_root/ranker/seed1337/ranker_epoch20_weights.pth" > "$output_root/results/c4_csr_ranker_weights.sha256"

echo "C4 CSR development complete: $output_root/results"
