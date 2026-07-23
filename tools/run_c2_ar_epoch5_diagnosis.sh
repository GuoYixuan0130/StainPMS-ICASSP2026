#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Read-only p7/p8 diagnosis after both C2-AR train-only runs completed.
# Usage:
#   bash tools/run_c2_ar_epoch5_diagnosis.sh C2_SEED2027_ROOT C2_SEED1337_ROOT EXISTING_C0C1_ORACLE_ROOT OUTPUT_ROOT
c2_2027_root="${1:?missing C2 seed-2027 root}"
c2_1337_root="${2:?missing C2 seed-1337 root}"
existing_oracle_root="${3:?missing existing two-seed C0/C1 oracle root}"
output_root="${4:?missing C2 diagnosis output root}"
repo_root="${5:-/root/autodl-tmp/projects/StainPMS-ICASSP2026-f3c}"

if [[ -e "$output_root" ]]; then
  echo "Refusing to reuse output root: $output_root" >&2
  exit 2
fi
mkdir -p "$output_root/reports"
cd "$repo_root"

for seed in 2027 1337; do
  if [[ "$seed" == "2027" ]]; then root="$c2_2027_root"; else root="$c2_1337_root"; fi
  checkpoint=("$root"/c2_ar/checkpoints/epoch_0005_*.pth)
  declaration=("$root"/c2_ar/checkpoint_declarations/epoch_0005_*.json)
  if [[ ${#checkpoint[@]} -ne 1 || ${#declaration[@]} -ne 1 ]]; then
    echo "Expected exactly one retained epoch-5 state and declaration under $root/c2_ar" >&2
    exit 2
  fi
  conda run -n agentseg python tools/run_zero_training_oracle_diagnosis.py \
    --manifest /root/autodl-tmp/f3c_phase1/manifests/tnbc_p7_8_phase1.json \
    --checkpoint "${checkpoint[0]}" --checkpoint-declaration "${declaration[0]}" \
    --data-path /root/autodl-tmp/projects/AgentSeg-CA-SAM2/data/tnbc \
    --output-dir "$output_root/seed${seed}_c2_ar" --seed "$seed" --arm c2_ar \
    --crop-size 256 --out-size 256 --overlap 32 --load unclockwise \
    --point-nms-thr 12 --instance-nms-iou 0.5 --prompt-chunk-size 64 \
    --point-filtering --texture --context \
    2>&1 | tee "$output_root/reports/seed${seed}_c2_ar.log"
done

conda run -n agentseg python tools/summarize_c2_ar_results.py \
  --input "2027:c0=$existing_oracle_root/seed2027_c0/summary.json" \
  --input "2027:c1=$existing_oracle_root/seed2027_c1/summary.json" \
  --input "2027:c2_ar=$output_root/seed2027_c2_ar/summary.json" \
  --input "1337:c0=$existing_oracle_root/seed1337_c0/summary.json" \
  --input "1337:c1=$existing_oracle_root/seed1337_c1/summary.json" \
  --input "1337:c2_ar=$output_root/seed1337_c2_ar/summary.json" \
  --output-dir "$output_root/results" \
  2>&1 | tee "$output_root/reports/summarize_c2_ar.log"

conda run -n agentseg python tools/render_c2_ar_cases.py \
  --input "2027:c0=$existing_oracle_root/seed2027_c0" \
  --input "2027:c1=$existing_oracle_root/seed2027_c1" \
  --input "2027:c2_ar=$output_root/seed2027_c2_ar" \
  --input "1337:c0=$existing_oracle_root/seed1337_c0" \
  --input "1337:c1=$existing_oracle_root/seed1337_c1" \
  --input "1337:c2_ar=$output_root/seed1337_c2_ar" \
  --output-dir "$output_root/results" \
  2>&1 | tee "$output_root/reports/render_c2_ar_cases.log"

echo "C2-AR read-only development diagnosis complete: $output_root/results"
