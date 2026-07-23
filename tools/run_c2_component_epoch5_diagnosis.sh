#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Read-only development attribution after all four C2 component train-only
# arms complete.  Usage:
#   bash tools/run_c2_component_epoch5_diagnosis.sh E2027 U2027 E1337 U1337 C0C1_ORACLE C2EU_ORACLE C2EU_TRAIN2027 C2EU_TRAIN1337 OUT [REPO]
e2027_root="${1:?missing C2-E seed2027 root}"
u2027_root="${2:?missing C2-U seed2027 root}"
e1337_root="${3:?missing C2-E seed1337 root}"
u1337_root="${4:?missing C2-U seed1337 root}"
c0c1_oracle_root="${5:?missing existing C0/C1 oracle root}"
c2eu_oracle_root="${6:?missing existing C2-EU oracle root}"
c2eu_train2027_root="${7:?missing C2-EU seed2027 train root}"
c2eu_train1337_root="${8:?missing C2-EU seed1337 train root}"
output_root="${9:?missing output root}"
repo_root="${10:-/root/autodl-tmp/projects/StainPMS-ICASSP2026-f3c}"

if [[ -e "$output_root" ]]; then echo "Refusing to reuse output root: $output_root" >&2; exit 2; fi
mkdir -p "$output_root/reports" "$output_root/mechanisms"
cd "$repo_root"

declare -A roots=(
  ["2027:c2_e"]="$e2027_root"
  ["2027:c2_u"]="$u2027_root"
  ["1337:c2_e"]="$e1337_root"
  ["1337:c2_u"]="$u1337_root"
)
for key in "2027:c2_e" "2027:c2_u" "1337:c2_e" "1337:c2_u"; do
  seed="${key%%:*}"; arm="${key##*:}"; root="${roots[$key]}"
  checkpoint=("$root"/"$arm"/checkpoints/epoch_0005_*.pth)
  declaration=("$root"/"$arm"/checkpoint_declarations/epoch_0005_*.json)
  if [[ ${#checkpoint[@]} -ne 1 || ${#declaration[@]} -ne 1 ]]; then
    echo "Expected exactly one retained epoch-5 state/declaration under $root/$arm" >&2; exit 2
  fi
  conda run -n agentseg python tools/run_zero_training_oracle_diagnosis.py \
    --manifest /root/autodl-tmp/f3c_phase1/manifests/tnbc_p7_8_phase1.json \
    --checkpoint "${checkpoint[0]}" --checkpoint-declaration "${declaration[0]}" \
    --data-path /root/autodl-tmp/projects/AgentSeg-CA-SAM2/data/tnbc \
    --output-dir "$output_root/seed${seed}_${arm}" --seed "$seed" --arm "$arm" \
    --crop-size 256 --out-size 256 --overlap 32 --load unclockwise \
    --point-nms-thr 12 --instance-nms-iou 0.5 --prompt-chunk-size 64 \
    --point-filtering --texture --context \
    2>&1 | tee "$output_root/reports/seed${seed}_${arm}.log"
done

# Existing C1/C2-EU artifact runs remain read-only references. Their hard
# leakage/overlap and score accounting are reconstructible from their stored
# masks. New C2-E/U artifacts also include compact soft scalar diagnostics.
for spec in \
  "2027 c1 $c0c1_oracle_root/seed2027_c1" \
  "1337 c1 $c0c1_oracle_root/seed1337_c1" \
  "2027 c2_ar $c2eu_oracle_root/seed2027_c2_ar" \
  "1337 c2_ar $c2eu_oracle_root/seed1337_c2_ar" \
  "2027 c2_e $output_root/seed2027_c2_e" \
  "2027 c2_u $output_root/seed2027_c2_u" \
  "1337 c2_e $output_root/seed1337_c2_e" \
  "1337 c2_u $output_root/seed1337_c2_u"
do
  read -r seed arm source <<< "$spec"
  logical_arm="$arm"; [[ "$arm" == "c2_ar" ]] && logical_arm="c2_eu"
  conda run -n agentseg python tools/audit_c2_component_mechanisms.py \
    --oracle-dir "$source" --output-dir "$output_root/mechanisms/seed${seed}_${logical_arm}" \
    --seed "$seed" --arm "$arm" \
    2>&1 | tee "$output_root/reports/mechanism_seed${seed}_${logical_arm}.log"
done

conda run -n agentseg python tools/summarize_c2_component_ablation.py \
  --oracle "2027:c0=$c0c1_oracle_root/seed2027_c0/summary.json" \
  --oracle "2027:c1=$c0c1_oracle_root/seed2027_c1/summary.json" \
  --oracle "2027:c2_eu=$c2eu_oracle_root/seed2027_c2_ar/summary.json" \
  --oracle "2027:c2_e=$output_root/seed2027_c2_e/summary.json" \
  --oracle "2027:c2_u=$output_root/seed2027_c2_u/summary.json" \
  --oracle "1337:c0=$c0c1_oracle_root/seed1337_c0/summary.json" \
  --oracle "1337:c1=$c0c1_oracle_root/seed1337_c1/summary.json" \
  --oracle "1337:c2_eu=$c2eu_oracle_root/seed1337_c2_ar/summary.json" \
  --oracle "1337:c2_e=$output_root/seed1337_c2_e/summary.json" \
  --oracle "1337:c2_u=$output_root/seed1337_c2_u/summary.json" \
  --mechanism "2027:c1=$output_root/mechanisms/seed2027_c1/component_mechanisms.json" \
  --mechanism "2027:c2_eu=$output_root/mechanisms/seed2027_c2_eu/component_mechanisms.json" \
  --mechanism "2027:c2_e=$output_root/mechanisms/seed2027_c2_e/component_mechanisms.json" \
  --mechanism "2027:c2_u=$output_root/mechanisms/seed2027_c2_u/component_mechanisms.json" \
  --mechanism "1337:c1=$output_root/mechanisms/seed1337_c1/component_mechanisms.json" \
  --mechanism "1337:c2_eu=$output_root/mechanisms/seed1337_c2_eu/component_mechanisms.json" \
  --mechanism "1337:c2_e=$output_root/mechanisms/seed1337_c2_e/component_mechanisms.json" \
  --mechanism "1337:c2_u=$output_root/mechanisms/seed1337_c2_u/component_mechanisms.json" \
  --training-summary "2027:c2_eu=$c2eu_train2027_root/c2_ar/training_summary.json" \
  --training-summary "1337:c2_eu=$c2eu_train1337_root/c2_ar/training_summary.json" \
  --training-summary "2027:c2_e=$e2027_root/c2_e/training_summary.json" \
  --training-summary "2027:c2_u=$u2027_root/c2_u/training_summary.json" \
  --training-summary "1337:c2_e=$e1337_root/c2_e/training_summary.json" \
  --training-summary "1337:c2_u=$u1337_root/c2_u/training_summary.json" \
  --output-dir "$output_root/results" \
  2>&1 | tee "$output_root/reports/summarize.log"

conda run -n agentseg python tools/render_c2_component_cases.py \
  --input "2027:c1=$c0c1_oracle_root/seed2027_c1" \
  --input "2027:c2_eu=$c2eu_oracle_root/seed2027_c2_ar" \
  --input "2027:c2_e=$output_root/seed2027_c2_e" \
  --input "2027:c2_u=$output_root/seed2027_c2_u" \
  --input "1337:c1=$c0c1_oracle_root/seed1337_c1" \
  --input "1337:c2_eu=$c2eu_oracle_root/seed1337_c2_ar" \
  --input "1337:c2_e=$output_root/seed1337_c2_e" \
  --input "1337:c2_u=$output_root/seed1337_c2_u" \
  --output-dir "$output_root/results" \
  2>&1 | tee "$output_root/reports/render_cases.log"

echo "C2 component read-only diagnosis complete: $output_root/results"
