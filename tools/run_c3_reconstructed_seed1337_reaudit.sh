#!/usr/bin/env bash
# Read-only C3 re-audit for a frozen reconstructed seed-1337 C1 epoch-5
# lineage.  The accepted seed-2027 C3 result is reused verbatim, not replayed.
set -euo pipefail

if [[ $# -ne 9 ]]; then
  echo "usage: $0 REPO_ROOT RECONSTRUCTION_ROOT DEV_MANIFEST TNBC_DATA OLD_C1_2027_ORACLE OLD_C3_AUDIT OLD_C2_MECH_2027 OUTPUT_ROOT INSTANCE_NMS_IOU" >&2
  exit 2
fi

repo_root=$1
reconstruction_root=$2
dev_manifest=$3
tnbc_data=$4
old_c1_2027=$5
old_c3=$6
old_c2_mechanism_2027=$7
output_root=$8
nms_iou=$9

[[ -d "$repo_root/.git" ]] || { echo "invalid repository: $repo_root" >&2; exit 2; }
[[ -f "$dev_manifest" ]] || { echo "missing p7/p8 manifest" >&2; exit 2; }
[[ -d "$old_c1_2027/completed_images" ]] || { echo "missing retained seed-2027 C1 compact artifacts" >&2; exit 2; }
[[ -f "$old_c3" ]] || { echo "missing accepted historical seed-2027 C3 audit" >&2; exit 2; }
[[ -f "$old_c2_mechanism_2027" ]] || { echo "missing historical seed-2027 full-oracle reference" >&2; exit 2; }
[[ ! -e "$output_root" ]] || { echo "refusing to overwrite output root: $output_root" >&2; exit 2; }

run_root="$reconstruction_root/c1_reconstructed"
frozen_manifest="$run_root/epoch5_frozen/frozen_epoch5_manifest.json"
[[ -f "$frozen_manifest" ]] || { echo "missing frozen reconstructed epoch-5 manifest" >&2; exit 2; }
shopt -s nullglob
states=("$run_root/checkpoints"/epoch_0005_*.pth)
if [[ ${#states[@]} -ne 1 ]]; then
  echo "reconstructed lineage requires exactly one retained epoch-5 full state; found ${#states[@]}" >&2
  exit 2
fi
state=${states[0]}
declaration="$run_root/checkpoint_declarations/$(basename "${state%.pth}").json"
[[ -f "$declaration" ]] || { echo "missing reconstructed epoch-5 declaration" >&2; exit 2; }

mkdir -p "$output_root"
cd "$repo_root"

conda run -n agentseg python tools/run_zero_training_oracle_diagnosis.py \
  --manifest "$dev_manifest" \
  --checkpoint "$state" \
  --checkpoint-declaration "$declaration" \
  --frozen-epoch5-manifest "$frozen_manifest" \
  --data-path "$tnbc_data" \
  --output-dir "$output_root/seed1337_c1_reconstructed" \
  --seed 1337 --arm c1_reconstructed \
  --crop-size 256 --out-size 256 --overlap 32 --load unclockwise \
  --point-nms-thr 12 --instance-nms-iou "$nms_iou" --prompt-chunk-size 64 \
  --point-filtering --texture --context

conda run -n agentseg python tools/audit_c3_score_control.py \
  --config configs/phase2a/tnbc_c3_score_control_audit_v1.json \
  --historical-seed-audit "2027=$old_c3" \
  --input "1337=$output_root/seed1337_c1_reconstructed" \
  --reconstructed-seed 1337 \
  --output-dir "$output_root/c3_results" \
  --instance-nms-iou "$nms_iou"

conda run -n agentseg python tools/render_c3_score_control_cases.py \
  --input "2027=$old_c1_2027" \
  --input "1337=$output_root/seed1337_c1_reconstructed" \
  --audit "$output_root/c3_results/c3_score_control_audit.json" \
  --output-dir "$output_root/cases"

conda run -n agentseg python tools/summarize_c3_reconstructed_joint_gate.py \
  --c3-audit "$output_root/c3_results/c3_score_control_audit.json" \
  --reconstructed-freeze-manifest "$frozen_manifest" \
  --output-dir "$output_root/joint_gate"

echo "C3 reconstructed seed-1337 re-audit complete: $output_root"
