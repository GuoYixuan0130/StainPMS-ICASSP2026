#!/usr/bin/env bash
# Read-only C3 score-control feasibility audit.  It replays compact p7/p8 C1
# artifacts only; it does not construct a model, load a checkpoint, or train.
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 C1_2027_DIR C1_1337_DIR C1_MECHANISMS_2027_JSON C1_MECHANISMS_1337_JSON OUTPUT_ROOT REPO_ROOT" >&2
  exit 2
fi

c1_2027=$1
c1_1337=$2
mechanisms_2027=$3
mechanisms_1337=$4
output_root=$5
repo_root=$6

if [[ -e "$output_root" ]]; then
  echo "refusing to overwrite existing output root: $output_root" >&2
  exit 2
fi
for source in "$c1_2027" "$c1_1337"; do
  count=$(find "$source/completed_images" -maxdepth 1 -type f -name '*.json.gz' -printf . | wc -c)
  if [[ "$count" != 7 ]]; then
    echo "C3 requires exactly seven compact p7/p8 artifacts in $source; found $count" >&2
    exit 2
  fi
done
for source in "$mechanisms_2027" "$mechanisms_1337"; do
  [[ -f "$source" ]] || { echo "missing C1 full-oracle mechanism reference: $source" >&2; exit 2; }
done

mkdir -p "$output_root"
cd "$repo_root"
config="$repo_root/configs/phase2a/tnbc_c3_score_control_audit_v1.json"

conda run -n agentseg python tools/audit_c3_score_control.py \
  --config "$config" \
  --input "2027=$c1_2027" \
  --input "1337=$c1_1337" \
  --reference-full-oracle "2027=$mechanisms_2027" \
  --reference-full-oracle "1337=$mechanisms_1337" \
  --output-dir "$output_root/results"

conda run -n agentseg python tools/render_c3_score_control_cases.py \
  --input "2027=$c1_2027" \
  --input "1337=$c1_1337" \
  --audit "$output_root/results/c3_score_control_audit.json" \
  --output-dir "$output_root/cases"

echo "C3 read-only audit complete: $output_root"
