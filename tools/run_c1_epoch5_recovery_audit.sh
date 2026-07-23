#!/usr/bin/env bash
# Read-only C1 epoch-5 provenance audit.  No dataset/model inference/training.
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 REPO_ROOT C1_STATE_2027_ROOT C1_STATE_1337_ROOT C1_ORACLE_ROOT C3_AUDIT_JSON OUTPUT_ROOT" >&2
  exit 2
fi

repo_root=$1
root_2027=$2
root_1337=$3
c1_oracle_root=$4
c3_audit=$5
output_root=$6

[[ ! -e "$output_root" ]] || { echo "refusing existing audit output: $output_root" >&2; exit 2; }
[[ -f "$c3_audit" ]] || { echo "missing accepted C3 audit: $c3_audit" >&2; exit 2; }
cd "$repo_root"

set +e
conda run -n agentseg python tools/audit_c1_epoch5_recovery.py \
  --seed-root "2027=$root_2027" --seed-root "1337=$root_1337" \
  --c1-oracle-root "$c1_oracle_root" --c3-audit "$c3_audit" \
  --repo-root "$repo_root" --search-root /root/autodl-tmp \
  --output-dir "$output_root"
code=$?
set -e

if [[ -f "$output_root/c1_epoch5_recovery_audit.json" ]]; then
  echo "C1 epoch-5 recovery audit completed with exit_code=$code: $output_root"
  exit 0
fi
echo "C1 epoch-5 recovery audit failed before writing a report (exit_code=$code)" >&2
exit "$code"
