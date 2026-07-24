#!/usr/bin/env bash
# Fixed C4-CSR development run.  It accepts only the project-lead-approved
# seed-2027 recovery audit and seed-1337 reconstructed freeze manifest.
set -euo pipefail

continue_after_zero_residual_gate_bug=false
if [[ $# -eq 8 ]]; then
  repo_root=$1
  expected_commit=$2
  recovery_2027=$3
  freeze_1337=$4
  dev_2027_root=$5
  reconstructed_c3_root=$6
  c0_1337_dev=$7
  output_root=$8
elif [[ $# -eq 7 && "$1" == "--continue-after-zero-residual-gate-bug" ]]; then
  continue_after_zero_residual_gate_bug=true
  repo_root=$2
  expected_commit=$3
  dev_2027_root=$4
  reconstructed_c3_root=$5
  c0_1337_dev=$6
  output_root=$7
else
  echo "usage: $0 REPO_ROOT EXPECTED_COMMIT RECOVERY_2027_MANIFEST RECONSTRUCTED_1337_FREEZE_MANIFEST DEV_2027_ROOT RECONSTRUCTED_C3_ROOT C0_1337_DEV_DIR OUTPUT_ROOT" >&2
  echo "   or: $0 --continue-after-zero-residual-gate-bug REPO_ROOT EXPECTED_COMMIT DEV_2027_ROOT RECONSTRUCTED_C3_ROOT C0_1337_DEV_DIR OUTPUT_ROOT" >&2
  exit 2
fi

expected_branch=research/f3c-stainpms
[[ "$(git -C "$repo_root" branch --show-current)" == "$expected_branch" ]] || { echo "unexpected branch" >&2; exit 2; }
[[ "$(git -C "$repo_root" rev-parse HEAD)" == "$expected_commit" ]] || { echo "unexpected commit" >&2; exit 2; }
[[ -z "$(git -C "$repo_root" status --porcelain)" ]] || { echo "worktree must be clean before C4" >&2; exit 2; }
c3_audit="$reconstructed_c3_root/c3_results/c3_score_control_audit.json"

[[ -f "$c3_audit" ]] || { echo "missing reconstructed joint C3 audit" >&2; exit 2; }
[[ -f "$dev_2027_root/seed2027_c0/summary.json" && -f "$dev_2027_root/seed2027_c1/summary.json" ]] || { echo "missing historical seed-2027 C0/C1 compact development sources" >&2; exit 2; }
[[ -f "$c0_1337_dev/summary.json" && -f "$reconstructed_c3_root/seed1337_c1_reconstructed/summary.json" ]] || { echo "missing seed-1337 reconstructed C1 or paired C0 development source" >&2; exit 2; }

if [[ "$continue_after_zero_residual_gate_bug" == false ]]; then
  [[ ! -e "$output_root" ]] || { echo "refusing existing C4 output root: $output_root" >&2; exit 2; }
  c1_2027_checkpoint=$(python -c "import json,sys; print(json.load(open(sys.argv[1],encoding='utf-8'))['best_pq']['path'])" "$recovery_2027")
  c1_2027_declaration=$(python -c "import json,sys; print(json.load(open(sys.argv[1],encoding='utf-8'))['declaration_path'])" "$recovery_2027")
  c1_1337_checkpoint=$(python -c "import json,sys; print(json.load(open(sys.argv[1],encoding='utf-8'))['complete_state']['path'])" "$freeze_1337")
  c1_1337_declaration=$(python -c "import json,sys; print(json.load(open(sys.argv[1],encoding='utf-8'))['complete_state']['declaration_path'])" "$freeze_1337")
  [[ -f "$recovery_2027" && -f "$freeze_1337" ]] || { echo "missing C1 lineage contract" >&2; exit 2; }
  [[ -f "$c1_2027_checkpoint" && -f "$c1_2027_declaration" ]] || { echo "missing recovery-audited seed-2027 weights/declaration" >&2; exit 2; }
  [[ -f "$c1_1337_checkpoint" && -f "$c1_1337_declaration" ]] || { echo "missing reconstructed seed-1337 full state/declaration" >&2; exit 2; }
  mkdir -p "$output_root/reports"
else
  [[ -d "$output_root/prepared" && -d "$output_root/frozen_train/seed2027" && -d "$output_root/frozen_train/seed1337" && -f "$output_root/preflight/seed2027_failure.json" ]] || { echo "missing the exact failed zero-residual preflight state" >&2; exit 2; }
  [[ ! -e "$output_root/preflight_corrected" && ! -e "$output_root/ranker" && ! -e "$output_root/results" && ! -e "$output_root/cases" ]] || { echo "continuation accepts no prior learned C4 outputs" >&2; exit 2; }
  python -c "import json,sys; d=json.load(open(sys.argv[1],encoding='utf-8')); rows=d.get('rows',[]); ok=bool(rows) and all(not r['native_reproduction']['metric_mismatch'] and r['native_reproduction']['final_map_identical'] and {k for k,v in r['invariance'].items() if not v}=={'inference_uses_gt','inference_uses_evaluator_matching'} for r in rows); raise SystemExit(0 if ok else 'failed preflight is not the known negative-invariance gate bug')" "$output_root/preflight/seed2027_failure.json"
fi

cd "$repo_root"

if [[ "$continue_after_zero_residual_gate_bug" == false ]]; then
conda run -n agentseg python tools/export_c4_csr_frozen_train_outputs.py \
  --manifest /root/autodl-tmp/f3c_phase1/manifests/tnbc_p1_6_phase1.json \
  --checkpoint "$c1_2027_checkpoint" --checkpoint-declaration "$c1_2027_declaration" --lineage-contract "$recovery_2027" \
  --data-path /root/autodl-tmp/projects/AgentSeg-CA-SAM2/data/tnbc \
  --output-dir "$output_root/frozen_train/seed2027" --seed 2027 \
  --crop-size 256 --out-size 256 --overlap 32 --load unclockwise \
  --point-nms-thr 12 --instance-nms-iou 0.5 --prompt-chunk-size 64 \
  --point-filtering --texture --context 2>&1 | tee "$output_root/reports/frozen_c1_seed2027.log"

conda run -n agentseg python tools/export_c4_csr_frozen_train_outputs.py \
  --manifest /root/autodl-tmp/f3c_phase1/manifests/tnbc_p1_6_phase1.json \
  --checkpoint "$c1_1337_checkpoint" --checkpoint-declaration "$c1_1337_declaration" --lineage-contract "$freeze_1337" \
  --data-path /root/autodl-tmp/projects/AgentSeg-CA-SAM2/data/tnbc \
  --output-dir "$output_root/frozen_train/seed1337" --seed 1337 \
  --crop-size 256 --out-size 256 --overlap 32 --load unclockwise \
  --point-nms-thr 12 --instance-nms-iou 0.5 --prompt-chunk-size 64 \
  --point-filtering --texture --context 2>&1 | tee "$output_root/reports/frozen_c1_seed1337.log"

# These three files are created from p1-p6-only exports and frozen before the
# first p7/p8 C4 preflight or learned-ranker evaluation is allowed to read.
conda run -n agentseg python tools/run_c4_csr.py prepare \
  --config configs/phase2a/tnbc_c4_csr_v1.json \
  --train-source "2027=$output_root/frozen_train/seed2027" --train-source "1337=$output_root/frozen_train/seed1337" \
  --c3-audit "$c3_audit" --output-dir "$output_root/prepared" 2>&1 | tee "$output_root/reports/prepare.log"

conda run -n agentseg python tools/run_c4_csr.py preflight \
  --prepared-dir "$output_root/prepared" \
  --c1-source "2027=$dev_2027_root/seed2027_c1" --c1-source "1337=$reconstructed_c3_root/seed1337_c1_reconstructed" \
  --c0-source "2027=$dev_2027_root/seed2027_c0" --c0-source "1337=$c0_1337_dev" \
  --c3-audit "$c3_audit" --output-dir "$output_root/preflight" 2>&1 | tee "$output_root/reports/preflight.log"
else
  conda run -n agentseg python tools/run_c4_csr.py preflight \
    --prepared-dir "$output_root/prepared" \
    --c1-source "2027=$dev_2027_root/seed2027_c1" --c1-source "1337=$reconstructed_c3_root/seed1337_c1_reconstructed" \
    --c0-source "2027=$dev_2027_root/seed2027_c0" --c0-source "1337=$c0_1337_dev" \
    --c3-audit "$c3_audit" --output-dir "$output_root/preflight_corrected" 2>&1 | tee "$output_root/reports/preflight_corrected.log"
fi

for seed in 2027 1337; do
  conda run -n agentseg python tools/run_c4_csr.py train \
    --prepared-dir "$output_root/prepared" --train-source "$output_root/frozen_train/seed${seed}" \
    --seed "$seed" --output-dir "$output_root/ranker/seed${seed}" 2>&1 | tee "$output_root/reports/train_seed${seed}.log"
done

conda run -n agentseg python tools/run_c4_csr.py evaluate \
  --prepared-dir "$output_root/prepared" \
  --c1-source "2027=$dev_2027_root/seed2027_c1" --c1-source "1337=$reconstructed_c3_root/seed1337_c1_reconstructed" \
  --c0-source "2027=$dev_2027_root/seed2027_c0" --c0-source "1337=$c0_1337_dev" \
  --ranker-weights "2027=$output_root/ranker/seed2027/ranker_epoch20_weights.pth" --ranker-weights "1337=$output_root/ranker/seed1337/ranker_epoch20_weights.pth" \
  --c3-audit "$c3_audit" --output-dir "$output_root/results" 2>&1 | tee "$output_root/reports/evaluate.log"

conda run -n agentseg python tools/render_c4_csr_cases.py \
  --evaluation "$output_root/results/c4_csr_results.json" --prepared-dir "$output_root/prepared" \
  --c1-source "2027=$dev_2027_root/seed2027_c1" --c1-source "1337=$reconstructed_c3_root/seed1337_c1_reconstructed" \
  --ranker-weights "2027=$output_root/ranker/seed2027/ranker_epoch20_weights.pth" --ranker-weights "1337=$output_root/ranker/seed1337/ranker_epoch20_weights.pth" \
  --output-dir "$output_root/cases" 2>&1 | tee "$output_root/reports/cases.log"

cp "$output_root/prepared/c4_csr_design.md" "$output_root/results/c4_csr_design.md"
cp "$output_root/prepared/c4_csr_preregistered_config.json" "$output_root/results/c4_csr_preregistered_config.json"
cp "$output_root/prepared/c4_csr_feature_schema.json" "$output_root/results/c4_csr_feature_schema.json"
cp "$output_root/cases/c4_csr_case_index.json" "$output_root/results/c4_csr_case_index.json"
sha256sum "$output_root/ranker/seed2027/ranker_epoch20_weights.pth" "$output_root/ranker/seed1337/ranker_epoch20_weights.pth" > "$output_root/results/c4_csr_ranker_weights.sha256"
sha256sum "$output_root/ranker/seed2027/ranker_epoch20_weights.pth" "$output_root/ranker/seed2027/ranker_final_training_state.pth" "$output_root/ranker/seed1337/ranker_epoch20_weights.pth" "$output_root/ranker/seed1337/ranker_final_training_state.pth" > "$output_root/results/c4_csr_ranker_artifacts.sha256"
find "$output_root/cases" -type f -print0 | sort -z | xargs -0 sha256sum > "$output_root/results/c4_csr_cases.sha256"
git rev-parse HEAD > "$output_root/results/repository_commit.txt"
git status --short > "$output_root/results/worktree_status.txt"
find "$output_root" -type f ! -path "$output_root/reports/*" ! -path "$output_root/results/SHA256SUMS" -print0 | sort -z | xargs -0 sha256sum > "$output_root/results/SHA256SUMS"

echo "C4 CSR development complete: $output_root/results"
