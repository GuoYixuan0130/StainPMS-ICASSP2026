# StainRoute Stage 1 — AutoDL execution

Run this document only on the `research/stainroute` branch and only on the
AutoDL 4090. Stage 1 is a train/calibration oracle feasibility study: it does
not run any action on the official MoNuSeg test split or TNBC patients 9–11.

All commands below use the frozen Development Baseline v1: StainPMS e156 for
TNBC, `monuseg_pms_best_pq.pth` for MoNuSeg, NMS 12, disabled TTA, seed 3407,
batch size 1, and enabled texture/context. Do not change an action-generator
parameter after inspecting any Stage 1 result.

## 1. Update and freeze Baseline v1

```bash
set -euo pipefail
cd /root/autodl-tmp/projects/StainPMS-ICASSP2026
conda activate agentseg
git fetch origin
git switch research/stainroute
git pull --ff-only origin research/stainroute
git rev-parse HEAD
git status --short

python tools/stainroute_make_splits.py \
  --monuseg-root ./data/monuseg \
  --tnbc-root ./data/tnbc

python tools/stainroute_freeze_baseline.py \
  --config configs/stainroute/baseline_v1.yaml \
  --monuseg-root ./data/monuseg \
  --tnbc-root ./data/tnbc \
  --monuseg-split configs/splits/stainroute_monuseg.json \
  --tnbc-split configs/splits/stainroute_tnbc.json \
  --out logs/stainroute/stage1/baseline_v1_manifest.json
```

Before any GPU oracle run, save these three lightweight files for review and
commit: `configs/splits/stainroute_monuseg.json`,
`configs/splits/stainroute_tnbc.json`, and
`logs/stainroute/stage1/baseline_v1_manifest.json`. The script refuses a
checkpoint or split checksum mismatch later.

## 2. Targeted checks and two one-image smoke runs

```bash
mkdir -p logs/stainroute/stage1/checks
python -m unittest discover -s tests/stainroute -t . -v \
  2>&1 | tee logs/stainroute/stage1/checks/tests.txt
python -m unittest tests.test_stainroute_metrics tests.test_stainroute_stage0_reconcile -v \
  2>&1 | tee -a logs/stainroute/stage1/checks/tests.txt

mkdir -p logs/stainroute/stage1/monuseg_router_train_smoke
python main.py --eval --dataset monuseg --data_path ./data/monuseg \
  --sam_ckpt ../CA-SAM2-HRC/deliver_ckpts/monuseg_pms_best_pq.pth \
  --sam_config sam2_hiera_l --texture --context --overlap 92 \
  --test_nms_thr 12 --b 1 --seed 3407 \
  --stage1_stainroute_oracle \
  --stainroute_split_manifest configs/splits/stainroute_monuseg.json \
  --stainroute_split router_train \
  --stainroute_baseline_manifest logs/stainroute/stage1/baseline_v1_manifest.json \
  --stainroute_out_dir logs/stainroute/stage1/monuseg_router_train_smoke \
  --stainroute_max_images 1 --exp_name stainroute_stage1_monuseg_smoke \
  2>&1 | tee logs/stainroute/stage1/monuseg_router_train_smoke/stdout.log

mkdir -p logs/stainroute/stage1/tnbc_router_train_smoke
python main.py --eval --dataset monuseg --data_path ./data/tnbc \
  --sam_ckpt ../CA-SAM2-HRC/deliver_ckpts/tnbc_pms_best_e156.pth \
  --sam_config sam2_hiera_l --texture --context --overlap 32 \
  --test_nms_thr 12 --b 1 --seed 3407 \
  --stage1_stainroute_oracle \
  --stainroute_split_manifest configs/splits/stainroute_tnbc.json \
  --stainroute_split router_train \
  --stainroute_baseline_manifest logs/stainroute/stage1/baseline_v1_manifest.json \
  --stainroute_out_dir logs/stainroute/stage1/tnbc_router_train_smoke \
  --stainroute_max_images 1 --exp_name stainroute_stage1_tnbc_smoke \
  2>&1 | tee logs/stainroute/stage1/tnbc_router_train_smoke/stdout.log
```

Inspect both `cached_decode_equivalence.json` files. Every eligible image must
have `passed: true`; any nonzero failure is a hard stop. These directories are
explicitly marked `is_smoke_run: true` and are not formal results.

## 3. Formal frozen oracle runs

Run all four commands without `--stainroute_max_images`. They decode only
actions from `router_train` or `calibration`; none accesses a test split.

```bash
for split in router_train calibration; do
  out="logs/stainroute/stage1/monuseg_${split}_v1"
  mkdir -p "$out"
  python main.py --eval --dataset monuseg --data_path ./data/monuseg \
    --sam_ckpt ../CA-SAM2-HRC/deliver_ckpts/monuseg_pms_best_pq.pth \
    --sam_config sam2_hiera_l --texture --context --overlap 92 \
    --test_nms_thr 12 --b 1 --seed 3407 \
    --stage1_stainroute_oracle \
    --stainroute_split_manifest configs/splits/stainroute_monuseg.json \
    --stainroute_split "$split" \
    --stainroute_baseline_manifest logs/stainroute/stage1/baseline_v1_manifest.json \
    --stainroute_out_dir "$out" --exp_name "stainroute_stage1_monuseg_${split}" \
    2>&1 | tee "$out/stdout.log"
done

for split in router_train calibration; do
  out="logs/stainroute/stage1/tnbc_${split}_v1"
  mkdir -p "$out"
  python main.py --eval --dataset monuseg --data_path ./data/tnbc \
    --sam_ckpt ../CA-SAM2-HRC/deliver_ckpts/tnbc_pms_best_e156.pth \
    --sam_config sam2_hiera_l --texture --context --overlap 32 \
    --test_nms_thr 12 --b 1 --seed 3407 \
    --stage1_stainroute_oracle \
    --stainroute_split_manifest configs/splits/stainroute_tnbc.json \
    --stainroute_split "$split" \
    --stainroute_baseline_manifest logs/stainroute/stage1/baseline_v1_manifest.json \
    --stainroute_out_dir "$out" --exp_name "stainroute_stage1_tnbc_${split}" \
    2>&1 | tee "$out/stdout.log"
done
```

The formal directory contains separate `action_features.csv` and
`action_labels.csv`, replay masks under `decoded_actions/`, complete
per-image/summary oracle and control tables, bootstrap summaries (2,000 image
resamples, seed 3407), error diagnostics, cached-decoding equivalence, and
runtime/manifest records. Large masks/logits and `logs/` remain untracked.

After all four runs, copy only lightweight summaries and manifests into the
shared `temple/` directory for review. Do not launch router training, test
actions, or any Stage 2 work.
