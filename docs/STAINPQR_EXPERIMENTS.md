# StainPQR Experiment Workflow on AutoDL

This file tracks the ICASSP-stage experiment workflow built on top of the
StainPMS repository.

Local development happens in this repository, but all model evaluation/training
is expected to run on the AutoDL Linux machine. After local changes are pushed,
run the following on AutoDL:

```bash
cd /path/to/StainPMS-ICASSP2026
git pull origin main
conda activate CA-SAM2
```

## Stage 0: Baseline Reproduction and Error Audit

Goal:

1. Reproduce CA-SAM2 and StainPMS metrics.
2. Dump per-image artifacts needed by StainPQR.
3. Decompose the remaining PQ errors before adding any selective correction.

Artifacts produced by `--dump_eval_artifacts_dir`:

- `<image>_gt.npy`: GT instance map.
- `<image>_pred.npy`: final predicted instance map.
- `<image>_meta.json`: mask-level assembly records, including prompt point,
  bbox, predicted IoU, stability score, crop box, edge penalty flag, and selected
  final instance sources.

### MoNuSeg CA-SAM2

```bash
python main.py --eval \
  --dataset monuseg \
  --data_path ./data/monuseg \
  --sam_ckpt ./checkpoints/CA-SAM2_monuseg.pth \
  --sam_config sam2_hiera_l \
  --texture --context \
  --overlap 92 \
  --test_nms_thr 12 \
  --b 1 \
  --exp_name stage0_casam2_monuseg \
  --dump_eval_artifacts_dir ./logs/stainpqr_stage0/casam2_monuseg_test
```

```bash
python tools/analyze_eval_artifacts.py \
  --artifacts_dir ./logs/stainpqr_stage0/casam2_monuseg_test
```

### MoNuSeg StainPMS

```bash
python main.py --eval \
  --dataset monuseg \
  --data_path ./data/monuseg \
  --sam_ckpt ./logs/<stainpms_monuseg_exp>/Model/base_pq_epoch.pth \
  --sam_config sam2_hiera_l \
  --texture --context \
  --overlap 92 \
  --test_nms_thr 12 \
  --b 1 \
  --exp_name stage0_stainpms_monuseg \
  --dump_eval_artifacts_dir ./logs/stainpqr_stage0/stainpms_monuseg_test
```

```bash
python tools/analyze_eval_artifacts.py \
  --artifacts_dir ./logs/stainpqr_stage0/stainpms_monuseg_test
```

### TNBC CA-SAM2

```bash
python main.py --eval \
  --dataset monuseg \
  --data_path ./data/tnbc \
  --sam_ckpt ./checkpoints/CA-SAM2_tnbc.pth \
  --sam_config sam2_hiera_l \
  --texture --context \
  --overlap 32 \
  --test_nms_thr 12 \
  --b 1 \
  --exp_name stage0_casam2_tnbc \
  --dump_eval_artifacts_dir ./logs/stainpqr_stage0/casam2_tnbc_test
```

```bash
python tools/analyze_eval_artifacts.py \
  --artifacts_dir ./logs/stainpqr_stage0/casam2_tnbc_test
```

### TNBC StainPMS

```bash
python main.py --eval \
  --dataset monuseg \
  --data_path ./data/tnbc \
  --sam_ckpt ./logs/<stainpms_tnbc_exp>/Model/base_pq_epoch.pth \
  --sam_config sam2_hiera_l \
  --texture --context \
  --overlap 32 \
  --test_nms_thr 12 \
  --b 1 \
  --exp_name stage0_stainpms_tnbc \
  --dump_eval_artifacts_dir ./logs/stainpqr_stage0/stainpms_tnbc_test
```

```bash
python tools/analyze_eval_artifacts.py \
  --artifacts_dir ./logs/stainpqr_stage0/stainpms_tnbc_test
```

## Stage 0 Success Criteria

Proceed to Stage 1 only after the reproduced metrics are close to the VCIP
numbers:

| Dataset | Method | Expected PQ |
| --- | --- | ---: |
| MoNuSeg | CA-SAM2 | 0.620 |
| MoNuSeg | StainPMS | 0.658 |
| TNBC | CA-SAM2 | 0.676 |
| TNBC | StainPMS | 0.682 |

The error audit should show whether the remaining failures are mostly:

- missed low-overlap nuclei,
- near-threshold unmatched GT,
- weak matched masks,
- split-like unmatched GT,
- merge-like unmatched predictions.

Stage 1 will use these artifacts to build the oracle corrective-action dataset.

## Stage 1A: Candidate Audit Before Decoder Oracle

Goal:

1. Check whether residual hematoxylin peaks outside current predicted coverage
   hit the remaining FN nuclei.
2. Check whether internal multi-peak masks cover merge-like predictions.
3. Check whether raw proxy scores can rank weak/FP selected instances under a
   small per-image budget.

Run this first on the StainPMS artifacts, because StainPQR is meant to refine
the StainPMS first-pass output.

### MoNuSeg StainPMS Candidate Audit

```bash
python tools/stage1_candidate_audit.py \
  --artifacts_dir ./logs/stainpqr_stage0/stainpms_monuseg_test \
  --data_path ./data/monuseg \
  --split test
```

### TNBC StainPMS Candidate Audit

```bash
python tools/stage1_candidate_audit.py \
  --artifacts_dir ./logs/stainpqr_stage0/stainpms_tnbc_test \
  --data_path ./data/tnbc \
  --split test
```

Optional comparison against CA-SAM2 first-pass artifacts:

```bash
python tools/stage1_candidate_audit.py \
  --artifacts_dir ./logs/stainpqr_stage0/casam2_monuseg_test \
  --data_path ./data/monuseg \
  --split test

python tools/stage1_candidate_audit.py \
  --artifacts_dir ./logs/stainpqr_stage0/casam2_tnbc_test \
  --data_path ./data/tnbc \
  --split test
```

Key fields in `stage1a_candidate_audit.json`:

- `coverage_recall_fn`: fraction of remaining FNs touched by residual H peaks.
- `coverage_recall_near_fn`: residual-peak recall on near-threshold FNs.
- `coverage_recall_missed_fn`: residual-peak recall on low-overlap missed FNs.
- `merge_peak_recall`: internal multi-peak recall on merge-like predictions.
- `proxy_topk`: precision/recall of simple proxy ranking at budgets 2/4/8/12.

Proceed to the GPU decoder oracle only if at least one candidate family has
non-trivial recall on the residual error pool.

## Stage 1B: Coverage-Action Decoder Oracle

Goal:

Measure whether residual-coverage candidates actually improve PQ after one
frozen SAM2 mask-decoder pass. This produces action-level labels for later
utility/risk learning.

The first oracle is intentionally limited to coverage actions:

```text
residual H peak outside current predicted coverage
  -> one positive point prompt
  -> frozen decoder mask
  -> insert uncovered region as a new instance
  -> compute global Delta PQ / DQ / SQ / AJI
```

### MoNuSeg StainPMS Coverage Oracle

```bash
python main.py \
  --stage1_coverage_oracle \
  --dataset monuseg \
  --data_path ./data/monuseg \
  --sam_ckpt ../CA-SAM2-HRC/deliver_ckpts/monuseg_pms_best_pq.pth \
  --sam_config sam2_hiera_l \
  --texture --context \
  --overlap 92 \
  --test_nms_thr 12 \
  --b 1 \
  --oracle_split test \
  --oracle_artifacts_dir ./logs/stainpqr_stage0/stainpms_monuseg_test \
  --oracle_out_dir ./logs/stainpqr_stage1b/coverage_oracle_stainpms_monuseg
```

Debug on the first two images:

```bash
python main.py \
  --stage1_coverage_oracle \
  --dataset monuseg \
  --data_path ./data/monuseg \
  --sam_ckpt ../CA-SAM2-HRC/deliver_ckpts/monuseg_pms_best_pq.pth \
  --sam_config sam2_hiera_l \
  --texture --context \
  --overlap 92 \
  --test_nms_thr 12 \
  --b 1 \
  --oracle_split test \
  --oracle_max_images 2 \
  --oracle_artifacts_dir ./logs/stainpqr_stage0/stainpms_monuseg_test \
  --oracle_out_dir ./logs/stainpqr_stage1b/debug_coverage_oracle_stainpms_monuseg
```

### TNBC StainPMS Coverage Oracle

```bash
python main.py \
  --stage1_coverage_oracle \
  --dataset monuseg \
  --data_path ./data/tnbc \
  --sam_ckpt ../CA-SAM2-HRC/deliver_ckpts/tnbc_pms_best_e156.pth \
  --sam_config sam2_hiera_l \
  --texture --context \
  --overlap 32 \
  --test_nms_thr 12 \
  --b 1 \
  --oracle_split test \
  --oracle_artifacts_dir ./logs/stainpqr_stage0/stainpms_tnbc_test \
  --oracle_out_dir ./logs/stainpqr_stage1b/coverage_oracle_stainpms_tnbc
```

Key outputs:

- `actions.csv`: one row per decoded corrective action, with Delta PQ/DQ/SQ/AJI.
- `images.csv`: per-image action counts.
- `summary.json`: positive/harmful action rates and Delta PQ grouped by target type.

Analyze simple ranking baselines after the oracle finishes:

```bash
python tools/analyze_oracle_actions.py \
  --actions_csv ./logs/stainpqr_stage1b/coverage_oracle_stainpms_monuseg/actions.csv

python tools/analyze_oracle_actions.py \
  --actions_csv ./logs/stainpqr_stage1b/coverage_oracle_stainpms_tnbc/actions.csv
```

Combined MoNuSeg + TNBC analysis:

```bash
python tools/analyze_oracle_actions.py \
  --actions_csv \
    ./logs/stainpqr_stage1b/coverage_oracle_stainpms_monuseg/actions.csv \
    ./logs/stainpqr_stage1b/coverage_oracle_stainpms_tnbc/actions.csv \
  --out_prefix ./logs/stainpqr_stage1b/coverage_oracle_combined_action_analysis
```
