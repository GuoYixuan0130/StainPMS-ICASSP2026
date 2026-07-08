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

## Stage 2A: Coverage Utility Selector

Goal:

Train a lightweight selector on Stage 1B oracle labels and compare it against
the strongest rule baselines. The first model is intentionally small:

```text
features: residual evidence, decoded IoU, stability, decoded/added area, action rank
heads: logistic P(Delta PQ > 0) + ridge E[Delta PQ]
initial score: P(Delta PQ > 0) * max(E[Delta PQ], 0)
```

The first AutoDL summaries showed that `selector_prob` can be useful on
MoNuSeg, but the regression-based expected-utility score is not yet reliable
and TNBC/combined runs need strict NaN handling. The current script therefore
also reports hybrid scores that test whether learned probability improves the
strong hand-built rules:

- `selector_prob_added_area`
- `selector_prob_missed_like`
- `selector_prob_iou_area`

Use `best_budget_methods` in `summary.json` to see which non-oracle ranking wins
each budget. For this stage, a useful selector must beat `missed_like_proxy` and
`added_area` at budgets 1/2/4; action-level AUROC/AP alone is not enough.

Current test-oracle group-CV reading:

- MoNuSeg: `selector_prob_iou_area` is the strongest stable learned score at
  budgets 2/4.
- TNBC: `selector_prob_added_area` is strongest at budget 1, while
  `missed_like_proxy` is still competitive at budget 2.
- Combined: learned probability hybrids beat the hand-built rules at budgets
  1/2, but larger budgets can over-correct.

Treat these as feasibility results only, because the labels come from test
oracle actions. The next paper-grade check is train-oracle to test-oracle
holdout.

Group-CV on MoNuSeg:

```bash
python tools/train_coverage_selector.py \
  --train_actions ./logs/stainpqr_stage1b/coverage_oracle_stainpms_monuseg/actions.csv \
  --out_dir ./logs/stainpqr_stage2a/coverage_selector_monuseg_cv
```

Group-CV on TNBC:

```bash
python tools/train_coverage_selector.py \
  --train_actions ./logs/stainpqr_stage1b/coverage_oracle_stainpms_tnbc/actions.csv \
  --out_dir ./logs/stainpqr_stage2a/coverage_selector_tnbc_cv
```

Cross-dataset checks:

```bash
python tools/train_coverage_selector.py \
  --train_actions ./logs/stainpqr_stage1b/coverage_oracle_stainpms_monuseg/actions.csv \
  --test_actions ./logs/stainpqr_stage1b/coverage_oracle_stainpms_tnbc/actions.csv \
  --out_dir ./logs/stainpqr_stage2a/coverage_selector_train_monuseg_test_tnbc

python tools/train_coverage_selector.py \
  --train_actions ./logs/stainpqr_stage1b/coverage_oracle_stainpms_tnbc/actions.csv \
  --test_actions ./logs/stainpqr_stage1b/coverage_oracle_stainpms_monuseg/actions.csv \
  --out_dir ./logs/stainpqr_stage2a/coverage_selector_train_tnbc_test_monuseg
```

Combined group-CV:

```bash
python tools/train_coverage_selector.py \
  --train_actions \
    ./logs/stainpqr_stage1b/coverage_oracle_stainpms_monuseg/actions.csv \
    ./logs/stainpqr_stage1b/coverage_oracle_stainpms_tnbc/actions.csv \
  --out_dir ./logs/stainpqr_stage2a/coverage_selector_combined_cv
```

Key outputs:

- `summary.json`: AUROC/AP, Brier/ECE, and budget curves for learned/rule scores.
- `best_budget_methods`: best non-oracle score per budget by selected
  `delta_pq_sum`.
- `selector_model.json`: feature normalization and learned linear weights.
- `predictions.csv`: action-level predictions for plotting and error analysis.

## Stage 2B: Train-Oracle to Test-Oracle Holdout

Goal:

Generate oracle labels on the training split, train the selector only on those
training actions, and evaluate selection on the held-out test action CSVs from
Stage 1B. This removes the main leakage concern from the Stage 2A group-CV
feasibility audit.

### MoNuSeg Train Artifacts

```bash
rm -rf ./logs/stainpqr_stage0/stainpms_monuseg_train

python main.py --eval --eval_on_train \
  --dataset monuseg \
  --data_path ./data/monuseg \
  --sam_ckpt ../CA-SAM2-HRC/deliver_ckpts/monuseg_pms_best_pq.pth \
  --sam_config sam2_hiera_l \
  --texture --context \
  --overlap 92 \
  --test_nms_thr 12 \
  --b 1 \
  --exp_name stage0_stainpms_monuseg_train \
  --dump_eval_artifacts_dir ./logs/stainpqr_stage0/stainpms_monuseg_train

python tools/analyze_eval_artifacts.py \
  --artifacts_dir ./logs/stainpqr_stage0/stainpms_monuseg_train
```

### MoNuSeg Train Coverage Oracle

```bash
rm -rf ./logs/stainpqr_stage1b/coverage_oracle_stainpms_monuseg_train

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
  --oracle_split train \
  --oracle_artifacts_dir ./logs/stainpqr_stage0/stainpms_monuseg_train \
  --oracle_out_dir ./logs/stainpqr_stage1b/coverage_oracle_stainpms_monuseg_train
```

### TNBC Train Artifacts

```bash
rm -rf ./logs/stainpqr_stage0/stainpms_tnbc_train

python main.py --eval --eval_on_train \
  --dataset monuseg \
  --data_path ./data/tnbc \
  --sam_ckpt ../CA-SAM2-HRC/deliver_ckpts/tnbc_pms_best_e156.pth \
  --sam_config sam2_hiera_l \
  --texture --context \
  --overlap 32 \
  --test_nms_thr 12 \
  --b 1 \
  --exp_name stage0_stainpms_tnbc_train \
  --dump_eval_artifacts_dir ./logs/stainpqr_stage0/stainpms_tnbc_train

python tools/analyze_eval_artifacts.py \
  --artifacts_dir ./logs/stainpqr_stage0/stainpms_tnbc_train
```

### TNBC Train Coverage Oracle

```bash
rm -rf ./logs/stainpqr_stage1b/coverage_oracle_stainpms_tnbc_train

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
  --oracle_split train \
  --oracle_artifacts_dir ./logs/stainpqr_stage0/stainpms_tnbc_train \
  --oracle_out_dir ./logs/stainpqr_stage1b/coverage_oracle_stainpms_tnbc_train
```

### Holdout Selector Evaluation

```bash
python tools/train_coverage_selector.py \
  --train_actions ./logs/stainpqr_stage1b/coverage_oracle_stainpms_monuseg_train/actions.csv \
  --test_actions ./logs/stainpqr_stage1b/coverage_oracle_stainpms_monuseg/actions.csv \
  --out_dir ./logs/stainpqr_stage2b/coverage_selector_monuseg_train_to_test

python tools/train_coverage_selector.py \
  --train_actions ./logs/stainpqr_stage1b/coverage_oracle_stainpms_tnbc_train/actions.csv \
  --test_actions ./logs/stainpqr_stage1b/coverage_oracle_stainpms_tnbc/actions.csv \
  --out_dir ./logs/stainpqr_stage2b/coverage_selector_tnbc_train_to_test

python tools/train_coverage_selector.py \
  --train_actions \
    ./logs/stainpqr_stage1b/coverage_oracle_stainpms_monuseg_train/actions.csv \
    ./logs/stainpqr_stage1b/coverage_oracle_stainpms_tnbc_train/actions.csv \
  --test_actions \
    ./logs/stainpqr_stage1b/coverage_oracle_stainpms_monuseg/actions.csv \
    ./logs/stainpqr_stage1b/coverage_oracle_stainpms_tnbc/actions.csv \
  --out_dir ./logs/stainpqr_stage2b/coverage_selector_combined_train_to_test
```

Stage 2B interpretation so far:

- MoNuSeg train -> test: `selector_prob_iou_area` is the preferred learned
  score, especially at budgets 2/4.
- TNBC train -> test: `selector_prob_added_area` is strongest at budget 1 and
  remains competitive at budget 2.
- Combined train -> test: mixed calibration is less reliable because the TNBC
  train oracle has very few positive coverage actions. Prefer dataset-specific
  selectors unless later calibration fixes this.

## Stage 2C: True Selective Re-Decoding

Goal:

Validate that the selected actions still improve full-image metrics when they
are executed together. Stage 2A/2B budget curves add single-action Delta PQ
values, but selected actions can interact after insertion. Stage 2C replays the
selected actions with the frozen decoder, merges the masks into the first-pass
prediction, and recomputes Dice/AJI/DQ/SQ/PQ.

After pulling the Stage 2C code, rerun the relevant holdout selector commands
from Stage 2B once so `predictions.csv` contains action coordinates and baseline
score aliases.

Recommended first check on MoNuSeg:

```bash
rm -rf ./logs/stainpqr_stage2c/monuseg_b2_selector_prob_iou_area

python main.py \
  --stage2_selective_refine \
  --dataset monuseg \
  --data_path ./data/monuseg \
  --sam_ckpt ../CA-SAM2-HRC/deliver_ckpts/monuseg_pms_best_pq.pth \
  --sam_config sam2_hiera_l \
  --texture --context \
  --overlap 92 \
  --test_nms_thr 12 \
  --b 1 \
  --selective_split test \
  --selective_artifacts_dir ./logs/stainpqr_stage0/stainpms_monuseg_test \
  --selective_actions_csv ./logs/stainpqr_stage1b/coverage_oracle_stainpms_monuseg/actions.csv \
  --selective_predictions_csv ./logs/stainpqr_stage2b/coverage_selector_monuseg_train_to_test/predictions.csv \
  --selective_score selector_prob_iou_area \
  --selective_budget 2 \
  --selective_out_dir ./logs/stainpqr_stage2c/monuseg_b2_selector_prob_iou_area
```

Recommended first check on TNBC:

```bash
rm -rf ./logs/stainpqr_stage2c/tnbc_b1_selector_prob_added_area

python main.py \
  --stage2_selective_refine \
  --dataset monuseg \
  --data_path ./data/tnbc \
  --sam_ckpt ../CA-SAM2-HRC/deliver_ckpts/tnbc_pms_best_e156.pth \
  --sam_config sam2_hiera_l \
  --texture --context \
  --overlap 32 \
  --test_nms_thr 12 \
  --b 1 \
  --selective_split test \
  --selective_artifacts_dir ./logs/stainpqr_stage0/stainpms_tnbc_test \
  --selective_actions_csv ./logs/stainpqr_stage1b/coverage_oracle_stainpms_tnbc/actions.csv \
  --selective_predictions_csv ./logs/stainpqr_stage2b/coverage_selector_tnbc_train_to_test/predictions.csv \
  --selective_score selector_prob_added_area \
  --selective_budget 1 \
  --selective_out_dir ./logs/stainpqr_stage2c/tnbc_b1_selector_prob_added_area
```

Key outputs:

- `summary.json`: base/refined full-image metrics and deltas.
- `image_metrics.csv`: per-image base/refined metrics.
- `selected_actions.csv`: selected actions, scores, decode status, and applied
  added area.
- `<image>_pred.npy`: refined prediction maps for optional qualitative figures.

Current true-replay reading:

- MoNuSeg: `selector_prob_iou_area` improves PQ at budgets 1/2/4. Budget 4 is
  currently highest (`+0.00104` PQ, `+0.00462` AJI), but it also inserts more FP
  than budget 2. Budget 8 turns PQ negative (`-0.00029`), so the curve has a
  clear over-correction point.
- MoNuSeg: at budget 2, `selector_prob_iou_area` (`+0.00068` PQ) beats
  `missed_like_proxy` (`+0.00027` PQ), while raw `residual_evidence` hurts PQ
  (`-0.00048`). At budget 4, the learned score (`+0.00104` PQ) also beats
  `missed_like_proxy` (`+0.00014` PQ).
- TNBC: budget 1 is the useful setting. `selector_prob_added_area` improves PQ
  by about `+0.00828`, while budget 2 over-corrects and drops most of the gain.
  The rule baseline behaves similarly: `missed_like_proxy` is strong at budget 1
  (`+0.00818` PQ) but drops at budget 2 (`+0.00063` PQ).
- TNBC: raw `residual_evidence` at budget 1 is clearly harmful (`-0.01036` PQ),
  proving that stain residual peaks need risk selection.
- In both datasets the gain mainly comes from DQ/FN recovery, while SQ can
  decrease when inserted masks add FP or lower-quality matches.

Completed minimal checks:

1. Raw residual-evidence ranking at the same budgets, proving that selection is
   needed.
2. MoNuSeg budget 4 for the learned score, showing a larger but FP-heavier gain.
3. MoNuSeg budget 8 and TNBC budget 2, showing where over-correction begins.

MoNuSeg raw residual baseline:

```bash
rm -rf ./logs/stainpqr_stage2c/monuseg_b2_residual_evidence

python main.py \
  --stage2_selective_refine \
  --dataset monuseg \
  --data_path ./data/monuseg \
  --sam_ckpt ../CA-SAM2-HRC/deliver_ckpts/monuseg_pms_best_pq.pth \
  --sam_config sam2_hiera_l \
  --texture --context \
  --overlap 92 \
  --test_nms_thr 12 \
  --b 1 \
  --selective_split test \
  --selective_artifacts_dir ./logs/stainpqr_stage0/stainpms_monuseg_test \
  --selective_actions_csv ./logs/stainpqr_stage1b/coverage_oracle_stainpms_monuseg/actions.csv \
  --selective_predictions_csv ./logs/stainpqr_stage2b/coverage_selector_monuseg_train_to_test/predictions.csv \
  --selective_score residual_evidence \
  --selective_budget 2 \
  --selective_out_dir ./logs/stainpqr_stage2c/monuseg_b2_residual_evidence
```

MoNuSeg learned budget 4:

```bash
rm -rf ./logs/stainpqr_stage2c/monuseg_b4_selector_prob_iou_area

python main.py \
  --stage2_selective_refine \
  --dataset monuseg \
  --data_path ./data/monuseg \
  --sam_ckpt ../CA-SAM2-HRC/deliver_ckpts/monuseg_pms_best_pq.pth \
  --sam_config sam2_hiera_l \
  --texture --context \
  --overlap 92 \
  --test_nms_thr 12 \
  --b 1 \
  --selective_split test \
  --selective_artifacts_dir ./logs/stainpqr_stage0/stainpms_monuseg_test \
  --selective_actions_csv ./logs/stainpqr_stage1b/coverage_oracle_stainpms_monuseg/actions.csv \
  --selective_predictions_csv ./logs/stainpqr_stage2b/coverage_selector_monuseg_train_to_test/predictions.csv \
  --selective_score selector_prob_iou_area \
  --selective_budget 4 \
  --selective_out_dir ./logs/stainpqr_stage2c/monuseg_b4_selector_prob_iou_area
```

TNBC raw residual baseline:

```bash
rm -rf ./logs/stainpqr_stage2c/tnbc_b1_residual_evidence

python main.py \
  --stage2_selective_refine \
  --dataset monuseg \
  --data_path ./data/tnbc \
  --sam_ckpt ../CA-SAM2-HRC/deliver_ckpts/tnbc_pms_best_e156.pth \
  --sam_config sam2_hiera_l \
  --texture --context \
  --overlap 32 \
  --test_nms_thr 12 \
  --b 1 \
  --selective_split test \
  --selective_artifacts_dir ./logs/stainpqr_stage0/stainpms_tnbc_test \
  --selective_actions_csv ./logs/stainpqr_stage1b/coverage_oracle_stainpms_tnbc/actions.csv \
  --selective_predictions_csv ./logs/stainpqr_stage2b/coverage_selector_tnbc_train_to_test/predictions.csv \
  --selective_score residual_evidence \
  --selective_budget 1 \
  --selective_out_dir ./logs/stainpqr_stage2c/tnbc_b1_residual_evidence
```
