# StainRoute Stage 0: AutoDL execution

Run these commands on the AutoDL 4090 only after the `research/stainroute`
commit has been transferred there. Do not change a command, checkpoint, seed,
overlap, NMS threshold, test split, or TTA setting after seeing a metric.

All commands write their terminal output into the corresponding artifact
directory. They also write unrounded `main_eval_metrics.json`; the final
reconciliation command uses this structured value rather than hand-copied or
rounded terminal values. The stdout log remains a backward-compatible backup.

```bash
set -euo pipefail
cd /path/to/StainPMS-ICASSP2026
conda activate CA-SAM2
git rev-parse HEAD
mkdir -p logs/stainroute/stage0
```

Use the Git-tracked `configs/stainroute_stage0_runs.example.json` directly;
it contains the canonical shared-checkpoint paths. If a future environment
genuinely uses different paths, copy it to a distinct untracked filename and
pass that filename explicitly to the reconciliation command. Keep the
`command` fields consistent with the invoked commands and do not edit their
evaluation settings.

## MoNuSeg CA-SAM2

```bash
mkdir -p logs/stainroute/stage0/casam2_monuseg_test
python main.py --eval --dataset monuseg --data_path ./data/monuseg --sam_ckpt ../CA-SAM2-HRC/checkpoints/CA-SAM2_monuseg.pth --sam_config sam2_hiera_l --texture --context --overlap 92 --test_nms_thr 12 --b 1 --seed 3407 --exp_name stainroute_stage0_casam2_monuseg --dump_eval_artifacts_dir ./logs/stainroute/stage0/casam2_monuseg_test 2>&1 | tee logs/stainroute/stage0/casam2_monuseg_test/main_stdout.log
```

## MoNuSeg StainPMS

```bash
mkdir -p logs/stainroute/stage0/stainpms_monuseg_test
python main.py --eval --dataset monuseg --data_path ./data/monuseg --sam_ckpt ../CA-SAM2-HRC/deliver_ckpts/monuseg_pms_best_pq.pth --sam_config sam2_hiera_l --texture --context --overlap 92 --test_nms_thr 12 --b 1 --seed 3407 --exp_name stainroute_stage0_stainpms_monuseg --dump_eval_artifacts_dir ./logs/stainroute/stage0/stainpms_monuseg_test 2>&1 | tee logs/stainroute/stage0/stainpms_monuseg_test/main_stdout.log
```

## TNBC CA-SAM2

```bash
mkdir -p logs/stainroute/stage0/casam2_tnbc_test
python main.py --eval --dataset monuseg --data_path ./data/tnbc --sam_ckpt ../CA-SAM2-HRC/deliver_ckpts/tnbc_baseline_best_e147.pth --sam_config sam2_hiera_l --texture --context --overlap 32 --test_nms_thr 12 --b 1 --seed 3407 --exp_name stainroute_stage0_casam2_tnbc --dump_eval_artifacts_dir ./logs/stainroute/stage0/casam2_tnbc_test 2>&1 | tee logs/stainroute/stage0/casam2_tnbc_test/main_stdout.log
```

## TNBC StainPMS

```bash
mkdir -p logs/stainroute/stage0/stainpms_tnbc_test
python main.py --eval --dataset monuseg --data_path ./data/tnbc --sam_ckpt ../CA-SAM2-HRC/deliver_ckpts/tnbc_pms_best_e156.pth --sam_config sam2_hiera_l --texture --context --overlap 32 --test_nms_thr 12 --b 1 --seed 3407 --exp_name stainroute_stage0_stainpms_tnbc --dump_eval_artifacts_dir ./logs/stainroute/stage0/stainpms_tnbc_test 2>&1 | tee logs/stainroute/stage0/stainpms_tnbc_test/main_stdout.log
```

## Metric reconciliation

```bash
python tools/stainroute_stage0_reconcile.py --spec configs/stainroute_stage0_runs.example.json --out-dir logs/stainroute/stage0
```

The command creates the required raw evidence files:

- `logs/stainroute/stage0/baseline_manifest.json`
- `logs/stainroute/stage0/baseline_metrics.csv`
- `logs/stainroute/stage0/metric_reconciliation_diagnostics.csv`

It returns non-zero if the main evaluation, artifact analyzer, and factorized
PQ paths disagree beyond `2e-6`. The diagnostics CSV identifies the per-image
factorized-PQ difference and any exact-IoU-0.5 pairs. Do not continue to Stage
1 when it fails.

## TNBC historical-anchor diagnostic: NMS threshold 2

This is a Stage 0 diagnosis only, not a replacement for the canonical NMS-12
evaluation. Historical project records state that the older TNBC PQ anchor was
measured at NMS threshold 2, while the canonical StainRoute baseline above uses
threshold 12. Run these two frozen-checkpoint evaluations once to verify or
reject NMS threshold as the explanation for the `0.682` versus `0.6681`
difference.

```bash
mkdir -p logs/stainroute/stage0/tnbc_nms2_diagnostic/casam2_test
python main.py --eval --dataset monuseg --data_path ./data/tnbc --sam_ckpt ../CA-SAM2-HRC/deliver_ckpts/tnbc_baseline_best_e147.pth --sam_config sam2_hiera_l --texture --context --overlap 32 --test_nms_thr 2 --b 1 --seed 3407 --exp_name stainroute_stage0_tnbc_nms2_casam2 --dump_eval_artifacts_dir ./logs/stainroute/stage0/tnbc_nms2_diagnostic/casam2_test 2>&1 | tee logs/stainroute/stage0/tnbc_nms2_diagnostic/casam2_test/main_stdout.log

mkdir -p logs/stainroute/stage0/tnbc_nms2_diagnostic/stainpms_test
python main.py --eval --dataset monuseg --data_path ./data/tnbc --sam_ckpt ../CA-SAM2-HRC/deliver_ckpts/tnbc_pms_best_e156.pth --sam_config sam2_hiera_l --texture --context --overlap 32 --test_nms_thr 2 --b 1 --seed 3407 --exp_name stainroute_stage0_tnbc_nms2_stainpms --dump_eval_artifacts_dir ./logs/stainroute/stage0/tnbc_nms2_diagnostic/stainpms_test 2>&1 | tee logs/stainroute/stage0/tnbc_nms2_diagnostic/stainpms_test/main_stdout.log

python tools/stainroute_stage0_reconcile.py --spec configs/stainroute_stage0_tnbc_nms2_diagnostic.example.json --out-dir logs/stainroute/stage0/tnbc_nms2_diagnostic
```

Do not change the canonical NMS-12 configuration or proceed to Stage 1 based
on this diagnostic alone.
