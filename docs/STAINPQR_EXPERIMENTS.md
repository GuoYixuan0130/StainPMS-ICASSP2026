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
