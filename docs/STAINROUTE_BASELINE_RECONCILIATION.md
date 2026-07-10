# StainRoute Stage 0 Baseline Reconciliation

Status: **BLOCKED — do not enter Stage 1**

Date: 2026-07-10

## Scope and evidence

The frozen evaluations below used the standard test splits, TTA disabled,
batch size 1, seed 3407, texture/context enabled, and the artifact dump path.
The raw outputs are intentionally untracked and remain under
`logs/stainroute/stage0/` on AutoDL.

| Evidence run | Git SHA | Raw output |
| --- | --- | --- |
| Canonical NMS-12 reconciliation | `2a1348cb7a1158a6f77aae2f92c168f9552d8068` | `logs/stainroute/stage0/baseline_manifest.json`, `baseline_metrics.csv` |
| TNBC NMS-2 diagnostic | `79ba59655f2f94cba9722dfee913483a73250c7e` | `logs/stainroute/stage0/tnbc_nms2_diagnostic/baseline_manifest.json`, `baseline_metrics.csv` |
| Historical-code replay | `8c82290b0c8d05b8bdf8c4435689060cd9d31c15` | terminal output retained by the operator |

The exact AutoDL GPU, Python, PyTorch, and CUDA versions were not captured in
the returned manifests. They must be recorded before any future Stage 1 run.

## Checkpoints evaluated

| Dataset | Model | Path | SHA256 |
| --- | --- | --- | --- |
| MoNuSeg | CA-SAM2 | `../CA-SAM2-HRC/checkpoints/CA-SAM2_monuseg.pth` | `33bc933508c96b7b8332c27185fc1b24da83c90fa82598b265af6e72a3a059cd` |
| MoNuSeg | StainPMS | `../CA-SAM2-HRC/deliver_ckpts/monuseg_pms_best_pq.pth` | `6616c24626a162580d79aa7035d0b6c1a4ae6240a6c2f4599960dd4efac95db1` |
| TNBC | CA-SAM2 | `../CA-SAM2-HRC/deliver_ckpts/tnbc_baseline_best_e147.pth` | `a73f6b16544572dda931f5b3eac479f4c9574ab2417866a8a87c97bbda0f74c6` |
| TNBC | StainPMS | `../CA-SAM2-HRC/deliver_ckpts/tnbc_pms_best_e156.pth` | `44a3cb3e93051301d789e44f93769588abfa727d7a174b0270f55305ef023781` |

## Canonical NMS-12 result

| Dataset | Model | Expected PQ | Reproduced PQ | Difference | AJI | DQ | SQ | Test images |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MoNuSeg | CA-SAM2 | 0.620000 | 0.619861 | -0.000139 | 0.643610 | 0.826324 | 0.749444 | 14 |
| MoNuSeg | StainPMS | 0.658000 | 0.657768 | -0.000232 | 0.666702 | 0.852661 | 0.770973 | 14 |
| TNBC | CA-SAM2 | 0.676000 | 0.663411 | -0.012589 | 0.622006 | 0.831497 | 0.797116 | 13 |
| TNBC | StainPMS | 0.682000 | 0.668077 | -0.013923 | 0.647064 | 0.830578 | 0.803866 | 13 |

TNBC test is patients 9–11 and contains 13 images (`09_1` through `11_3`).

## Metric consistency

For all four NMS-12 runs, the main evaluation and
`tools/analyze_eval_artifacts.py` values are exactly equal. The factorized PQ
check is within `2e-6` for every run.

One MoNuSeg StainPMS pair on `TCGA-EJ-A46H-01A-03-TSC` has IoU exactly 0.5.
The old strict `>0.5` implementation produced a 0.00006803 dataset-mean PQ
difference. Commit `2a1348c` unified main evaluation, artifact analysis, and
factorized PQ on the specified inclusive `IoU >= 0.5` one-to-one matching
definition. This does not affect TNBC because no exact-threshold pair was
observed there.

## TNBC root-cause audit

An NMS-2 diagnostic used exactly the same checkpoints, data, split, overlap,
seed, and evaluation options except `--test_nms_thr 2`.

| TNBC model | PQ at NMS=12 | PQ at NMS=2 | Historical anchor |
| --- | ---: | ---: | ---: |
| CA-SAM2 | 0.663411 | 0.666447 | 0.676000 |
| StainPMS | 0.668077 | 0.669939 | 0.682000 |

NMS threshold therefore explains only part of the discrepancy.

A replay with historical source commit `8c82290` and NMS=2 produced terminal
PQ values of 0.6664 (CA-SAM2, epoch 147) and 0.6699 (StainPMS, epoch 156),
matching the current-code NMS-2 results to the printed precision. This excludes
release-code drift as the explanation.

The remaining unverified provenance is the identity of the checkpoint and
TNBC instance-label snapshot used for the historical paper values. The current
AutoDL project has no retained historical `logs/*tnbc*/Model/base_pq_epoch.pth`
or `base_aji_epoch.pth` from which to compare SHA256. Without either the
paper-time checkpoint SHA256/path or a paper-time TNBC data/label manifest,
the two remaining explanations (different model state versus different data
conversion snapshot) cannot be distinguished.

## Stop condition and required decision

The MoNuSeg anchors and all metric paths pass. The TNBC paper anchor cannot
yet be reproduced or attributed to one provable source artifact. Per the
StainRoute plan, Stage 1 is blocked.

The project lead must provide either:

1. the paper-time TNBC full/best checkpoint (or its SHA256 and recoverable
   path); or
2. a paper-time TNBC image/label manifest with checksums.

Only after that provenance check can the lead decide whether the current
NMS-12 `e147/e156` evaluation becomes the new canonical baseline. No model
fine-tuning or Stage 1 oracle/router work is authorized while this report is
blocked.

## Addendum — 2026-07-10: conditional Stage 1 authorization

The project lead reviewed this report and authorized a **conditional** Stage 1
oracle feasibility study. Historical TNBC provenance remains unresolved.

- Current e147/e156 checkpoints, the current TNBC data snapshot, and NMS=12
  are accepted as **StainRoute Development Baseline v1**.
- This authorization unblocks only train/calibration ADD/SPLIT oracle studies.
  It does not authorize router training, test-split oracle enumeration, model
  fine-tuning, boundary actions, or final paper experiments.
- All final TNBC comparisons must be rerun against Baseline v1. Historical
  paper TNBC values must never be mixed into a new-method comparison.
- A subsequently recovered historical artifact is a separate provenance
  experiment and must not silently replace Baseline v1.

The development-baseline specification is
`configs/stainroute/baseline_v1.yaml`; its generated host/data/checksum
manifest is `logs/stainroute/stage1/baseline_v1_manifest.json`.
