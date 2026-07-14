# PromptQ-v2 Primary-Metric Audit — Final NO-GO Record

**Status:** FORMALLY TERMINATED / NO-GO  
**Lead confirmation:** 2026-07-14

## Immutable audit identity

- Canonical full-supervision StainPMS baseline: `2a1348cb7a1158a6f77aae2f92c168f9552d8068`
- PromptQ-v2 audit code commit: `3bff25901921e5611a5f5185c4fb9ef2ecd308ee`
- Checkpoint: `deliver_ckpts/tnbc_pms_best_e156.pth`
- Checkpoint SHA256: `44a3cb3e93051301d789e44f93769588abfa727d7a174b0270f55305ef023781`
- Formal AutoDL artifact (read-only):
  `logs/promptq_v2/primary_metric/promptq_v2_primary_20260714_135742/`
- Artifact integrity manifest: `SHA256SUMS` in the formal artifact directory.

The audit used TTA-off, batch size 1, seed 3407, point-NMS radius 12, and
inclusive instance IoU >= 0.5.  It trained only on TNBC patients 1--6 and
performed one formal development audit on all seven patients-7--8 images.
TNBC patients 9--11 and MoNuSeg were not opened.

## Formal primary-metric result

| Path | AJI | PQ | DQ | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.750788 | 0.742225 | 0.927965 | 889 | 82 | 51 |
| PromptQ-v2 product score | 0.750549 | 0.740339 | 0.925236 | 889 | 87 | 51 |
| Quality-only diagnostic | 0.750367 | 0.739845 | 0.924790 | 889 | 88 | 51 |
| GT-IoU score oracle diagnostic | 0.751332 | 0.742947 | 0.929151 | 890 | 81 | 50 |

PromptQ-v2 product versus Baseline:

- ΔAJI = -0.000239
- ΔPQ = -0.001886
- ΔDQ = -0.002729

The GT-IoU score-only oracle has only +0.000544 AJI and +0.000722 PQ versus
Baseline.  This is insufficient remaining headroom for any score-only
intervention under the fixed candidate/mask/assembly contract.

## Final attribution and closure

With StainPMS candidate coordinates, decoded masks, and assembly rules fixed,
learned candidate-quality scoring has no usable AJI/PQ space.  Therefore the
**score-only, reranking, and quality-calibration routes are formally
terminated**.

This conclusion accepts both the primary metrics and the GT-IoU score oracle.
It must not be reopened through a rerun, new seed, longer training, changed
target/loss, alternate score formula, threshold search, or access to TNBC
patients 9--11 or MoNuSeg.  Spearman, ECE, and quality loss remain mechanism
descriptions only and are not grounds for a retry.

The formal artifact and all prior failed implementation-attempt directories
are retained read-only.  They are not Git inputs; this report is the Git
tracked custody record.
