# PromptQ TNBC development protocol

Date: 2026-07-11

This document fixes the only authorized next experiment after the immutable
PromptCredit-v1 corrected smoke.  PromptCredit-v1 remains automatic **FAIL**;
its directional mask-gradient path is retired and its artifact directory is
never modified.

## Scope and method boundary

PromptQ means **Frozen-Model Pre-Decode Mask-Utility Distillation**.  Every
inherited StainPMS and SAM2 parameter is frozen: point backbone, fusion conv,
deform/regression/classification heads, auxiliary mask head, image encoder,
prompt encoder, mask decoder, and memory modules.  The sole optimizer
parameter is the existing lightweight `quality_head` (under 0.1M parameters).

The quality head has no Dropout.  Its input is detached ROI feature data,
quantized FP16 in the one-pass cache and restored to FP32 before the head.  Its
last weight is zero and its final bias is the fixed `logit(0.01)` prior.  Thus
`prompt_credit_grad_scale=0`; coordinates are detached; mask loss is excluded
from backward; and the quality loss cannot reach shared features or inherited
parameters.

The frozen target and loss are:

```text
r = decoded hard-mask IoU with the existing nearest-matched GT
u = r * sigmoid((r - 0.5) / 0.1)
unmatched target = 0; duplicate source target = max(u)
Quality Focal Loss gamma = 2
```

The sole inference difference is the point-NMS ranking score:

```text
baseline: objectness
PromptQ: objectness * sigmoid(quality_logit)
```

Coordinates, logits, class decisions, semantic filtering, decoder calls,
NMS radius 12, instance assembly, texture/context, unclockwise crop traversal,
crop size 256, overlap 32, TTA-off and every threshold are fixed.  No
calibration/test-time action, additional candidate, image encoding, or decoder
call is allowed.

## Data and time limits

- Cache and quality-head training: TNBC patients 1-6 only.
- Development: the seven fixed patients-7-8 images only.  They may have been
  seen by the StainPMS initialization and are therefore explicitly not an
  independent leakage-free validation claim.
- Patients 9-11 remain closed.  The runner resolves only direct allowed IDs;
  it neither lists nor opens those paths.
- MoNuSeg, StainRoute, threshold tuning, a second seed, full TNBC training,
  and directional credit are prohibited.

The runner forecasts total PromptQ work at the first 10% of train-cache
extraction.  A forecast above six GPU hours saves partial artifacts and stops.

## Scalar-isolation gate

The first phase uses the immutable corrected-smoke two crops from `02_1`, 100
AdamW steps (`lr=1e-4`, `weight_decay=1e-4`, seed 3407), with a frozen baseline
and a quality-head-only PromptQ arm.  It must prove unchanged inherited
checksums and base outputs, finite execution, decreasing quality loss,
nonconstant quality prediction, target Spearman at least 0.60, product-score
IoU Spearman at least 0.20 above raw objectness, and no fixed-crop product-NMS
mean-IoU decrease.  Failure is NO-GO and prevents development extraction.

## Fixed development procedure

One no-grad/eval cache pass saves detached features and existing standard-path
utility only.  Unmatched proposals get target zero without extra decoding.
The cache head trains for exactly 20 epochs with AdamW (`lr=1e-4`,
`weight_decay=1e-4`, seed 3407); batch starts at 4096 and only halves on OOM.
Epoch 20 is fixed—no early stopping or selection.  Cached and online quality
logits must differ by less than `1e-6`.

Two frozen standard inference paths then run on the seven development images:
objectness baseline and objectness-times-quality PromptQ.  The output includes
per-image Dice1/Dice2, AJI, AJI+, DQ, SQ, PQ, prompt/decoder counts, runtime,
memory, calibration, NMS conflict-winner analysis, and a 2000-resample paired
image bootstrap (seed 3407).

## Command

Run on AutoDL 4090 only, with a new, non-existing run ID.  Do not use
`temple/` for output or transfer any cache/checkpoint/log artifact to Git.

```bash
git pull --ff-only origin research/promptcredit
python -m unittest discover -s tests/promptcredit -v
python tools/run_promptq_tnbc_dev.py \
  --data-root ./data/tnbc \
  --checkpoint ../CA-SAM2-HRC/deliver_ckpts/tnbc_pms_best_e156.pth \
  --out-dir logs/promptcredit/promptq_tnbc_dev/<new-run-id>
```

The runner writes a non-overwritable directory containing manifests, cache
manifests, calibration curves, per-image metrics, NMS analysis, bootstrap,
runtime, tests command, environment, report, and `SHA256SUMS`.  Stop after the
run and await the project lead; the script never accesses TNBC test or MoNuSeg.
