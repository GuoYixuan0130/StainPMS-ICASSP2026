# PromptCredit-v1 to PromptQ amendment

Date: 2026-07-11

## Frozen PromptCredit-v1 outcome

The corrected paired PromptCredit-v1 smoke completed at code SHA
`2ab64aca932de3c6a21ab3be1271601c483de87c` and its automatic outcome is
**FAIL**.  This result and all artifacts are immutable:

```text
logs/promptcredit/stage1_smoke_corrected/2ab64ac/
```

The common-score comparison against the head-only control was:

```text
delta IoU       = -0.000202
delta mask loss = +0.000057
```

These very small differences are not interpreted as statistically significant
harm.  They are nevertheless insufficient to support Directional Credit as an
incremental mechanism.  The automatic verdict and its pre-specified criterion
are not changed after observing the result.  Directional Credit is therefore
retired: no subsequent PromptQ run may backpropagate a mask loss to point
coordinates or update point-generator parameters with a mask gradient.

## Scalar-credit evidence retained

The same immutable smoke established scalar quality-learning evidence:

```text
quality-target Spearman at step 100              = +0.7706
deployment-score/decoded-mask-IoU Spearman step 0 = -0.1201
deployment-score/decoded-mask-IoU Spearman step100= +0.7357
```

This evidence earns **CONDITIONAL GO** only for scalar quality credit.  It is
not a generalization claim and does not change PromptCredit-v1's FAIL result.

## PromptQ definition fixed before development observations

The next version is **PromptCredit-Q / PromptQ: Frozen-Model Pre-Decode
Mask-Utility Distillation**.  Before observing any TNBC development result, it
is constrained as follows:

- Every existing StainPMS and SAM2 parameter is frozen; only `quality_head` is
  trainable.
- Point coordinates, ROI features, decoder targets, and utility targets are
  detached at the required boundaries.  `prompt_credit_grad_scale` is zero and
  mask loss never participates in backward.
- The quality head has no dropout and keeps its fixed neutral final-layer
  initialization: zero weight and `logit(0.01)` bias.
- The target remains `hard_iou * sigmoid((hard_iou - 0.5) / 0.1)`, unmatched
  sources remain zero, duplicate sources use maximum utility, and Quality
  Focal Loss uses gamma 2.
- The only inference change is point-NMS ranking from objectness to
  `objectness * sigmoid(quality_logit)`.  Coordinates, class decisions,
  semantic filtering, decoder calls, NMS radius, assembly, and traversal are
  unchanged.

PromptQ training may use only TNBC patients 1-6.  Patients 7-8 are a
development set whose prior use by the StainPMS initialization must be
disclosed; they are not an independent leakage-free validation claim.  TNBC
patients 9-11, MoNuSeg, StainRoute, threshold tuning, a second seed, and any
new method module remain prohibited pending a future project-lead decision.
