# REPORT FOR PROJECT LEAD — PROMPTQ TNBC DEVELOPMENT

Date: 2026-07-11

## 1. PromptCredit-v1 record

PromptCredit-v1 remains automatic **FAIL**.  Its immutable corrected paired
smoke artifact at `logs/promptcredit/stage1_smoke_corrected/2ab64ac/` was not
modified.  Directional Credit remains retired; this run used
`prompt_credit_grad_scale=0` and did not backpropagate a mask loss to point
coordinates.

## 2. Scalar-isolation smoke result

**Verdict: NO-GO.**  The pre-registered scalar-isolation gate failed, so the
runner stopped before TNBC development cache extraction, offline 20-epoch
quality-head training, or patients-7--8 full inference.

The sole failed criterion was:

```text
quality-target Spearman at step 100 = 0.5411764706 < 0.60
```

It must not be repaired through threshold tuning, a new seed, changing the two
crops, changing the target/loss, or a retry.  The automatic NO-GO is retained.

## 3. Frozen-model and baseline checks

- Baseline-to-PromptQ step-0 maximum absolute error was exactly zero for
  `pred_coords`, `pred_logits`, and decoded mask logits.
- The corresponding post-training maximum errors were also exactly zero.
- Inherited point-model checksum was unchanged:
  `97d83fae9587427b6b6efc3cb6822cb2c4df1d963b4da43ea2925554f7db8a1d`.
- SAM2 checksum was unchanged:
  `cd01d3fde4f542b52c21474d6c3ede63f644b18584ff703f976b26546c5bba88`.
- Only `quality_head` was trainable: 66,049 parameters; all 356,211,719
  inherited point-model and 374,772,797 SAM2 parameters were frozen.

## 4. Scalar-credit observations

- Quality loss decreased from `2.2187236547` to `0.0047489833`.
- Quality prediction became nonconstant (standard deviation `1.4823574`).
- Raw-objectness/decoded-IoU Spearman was `-0.1176471`; product-score/decoded-
  IoU Spearman reached `+0.1735294`, a `+0.2911765` improvement, satisfying
  the required `+0.20` relation check.
- Fixed-crop product-NMS mean hard-mask IoU was unchanged at `0.82476515`.
- Execution remained finite and detached ROI features were used.

These positive mechanical and calibration-adjacent observations do not
override the failed pre-registered target-Spearman criterion.

## 5. TNBC development status

No TNBC development cache, development quality training, development full
inference, paired bootstrap, or NMS conflict analysis was run.  TNBC patients
9--11 and MoNuSeg were not accessed.  There are consequently no development
segmentation metrics, runtime comparison, or test recommendation.

## 6. Tests

The AutoDL PyTorch environment passed all 27 PromptCredit/PromptQ tests.

## 7. Artifact record

The formal AutoDL run directory is:

```text
logs/promptcredit/promptq_tnbc_dev/20260711_0ddc7a6/
```

Its artifact `SHA256SUMS` records, among others:

```text
report.json                              6e66a2d48373eea8a7c471a50d5ee722a7f51871586dd24a6b9fde51dec188b2
scalar_isolation_report.json             fe354c4f9c9da3bb7f199d750b5909e0fe9cd1c4ee7cac0691c20c421bce9a26
scalar_isolation_curve.csv               fddb6bd07ef8220029b937ee93f7f45f41e4f741fc4e9a687c41fd72be1125b1
scalar_smoke_crop_selection.json         a8008eef8e6b841a8a43a411ea601360625ea4a3239ac2fad1707574c9317c94
```

The local `temple/` copies were inspected read-only and were neither modified
nor staged.  The AutoDL `logs/` directory is the formal artifact location.

## 8. Recommendation

Do not authorize TNBC test, MoNuSeg, a full TNBC PromptQ experiment, or a
second seed under this method version.  PromptQ stops here and awaits any new
project-lead decision.
