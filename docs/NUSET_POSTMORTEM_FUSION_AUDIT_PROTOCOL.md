# NuSet Postmortem-A Protocol

## Immutable scope

This is a one-time read-only diagnosis after the formal NuRank-v1 NO-GO. It never changes `logs/nurank/stage1_tnbc_dev/20260711_nurank_stage1/`, its checksums, report, thresholds, or verdict. It does not train any model, retry NuRank, change its epoch/loss/seed/features/width, access TNBC patients 9–11, run MoNuSeg, or invoke StainRoute, PromptCredit, or PromptQ.

Only patients 1–6 are train-side diagnostics and only patients 7–8 are development diagnostics. The frozen baseline-v1 checkpoint SHA256 remains `44a3cb3e93051301d789e44f93769588abfa727d7a174b0270f55305ef023781`; seed is 3407, TTA off, batch 1, and the GPU time cap is 45 minutes.

## Cache policy

The audit first reads the immutable NuRank train/development caches. It checks for low-resolution four-token logits. A saved `mask_logits` tensor of `256×256` is also accepted as an exact low-resolution alias in this frozen path: SAM2 emits `256×256` mask logits and the existing path applies bilinear resize to the same `256×256` size; NuSet Stage 0 recorded zero token-0 error for that operation. If neither representation exists, the audit writes a new `reextracted_lowres_cache/` only under its new artifact directory, runs one deterministic frozen extraction for patients 1–8, and checks group order, prompt IDs/classes, coordinates, original predicted IoU, and upsampled logits against the formal NuRank cache. Any mismatch stops the audit. Point model, point encoder, and SAM2 checksums are written by the extraction and must remain unchanged.

## Failure attribution

Single, Existing-All-Pred, frozen NuRank, and Token-Oracle are reported on train and development. Token-Oracle uses true IoU only for matched prompts; unmatched prompts remain token 0. Failure labels are fixed:

- `representation_or_objective_failure`: train NuRank vs Existing top-1 improvement `<5` pp or regret reduction `<10%`.
- `cross_patient_generalization_failure`: train clears both thresholds but development top-1 does not improve or regret increases.
- `assembly_mismatch`: development NuRank selected-mask mean IoU improves by `>=.005` vs Single while PQ improves by `<=.001`.
- Multiple labels produce `mixed`.

The labels explain the frozen NO-GO only; they cannot alter it.

## Preregistered parameter-free fusion library

All candidate-token logits come from the same cached one-call `predict_masks()` output. No image/prompt/decoder call occurs during cache replay. Fixed candidates are token 0, Existing-All-Pred, NuRank, equal logit mean, equal probability mean, logit median, hard majority with token-0 2:2 tie-break, logit max, and logit min. Logit operators fuse at low resolution and use baseline bilinear upsampling. Probability mean uses its mathematically equivalent logit before the fixed zero threshold. Hard majority votes final-resolution hard masks. Every fusion uses token-0 predicted IoU as its fixed baseline-anchored assembly score; individual token paths retain their own existing predicted IoU. No threshold, temperature, or weight is selected from development results.

Fixed-Library Oracle is GT-only: each matched prompt picks the best of four single-token masks plus fixed fusions; unmatched prompts remain token 0. Convex-Fusion Oracle is GT-only: it enumerates exactly 35 nonnegative four-token logit weights on `{0,.25,.5,.75,1}` summing to one. Its unmatched prompts remain token 0. Both oracle paths are upper bounds, never deployment proposals.

## Decision

`FIXED_FUSION_SIGNAL=YES` requires one fixed fusion on development with `ΔPQ>=+.003`, nondecreasing AJI, at least 5/7 nondecreasing PQ images, FP no higher than Single, and largest positive contribution `<=60%`.

`LEARNED_FUSION_HEADROOM=YES` requires Convex-Fusion Oracle versus Token-Oracle with extra `ΔPQ>=+.003`, extra mean prompt IoU `>=.005`, non-one-hot optimal fraction `>=25%`, at least 5/7 nondecreasing images, and largest extra contribution `<=60%`.

`MULTIMASK_CONTINUE=CONDITIONAL GO` only if either signal is YES; otherwise it is NO-GO. The audit ends after its report and checksum are written; it does not implement NuFuse or another model.
