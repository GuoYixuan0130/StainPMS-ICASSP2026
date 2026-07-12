# SemiPMS Stage 1 — TNBC development protocol

This stage follows the Phase-0 STRONG GO decision.  It is a development
experiment, not a paper-level conclusion or a license to access TNBC test.

- Train images: TNBC patients 1–6 only, using the Phase-0 deterministic six
  labelled images plus the remaining 24 image-only records.
- Development: patients 7–8 only.  Patients 9–11 and MoNuSeg are forbidden.
- Initialization: `sam2_hiera_large.pt` official SAM2 only; point head random
  initialization with seed 3407.  TNBC/e147/e156/PMS-derived checkpoints are
  rejected before being loaded.
- The valid Phase-0 `frozen_acceptance_rule.json` is copied by checksum and is
  immutable for the whole run.  Train-side unlabeled GT cannot be read until
  every path reaches its final fixed optimizer step.

## Fair schedule

All methods share an exact model-only 240-step StainPMS warm-up checkpoint.
Adam is reset at this one fixed boundary for every path, so no method receives
a private optimizer-state advantage. They then continue to 960 total model
updates with the
same labelled loader, labelled augmentations, initialization, warm-up and
standard inference/assembly:

1. `Supervised-StainPMS-20`: labelled branch only.
2. `MeanTeacher-PMS`: EMA standard-deployment base pseudo instances only.
3. `SemiPMS`: the same base pseudo instances plus residual support expansion.

Development measurements are reported at steps 240, 480, 720 and 960.  The
step-960 checkpoint is fixed for every method; development curves do not cause
early stopping.

The standard-deployment replay is compared with the repository validation path
at the shared 240-step checkpoint in explicit evaluation mode, before the
three long continuations. Since no path changes inference code afterwards, this
is the retained inference-equivalence preflight rather than a late training
termination condition.

## Pseudo labels

EMA teacher inference uses the unchanged CA-SAM2/StainPMS deployment replay.
Base pseudo instances are its area-filtered standard instance maps.  SemiPMS
adds H optical-density residual candidates outside dilated teacher coverage,
then applies the Phase-0 frozen cross-view rule.  Masks are deduplicated against
base instances and each other at mask IoU 0.50 before artifact-owned cache maps
are written.

Pseudo losses are foreground-positive only: unconfirmed unlabeled pixels are
not used as negative background supervision.  Base prompt/mask loss is logged
separately as a preservation loss; residual prompt/mask loss has a 240-step
ramp.  Caches refresh every 240 updates without using any train-side hidden GT.

## Required evidence

The artifact contains development Dice/AJI/AJI+/DQ/SQ/PQ, TP/FP/FN per image
and patient, standard-inference equivalence checks, training cost, pseudo-cache
statistics, and post-training-only hidden-GT cache diagnostics.  The diagnostic
tables decompose proposal and accepted-mask precision/recall, pseudo-set
precision/recall, decoded-IoU distribution, duplicate rate, and the requested
FP sources.

The reference DeltaPQ bands are interpretive only.  The runner emits no
automatic GO/NO-GO verdict and stops after the development comparison.
