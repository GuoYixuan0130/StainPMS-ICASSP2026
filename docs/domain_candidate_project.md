# F3C-StainPMS project contract

## Scientific question

F3C-StainPMS (Factorized Counterfactual Candidate Coverage) asks which of
stain, patient, organ, disease, and morphology changes first damages automatic
point generation or mask-candidate generation.  Only after that diagnosis may
training optimize candidate-set coverage across controlled, geometry-preserving
stain views.  Frozen-output rescoring, an external patient-specific ranker, and
the existing pseudo-label loop are out of scope.

AJI is the primary endpoint and PQ is the key co-endpoint.  Dice, DQ, SQ,
point recall/precision, CCR at 0.5/0.6/0.7, error decomposition, and grouped
consistency are explanatory outcomes; CCR never replaces final instance metrics.

## Immutable data boundary

- TNBC optimization is patients 1--6.  Patients 7--8 are development only and
  cannot be merged into training without project-lead approval.  Patients 9--11
  are closed: code must reject them before opening an image, label, prediction,
  or metadata record.
- MoNuSeg uses the current official-download 37/14 StainPMS-continuity
  protocol.  No internal subdivision of its 37 training images is authorized.
  During Phase 0.5, test14 access is limited to case ID, filename, byte size
  and raw-image SHA256; test images are never decoded and test labels are never
  opened.  Patches from one source image/case cannot cross splits.
- CPM-17 is diagnostic only unless its disease/tissue/image mapping is reliable;
  it is not a default training source or a replacement endpoint.
- Every runnable split is an explicit, ordered manifest with content hashes.
  Directory discovery is allowed only to audit a named training pool and cannot
  be used as the training order.

The pending protocol descriptions are
[`configs/splits/tnbc_p1_6_dev_p7_8.json`](../configs/splits/tnbc_p1_6_dev_p7_8.json)
and
[`configs/splits/monuseg_grouped_dev.json`](../configs/splits/monuseg_grouped_dev.json).
They deliberately contain no invented sample rows and are not yet training
manifests.

## Stage gates

Phase 0 and Phase 0.5 must establish code paths, label semantics, case/organ provenance,
split isolation, preprocessing effects, evaluator invariants, checkpoint
identity, and a budgeted baseline plan.  Phase 1 cannot begin while any of
these protocol gates remains unresolved.

No MoNuSeg internal development split is currently authorized.  The current
37-image set remains the continuity training scope; official test14 remains
sealed until final evaluation.

TNBC's existing prepared ``.mat`` labels are permitted for the manifest-safe
1--2 batch Phase 0.5 smoke only.  They cannot become the final label protocol
until the separate raw-binary versus connected-components/watershed audit is
resolved.

Phase 1 is read-only diagnosis from frozen checkpoints/inference where
possible: automatic-point behavior, GT-point single/four-candidate CCR, final
assembly, interpretable morphology/appearance groups, and paired controlled
restaining.  It stops for owner review before training changes.

Phase 2 modules are evidence-triggered.  Multimask training, worst-view
coverage, permutation-invariant consistency, point stability, morphology
CVaR/GroupDRO, quality-head supervision, and optional late LoRA must remain
independently switchable.  B0 continued training uses the same initialization,
steps, crop/view budget, optimizer, evaluator, and seed as every method arm.

## Reproducibility record required per run

Each run must store the branch/commit and dirty diff identity, complete command,
resolved config, manifest and checkpoint SHA256, Python/PyTorch/CUDA/dependency
versions, seeds and deterministic settings, optimizer/scheduler state,
effective crop count and update count, evaluator/postprocessing settings,
wall time, peak memory, machine-readable per-image/group metrics, and a human
summary.  Logs, data, artifacts, and checkpoints remain outside Git.

Before a long run: static checks, CPU unit tests, then a manifest-safe 1--2
batch GPU smoke test.  Any change to split, seed, evaluation region, point NMS,
mask NMS, overlap, threshold, or metric is a separate, predeclared ablation.
