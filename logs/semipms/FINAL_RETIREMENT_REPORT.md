# SemiPMS Final Retirement Report

**Route status: RETIRED / NO-GO.**  This report is a read-only synthesis of
the completed artifacts.  It introduces no model run, data access, threshold,
candidate budget, checkpoint selection, or other experiment.

## Scope and immutable evidence

| Phase | Artifact | Status |
| --- | --- | --- |
| Phase 0 | `StainPMS-SemiPMS/logs/semipms/phase0/semipms_20260712_223233_52e148d/` | Valid offline discovery audit |
| Stage 1 | `StainPMS-SemiPMS-Stage1/logs/semipms/stage1_tnbc_dev/semipms_stage1_20260713_011926_aab0a3d/` | Online EMA/dynamic-cache/mask-pseudo path: NO-GO |
| Stage 1B anchor | `StainPMS-SemiPMS-Stage1B/logs/semipms/stage1b_anchor_reconstruction/semipms_anchor_20260713_110735_178b5b3/` | Eligible reconstructed 720-step common anchor |
| Stage 1B | `StainPMS-SemiPMS-Stage1B/logs/semipms/stage1b_anchored_tnbc_dev/semipms_stage1b_20260713_120905_e98ce72/` | Frozen static point-pseudo rescue: NO-GO |

The Phase 0, Stage 1, anchor, and Stage 1B artifacts are read-only.  The
Stage 1B `SHA256SUMS` verification completed successfully for every reported
checkpoint, static-cache file, metric table, manifest, test log, and report.

All work was restricted to TNBC patients 1--6 for training and, where
authorised for Stage 1/1B, patients 7--8 for development.  Patients 9--11
were not accessed.  TNBC test and MoNuSeg were not run.

## Phase 0: offline residual-discovery and decoder headroom is real

Phase 0 trained a weak 20%-label teacher from the clean official SAM2
initialization, using one deterministic labelled image per patient 1--6 and
keeping the other 24 images' labels hidden until the single offline audit.
On that offline audit, the weak teacher had PQ **0.1319** (TP/FP/FN
184/237/1448).

Hematoxylin-residual proposals were informative in this *offline diagnostic*
setting:

- Raw residual proposals recalled **50.69%** of teacher false negatives at
  **72.53%** precision.
- The labelled-only frozen cross-view rule achieved **81.99%** precision in
  the subsequent offline audit.
- Correctly missed instances had decoded-mask IoU mean/median
  **0.639/0.683** (n=1114).
- Oracle Addition produced ΔPQ **+0.2839** and ΔAJI **+0.3064**.
- Frozen selected addition produced ΔPQ **+0.1855** and ΔAJI **+0.2510**;
  it added 528 TP and 615 FP, with positive selected PQ contribution in all
  six training patients.

This is retained as evidence that residual H-channel cues plus the frozen
decoder expose **offline proposal/decoder headroom** beyond the weak teacher's
coverage.  It is **not** evidence that SemiPMS improves semi-supervised
training or deployment performance.

## Stage 1: online pseudo-training collapsed

The properly extended supervised 20%-label path demonstrated that the original
240-step teacher was undertrained: its development PQ progressed from 0.0627
(240 steps) to a supervised maximum of **0.1654** at 720 steps, then 0.1003
at 960 steps.

The two online pseudo-training paths instead deteriorated:

| Method | PQ @240 | PQ @480 | PQ @720 | PQ @960 |
| --- | ---: | ---: | ---: | ---: |
| Supervised-StainPMS-20 | 0.0627 | 0.0311 | **0.1654** | 0.1003 |
| MeanTeacher-PMS | 0.0627 | 0.0043 | 0.0000 | 0.0000 |
| SemiPMS online EMA/dynamic cache | 0.0627 | 0.0000 | 0.0081 | 0.0000 |

The diagnostic cache audit identifies the mechanism of this failure.  At the
first SemiPMS cache update, 794 residual masks were accepted, but accepted-mask
precision was only 0.566 and pseudo-set precision 0.498.  At later updates,
the accepted set reduced to 95 masks with decoded IoU mean about 0.013,
duplicate rate about 0.896, and pseudo-set precision 0.  Thus the dynamic
teacher/cache/mask-pseudo feedback loop became self-reinforcing noise rather
than useful additional support.

The online EMA, dynamic cache, and mask-level pseudo-supervision construction
is therefore permanently rejected.

## Stage 1B: frozen static point-centre isolation also failed

Stage 1B removed the principal Stage 1 confounders.  It used the one-time
reconstructed 720-step supervised anchor (development PQ **0.0403**) as the
only initializer.  Every path started from that same checkpoint, and only the
auto-point heads (`deform_layer`, `reg_head`, `cls_head`, `conv`) could change.
The image encoder, prompt encoder, mask decoder, mask-quality/multimask
modules, static teacher, and static cache were checksum-verified frozen.

The cache was constructed once without reading the 24 training-side hidden GT
labels.  LOPO calibration on the six labelled images met the 0.90 target in
6/6 folds and froze view-IoU=0.80 with proposal budget=32.  It applied
cross-view one-to-one matching, one instance per H component, teacher/residual
mask NMS, residual/residual NMS, and centre-distance suppression before
producing pseudo centres.

Despite this deliberately conservative construction, no residual-supervision
benefit appeared:

| Stage 1B method | Best development PQ (step) | Final PQ @240 | Final TP/FP/FN |
| --- | ---: | ---: | ---: |
| Labeled-Only Continuation | **0.0981** (120) | 0.00175 | 1/1/939 |
| Static-Base Self-Training | 0.0674 (60) | 0.0000 | 0/0/940 |
| Anchored SemiPMS | 0.0403 (**0**, anchor) | 0.0000 | 0/0/940 |

Anchored SemiPMS never exceeded its own untrained anchor and never exceeded
either control.  At the preregistered final step, Anchored minus Labeled-Only
was ΔPQ **-0.00175**; Anchored minus Static-Base was **0.00000** because both
had zero predicted instances.  All three 240-step continuations showed late
degeneration, but this cannot rescue SemiPMS: the residual arm showed no
positive interval above its step-0 anchor, and the only authorised rescue has
been exhausted.

## Causal conclusion

The evidence chain is consistent across the stages:

1. Residual staining cues can point toward instances missed by a weak teacher,
   and a frozen decoder can produce good masks for many such points when
   assessed offline.
2. Those signals do not form a stable self-training target.  The online route
   collapsed through cache impurity, duplicate masks, poor later decoded masks,
   and feedback from pseudo supervision.
3. Eliminating EMA, cache refresh, pseudo-mask loss, and trainable decoder
   components did not solve the problem.  A once-only high-precision static
   residual-centre experiment also produced no development gain.

Accordingly, the supported conclusion is limited to offline
proposal/decoder headroom.  There is no validated semi-supervised learning
improvement from SemiPMS.

## Permanent stop conditions

SemiPMS is retired.  Do not run or implement:

- alternative pseudo-loss weights, candidate thresholds, or budgets;
- EMA, dynamic cache, mask-level pseudo supervision, or point-level pseudo
  supervision;
- new seeds, longer schedules, or a 960-step initialization retry;
- TNBC test evaluation or any MoNuSeg access.

No further SemiPMS model execution is authorised.  Await a new research
direction from the project lead.
