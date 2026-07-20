# Phase 0 code-path audit

Phase 0 audit base: `research/f3c-stainpms` at
`0f2fa9822146842f88aef2e20bc07c6b36bd39d7`.  The checkout was copied from
AgentSeg-CA-SAM2 commit `51c2eac`; local `data/` and `checkpoints/` are empty.
No model inference or training was run in this audit.

The executive findings below describe that audited base.  Phase 0.5 has since
added an opt-in manifest loader and strict evaluator while preserving legacy
defaults; remote data materialization and the GPU smoke are still pending.

## Executive findings

1. At the Phase 0 base, the loader was not manifest-backed.  Training enumerated
   `train_12/images`; validation/model selection enumerates `test/images` and
   saves best-PQ/best-AJI checkpoints from that loader.  It cannot implement
   TNBC p1--6/p7--8 or grouped MoNuSeg development safely without a loader
   change.
2. The decoder owns four mask tokens, but its public
   `multimask_output=True` path returns tokens 1--3 only.  The single-mask path
   returns token 0, with an evaluation-time stability fallback that may replace
   it by the quality-head-best token among 1--3.  A future four-candidate audit
   must expose all outputs from `predict_masks`; merely toggling the Boolean is
   not a four-mask implementation.
3. Every direct StainPMS training and inference decoder call passes
   `multimask_output=False`.  Consequently the ordinary segmentation losses and
   quality loss supervise the selected single result, not an explicit set of
   four task candidates.
4. The Phase 0 evaluator silently omitted an image when GT or prediction was empty.  That
   changes the estimand and must be fixed or explicitly frozen before baseline
   reproduction and all later arms.
5. Baseline checkpoints/logs do not contain enough provenance for a fully
   auditable continuation: no optimizer/scheduler/RNG state, resolved config,
   command, manifest hash, checkpoint hash, Git SHA, or environment snapshot.
6. A supplied initialization checkpoint is protocol-valid only if its training
   manifest excludes the newly designated development groups.  SHA256 proves
   byte identity, not absence of p7--8/MoNuSeg-fold label exposure.

These are protocol/blocking findings, not evidence that F3C will improve AJI or
PQ.

## Network and prompt path

| Component | Located path | Audited behavior |
|---|---|---|
| Point generator | `sam2_train/modeling/dpa_p2pnet.py` | `convnext_xlarge_in22k` (configured `pretrained=True`) plus FPN; 16-pixel anchor grid; deform/refine coordinates, foreground/no-object logits, and a semantic mask.  For a 256 crop the anchor count is 16x16=256, so the main outputs are coordinates `[B,256,2]`, logits `[B,256,2]`, and mask `[B,1,256,256]`. |
| Image encoder | `sam2_train/modeling/backbones/hieradet.py`, `sam2_train/sam2_hiera_l.yaml` | Hiera-L, embedding 144, stages `[2,6,36,4]`, global blocks `[23,33,43]`, FPN output width 256. |
| CA-SAM2 adaptation | `hieradet.py:307-430,501-1025` | `PromptGenerator` combines FFT high-pass handcrafted prompts, learned embedding prompts, and point-feature/SAM-feature interactions across Hiera stages.  `main.py:206-209` freezes image-encoder parameters except names containing `prompt_generator`; point net, prompt/mask decoder, and other non-image-encoder SAM parameters remain trainable. |
| Prompt encoder | `sam2_train/modeling/sam/prompt_encoder.py` | Positive label 1 and negative label 0 have separate embeddings; no-mask input uses a learned dense embedding, forcibly interpolated to 16x16. |
| Mask decoder | `sam2_train/modeling/sam/mask_decoder.py` | Separate object, IoU, and four mask tokens; high-resolution features enabled; sigmoid quality head and MLP object head enabled by YAML. |

Two implementation mismatches need a later engineering decision, not silent
workarounds:

- YAML declares image size 1024, but `SAM2Base.__init__` replaces it with 256
  (`sam2_train/modeling/sam2_base.py:159-160`).  Effective embedding/mask sizes
  therefore follow the 256 crop path.
- `PromptGenerator.fft` allocates a tensor with `.to('cuda')`
  (`hieradet.py:957`), so a full CPU forward is not device-agnostic.  The point
  backbone also has `pretrained=True`, which can cause an unapproved download
  if its cached weights are absent.

The generic `SAM2Base._forward_sam_heads` wrapper does not pass the locally
added mandatory `cell_nums` argument to `MaskDecoder.forward`.  The direct
StainPMS path passes it and is the exercised path; the generic wrapper is a
latent compatibility defect and must not be assumed usable without a smoke
test.

## Four-token tensor and selection audit

With `N` prompts and the current 256/16 feature geometry,
`MaskDecoder.predict_masks` constructs:

- mask logits `[N,4,64,64]`;
- predicted IoU/quality `[N,4]`;
- mask-token embeddings `[N,4,256]`;
- object-presence logit `[N,1]` from its own token/head.

Selection is:

- `multimask_output=False`, training: token 0 and quality 0, each retaining a
  singleton candidate dimension;
- `multimask_output=True`: tokens 1--3 and qualities 1--3, hence three masks;
- `multimask_output=False`, evaluation with default builder postprocessing:
  token 0 if its stability is at least 0.98, otherwise the largest predicted-IoU
  mask among tokens 1--3 (`delta=0.05`).

All direct calls in `run/run_on_epoch.py:376-383,539-546,1147-1159` pass false.
The public SAM2 wrapper would pick the largest predicted quality when given its
three-mask output, but that wrapper is not the direct training/evaluation path.

## Training supervision and StainPMS mining

Training samples one random foreground pixel from every GT instance
(`run/dataset/monuseg.py:174-189`).  The segmentation prompt is then the nearest
predicted point to each GT-selected point (`run/run_on_epoch.py:199-207,300`),
whereas validation uses only automatic predicted points.  This GT-dependent
nearest-point bridge is important when separating point failure from decoder
failure.

The base criterion combines Hungarian point matching, coordinate/class losses,
semantic foreground loss, mask Dice/focal terms, and MSE-style IoU quality
supervision.  Names are inverted in `criterion.py:116-117`: `loss_focal` calls
`DiceLoss` and `loss_dice` calls `BinaryFocalLoss`.  This is reproducibility
debt; it must not be “cleaned up” inside one experimental arm.

`stainpms/candidate.py` performs RGB-to-HED conversion, robust H evidence
normalization, Gaussian smoothing, Otsu thresholding, morphology, subtraction
of a dilated baseline map, and `peak_local_max`; candidates are matched to GT
within a configurable radius and split into positive/negative prompt targets.
Optional merge-aware intra-cell peaks and baseline-center preservation prompts
are present.  The loader attaches cached coverage as an extra mask channel
before geometric augmentation so image, GT, and coverage stay aligned.

Self-bootstrap behavior is in `main.py:435-524`: initial refresh occurs before
epoch 0 or at the deferred PMS start; later refreshes use
`(epoch - max(0,start)) % interval == 0`.  When self-bootstrap is requested with
a non-positive interval, code silently sets 10, while the reproducibility doc
recommends 20.  Coverage accumulation keeps a new instance only when its
overlap fraction with prior coverage is below 0.5
(`run/run_on_epoch.py:139-164`).

## Data preprocessing and augmentation

`MONUSEG` is also used for TNBC and assumes `.mat` labels with `inst_map`.
The legacy-compatible path uses `sorted(os.listdir(...))`.  Phase 0.5 adds an
explicit ordered, hash-verifiable manifest path; formal work must use it.
Training expands every
image to overlapping 256 crops, batch size 1 and `shuffle=False`.  Both dataset
and inference crop helpers hard-code stride as `256-overlap` rather than
`split_size-overlap`; this only agrees while crop size stays 256.

The default augmentation chain contains RandomCrop, 4x4 grid shuffle, two
ColorJitter entries, RandomRotate90, two horizontal and two vertical flips,
Downscale, Blur, Gaussian noise, Superpixels, ZoomBlur, ShiftScaleRotate,
padding, and normalization (`args.py:23-48`).  Loader construction mutates each
transform dictionary by `pop('type')`; constructing another loader from the
same resolved config may therefore be order-dependent and needs a smoke test.

`tools/prep_tnbc.py` treats official GT as binary and creates instance labels by
distance-transform watershed (Gaussian sigma 1, peak minimum distance 10 by
default), then relabels contiguously.  This can split one connected foreground
component into several instances while preserving foreground pixels, directly
changing instance count, AJI, DQ and PQ.  The converter stores no raw/prepared
hash sidecar or per-image split delta.  Its default patient split references the
closed cohort, so it must not be run under the new protocol.  The new audit tool
compares raw 8-connected components, prepared instances, split/merge counts,
foreground XOR, and the project evaluator only when an owner-approved raw GT
manifest/path is supplied.

## Evaluator and instance assembly

The fixed defaults found in code are:

| Item | Effective setting/path |
|---|---|
| Point-distance NMS | 12 pixels (`args.test.nms_thr`); this is not mask NMS. |
| Crop overlap | MoNuSeg 92 and TNBC 32 in existing reproduction docs. |
| Mask logit threshold | 0.0. |
| Per-crop box NMS | IoU 1.0 (effectively no suppression below identical boxes). |
| Cross-crop/final box NMS | IoU 0.5 from `args.data.post.iou_threshold`. |
| Predicted-IoU/stability filters | Disabled at threshold 0.0. |

Assembly first keeps the largest quality for repeated point IDs, then applies
class-agnostic box NMS.  Masks are painted in reverse keep order only when the
target region is currently all zero.  Candidate quality, crop-edge penalty
(0.3), crop order, and overlap can therefore affect the final label map.

`stats_utils.py` implements image-wise Dice1/Dice2, AJI/AJI+, DQ, SQ, and PQ;
the epoch result is `nanmean` over appended images.  DQ is
`TP/(TP+0.5FP+0.5FN)`, SQ is paired-IoU sum divided by TP, and PQ is DQ*SQ.
At match IoU 0.5 the current code includes equality through a special branch.
At the Phase 0 base, `_append_metric_scores` returned without appending when GT
or prediction was empty.  Phase 0.5 now exposes `strict_empty_handling_v1` and
the historical `legacy_skip_empty_handling_v1`; clean baseline and future F3C
must use strict under the same evaluator and postprocessing settings.

## Checkpoint, log, and environment audit

The training checkpoint contains SAM state, point-model state, `net._parameters`,
epoch, and texture bank.  It saves only new best-PQ and best-AJI files.  The
declared `SAVE_EPOCH=10` is unused; optimizer/scheduler/RNG state and periodic
resume checkpoints are absent.  Logger output captures the argument namespace,
not a resolved immutable config or repository identity.

Optional evaluation JSON stores metrics, checkpoint path, overlap, CLI NMS,
seed, and command, but not hashes/environment/manifest.  It records the raw
`--test_nms_thr` value (default -1), not necessarily the effective config value
(12), which is a provenance defect.

The requested AutoDL checkpoints are absent locally; their supplied SHA256
values are recorded in the baseline plan but not independently verified.  The
local default Python is 3.12.7 with NumPy 1.26.4, SciPy 1.13.1, and scikit-image
0.24.0; PyTorch/Hydra/mmengine/albumentations/OpenCV are unavailable in that
interpreter.  `environment.yml` specifies Python 3.12.4, PyTorch 2.3.1,
CUDA/cuDNN 11.8/8.7, and the pip versions needed by the project.  It also lists
`cuda-version=12.6` alongside the CUDA 11.8 runtime stack, so the actual AutoDL
environment must be captured rather than inferred from the YAML.

## Protocol status

| Gate | Status | Required evidence |
|---|---|---|
| Code paths and candidate selection | Complete (static) | GPU shape smoke still required before Phase 1. |
| TNBC p1--6 / p7--8 identity | Blocked locally | Run safe-manifest audit on AutoDL; lock rows and hashes. |
| TNBC raw-vs-watershed effect | Blocked | Owner-approved raw p1--8 label manifest/root. |
| MoNuSeg version provenance | Phase 0.5 in progress | Version scopes and case sets are declared; official archive/member hashes remain to be materialized on AutoDL. |
| MoNuSeg development candidate | Awaiting Phase 0.5 evidence and owner lock | Audit `classic30 -> extended7`; random/hash 23/7 and five-fold are not approved. |
| Extended7 GDC/TSS metadata | Complete locally; remote replay pending | GDC returned all seven cases with no missing field or TSS-code mismatch; preserve the raw response and SHA256 on AutoDL. |
| Checkpoint/config correspondence | Partial | Verify supplied hashes; checkpoint payload/config audit on AutoDL. |
| Initialization/dev isolation | Blocked | Prove the TNBC checkpoint excluded p7--8 and the MoNuSeg checkpoint excluded the chosen fold, or use a protocol-clean initialization. |
| Strict evaluator | Implemented locally; 11 boundary/output tests pass | Re-run the repository suite in `agentseg`; legacy skip remains available only for historical reproduction. |
| Baseline runtime/memory | Not measured | Manifest-safe train-only 1--2 batch GPU smoke on classic30 or TNBC p1--6. |

No Phase 1 diagnostic or Phase 2 method should start while these data gates are
unresolved.

During repository-wide static text search, existing tracked legacy documents
were found to contain historical closed-cohort filenames/metrics.  No closed
data file, label, prediction, checkpoint evaluation, or visualization was
opened, and those values were not used.  Future F3C commands and reports must
not consume those legacy closed-test references for selection or diagnosis.
