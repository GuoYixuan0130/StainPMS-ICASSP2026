# Phase 0 baseline reproduction plan

This is a gated plan, not authorization to train.  The current `main.py` uses a
`test/` loader for model selection and therefore no formal command is runnable
until explicit-manifest loading is implemented and its CPU/GPU smoke tests pass.

## Fixed identities

| Asset | AutoDL path | Required SHA256 |
|---|---|---|
| Official SAM2 Hiera-L initialization | `/root/autodl-tmp/projects/CA-SAM2-HRC/checkpoints/sam2_hiera_large.pt` | Not supplied; compute before use. |
| TNBC StainPMS warm start | `/root/autodl-tmp/projects/CA-SAM2-HRC/deliver_ckpts/tnbc_pms_best_e156.pth` | `44a3cb3e93051301d789e44f93769588abfa727d7a174b0270f55305ef023781` |
| MoNuSeg StainPMS warm start | `/root/autodl-tmp/projects/CA-SAM2-HRC/deliver_ckpts/monuseg_pms_best_pq.pth` | `6616c24626a162580d79aa7035d0b6c1a4ae6240a6c2f4599960dd4efac95db1` |

Every run must record the computed value and fail on mismatch.  The two supplied
hashes have not been verified from the Windows checkout.  Neither checkpoint is
approved for model selection until its training manifest is shown to exclude
the new development images; a correct hash alone does not establish that.

## TNBC protocol to materialize

- Optimization: patients 1--6, expected 30 images, ordered only by
  `/root/autodl-tmp/resimix_tnbc_train.json`.
- Model selection: patients 7--8, expected 7 images, ordered only by
  `/root/autodl-tmp/resimix_tnbc_dev.json`; always report p7 and p8 separately.
- Patients 9--11: reject before data/metadata I/O; never construct a loader.
- Crop 256, overlap 32, batch 1, point NMS 12, final box-NMS/assembly IoU 0.5,
  mask threshold 0, seed 3407, Hiera-L with the existing texture/context path.
  These are code/reproduction anchors, not permission to change any value.
- Evaluator and empty-image policy must be locked before the first comparison.

The exact original checkpoint training command cannot be reconstructed from its
payload because optimizer, scheduler, resolved config, manifest and command are
not embedded.  Therefore the first formal baseline is explicitly a warm-started
continued-training control, not a claim of bitwise reproduction of the paper
training run.

### Original 37/13 versus new 30/7/13

The original TNBC accounting used the 37 images from patients 1--8 for training
and 13 images from patients 9--11 for test.  The new protocol does not change or
inspect that closed 13-image cohort; it removes the seven p7--8 images from the
old 37-image optimization pool and reserves them for development, leaving 30
p1--6 training images.  Thus historical and new results differ in both training
data budget and model-selection protocol and are not directly comparable.

This creates an initialization gate: if
`tnbc_pms_best_e156.pth` was trained on all 37 old training images, it has already
seen p7--8 labels and cannot be used to select models on p7--8.  Acceptable
alternatives are (A) prove from a frozen manifest that it was trained only on
p1--6, or (B) initialize from a checkpoint with no p7--8 supervision and train a
new p1--6 baseline.  Option B costs substantially more GPU time but preserves
the development boundary.  The project lead must choose after the checkpoint
provenance audit.

## MoNuSeg version reconciliation

The [official challenge paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10439521/)
describes 30 training images from 30 patients, and the
[official evaluation page](https://monuseg.grand-challenge.org/Evaluation/)
describes 14 test images.  The supplied local training pool is expected to have
37 images, which is not the official 30-image training set until its provenance
is reconciled.  Before choosing a fold, the AutoDL audit must answer:

1. which seven images extend or differ from the official 30 training images;
2. whether each filename is one WSI/patient or a derivative/duplicate;
3. which case and organ each image belongs to;
4. whether the seven additional/different training-pool images come from a
   documented release or are derived/duplicated images.

The official 14-image test remains closed.  The frozen MoNuSeg-Lite directory
may be used only if its manifest, patch schedule and `SHA256SUMS` all validate;
it does not define the new grouped split by itself.

The same initialization gate applies to
`monuseg_pms_best_pq.pth`: if it was trained with labels from all 37 images, it
cannot initialize an experiment that treats a subset of those images as grouped
development.  Options are (A) prove the chosen fold was excluded, (B) build a
protocol-clean initialization per fixed split/fold, or (C) use the existing
checkpoint only for a clearly labeled, non-selective diagnostic that cannot
choose methods or hyperparameters.  G2 may require fold-specific clean
initializations and therefore has substantially higher cost than validation
inference alone.

## Grouped-development options requiring owner selection

| Option | Value | Cost/risk |
|---|---|---|
| G1 fixed 30/7 | One deterministic seven-image dev group, maximizing organ coverage subject to no case overlap. | Lowest screening cost and closest to current accounting; high variance from one fold. |
| G2 grouped 5-fold | Predeclare all case-disjoint, organ-balanced folds and one aggregate selection rule. | Roughly 5x validation inference/bookkeeping; more robust selection. |
| G3 leave-one-organ diagnostic | Evaluate organ transfer after G1/G2 is fixed. | Diagnostic only; too variable and expensive as sole model-selection rule. |

No option can be materialized until case/organ/source metadata is verified.

## Continued-training control and screening fairness

`B0` and every method arm must start from the same byte-identical,
protocol-valid warm-start
checkpoint and use the same:

- ordered optimization/dev manifests and crop schedule;
- seed, optimizer, LR schedule, update count, batch/crop budget and number of
  image views per update;
- trainable parameter set except in an explicit LoRA ablation;
- PMS refresh state and cadence where applicable;
- evaluator, thresholds, point NMS, crop overlap, assembly and validation
  frequency.

`B0` receives the same continued-training updates but no stronger stain view,
candidate loss, set consistency, point stability, GroupDRO/CVaR, or LoRA.
`B1` isolates stronger stain augmentation; `B2` adds ordinary prediction
consistency; `B3` adds ordinary four-mask best-of-K.  Later arms are allowed only
after Phase 1 evidence and must preserve the same budget.  If a paired-view arm
doubles image encodes, B0 must receive an equivalent compute/view control or the
comparison must explicitly report the unequal compute budget.

Single-seed short screening precedes three-seed full experiments.  Step/epoch
counts are intentionally not chosen in Phase 0 because they affect compute and
scientific comparability and no manifest-safe timing exists yet.  After a smoke
test, the project lead should lock updates rather than relying only on epochs.

## Outputs and checkpoint cadence proposal

Use ignored paths such as `logs/f3c/<dataset>/<arm>/<run_id>/`.  Each run should
contain `run_manifest.json`, `resolved_config.json`, `environment.json`,
`command.txt`, `metrics_per_image.csv`, `metrics_grouped.json`,
`metrics_summary.json`, timing/memory JSON, and checkpoints.  The run manifest
must hash every other identity file.

For short screening, save `latest` plus every validation epoch, including
optimizer/scheduler/RNG state; retain best AJI and best PQ as references without
using either to change evaluator settings.  For a full run, the project lead
should set a storage-aware periodic cadence after the first checkpoint size is
measured.  No existing checkpoint is overwritten.

## Runtime and memory

No defensible estimate is available locally: the full stack and GPU are absent,
and the hard-coded CUDA allocation prevents a full CPU forward.  The historical
environment states one RTX 4090, but that is not a timing measurement.  A
manifest-safe 1--2 batch smoke must record peak allocated/reserved memory,
data-loading time, forward/backward/update time, candidate count, and crop count;
only then should the report extrapolate total GPU hours.  Until the manifest
loader exists, do not run `main.py` even for timing because it constructs the
wrong validation boundary.

## Safe AutoDL Phase 0 commands

These commands audit only the named training/development pools and asset
identities; they do not run a model or access a closed test.

```bash
cd /root/autodl-tmp/projects/StainPMS-ICASSP2026
git branch --show-current
git rev-parse HEAD
git status --short

conda run -n agentseg python tools/audit_dataset.py \
  --config configs/splits/tnbc_p1_6_dev_p7_8.json \
  --config configs/splits/monuseg_grouped_dev.json \
  --output reports/dataset_audit.autodl.json \
  --summary-output reports/dataset_audit.autodl.md

sha256sum \
  /root/autodl-tmp/projects/CA-SAM2-HRC/checkpoints/sam2_hiera_large.pt \
  /root/autodl-tmp/projects/CA-SAM2-HRC/deliver_ckpts/tnbc_pms_best_e156.pth \
  /root/autodl-tmp/projects/CA-SAM2-HRC/deliver_ckpts/monuseg_pms_best_pq.pth

cd /root/autodl-tmp/projects/StainPMS-ICASSP2026/.setpms/logs/setpms/stage1_dual_dev/20260714_172429_6a40ce194788
sha256sum -c SHA256SUMS
cd /root/autodl-tmp/projects/StainPMS-ICASSP2026

conda run -n agentseg python -m pip freeze
conda run -n agentseg python -m unittest -v tests.test_audit_dataset
git diff --check
```

Return the two generated audit files, hash output, `pip freeze`, GPU model/driver,
and `nvidia-smi` memory state.  Do not run the TNBC converter, current baseline
evaluation, or any official-test command.

## Phase 0 decisions still needed

1. Provide an owner-approved raw TNBC p1--8 GT root/manifest so watershed impact
   can be measured without discovering closed patients.
2. Prove development isolation for both proposed initialization checkpoints or
   select protocol-clean initialization alternatives.
3. After the AutoDL identity audit, select G1 or G2 for MoNuSeg.
4. Decide the evaluator policy for empty GT/pred images before reproducing a
   baseline.
5. After a manifest-safe GPU smoke, lock screening/full update budgets and
   checkpoint cadence.
