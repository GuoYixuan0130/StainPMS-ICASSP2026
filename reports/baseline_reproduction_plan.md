# Phase 0 baseline reproduction plan

This is a gated plan, not authorization to train.  Phase 0.5 implements an
opt-in explicit-manifest loader and a train-only smoke path; legacy directory
loading remains the default for historical compatibility.  No formal baseline
command is authorized until the manifest path passes the AutoDL GPU smoke and
the project lead locks the development protocol and training budget.

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

The raw-data source is the corrected official Zenodo v1.1 record
`10.5281/zenodo.2579118`, not v1.0.  Its archive is
`TNBC_NucleiSegmentation.zip`, 25,232,361 bytes, with publisher MD5
`1605712a752b201b57eacc8f866adb4f`; a local SHA256 must still be recorded.  The
paper treats GT as binary and AJI objects as connected components.  Therefore
connected components, the current distance-transform watershed and prepared
labels must be compared before the project lead freezes an instance-GT version.

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
describes 14 test images. The official data page still states 30 training
images, but its current Training Data link (Google Drive file ID
`1ZgqFJomqQGNnsx7w7QBzQQMVA16lbVCA`) supplies a 37-image package according to
the project lead's independent inspection. These are two explicit version
scopes, not evidence that StainPMS used an erroneous third-party mirror:

- `monuseg_download37_v1`: current official-download 37/14 protocol and the
  StainPMS continuity primary protocol;
- `monuseg_challenge30_v1`: original 2018 Challenge paper 30/14 protocol and a
  later protocol-sensitivity experiment.

The seven cases in the current download beyond classic30 are called
`extended7` until Phase 0.5 resolves:

1. whether each filename is one WSI/patient or a derivative/duplicate;
2. which TCGA project, organ, disease and tissue source site each case has;
3. whether any extended7 image overlaps or derives from test14;
4. whether XML annotations and conversion behavior match classic30.

The official 14-image test remains sealed for decoding, annotation access,
inference and statistics; Phase 0.5 permits identity, size and raw-image SHA256
checks only.  The frozen MoNuSeg-Lite directory
may be used only if its manifest, patch schedule and `SHA256SUMS` all validate;
it does not define the new grouped split by itself.

The same initialization gate applies to
`monuseg_pms_best_pq.pth`: if it was trained with labels from all 37 images, it
cannot initialize an experiment that treats a subset of those images as grouped
development.  Options are (A) prove the chosen fold was excluded, (B) build a
protocol-clean initialization per fixed split/fold, or (C) use the existing
checkpoint only for a clearly labeled, non-selective diagnostic that cannot
choose methods or hyperparameters. Formal extended7 development work requires
a clean baseline initialized without extended7 task supervision.

## Phase 0.5 development candidate requiring owner lock

| Candidate | Value | Cost/risk |
|---|---|---|
| `classic30 -> extended7` | Train on classic30, diagnose on cases later present in the current official download, then retrain locked final methods on download37. | Scientifically interpretable version/domain transition, but invalid if extended7 overlaps or derives from test14, has incompatible XML conversion, or lacks reliable TCGA metadata. |

This candidate is not locked by the engineer. Five-fold and random/hash 23/7
splits are not approved in Phase 0.5.

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
only then should the report extrapolate total GPU hours.  The manifest loader
and smoke-only exit now exist but have not yet run on AutoDL.

## Safe AutoDL commands

Phase 0.5 commands are maintained in
[`docs/F3C_PHASE05_AUTODL.md`](../docs/F3C_PHASE05_AUTODL.md).  The commands
below are retained only as the earlier Phase 0 audit record.

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
conda run -n agentseg python -m unittest discover -s tests -p 'test_audit_dataset.py' -v
git diff --check
```

Return the two generated audit files, hash output, `pip freeze`, GPU model/driver,
and `nvidia-smi` memory state.  Do not run the TNBC converter, current baseline
evaluation, or any official-test command.

## Phase 0.5 gates still open

1. Provide an owner-approved raw TNBC p1--8 GT root/manifest so watershed impact
   can be measured without discovering closed patients.
2. Materialize download37/classic30/extended7/test14 identities and hashes from
   the current official archives without decoding or analyzing sealed test images.
3. Audit extended7 GDC/TSS metadata and XML conversion, then return evidence for
   the owner to accept or reject `classic30 -> extended7`.
4. Use the approved strict evaluator for new work and retain `legacy_skip` only
   for historical reproduction.
5. After a classic30/TNBC-p1--6 train-only GPU smoke, lock screening/full update budgets and
   checkpoint cadence.
