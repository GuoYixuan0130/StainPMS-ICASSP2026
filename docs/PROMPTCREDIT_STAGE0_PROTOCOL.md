# PromptCredit PC-Stage 0 Protocol

Status: implementation and CPU tests prepared on 2026-07-11.  No model audit
has been run from this repository state.

## Scope lock

PromptCredit is a separate, low-cost mechanism audit.  It must not change the
formal model, training loop, data split, evaluation standard, or paper claim.
It must not access TNBC calibration patients 7--8, TNBC test patients 9--11,
or any MoNuSeg file.  It must not invoke StainRoute code or run a StainRoute
oracle, router, training, or inference experiment.

The tool validates its scope before opening a data file.  It accepts only a
TNBC root named `tnbc`, validates the committed router-train split, resolves
only six exact `train_12/images/<id>.*` and `train_12/labels/<id>.mat` pairs,
and checks the frozen StainPMS baseline v1 checkpoint SHA256:

```text
44a3cb3e93051301d789e44f93769588abfa727d7a174b0270f55305ef023781
```

It refuses CPU model execution.  CPU-only unit tests are allowed locally; the
model audit command below is for AutoDL 4090 only.

## Fixed image selection

Before any image, GT, or model output is read, image IDs are ordered by
`SHA256("3407:" + image_id)` (with image ID as a deterministic tie-breaker)
over TNBC `router_train`, and the first six are frozen in
`configs/promptcredit/pc_stage0_tnbc_router_train_six.json`.

| Rank | Image ID |
| ---: | --- |
| 1 | `02_1` |
| 2 | `05_1` |
| 3 | `04_5` |
| 4 | `05_4` |
| 5 | `03_3` |
| 6 | `04_7` |

The ordered-list SHA256 is
`d172ba88ebe645c4abd1a4bf78f8b7b66da60a58ba093b153754611f3bdf2ea6`.
The selection-manifest content SHA256 is
`e1bfcdd57a526e435c95b4e91aa99f5f2f12a5ec75ec7a985f6353f4f89393ff`.

## Frozen preprocessing and measurements

The audit uses the baseline TNBC crop geometry: 256-pixel crops, overlap 32,
`unclockwise` traversal, no TTA, and the existing Albumentations `Normalize`
preprocessing.  Local density is the number of *other* GT centroids within a
fixed 64-pixel crop-radius.  Area, density, and point-distance subgroup
summaries use deterministic equal-frequency terciles; they are descriptive and
not tuning parameters.

Audit A reproduces independent nearest proposal assignment and the existing
Hungarian cost (`0.1 * coordinate distance - foreground probability`).  It
records the required GT/crop/proposal fields, collision groups, disagreement,
out-of-mask prompts, and density concentration.

Audit B sends the nearest-assigned prompts through one existing decoder call
per crop; duplicated nearest prompts stay duplicated, as in the current mask
branch.  It performs no prompt enumeration, test-time action, correction, or
manual thresholding.  It records hard/soft IoU, SAM predicted IoU, point score,
area, density, distance, and 10 equal-frequency-bin calibration.

Audit C is restricted to the first fixed image and the first 20 crop-order,
in-bounds Hungarian-matched prompts.  It freezes and clears all prompt-encoder
and mask-decoder parameter gradients, keeps only coordinates differentiable,
uses focal(gamma=2)+soft-dice loss, and performs one diagnostic-only one-pixel
gradient step plus one re-decode.  The stage report explicitly labels that
second call as diagnostic-only.  The tool uses the frozen checkpoint texture
bank and an ephemeral, per-image context bank; it never mutates the checkpoint
or any historical artifact.

## CPU checks and AutoDL command

Run locally only:

```bash
python -m unittest discover -s tests/promptcredit -p "test_*.py" -v
python tools/audit_prompt_credit.py --write-selection
```

Run the following only on the project lead's AutoDL 4090 environment.  Do not
run it on the local RTX 4060.  The output is ignored local audit evidence and
must not be committed or pushed.

```bash
python tools/audit_prompt_credit.py \
  --data-root ./data/tnbc \
  --checkpoint ../CA-SAM2-HRC/deliver_ckpts/tnbc_pms_best_e156.pth \
  --out-dir logs/promptcredit/stage0_tnbc_router_train_six
```

The run writes `assignment_records.csv`, `utility_records.csv`,
`gradient_records.csv`, `reliability_diagram.png`, `run_manifest.json`, and
`report.json`.  It refuses to overwrite an existing output directory.

## Pre-registered decision rule

- Assignment gap: collision excess rate >= 1% or nearest/Hungarian source
  disagreement >= 10%.
- Quality gap: point-score/hard-IoU Spearman <= 0.60 or 10-bin ECE >= 0.08.
- Actionable gradient: at least 95% finite nonzero coordinate gradients, mean
  one-pixel Delta mask loss below zero, and at least 60% prompt loss improved.

GO requires an actionable gradient, acceptable cost, and at least one of the
assignment or quality gaps.  A weak single-gap result may be explicitly marked
CONDITIONAL GO; simply having one positive gap is not weak by definition.  See
the dated verdict addendum for the correction to the original report logic.
