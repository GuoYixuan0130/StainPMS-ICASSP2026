# NuSet Stage 0 protocol

Date: 2026-07-11

NuSet is independent from the terminated StainRoute, PromptCredit-v1, and
PromptQ projects. It neither trains nor invokes prompt utility, routing,
corrective actions, or coordinate credit. Stage 0 is a GPU-only, no-training
audit of the four SAM2 mask tokens already produced by one decoder call.

## Fixed scope

- Restore the immutable six-image TNBC manifest at
  `configs/promptcredit/pc_stage0_tnbc_router_train_six.json`; do not select
  images again.
- Use seed 3407; baseline-v1 checkpoint SHA256
  `44a3cb3e93051301d789e44f93769588abfa727d7a174b0270f55305ef023781`;
  NMS 12; TTA off; texture/context on; 256px, overlap 32, unclockwise crops;
  and the repository's inclusive IoU >= 0.5 evaluator.
- Resolve only patients 1-6 by direct filenames. Patients 7-11, MoNuSeg,
  training, second seeds, threshold tuning, prompt perturbation, action
  enumeration, and model changes are prohibited.

## Single-call contract

SAM2 has `num_multimask_outputs=3` and `num_mask_tokens=4`. For every existing
prompt, NuSet calls `MaskDecoder.predict_masks()` once, after one image and one
prompt encoding, and retains token 0-3 plus four predicted-IoU values.

`MaskDecoder.forward(multimask_output=False)` uses that same call and selects
token 0. NuSet applies the exact token-0 selector without a second decoder
forward. The baseline token-0 memory/context state is shared for all
post-decoder token-selection paths, preventing branch-specific re-encoding or
re-decoding.

## Required analyses

- GT-associated prompts: Single, Multi-Pred, All-Pred, Multi-Oracle, and
  All-Oracle IoU/loss/diversity/headroom analysis.
- Standard automatic prompts: points inside GT use that GT only for analysis;
  unmatched points remain unchanged and use predicted-IoU selection in Oracle
  assembly.
- Full assembly: Baseline-Single, Deployable-All-Pred, and Oracle-All share
  points, NMS, edge penalty, filtering, overlap processing, and assembly.
- IoU-head ranking: correlation, top-1 accuracy, regret, MRR, Brier, ECE, and
  token-wise calibration.

The run forecasts total time after its first 10% of fixed crop work and stops
if the forecast exceeds one GPU hour.

## Verdict rules

STRONG GO requires all pre-registered conditions: automatic matched All-Oracle
mean ΔIoU >= .010; at least 15% with ΔIoU >= .020; Oracle-All ΔPQ >= .005;
at least 4/6 non-decreasing PQ images; largest contribution <=60%; non-token-0
oracle best >=10%; identical call counts; and <=5% overhead.

CONDITIONAL covers prompt-level headroom with limited assembly gain or poor
ranking. NO-GO covers automatic mean oracle ΔIoU < .005, Oracle-All ΔPQ <
.003, severe collapse, GT-only headroom, extra decoder calls, or a budget
violation. The rules, token set, image list, and thresholds are frozen.

## Artifacts and command

Use only a fresh `logs/nuset/stage0/<run-id>/` directory; never use `temple/`
as formal storage. The runner writes manifests, prompt records, per-image
metrics, runtime/call counts, report, environment, tests command, stdout log,
and SHA256SUMS.

```bash
git pull --ff-only origin research/nuset && \
git rev-parse HEAD && \
python -m unittest discover -s tests/nuset -v && \
python tools/audit_multimask_headroom.py \
  --data-root ./data/tnbc \
  --checkpoint ../CA-SAM2-HRC/deliver_ckpts/tnbc_pms_best_e156.pth \
  --out-dir logs/nuset/stage0/<new-run-id>
```

Stop after one run and await the project lead's decision.
