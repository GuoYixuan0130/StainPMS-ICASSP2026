# DeployPMS Phase 0: Training–Deployment Prompt Exposure Gap Audit

This is a frozen diagnostic, not a training entry point. It is isolated on
`research/deploypms-phase0` from canonical baseline
`2a1348cb7a1158a6f77aae2f92c168f9552d8068`.

The runner accepts only the handover e156 checkpoint with SHA256
`44a3cb3e93051301d789e44f93769588abfa727d7a174b0270f55305ef023781`; it
verifies that checksum before loading the model. It manifests TNBC patients
1–6 as train-only provenance, evaluates only patients 7–8 (exactly seven
images), never opens patients 9–11, and rejects MoNuSeg. No optimiser is
created and no model parameter is trainable.

Teacher prompts use a deterministic replay of the training dataset’s interior
GT point selector, then the exact `find_nearest_points` query-coordinate rule
from `train_on_epoch`; token 0 is decoded with a positive SAM label. The
historical e156 augmentation RNG trajectory was not recoverable, so the report
records this fixed audit seed rather than claiming it has been reconstructed.
The teacher coordinate itself is the actual nearest model query, not the GT
point.

Deployment prompts replay validation classification, filtering, progressive
point NMS, positive prompt labels, token-0 decoding, and formal mask assembly.
GT is never used in that prompt path. The teacher and deployment prompt groups
for a crop use the same frozen encoded image features and decoder weights;
deployment retains the formal validation decoder call, while teacher explicitly
selects token 0 without the validation-only dynamic fallback.
No StainPMS residual prompt or coverage statistic enters any Phase 0 result.

Run on the AutoDL workspace (only after the short asset locator identifies the
real e156 file):

```powershell
Get-ChildItem checkpoints,deliver_ckpts,logs -Recurse -File -Filter '*e156*.pth' | Select-Object FullName,Length
```

Then use the returned path directly:

```powershell
python tools/run_deploypms_phase0.py --checkpoint <returned-e156-file>
```

The sole output is `logs/deploypms/phase0/<run_id>/`. It includes the manifest,
environment, model/checkpoint checksums, exact paths, call counts, per-image
and per-instance CSVs, `report.json`, and `SHA256SUMS`. `report.json` applies
the preregistered availability, conditioning, and shared-GT-swap gates and
stops immediately after the final verdict.
