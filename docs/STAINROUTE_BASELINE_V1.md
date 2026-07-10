# StainRoute Development Baseline v1

Status: frozen for the Stage 1 oracle feasibility study on 2026-07-10.

The canonical settings are versioned in
`configs/stainroute/baseline_v1.yaml`. The manifest is generated on AutoDL at
`logs/stainroute/stage1/baseline_v1_manifest.json` by
`tools/stainroute_freeze_baseline.py`; it captures the actual host, software,
GPU, checkpoint, data-file, split, and configuration checksums.

## Frozen development baseline

| Dataset | Frozen model | PQ at NMS=12 |
| --- | --- | ---: |
| MoNuSeg | CA-SAM2 | 0.619861 |
| MoNuSeg | StainPMS | 0.657768 |
| TNBC | CA-SAM2 e147 | 0.663411 |
| TNBC | StainPMS e156 | 0.668077 |

All Stage 1 oracle candidates and action utilities must use the matching
checkpoint for their dataset. Evaluation uses inclusive IoU >= 0.5, one-to-one
matching, TTA disabled, NMS threshold 12, batch size 1, seed 3407, and enabled
texture/context.

## Scope boundary

This baseline is accepted only for StainRoute development and oracle
feasibility. Historical TNBC provenance remains unresolved. No Stage 1 or
future StainRoute result may claim to reproduce, compare directly with, or
silently substitute the historical TNBC paper values. Any subsequently found
historical artifact is a separate provenance experiment, not a replacement for
Baseline v1.

## Required initialization on AutoDL

```bash
python tools/stainroute_make_splits.py \
  --monuseg-root ./data/monuseg \
  --tnbc-root ./data/tnbc

python tools/stainroute_freeze_baseline.py \
  --config configs/stainroute/baseline_v1.yaml \
  --monuseg-root ./data/monuseg \
  --tnbc-root ./data/tnbc \
  --monuseg-split configs/splits/stainroute_monuseg.json \
  --tnbc-split configs/splits/stainroute_tnbc.json \
  --out logs/stainroute/stage1/baseline_v1_manifest.json
```

Copy the two generated `configs/splits/*.json` manifests and the resulting
`baseline_v1_manifest.json` back to the shared workspace before Stage 1 GPU
execution. The split files are lightweight and must be committed after review;
the manifest remains under ignored `logs/`.
