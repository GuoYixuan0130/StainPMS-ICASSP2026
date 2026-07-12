# NuPart Stage 0 protocol

This is a read-only, cache-first ownership-conflict audit. It uses only token-0 cached logits and automatic prompts from the frozen StainPMS baseline, with TNBC patients 1–6 for train diagnosis and patients 7–8 for development diagnosis. It never reads patients 9–11 or MoNuSeg, never loads the checkpoint weights, and never performs training or an optimizer step.

The runner requires the immutable formal cache directories and an immutable formal token-0 baseline instance-map archive. Before any ownership conclusion it verifies the checkpoint hash, cache checksums, frozen point/SAM2 checksums, token-0 low-resolution-to-upsampling equality, exact final instance-map equality, and metric equality within `1e-7`. A failed check writes `PROTOCOL INVALID` and stops.

Run on AutoDL Bash from the repository root (not on this CPU-only workstation):

```bash
python tools/run_nupart_stage0.py \
  --train-cache logs/nurank/stage1_tnbc_train/<formal_train_cache> \
  --development-cache logs/nurank/stage1_tnbc_dev/<formal_development_cache> \
  --data-root <TNBC_ROOT> \
  --checkpoint ../CA-SAM2-HRC/deliver_ckpts/tnbc_pms_best_e156.pth \
  --baseline-maps <FORMAL_TOKEN0_ASSEMBLY_NPZ> \
  --out-dir logs/nupart/stage0/<run_id>
```

The runner does not fall back to extraction. If a required cache field or formal baseline map is unavailable, it stops rather than generating a new model output. The created artifact contains the preregistered conflict, resolver, strict conflict-only oracle, detached-logit partition-gradient, fixed visual, checksum, and final-decision files. Completion is a stopping point; NuPart Stage 1 requires a new explicit project-lead authorization.
