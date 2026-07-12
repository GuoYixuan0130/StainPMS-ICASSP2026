# SafePMS Stage 0-1 compact validation

SafePMS is the only authorized training-feasibility experiment after NuPart.
It fine-tunes only the shared `net.sam_mask_decoder`. The point net/encoder,
SAM2 image and prompt encoders, memory modules, and every non-decoder parameter
are frozen and checksum-verified. It never reads patients 9-11 or MoNuSeg,
never refreshes coverage maps, and does not change inference, NMS, or assembly.

The runner consumes the formal NuRank cache manifests only as named closed-split
manifests: train patients 1-6 and development patients 7-8. It does not
enumerate `train_12`, so held-out patient names are never read. Recover the
coverage directory and continuation JSON from the immutable e156 setup. The
JSON must provide every field in
`configs/safepms/pms_settings.schema.json`, including the PMS coefficients and
the original runtime settings. The runner refuses to infer those settings.

Stage 0 records up to 36 deterministic patient-balanced effective batches. It
stops with `NO-GO` unless every preregistered conflict, projection, finite,
frozen-artifact, and closed-split guard passes. Only `GO` automatically starts
the single paired five-epoch Stage 1 Control-Sum / SafePMS comparison. There is
no development evaluation during epochs 1-4.

Run on AutoDL Bash from the repository root:

```bash
python tools/run_safepms.py \
  --data-root <TNBC_ROOT> \
  --checkpoint ../CA-SAM2-HRC/deliver_ckpts/tnbc_pms_best_e156.pth \
  --coverage-dir <IMMUTABLE_E156_COVERAGE_DIR> \
  --train-manifest logs/nurank/stage1_tnbc_dev/20260711_nurank_stage1/cache/train/manifest.json \
  --development-manifest logs/nurank/stage1_tnbc_dev/20260711_nurank_stage1/cache/development/manifest.json \
  --continuation-config <IMMUTABLE_E156_CONTINUATION_SETTINGS_JSON> \
  --stage0-out logs/safepms/stage0/<run_id> \
  --stage1-out logs/safepms/stage1_tnbc_dev/<run_id>
```

If the continuation LR is uniquely recorded in immutable run metadata, pass it
once with `--lr`; otherwise omit it and the preregistered fixed fallback
`1e-5` is recorded. Completion is a stopping point: do not run a second seed,
test/calibration, a new inference method, or SafePMS Stage 2.
