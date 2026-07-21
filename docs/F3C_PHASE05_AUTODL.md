# F3C-StainPMS Phase 0.5 AutoDL runbook

This runbook validates only the existing TNBC p1--p6 data path for one or two
training updates.  It is not authorization for baseline training, model
selection, Phase 1 diagnosis, or any MoNuSeg internal split.

## Fixed boundary

- Use the existing prepared TNBC data at
  `/root/autodl-tmp/projects/AgentSeg-CA-SAM2/data/tnbc/train_12`.
- The source manifest is the existing p1--p6 manifest only.  p7--p8 are not
  opened by this command; p9--p11 are rejected before file access.
- The smoke does not construct an evaluation loader, load task-specific TNBC
  weights, write a checkpoint, or run an epoch loop.
- The prepared MAT labels are labelled `smoke_only_pending_raw_binary_gt_audit`.
  This does not decide the eventual TNBC ground-truth protocol.
- MoNuSeg remains the current 37/14 continuity protocol.  No internal 37-image
  train/development split is created or used here.

## Commands

Run from the clean F3C worktree after pulling the current branch.

```bash
cd /root/autodl-tmp/projects/StainPMS-ICASSP2026-f3c
phase05_root=/root/autodl-tmp/f3c_phase05
mkdir -p "$phase05_root/manifests" "$phase05_root/reports"

conda run -n agentseg python tools/capture_phase05_environment.py \
  --output "$phase05_root/reports/environment_phase05.json" \
  --pip-freeze-output "$phase05_root/reports/pip_freeze_phase05.txt"

conda run -n agentseg python -m unittest discover \
  -s tests -p 'test_strict_evaluator.py' -v \
  2>&1 | tee "$phase05_root/reports/strict_evaluator_tests.txt"

conda run -n agentseg python -m unittest discover \
  -s tests -p 'test_manifest_loader.py' -v \
  2>&1 | tee "$phase05_root/reports/manifest_loader_tests.txt"

conda run -n agentseg python -m unittest discover \
  -s tests -p 'test_tnbc_smoke_manifest.py' -v \
  2>&1 | tee "$phase05_root/reports/tnbc_smoke_manifest_tests.txt"
```

Freeze the existing p1--p6 source manifest into a loader-runnable manifest.
This is explicit-list only: it never enumerates the TNBC image directory.

```bash
conda run -n agentseg python tools/freeze_tnbc_smoke_manifest.py \
  --source-manifest /root/autodl-tmp/resimix_tnbc_train.json \
  --image-root /root/autodl-tmp/projects/AgentSeg-CA-SAM2/data/tnbc/train_12/images \
  --label-root /root/autodl-tmp/projects/AgentSeg-CA-SAM2/data/tnbc/train_12/labels \
  --allowed-patients 1 2 3 4 5 6 \
  --expected-count 30 \
  --output "$phase05_root/manifests/tnbc_p1_6_smoke_prepared_labels_v1.json"
```

Run exactly one update first.  It uses only generic SAM2 initialization and
records peak memory, iteration time, hashes, command, and sealed-data
attestation in the JSON output.

```bash
conda run -n agentseg python main.py \
  --dataset tnbc \
  --data_path /root/autodl-tmp/projects/AgentSeg-CA-SAM2/data/tnbc \
  --train_manifest "$phase05_root/manifests/tnbc_p1_6_smoke_prepared_labels_v1.json" \
  --verify_manifest_hashes \
  --train_only_smoke_steps 1 \
  --smoke_output "$phase05_root/reports/tnbc_p1_6_smoke_1batch.json" \
  --sam_ckpt /root/autodl-tmp/projects/CA-SAM2-HRC/checkpoints/sam2_hiera_large.pt \
  --sam_config sam2_hiera_l \
  --evaluator_mode strict \
  --exp_name f3c_phase05_tnbc_smoke
```

If—and only if—the one-batch JSON reports finite losses and at least one
optimizer step, repeat the same command with
`--train_only_smoke_steps 2` and change only `--smoke_output` to
`tnbc_p1_6_smoke_2batch.json`.

Stop after the smoke.  Do not add `--eval`, `--eval_manifest`, task-specific
TNBC checkpoints, or extra epochs.  Generated manifests and reports remain
outside Git.
