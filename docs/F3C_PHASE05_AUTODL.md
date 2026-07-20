# F3C-StainPMS Phase 0.5 AutoDL protocol

This runbook materializes evidence and performs a 1--2 image train-only smoke.
It is not authorization for baseline training or Phase 1.  It must be run on
`research/f3c-stainpms` in the frozen `agentseg` environment.

## Safety boundary

- MoNuSeg test14: the manifest tool may read ZIP directory metadata and raw
  source-TIFF bytes only to record filename, size, CRC and SHA256.  It does not
  decode test images and never opens test XML/PNG/MAT annotations.
- TNBC patients 9--11 are never enumerated or read.
- XML and prepared-label audit is accepted only for a training manifest; the
  tool rejects a test-role manifest before opening its archive.
- Smoke mode requires a hash-verified train manifest, constructs no evaluation
  loader, writes no checkpoint and exits after the requested manifest-ordered
  training images.
- All generated reports, regenerated candidate labels, archives, logs and
  checkpoints remain outside Git.

## Inputs to place outside the repository

Use a dedicated directory such as `/root/autodl-tmp/f3c_phase05/source` and
preserve the original downloaded filenames.  Required inputs are:

1. current official MoNuSeg Training Data ZIP, Drive file ID
   `1ZgqFJomqQGNnsx7w7QBzQQMVA16lbVCA`;
2. current official MoNuSeg test ZIP, Drive file ID
   `1NKkSQ5T0ZNQ8aUhh0a8Dt2YKYCQXIViw`;
3. preferably, official training-organ information and XML-to-mask MATLAB
   converter files, Drive IDs `1xYyQ31CHFRnvTCTuuHdconlJCMk2SK7Z` and
   `1YDtIiLZX0lQzZp_JbqneHXHvRo45ZWGX`.

Record the actual download time in UTC.  Do not extract or inspect test labels.

## Commands

Create local output directories and capture the frozen environment:

```bash
cd /root/autodl-tmp/projects/StainPMS-ICASSP2026-f3c
mkdir -p /root/autodl-tmp/f3c_phase05/reports

conda run -n agentseg python tools/capture_phase05_environment.py \
  --output /root/autodl-tmp/f3c_phase05/reports/environment.json \
  --pip-freeze-output /root/autodl-tmp/f3c_phase05/reports/pip_freeze.txt
```

Run all CPU tests in `agentseg`:

```bash
conda run -n agentseg python -m unittest discover -s tests -p 'test_*.py' -v \
  2>&1 | tee /root/autodl-tmp/f3c_phase05/reports/unit_tests.txt
```

Materialize the four versioned manifests.  Replace the four source filenames
and timestamp with the actual values; omit the two optional arguments if those
official auxiliary files are not present.

```bash
conda run -n agentseg python tools/build_monuseg_manifests.py \
  --train-archive /root/autodl-tmp/f3c_phase05/source/TRAIN_ARCHIVE.zip \
  --test-archive /root/autodl-tmp/f3c_phase05/source/TEST_ARCHIVE.zip \
  --prepared-image-root /root/autodl-tmp/projects/AgentSeg-CA-SAM2/data/monuseg/train_12/images \
  --legacy-label-root /root/autodl-tmp/projects/AgentSeg-CA-SAM2/data/monuseg/train_12/labels \
  --downloaded-at-utc 2026-07-21T00:00:00Z \
  --organ-info /root/autodl-tmp/f3c_phase05/source/ORGAN_INFORMATION_FILE \
  --official-converter /root/autodl-tmp/f3c_phase05/source/XML_CONVERTER_FILE \
  --output-dir /root/autodl-tmp/f3c_phase05/manifests
```

Query GDC case metadata only (no image request) and retain the raw response:

```bash
conda run -n agentseg python tools/audit_tcga_metadata.py \
  --output /root/autodl-tmp/f3c_phase05/reports/extended7_tcga_metadata.json \
  --save-raw-response /root/autodl-tmp/f3c_phase05/reports/extended7_gdc_raw.json
```

Audit source XML, legacy labels and train-image preprocessing.  Candidate labels
are deliberately written to a new directory and never overwrite legacy labels:

```bash
conda run -n agentseg python tools/audit_monuseg_xml_labels.py \
  --manifest /root/autodl-tmp/f3c_phase05/manifests/monuseg_download37_v1.json \
  --output /root/autodl-tmp/f3c_phase05/reports/monuseg_xml_label_audit.json \
  --summary-output /root/autodl-tmp/f3c_phase05/reports/monuseg_xml_label_audit.md \
  --regenerated-label-root /root/autodl-tmp/f3c_phase05/candidate_labels/xml_region_skimage_v1
```

Run the train-only smoke from generic SAM2 initialization on the first one or
two classic30 manifest records.  This command does not load extended7 or test14:

```bash
conda run -n agentseg python main.py \
  --dataset monuseg \
  --data_path /root/autodl-tmp/projects/AgentSeg-CA-SAM2/data/monuseg \
  --train_manifest /root/autodl-tmp/f3c_phase05/manifests/monuseg_challenge30_v1.json \
  --verify_manifest_hashes \
  --train_only_smoke_steps 1 \
  --smoke_output /root/autodl-tmp/f3c_phase05/reports/monuseg_classic30_smoke.json \
  --sam_ckpt /root/autodl-tmp/projects/CA-SAM2-HRC/checkpoints/sam2_hiera_large.pt \
  --evaluator_mode strict \
  --exp_name f3c_phase05_monuseg_smoke
```

Stop after the smoke.  Do not add `--eval`, do not provide an eval manifest,
and do not run any baseline epoch loop.

## Return bundle

Return the whole `/root/autodl-tmp/f3c_phase05/reports` directory and the five
JSON files under `/root/autodl-tmp/f3c_phase05/manifests`.  The candidate-label
MAT files are large and need not be returned unless a discrepancy requires
pixel-level inspection.  Do not commit or push any generated file.
