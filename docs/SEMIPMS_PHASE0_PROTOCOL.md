# SemiPMS Phase 0

`research/semipms` is an independent descendant of canonical baseline
`2a1348cb7a1158a6f77aae2f92c168f9552d8068`. It does not revive
SafePMS, DeployPMS, StainRoute, PromptCredit/PromptQ, NuSet/NuRank, or NuPart.

Phase 0 permits only TNBC patients 1–6. It reads one deterministic labeled
image per patient and treats the other 24 images as unlabeled until its
labeled-only leave-one-patient-out acceptance rule is written and checksummed.
Patients 7–11 and MoNuSeg are blocked. The runner refuses e147/e156, TNBC, PMS,
and any checkpoint with point-head/training-state keys as initialization.

The weak teacher begins from an official `sam2_hiera_*.pt` SAM2 model state
and a random point head. It performs exactly 240 optimizer updates using the
six labeled images, saving the final fixed-step checkpoint rather than choosing
a checkpoint from unlabeled images. It then runs a GT-free weak-teacher replay,
residual proposal generation, and cross-view decoding. Only after the frozen
rule is recorded can it open unlabeled labels once for the offline audit.

Find an eligible official SAM2 checkpoint by filename only:

```bash
find checkpoints deliver_ckpts -type f -name 'sam2_hiera_*.pt' -printf '%p\t%s bytes\n' 2>/dev/null
```

Run the audit with the returned official path. It writes exactly one immutable
artifact under `logs/semipms/phase0/<timestamp_sha>/`, including all requested
CSV/JSON records, `tests.txt`, and `SHA256SUMS`. The output reports continuous
evidence curves and deliberately does not auto-declare a research verdict.

After the report is written, stop: do not access patients 7–11/MoNuSeg and do
not implement EMA-student training.
