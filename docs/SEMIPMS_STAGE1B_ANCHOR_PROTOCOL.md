# Anchored SemiPMS Stage 1B — one-time anchor reconstruction

This artifact reconstructs the missing supervised 720-step checkpoint exactly
once from the original Stage-1 training code path (`aab0a3d`). It uses only the
six labelled TNBC images selected in the valid Stage-1 manifest and official
`sam2_hiera_large.pt`, with seed 3407 and the original batch size, augmentation,
AdamW settings, StainPMS loss, no scheduler and AMP disabled.

No unlabeled train image, pseudo label, EMA, patient 9--11, or MoNuSeg input is
opened. Full checkpoints at 0/240/480/720 include optimizer, scheduler (null),
GradScaler (null) and Python/NumPy/PyTorch/CUDA RNG state. The runner preflights
disk space for all four checkpoints before its first optimizer update.

At step 240 it compares the reconstructed checkpoint against the valid prior
240-step checkpoint. Differences are recorded rather than hidden. Step 720 is
evaluated once on patients 7--8. A finite, recoverable, nonzero-PQ 720 checkpoint
becomes the sole permitted Stage-1B anchor; the reconstruction itself stops.
