# TNBC C0/C1 5-epoch screen

This is the owner-approved exploratory warm-start screen. It trains only TNBC p1–6 and evaluates immutable epoch checkpoints only on p7–8. Patients 9–11 are neither present in the training process nor in the evaluation commands.

`C0` continues the historical StainPMS objective. `C1` starts independently from identical task weights and adds the frozen native best-of-K coverage plus quality-calibration terms. Both runs consume the same hash-locked train-only coverage cache; coverage is not refreshed during the five epochs.

Run `tools/run_phase2a_tnbc_c0c1_screen.sh SMOKE_ROOT` on AutoDL. It:

- requires a clean, committed worktree and the previously passing TNBC C0/C1 smoke gate;
- computes one shared p7–8 epoch-0 strict diagnosis before training;
- runs C0 and C1 in separate processes, each for exactly 1,350 attempted crop batches (five complete epochs); no-prompt batches retain the legacy no-step behavior and actual optimizer updates are reported afterward;
- fails closed if the arms have different no-prompt positions, attempted crop counts, effective optimizer updates, or scheduler states;
- saves model, point head, optimizer, scheduler, RNG, runtime state, checkpoint SHA256, and declaration after every epoch;
- evaluates all ten immutable epoch checkpoints with the same frozen Phase 1 decoder, NMS, assembly, and strict evaluator;
- retains the complete machine-readable diagnosis but removes each diagnosis's resumability-only `texture_memory_bank.pt` after it completes successfully;
- creates p7, p8, equal patient-macro, and C1−C0 tables; and
- applies the promotion rule only to epoch 5. It never starts epoch 10 or MoNuSeg.

The generated result remains single-seed exploratory warm-start evidence. A pass indicates a stable exploratory signal, not final performance validation.

## Interrupted C1 recovery

If C0 completed and C1 stopped only after an epoch checkpoint was atomically saved,
do not rerun C0 or restart C1 from the historical checkpoint. After restoring enough
disk space, run `tools/resume_phase2a_tnbc_c1_screen.sh SCREEN_ROOT SMOKE_ROOT`.
The recovery path accepts only the latest C1 checkpoint in that screen root and
fails closed unless every prior C1 epoch is contiguous and has matching arm,
train-manifest, coverage, screen-config, checkpoint declaration, no-prompt audit,
optimizer, scheduler, and RNG provenance. It restores model, point head, optimizer,
scheduler, and RNG, completes only the remaining epochs, then runs the unchanged
fairness gate and fixed p7/p8 evaluation.
