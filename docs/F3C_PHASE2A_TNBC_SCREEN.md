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
