# TNBC C0/C1 second-seed reproduction

This protocol freezes the approved second-seed (`2027`) reproduction of the
five-epoch TNBC warm-start comparison.  It uses patients 1--6 for optimizer
updates and patients 7--8 only after each complete epoch for frozen strict
development diagnosis.  Patients 9--11 are sealed and are never constructed
as datasets in this stage.

The primary comparison is paired C1-full minus C0 at fixed epoch 5.  The
per-epoch equal-patient-macro PQ-best checkpoint is retained for recordkeeping
but cannot replace the fixed-epoch primary comparison.

## Storage contract

Each arm atomically retains only:

- `checkpoints/last_complete_state.pth`: model, model1, optimizer, scheduler,
  and RNG for recovery (about 6 GiB);
- `best_pq/model_model1_weights.pth`: model/model1 weights only (about 2.9
  GiB);
- JSON/CSV metrics, declarations, logs, and read-only diagnosis summaries.

No permanent `epoch_*.pth` archive is created. The two-arm run is expected to
use about 18--20 GiB persistently, including diagnostics. Atomic replacement
of a `last` state briefly adds about 6 GiB, so the sequential two-arm peak is
about 24--26 GiB. Each arm refuses to start below 16 GiB free space.
