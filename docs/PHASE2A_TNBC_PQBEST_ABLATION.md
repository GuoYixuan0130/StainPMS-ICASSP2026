# Phase 2A TNBC PQ-best loss ablation

This protocol keeps the completed C0/C1 screen intact.  It adds only two
five-epoch exploratory weight-warm-start arms: `coverage_only` and
`quality_only`.  Both use p1--6 for training and p7--8 only for development
checkpoint selection.  TNBC p9--11 are not part of this workflow.

Each completed epoch is diagnosed with the frozen strict evaluator.  The
selected model is the strictly highest equal-patient p7/p8 macro PQ; an exact
tie retains the earlier epoch.  Fixed epoch 5 remains in the report for an
equal-training-length comparison with C0/C1.

Retention is deliberately bounded per arm:

- `checkpoints/last_complete_state.pth`: one rolling model/model1/optimizer/
  scheduler/RNG state for recovery;
- `best_pq/model_model1_weights.pth`: model/model1 weights for the currently
  selected development PQ-best epoch;
- JSON/CSV epoch metrics and small diagnosis outputs.

No historical full epoch states are retained.  The runner checks for 12 GiB
free space before an arm starts; practical persistent storage is about 9 GiB
per arm, with extra temporary space required during an atomic overwrite.

`C0` is the continued-training control for an added-loss claim.  A comparison
against the historical StainPMS checkpoint is reported separately as overall
exploratory warm-start change and cannot by itself establish that either added
loss caused an improvement.
