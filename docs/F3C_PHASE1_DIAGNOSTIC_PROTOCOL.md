# F3C-StainPMS Phase 1: read-only candidate-failure diagnosis

This protocol implements the project-owner-approved Phase 1 only. It has no
optimizer, scheduler, gradient update, F3C loss, LoRA, threshold sweep, or
test-set path.

The frozen machine-readable specification is
`configs/phase1/metrics_frozen_v1.json`. The main strict matching IoU is 0.5;
CCR at 0.3 and 0.7 is descriptive sensitivity only.

The standard StainPMS path requests `multimask_output=False`, which selects
native SAM2 mask token 0. The candidate audit reads all native decoder tokens
through `predict_masks`: token 0 plus the three ambiguity tokens. This is a
read-only observation path. The actual selected candidate mirrors the existing
`multimask_output=False` call, including its enabled dynamic-stability fallback
to the highest-quality ambiguity token; NMS and instance assembly are unchanged.

For causal attribution, an automatic candidate is associated with a GT only
when its automatic prompt coordinate lies in that GT instance. Therefore the
conditional automatic CCR asks whether masks can cover an instance after an
automatic point reached it, while end-to-end CCR keeps all GT instances in the
denominator. Fixed GT points use a deterministic EDT-maximum interior pixel.

The five GT classes are exhaustive and mutually exclusive. A final strict TP
takes precedence; otherwise the earliest failed stage is reported: point,
candidate generation, default-token selection, then downstream assembly/NMS.
Split, merge, FP/FN and boundary/localization counts are supplementary and may
overlap these five causal categories.

Checkpoint declarations are mandatory. A checkpoint with unknown selection
history is labelled `historical_exploratory`; its diagnostic output cannot be
used as clean-baseline or final-performance evidence.
