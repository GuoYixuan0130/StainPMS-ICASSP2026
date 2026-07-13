# Anchored SemiPMS Stage 1B

This runner is the single authorised rescue experiment after the online
EMA/dynamic-cache/mask-pseudo-training Stage 1 route was declared NO-GO.

- The 720-step reconstructed supervised checkpoint is the only initializer.
- Its teacher, SAM2 image/prompt/mask modules, mask-quality modules, and the
  static cache are frozen. Only `deform_layer`, `reg_head`, `cls_head`, and
  `conv` in the auto-point model may update.
- The cache is built once from the 24 p1--6 images without opening their
  labels. It contains base standard-deployment instances and accepted residual
  centres. Residual masks are only a source for one centre and ROI; no pseudo
  mask/Dice/BCE loss is computed.
- LOPO calibration uses only the six labelled images, prioritises 90% point
  precision, and freezes the rule and proposal budget before cache creation.
- Labeled-only, static-base, and Anchored SemiPMS each continue for 240
  optimizer steps from a freshly loaded copy of exactly the same anchor.
- Development evaluation is standard StainPMS inference at 0/60/120/180/240
  on patients 7--8. Patients 9--11 and MoNuSeg are never enumerated or read.

The artifact is `logs/semipms/stage1b_anchored_tnbc_dev/<run_id>/`. It contains
the frozen cache manifest and checksum, deduplication and cross-view score
statistics, explicit trainable parameters, point-loss/gradient logs, per-image
and per-patient metrics, best/final point-head checkpoints, tests, and
`SHA256SUMS`.
