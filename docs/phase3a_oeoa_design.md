# Phase 3A — Orthogonal Error-Oracle Audit (OEOA)

## Status and scope

This is a preregistered, GT-only diagnostic of already exported TNBC p7/p8
compact C0/C1 artifacts. It does not run neural inference, train a model,
modify C1, adjust C4-CSR, read p1–6 for training, or access p9–11 or MoNuSeg.
Its oracle values are upper bounds and must not be described as implemented
model performance.

The baseline is strictly paired C0 and C1 for seed 2027 and seed 1337. Seed
2027 uses the recovery-audited original C1 epoch-5 lineage. Seed 1337 uses
only the frozen reconstructed C1 epoch-5 lineage, whose complete state,
weights-only state, and canonical `model/model1` tensor hashes are read in
full from the frozen manifest. Historical seed-1337 C1/C3 records are not
inputs and must not be mixed into this audit.

## Fixed evaluator and aggregation

All maps are evaluated by `stainpms.evaluator.evaluate_instance_pair` in
`strict` mode with `match_iou=0.5`, reporting Dice1, Dice2, AJI, DQ, SQ, and
PQ. “Standard PQ TP” means the exact pairing produced by that evaluator,
including its established compatibility treatment of an exact 0.5 IoU.

For each seed, images are first averaged within patient 7 and patient 8; the
seed patient-macro is the equal mean of those two patient values. The two-seed
macro is the equal mean of the two seed patient-macros. C0-relative and
C1-relative deltas always use the corresponding strictly paired value at the
same aggregation level.

## Final-instance overlap-component oracle

The native C1 final map and GT map form a bipartite graph. A prediction and a
GT instance are linked iff they share at least one pixel. Isolated prediction
and GT nodes remain components. Let P and G be the prediction and GT counts in
a component. Components are mutually exclusive and exhaustive:

| action | exact condition | GT-only operation |
|---|---|---|
| `tp_boundary` | P=1, G=1, standard-PQ TP | replace prediction mask with its GT mask |
| `subthreshold_1to1` | P=1, G=1, not a standard-PQ TP | replace prediction mask with its GT mask |
| `merge` | P=1, G≥2 | remove prediction; insert all component GT instances |
| `split_or_duplicate` | P≥2, G=1 | remove all predictions; insert component GT instance |
| `complex_topology` | P≥2, G≥2 | remove all predictions; insert all component GT instances |
| `pure_fn` | P=0, G≥1 | insert all component GT instances |
| `pure_fp` | P≥1, G=0 | remove all component predictions |

Only components in an enabled action class are modified. Output maps are
reindexed safely; a zero-action map is byte-identical to native C1 and the
all-seven-action map is byte-identical to GT. Every one of the 128 subsets is
evaluated, with output order checked under forward and reverse action orders.

## Declared analyses

The report contains every atomic action, all 128 combinations, declared
routes, pairwise interactions `Delta(A+B)-Delta(A)-Delta(B)`, and exact
seven-action Shapley values for AJI and PQ. It also enumerates minimum action
subsets meeting the stated +0.020 C0-relative thresholds; these are oracle
feasibility facts, not permission to implement an intervention.

Declared routes are: `mask_boundary`, `local_mask_rescue`, `topology`,
`coverage`, `precision`, `mask_quality_total`, and `detection_total`, exactly
as fixed in the companion configuration.

For each native-final standard-PQ FN GT, maximum IoU is recorded against the
native final prediction set, selected candidate pool, and all candidate pool.
The mutually exclusive priority labels are `assembly_or_keep_miss`,
`selection_miss`, `candidate_mask_near_miss`, and `generation_miss`. Selected
and all pools use the established native prompt-group-constrained one-to-one
matching: maximize IoU>0.5 match count, then total IoU. Their PQ ceilings are
explicitly theoretical candidate-set ceilings.

## Frozen decision arithmetic

For each declared route and AJI/PQ, the audit reports C1−C0, oracle−C1,
oracle−C0, distance to +0.020, and
`(0.020-(C1-C0))/(oracle-C1)`. A non-positive oracle gain is `impossible`; a
required recovery above one is `oracle_cannot_reach_target`. No result may add
an action class, alter a threshold, or start a subsequent experiment.
