# TNBC three-seed C0/C1 warm-start decision

The pre-specified seeds are `3407`, `2027`, and `1337`. Every seed uses five
complete epochs, p1--6 optimizer updates, and post-epoch p7--8 strict
development diagnosis. Patients 9--11 and all MoNuSeg test data remain sealed.

The fixed epoch-5 C1-full minus paired C0 comparison is primary. Per-epoch
development PQ-best records are retained for audit only and cannot replace the
fixed-epoch comparison.

The route advances only when all conditions hold:

- mean paired AJI delta is positive;
- mean paired PQ delta is positive;
- AJI is positive for at least two of the three seeds; and
- PQ is positive for at least two of the three seeds.

Otherwise the current warm-start route stops without a p1--8/p9--11 formal
run. Candidate coverage, selected coverage, and selection regret are reported
as mechanism context only.
