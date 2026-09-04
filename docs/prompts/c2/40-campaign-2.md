# Campaign 2 plan: improve the champion on every benchmark (planner, 4 Sep 03:30)

## Levers, ranked by expected value and by what campaign 1 measured
1. **Data scale** (WP-H): 940k events × 400 candidates vs 80k × 200. Also removes the 200-vs-400 train/eval mismatch.
2. **Capacity of the set encoder** (WP-I): a-pma (transformer body + PMA readout, 2.45M params) was separably ahead of the
   90k champion on the bench at R=20 (+0.0022) but not on the official grader. A small body (1-2 layers, 128-256 wide),
   more seed queries, and a larger latent are the middle ground nobody measured.
3. **Consistency objective done right** (WP-K): the two-view arm failed because a uniform latent offset was invisible to
   centred-cosine/normalised-MSE; the fix (penalise ||mean z||, or normalise) was identified and never run. The metric is
   literally "degraded latent close to clean latent under a probe fit on clean", so this is the most direct lever.
4. **Generator coverage** (WP-J): add the colleague's five TRAIN families; tune severity mix and curriculum; keep every
   held-out family out.
5. **Preprocessor and optimisation** (WP-L): sin/cos φ, dxysig squash, EMA/SWA weight averaging, degraded-val selection,
   longer schedule.
6. Measurement (WP-M) and figures/audit (WP-N) make the above decidable.

## Phases and wall clock (approximate; 5 GPUs, 8 cores)
- Phase 0 (03:30-04:15): WP-H ships `gpu_slot.sh`, branch `integration-2`, the full-data loader check, and launches the
  reference: champion × 3 seeds on the small file. Everyone else develops on CPU with `--test_mode`.
- Phase 1 (04:15-08:00): screening on the small file, 3 seeds per configuration, ≤ 4 configurations per package
  (12 runs × ~15 min + 12 benches × ~10 min per package). WP-H starts the full-data reference (3 seeds) as soon as slots allow.
- Phase 2 (08:00-13:00): the top configuration per package (planner ruling from Phase 1) retrained on the full file, 3 seeds,
  then one combined candidate if two arms are individually separable (measured, never assumed).
- Phase 3 (13:00-15:00): certification of the promoted candidate (or the reference if nothing is promoted): R=20 benches,
  held-out, colleague, official × 3, L1T block, fresh-clone accept.sh, submission branch, writeup.

## Slot budget
5 GPUs. Priority: reference runs > screening runs > benches > anything else. At most 5 training jobs and 2 benches
simultaneously (CPU). WP-M's benches take one slot continuously from Phase 1.

## What "improve as much as we can" does not license
- Training on any held-out family or on the colleague's held-out suite.
- Tuning to the bench maps (seeded ruler) or to the organisers' `degradation_eval.py` grid.
- Reading the organisers' private files.
- Claiming an ordering below the seed floor.
