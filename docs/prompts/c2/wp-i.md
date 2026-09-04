# WP-I: encoder capacity (session jovyan-72)
Read `prompts/c2/00-common-v2.md`, then `prompts/00-common.md`. Clone `integration-2` to `~/c2-i` when WP-H announces it; branch `c2-i`.

## Why
Campaign 1: PMAEncoder(num_layers=0) with 89,606 params (0.8255 ± 0.0007 at R=20) vs a-pma (4-layer transformer body + PMA readout,
2,445,478 params, 0.8277 ± 0.0015): separable on the bench, tied on the official grader, 27x the size. Nobody measured anything in between.
`latent_dim` is read from the config by the grader, so it is a legal knob; campaign 1 has one un-benched `lat16` config.

## Configurations (each = champion config + ONE change; 3 seeds each; screen on the small file)
I1 `num_layers: 1` (one pre-norm transformer block before PMA).      I2 `num_layers: 2`.
I3 `embed_size: 256` (num_layers 0).                                  I4 PMA seeds 8 (kwarg with default 4 → make the default the final choice only if promoted).
I5 `latent_dim: 16`.                                                   I6 `latent_dim: 32`.
Run I1, I3, I5 first (one seed each) to check nothing breaks, then all seeds. Report params for every row. If a row is separably above the
reference by the seed test, its combination with the next-best row is a NEW row (measured, never assumed).
Check `eval.py` still constructs the model from the config alone for every change (aliasing rule); if a change needs a kwarg, its default
must be the final choice.

## What you must not do
Change the preprocessor, generator, loss or schedule (other packages). Train on held-out or colleague families. Tune to bench maps.

## Deliverables
Phase 1: table of ≤ 6 configurations × 3 seeds with mean ± seed std, params, clean AUC; your recommendation for the one to carry to full data.
Phase 2 (after the planner's ruling): that configuration × 3 seeds on the full file. Reports `reports/i-<HHMM>.md`.
