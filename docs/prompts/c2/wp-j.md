# WP-J: generator coverage and curriculum (session jovyan-73)
Read `prompts/c2/00-common-v2.md`, then `prompts/00-common.md`. Clone `integration-2` to `~/c2-j`; branch `c2-j`.

## Why
The champion trains against WP-B's generator (five families, φ rotation, η reflection, curriculum warm-up, milder modes). It scores ~0.02
lower on shapes from neither generator nor bench (held-out) and lower still on the colleague's suite. Broader coverage of dead-region
geometry during training is the direct lever; the colleague's TRAIN families are an independent design we are allowed to use.

## Configurations (champion + ONE change; 3 seeds each; small file)
J1 add the colleague's five TRAIN families (`rectangle, eta_band, phi_wedge, multi_patch, candidate_loss`, ported from
   `~/hackathon-shared/colleague-group3/src/embedding/degradation.py` with attribution) as extra families in `Degradation`, equal mixture weight.
   NEVER port `ellipse`, `cell_dropout`, `edge_truncation` (their held-out suite) and never add anything resembling our held-out
   families (ellipse, annulus, diagonal band, E shapes).
J2 severity distribution: sample s ~ U(0,1) → mixture with 30% mass above 0.7 (the grader's area weights all severities equally; check
   the champion's effective severity histogram first and report it).
J3 `p_clean` sweep: 0.0 / 0.2 (whatever the champion is not).
J4 curriculum: remove the warm-up (if the champion has it) or double it; one of the two, argued from J2's histogram.
Deliver `plot_augmentation.py`-style examples of J1's new families (WP-N will render them).

## Rules
Generator changes only; `degradation.py` and configs. Do not touch bench maps. Report the dropped-fraction-vs-severity calibration of any new
family (target p_drop·s; campaign 1 register class 9).

## Deliverables
Phase 1 table (≤ 4 configs × 3 seeds) with dev bench AND held-out AND colleague-suite numbers (WP-M runs the latter two; ask), recommendation.
Phase 2 after the ruling. Reports `reports/j-<HHMM>.md`.
