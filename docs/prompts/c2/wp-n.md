# WP-N: figures, register, independent audit (session jovyan-17)
Read `prompts/c2/00-common-v2.md`, then `prompts/00-common.md`, then `runs/plots/README.md` (you own it now, and its register of twelve
silent-failure classes) and `reports/f-0130.md`. You inherit WP-F's plot scripts in `bench/plot_*.py`.

## N1 figures (continuous)
- Seed-spread figure: per configuration, three seed points + mean, reference band shaded, separated by data regime; one panel per suite.
- Keep `curves/heldout/colleague/altdata/official/progress` current under the ownership colour rule; L1T rows keep the triangle rule.
- Augmentation examples for WP-J's new families (`plot_augmentation.py`).
- Run `check_layout` on every render; no figure is published unless it passes. State on every figure whether it was visually inspected;
  if image reads work in your session, inspect the top three figures and say so.

## N2 audit (the new part)
Every gain any package claims gets an independent re-measurement by you before it is called real: clone their branch fresh, rebuild the
model through `eval.py`'s constructor path (`--use_repo_builder`), bench one seed yourself, and confirm the number within probe noise.
Also check: the config the checkpoint was trained with is the config in the repo; the tag's seed matches the config's seed; the
train_data field matches the file the training log names. Report mismatches to the planner immediately, then to the package.

## N3 register
Add a class for every new silent failure with what looked fine, what was wrong, and the defence. Campaign 2 starts at twelve.

## Rules
Never train. Never edit another package's code (report, do not fix). Reports `reports/n-<HHMM>.md` every 60 min.
