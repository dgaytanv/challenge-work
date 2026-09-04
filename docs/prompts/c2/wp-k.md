# WP-K: consistency objective with the offset fix (session jovyan-75)
Read `prompts/c2/00-common-v2.md`, then `prompts/00-common.md`, then `writeup/C-two-view-consistency.md` and
`writeup/D-batchnorm-selection-failure.md` (campaign 1's failure analysis, mandatory).

## Why
The grader fits a probe on CLEAN latents and applies it to DEGRADED latents of the same events; a loss that pulls z(degraded) toward
z(clean) targets the metric directly. Campaign 1's two-view arm failed because the latent's common offset inflated ~700x while both
consistency terms (centred cosine, population-normalised MSE) are blind to a uniform translation; the probe was left ill-conditioned.
The fix was written down and never run.

## Configurations (champion + the two-view loop + ONE consistency variant; 3 seeds; small file)
K1 stop-grad latent MSE z(deg) → z(clean) on UNnormalised latents + penalty λ·||mean_batch z||² (λ from a 10-step scan on
   `--test_mode`, report it), `val_bn_batch_stats: true`.
K2 latent L2-normalised before the loss (and the probe sees the normalised latent, so `PMAEncoder.forward` returns it; check the
   grader contract) + cosine consistency.
K3 logit-JSD: consistency through the classifier head (JSD between softmax(clean) and softmax(deg)), no latent term.
K4 K1 with the consistency weight ×3.
Log ||mean z||, per-dim std and the probe's condition number every epoch; a run whose offset grows is a failure even if its loss falls.

## Rules
Loss, training loop and configs only. Do not change the encoder or generator. A row that improves clean AUC but not mean_area is a null.

## Deliverables
Phase 1 table (≤ 4 × 3 seeds) with mean_area, clean AUC and the offset diagnostics; recommendation. Phase 2 after the ruling.
Reports `reports/k-<HHMM>.md`.
