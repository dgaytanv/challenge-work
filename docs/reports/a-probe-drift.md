# WP-A: probe-visible vs probe-invisible drift

**Hypothesis from WP-D and WP-C.** Measurement and analysis WP-A. Branch `wp-a` @ `fcfee2d`,
code `tests/probe_drift_decomposition.py`, JSON in `runs/probedrift-*.json`.
Read-only measurement on the shared bench families; nothing trains or tunes on them.

## The question
Deep Sets scores mean_area 0.80 while its latent moves a large fraction of the population
spread. Is that because most of the motion is in directions the frozen probe cannot read?

## Verdict: half true, and the half that is false is the half that mattered

**True:** Deep Sets' displacement really is less probe-aligned. **False:** it does not move less
— it moves far more, and the reduced alignment nowhere near compensates. "It moves a lot but
invisibly" does not survive measurement as an explanation of the score.

```
family   | STOCK   rel  alignX  shared     AUC | DEEPSETS  rel  alignX  shared     AUC
rect     |        1.03    1.37   0.459  0.7197 |         3.81    0.94   0.700  0.7021
wedge    |        0.95    1.44   0.547  0.7537 |         2.91    0.76   0.728  0.8235
strip    |        1.01    1.40   0.506  0.7581 |         3.84    0.89   0.719  0.7554
towers   |        1.00    1.49   0.573  0.7493 |         3.38    0.91   0.750  0.7721
cells    |        0.80    1.38   0.454  0.8253 |         2.00    0.86   0.615  0.8422
MEAN     |        0.96    1.42   0.508  0.7612 |         3.19    0.87   0.702  0.7791
```
`rel` = mean ||dz|| / clean spread. `alignX` = alignment with the probe gradient **as a multiple
of the random-direction baseline**. `shared` = ||mean(dz)|| / mean||dz||.

- **Alignment:** stock sits at **1.42x chance**, Deep Sets at **0.87x chance**. Deep Sets' motion
  is genuinely at-or-below chance alignment with what the probe reads; stock's is above it.
- **Magnitude:** Deep Sets moves **3.19x** its own spread against stock's **0.96x**. Net
  probe-visible displacement (`rel * along_g`) is therefore *larger* for Deep Sets, not smaller.
- **Score:** Deep Sets mean AUC 0.7791 vs stock 0.7612 — a real but modest edge, and it is
  **family-dependent** (Deep Sets is *worse* on rect: 0.7021 vs 0.7197).

## Read the random-direction baseline before reading anything else
At latent_dim 6 a random displacement already has mean |cos| = **0.34** with any fixed direction.
Deep Sets' raw fraction is ~0.29 and stock's ~0.47. Reported bare, "0.29 along the probe
direction" invites the conclusion that 71% of the motion is invisible — when 0.29 is *below
chance* and 0.47 is *40% above* it. Every alignment number here is a multiple of that baseline
for exactly this reason; the bare fractions are not interpretable.

## What actually predicts AUC
Pooled over 25 (family, severity) points, pearson / spearman:

```
                              stock                deepsets
raw |dz| / spread        -0.840 / -0.982      -0.861 / -0.938
Mahalanobis |dz|         -0.885 / -0.958      -0.814 / -0.908
fraction along grad      -0.401 / -0.141      -0.028 / +0.356
ABSOLUTE probe-visible   -0.910 / -0.912      -0.944 / -0.964
shared-shift fraction    -0.310 / -0.400      -0.687 / -0.912
```

**The absolute probe-visible displacement is the best predictor in both models.** The *fraction*
alone is useless and unstable — it flips sign between models and, per-family, ranges from -0.64
to +0.92. The fraction cannot distinguish "moves little" from "moves a lot but invisibly"; only
the product can, which is why it was reported.

## The most useful new number: the displacement is a shared translation
`shared` = the share of the displacement that is one common translation of the whole population,
rather than events scattering individually. Stock **0.51**, Deep Sets **0.70**, and for Deep Sets
it reaches **0.99** at high severity — at s>=0.8 essentially the entire displacement is a rigid
shift of the cloud.

That is a specific and different failure from "events cross the decision boundary". The cloud's
internal structure is largely preserved; it has walked off the manifold the probe was fit on.
It also explains the two things the alignment numbers could not: why Deep Sets tolerates a
displacement three times larger, and why `cells` (lowest shared fraction, 0.615) is the family
both encoders handle best.

**Consequence for WP-C:** a consistency term that penalises the *shared* component is attacking
the dominant mode — and the MSE term is the one best suited to it (see the correction below).
Run 1 weighted that term at 0.1 and the uncentred cosine at 1.0, which is backwards if the damage
is dominated by a rigid shift.

## Honest limits
1. **Cross-model comparison is confounded.** Each model has its own spread and its own separately
   fitted probe, so "units of own spread" is the only common footing and it is imperfect. Deep
   Sets also starts from a better clean latent (AUC 0.8854 vs 0.8604, spread 1.296 vs 14.705).
   A material part of its edge is a better-separated clean representation, not motion geometry.
   The measurements above do **not** establish that invisibility is why Deep Sets wins.
2. `alongW`, the alignment with the *global logistic* discriminant, is 0.021-0.026 for stock
   against a 0.341 baseline — systematically orthogonal. The drift barely moves events along the
   linear signal axis yet still destroys a nonlinear probe. Worth pursuing; not yet explained.
3. Deep Sets AUC is non-monotone in severity on rect (0.5424 at s=0.8, 0.5806 at s=1.0), so
   single-severity comparisons on that family are unsafe.
4. Deep Sets numbers are from `d-deepsets-clean` (no degradation training). An augmented Deep
   Sets may behave differently.

## Related: the latent offset (WP-C's thread, measured here)
`tests/latent_offset_audit.py`, same convention throughout (offset = ||E[z]||, spread =
mean||z - E[z]||):

```
                              offset   spread    ratio  ||W||_F  ||E[h]||  h-ratio  align
stock        eval  n=6000     7.2514  14.6887    0.49x   3.2207         -        -      -
stock        train n=512      2.6221  15.4311    0.17x   3.2207    4.5141    0.50x  0.174
C run1 ep8   train n=512     92.2164   3.0273   30.46x        -         -        -      -
C run1 ep17  train n=512    154.8795   4.1254   37.54x   6.4916   25.0160    7.56x  0.949
```
(ep8 row is WP-C's measurement; my independent reproduction was still queued at time of writing.)

- **Offset inflation continues; spread collapse does not.** ep8 -> ep17 the offset grew +68% but
  the spread also grew +36%. WP-C predicted a shrinking spread; it recovered. The pathology is
  specifically offset inflation, which is what the uncentred-cosine mechanism predicts and what a
  "spread collapse" reading does not.
- **The offset is not explained by the weights.** ||W||_F grew 2.0x and ||E[h]|| 5.5x, a norm
  product of 11.2x, against an offset that grew 59x. The missing factor is **alignment**:
  ||W E[h]|| / (||W||_F ||E[h]||) went 0.174 -> 0.949. The bottleneck rotated to map the common
  pre-bottleneck component almost maximally into the latent.
- **The pathology exists upstream of the bottleneck.** h's own offset ratio is 7.56x vs stock's
  0.50x. A bottleneck-only fix cannot remove it; the fix belongs upstream or in the loss.
- **Calibration warning.** The healthy reference is convention-dependent: stock is 0.49x on eval
  but **0.17x** on train. A train-measured ratio must be compared against the train-measured
  stock figure. Stock's offset is a small residual and is *not* stable across datasets (2.62 vs
  7.25); WP-C's is stable precisely because it is dominant. Stability is a symptom, not a
  property of the metric.

### Which loss term can even see the offset (WP-C, recorded here because it closes the thread)
WP-C tested whether the other terms are coupled to a common latent offset at all. The Projector is
`Linear -> BatchNorm1d -> GELU`, and `Linear(z + c) = Linear(z) + Wc` with `Wc` constant across the
batch, which BatchNorm then subtracts exactly. Measured in train mode, offsets up to 155:
max |proj(z+mu) - proj(z)| = 4.8e-07, 2.4e-06, 3.1e-06 at ||mu|| = 20, 90, 155. Numerically zero.

So CE and SupCon are **structurally** blind to a common latent offset — a property of the
Linear-then-BatchNorm ordering, not of the trained weights. With both MSE forms exactly
translation-invariant, the uncentred cosine is the only term in run 1's objective coupled to the
offset at all, and it rewards it by a factor of 190. WP-C's caveat is right and worth keeping:
this establishes it as the only source of incentive, not that the incentive alone produces 59x.

**The asymmetry that matters for the rest of us:** in *eval* mode the projector is no longer
blind (7.8e-01 at offset 90), because BatchNorm switches to running statistics. The offset is
invisible to the training gradient but visible at inference. Any validation number read off a
model in this state is measuring something the training signal never saw.

### CORRECTION: two different translations, and the loss terms treat them oppositely
An earlier version of this report claimed the shared-shift fraction and WP-C's latent offset were
"one quantity measured two ways", and that a translation-invariant MSE therefore could not
penalise 70-99% of the displacement. **That was wrong**, and it would have pushed the fix in the
wrong direction. WP-C caught it with a direct measurement. There are two distinct translations:

- **(A) shared shift between views**, `z_d = z_c + t` — what the `shared` column above measures.
- **(B) common offset in both views**, `z_c += c` and `z_d += c` — WP-C's `||E[z]||` pathology.

"Translation-invariant" refers only to (B). Measured by WP-C:

```
(A) z_d = z_c + t          ||t||    raw_mse   pop_nmse   centred_cos
                            0.00    0.00000    0.00000      0.000000
                            1.00    0.16667    0.01944      0.011598
                            2.00    0.66667    0.07775      0.044795
(B) both views += c        ||c||    raw_mse   pop_nmse   centred_cos   uncentred_cos
                            0.0     0.36402    0.04047      0.023853        0.023840
                           90.0     0.36402    0.04047      0.023853        0.000110
```

The MSE terms are invariant to (B) and **fully sensitive to (A)** — a rigid shift between views is
the easiest thing for an MSE to penalise, contributing `||t||^2` with no cancellation. So the MSE
is not blind to the dominant failure mode; it is the term best matched to it.

The grader still only cares about (A): a constant offset present in both passes is absorbed by a
probe fitted on clean latents. (B) matters because the uncentred cosine *rewards* inflating it,
which is how run 1 spent its largest loss weight on a term that was vacuous.

**The conclusion that survives, and it is stronger than the one it replaces:** if the damage is
70-99% rigid shift, run 1's weighting was backwards — 1.0 on a cosine term that was vacuous,
0.1 on the MSE term that penalises the dominant mode. Raising `consistency_mse_weight` relative
to `consistency_weight` is the concrete proposal, and it comes straight out of the shared-shift
measurement. WP-C is putting it to the planner as a run-3 candidate.

Two related traps worth recording. Centring each view by *its own* mean would remove `t` exactly
and make the centred cosine blind to (A); WP-C centres both views by the **clean** batch mean, so
`t` survives — the ordering matters. And the "MSE against a stop-gradient clean branch" I proposed
as a fix is already what the code does (z_c detached, MSE against it), so there was nothing to add.

## Credit
Hypothesis: WP-D and WP-C. Offset phenomenon and the uncentred-cosine mechanism: WP-C.
Measurement, baselines, decomposition and analysis: WP-A.
