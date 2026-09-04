# A: What the degradation does to the latent, and why the probe fails

Every number sourced to `runs/probedrift-*.json`, `runs/a-*.json`, or `reports/a-*.md`.
Code: `tests/latent_drift_diagnostic.py`, `tests/probe_drift_decomposition.py`. Branch `wp-a`.

## 1. The metric is driven by per-event latent drift

The premise every work package was built on, measured rather than assumed. Stock checkpoint,
8000 events, five bench families x five severities:

| family | r(cos, AUC) | rho(cos, AUC) | cos at s=1.0 |
|---|---|---|---|
| rect | +0.886 | **+1.000** | 0.102 |
| wedge | +0.852 | **+1.000** | 0.100 |
| strip | +0.936 | **+1.000** | 0.120 |
| towers | +0.884 | **+1.000** | 0.100 |
| cells | +0.975 | **+1.000** | 0.281 |
| pooled (25) | +0.850 | +0.985 | |

Rank correlation is **exactly 1.000 within every family**. Centring the latents on the clean-set
mean changes nothing (per-family rho still 1.000; pooled pearson +0.860 -> +0.866).

**Control that could have killed it, and did not.** Pairing each degraded event with a *different*
clean event gives cosine **-0.005** against **+0.424** for matched pairs. The metric reads
per-event identity, not global geometry. The sharpest single piece of evidence: at `rect` s=1.0
the centred cosine is -0.0099 and at `towers` s=1.0 it is -0.0107, both sitting exactly on that
shuffled floor — a degraded event is then statistically indistinguishable from an unrelated one.
Probe AUC at those two points is 0.5058 and 0.5000, i.e. chance. Two independent measurements
agreeing that per-event identity is gone.

**Caveat, and it is not a small one:** this ranks severities *within* a family perfectly, but
across families the rank correlation is only **+0.700** (cosine orders them
cells > wedge > towers > strip > rect; AUC orders them cells > strip > wedge > towers > rect).
And pooled pearson (+0.850) well below pooled spearman (+0.985) means the relationship is monotone
but **not linear**. "Cosine rose, robustness improved" is supported. "Cosine rose by X so
mean_area rises by Y" is not.

## 2. Read the random-direction baseline before any alignment number

At latent_dim 6, a random displacement already has mean |cos| = **0.34** with any fixed direction.
So a raw "0.29 of the motion lies along the probe direction" describes motion that is *below*
chance, and "0.47" is *40% above* it. Every alignment figure below is quoted as a multiple of that
baseline. Without it these numbers read backwards, and the first version of this analysis would
have concluded the opposite of the truth.

## 3. Stock vs Deep Sets: the hypothesis is half right

WP-D and WP-C proposed that Deep Sets scores well *because* its motion is probe-invisible.

| family | stock rel | stock alignX | stock shared | stock AUC | DS rel | DS alignX | DS shared | DS AUC |
|---|---|---|---|---|---|---|---|---|
| rect | 1.03 | 1.37 | 0.459 | 0.7197 | 3.81 | 0.94 | 0.700 | 0.7021 |
| wedge | 0.95 | 1.44 | 0.547 | 0.7537 | 2.91 | 0.76 | 0.728 | 0.8235 |
| strip | 1.01 | 1.40 | 0.506 | 0.7581 | 3.84 | 0.89 | 0.719 | 0.7554 |
| towers | 1.00 | 1.49 | 0.573 | 0.7493 | 3.38 | 0.91 | 0.750 | 0.7721 |
| cells | 0.80 | 1.38 | 0.454 | 0.8253 | 2.00 | 0.86 | 0.615 | 0.8422 |
| **mean** | **0.96** | **1.42** | **0.508** | **0.7612** | **3.19** | **0.87** | **0.702** | **0.7791** |

- **True:** Deep Sets' displacement is at-or-below chance alignment (0.87x) while stock's is above
  it (1.42x).
- **False:** Deep Sets does not move less. It moves **3.3x more** relative to its own spread, so
  its *absolute* probe-visible displacement is larger, not smaller.
- Its edge is real but modest (0.7791 vs 0.7612) and family-dependent — it is **worse** on rect.

**Cross-model comparison is confounded** and the measurements do not establish that invisibility
is why Deep Sets wins: each model has its own spread and its own separately fitted probe, and Deep
Sets starts from a better clean latent (AUC 0.8854 vs 0.8604). A material part of its advantage is
a better-separated clean representation, not motion geometry.

## 4. What actually predicts AUC

Pooled over 25 points, pearson / spearman:

| measure | stock | deepsets |
|---|---|---|
| raw \|dz\| / spread | -0.840 / -0.982 | -0.861 / -0.938 |
| Mahalanobis \|dz\| | -0.885 / -0.958 | -0.814 / -0.908 |
| **fraction** along probe grad | -0.401 / -0.141 | -0.028 / +0.356 |
| **absolute** probe-visible \|dz\| | **-0.910 / -0.912** | **-0.944 / -0.964** |
| shared-shift fraction | -0.310 / -0.400 | -0.687 / -0.912 |

The *fraction* alone is useless — it flips sign between models and ranges from -0.64 to +0.92
per-family. It cannot distinguish "moves little" from "moves a lot but invisibly"; only the
product can.

## 5. The displacement is largely a rigid translation

`shared` = `||mean(dz)|| / mean||dz||`. Stock 0.51, Deep Sets 0.70, reaching **0.99** at high
severity. The cloud's internal structure is preserved; it has walked off the manifold the probe
was fitted on.

**This is a range across severity cells, not one number.** It climbs steeply with severity — for
Deep Sets, 0.18 at s=0.2 to 0.99 at s=1.0. Comparing cell-means across models is fair only
because both are averaged over the identical severity grid.

**The ratio should not be the reported quantity, and no longer is.** Writing `dz_i = t + s_i` with
`mean(s_i) = 0`, the numerator is `||t||` and the denominator is `mean||t + s_i||`. A wider clean
cloud does not scale those together: `t` is how the degradation displaces the whole distribution,
`s_i` is how individual events scatter, and they respond to a wider cloud by different factors. So
`shared` can move because `t` moved, because the scatter moved, or because neither moved and the
mixture changed — and the ratio alone cannot say which (WP-C's point). Both components are now
reported in absolute units and `shared` is derived.

This matters concretely here: run 1's spread grew 40% over the second half of training
(3.027 -> 4.236) while its offset went flat, so a low `shared` from that model could mean the
consistency term reduced the rigid component *or* simply that the cloud got wider.

*Estimating `t` from the same sample biases `mean||s_i||` downward by `1 - 1/(2n)` — verified by
simulation against known scatter, matching to four decimals: 0.10% at n=512, 0.012% at n=4096. Not
the standard error of `t`, which would be ~4.4% at n=512 and in the other direction. So figures at
those two sample sizes are directly comparable.*

**The controlled comparison is run 1 vs run 2**, not run 1 vs stock. Same architecture, data,
generator and schedule, differing only in whether the cosine is centred — so `||t||` and
`mean||s_i||` for both is a direct read on what the consistency term does to each component.
Pending; run 2 is an ablation row and the number may arrive after the ruling. Even then, with
n=1 per arm it cannot separate the effect of centring from run-to-run variation, which we have no
repeats to estimate.

## 6. Open: is the failure extrapolation rather than misclassification?

Alignment with the *global linear discriminant* is **0.021-0.026** for stock against a 0.341
chance level, and consistently so across all 25 cells (the shared-shift direction's cosine with
`w` is 0.0003-0.0489 throughout). The degradation moves the latent along essentially one
characteristic direction regardless of which region dies, and that direction is near-orthogonal
to the linear discriminant.

That is a puzzle: motion orthogonal to the discriminant should barely trouble a linear probe, yet
AUC collapses to chance. It reconciles only through the probe's nonlinearity — `along_g`, the
per-event gradient of the grader's EvalMLP, is **1.42x above** chance while `along_w` is 14x
below. The motion is globally class-neutral but locally aligned with what the MLP reads.

**Hypothesis (not yet a finding):** the failure is extrapolation. Events are pushed along a
class-neutral axis out of the region where the MLP's boundary was fitted, into territory where it
was never constrained — they are not crossing a boundary toward the other class, they are leaving
the domain where the boundary means anything. Consistent with Mahalanobis (which measures
displacement in units of the clean covariance, i.e. how far outside the support) beating raw
displacement in the stock model.

**The extreme case is the `wedge` tail**, and it is the sharpest test available. At `wedge`
s=1.0 the grader's probe returns AUC **0.4473** — *below chance*, meaning the ranking inverts
rather than merely degrading. That is in direct tension with `along_w`: motion orthogonal to the
class axis cannot systematically swap classes. One of the two measurements has to give.

**Test, scheduled after the 22:50 ruling as writeup material.** Three quantities, from one run:
1. a logistic probe fitted on the same train split, scored on the same degraded latents;
2. nearest-clean-centroid accuracy on the degraded latents — pure geometry, no fitted probe;
3. the rigid shift's component along the clean class axis, in units of class separation.

Pre-registered outcomes:
- **Nearest-centroid >= 0.5 while EvalMLP is at 0.4473** -> the geometry did not invert; the
  inversion belongs to the nonlinear probe extrapolating outside its fitted support. Extrapolation
  confirmed, in its strongest form.
- **Nearest-centroid also below 0.5** -> the geometry inverted, and the `along_w` orthogonality
  result is wrong or misleading. That would need explaining before any conclusion about the
  encoder.
- **Linear probe stays >= 0.5 while EvalMLP inverts** -> inversion is a property of the probe.
  **Both invert** -> it is a property of the latents.

**The clean linear AUC must be checked first.** If the logistic probe is much weaker than EvalMLP
on clean latents, then "it degrades less" is a floor effect rather than robustness, and the
comparison says nothing. Result pending; it will be reported either way, including if it falsifies
the hypothesis.

See `A-masking-and-readouts.md` §4 for why the same `wedge` s=1.0 point should not be read as the
mask fix destroying signal: cosine-to-clean there is 0.0012 against a shuffled floor of -0.0134,
so the latent is degenerate and an AUC either side of chance is close to a coin flip.

## 7. Credit
Hypothesis in §3: WP-D and WP-C. Everything else measured here by WP-A. An earlier claim of mine
that the shared shift and WP-C's latent offset were "one quantity" was **wrong** and was retracted
in `reports/a-probe-drift.md`; see that file's CORRECTION section.
