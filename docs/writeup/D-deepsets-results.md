# Deep Sets: a permutation-symmetric encoder for dead-region robustness (WP-D)

All bench numbers from `bench_eval.py`, five scoring families, 20k eval events unless marked.
Every row names its source JSON in `~/hackathon-shared/runs/`.

## Why a set encoder

The metric rewards a *flat* AUC-vs-severity curve, not a high clean AUC. A dead eta-phi region
deletes a spatially correlated block of candidates, so the question is what happens to the latent
when elements are removed from the input set.

In the stock transformer every candidate contributes to the CLS token through attention, and a
zeroed candidate does not simply disappear: before the mask fix it entered attention as a constant
token, and even after the fix, removing candidates changes the attention normalisation for all the
survivors. In a Deep Sets encoder the latent is a masked **mean and max over surviving candidates**
of a per-particle embedding. Deleting a candidate removes its terms from the pool and rescales by
the new count. Nothing else moves. That is the whole mechanism.

The strong form of the property is testable exactly, and it is the acceptance test:
**a zeroed candidate is bit-equivalent to a candidate that was never there.**

## Results

| row | architecture | training | clean AUC | mean_area | source |
|---|---|---|---|---|---|
| stock anchor | transformer | stock | 0.8605 | 0.7734 | `anchor-stock-baseline_20260903_194212.json` |
| mask fix, frozen weights | transformer | none (stock ckpt) | 0.8605 | 0.7918 | `planner-maskfix-stockckpt_20260903_195138.json` |
| d-deepsets-clean | **Deep Sets** | no degradation at all | 0.8853 | 0.8036 | `d-deepsets-clean_20260903_202904.json` |
| d-deepsets-clean `--full` (70k) | **Deep Sets** | no degradation at all | 0.8887 | 0.8017 | `d-deepsets-clean-full_20260903_203928.json` |
| d-deepsets-aug | **Deep Sets** | WP-B generator | 0.8920 | **0.8183** | `d-deepsets-aug_20260903_205345.json` |

Per family, d-deepsets-aug vs the anchor: rect 0.7751 / 0.7283, wedge 0.8424 / 0.7982,
strip 0.8011 / 0.7485, towers 0.8141 / 0.7737, cells 0.8588 / 0.8185. Ahead in every family.

**The result worth stating first.** `d-deepsets-clean` never saw a dead region during training and
still beats the stock anchor by +0.030 mean_area, and beats the mask-fixed stock transformer by
+0.012. That increment is architecture alone, with no robustness training signal of any kind.
Augmentation then adds a further +0.015 on top. The two effects stack rather than substitute.

Clean AUC rises at every step (0.8605 -> 0.8853 -> 0.8920), so neither the architecture change nor
the augmentation buys robustness by giving up accuracy.

## Final measured table (the ruling's evidence)

`eval.py` seeds nothing, so the probe is refit differently on every run and every area carries that
noise. The comparison threshold between two rows is `3*sqrt(s_a^2/R_a + s_b^2/R_b)`; the official
metric additionally carries the organisers' unseeded Bernoulli, so its spreads are quoted as
observed ranges over n runs and never as variances.

| candidate | bench mean_area | official area (mean, range) | held-out mean_area | clean AUC |
|---|---|---|---|---|
| **d-pma0-aug-meanpt** (submission) | **0.8260 +- 0.0009** | **0.8788, 0.0004 (n=2)** | **0.8021 +- 0.0006** | 0.9147 |
| d-deepsets-aug-meanpt | 0.8186 +- 0.0007 | 0.8703, 0.0007 (n=3) | 0.7977 +- 0.0009 | 0.9066 |
| d-deepsets-aug | 0.8174 +- 0.0020 (R=10) | 0.8598, 0.0024 (n=2) | 0.7962 +- 0.0011 | 0.8921 |
| d-pma0-aug | 0.8162 +- 0.0038 | 0.8669, 0.0089 (n=3) | 0.7983 +- 0.0048 | 0.9215 |
| d-deepsets-clean (no augmentation) | 0.8036 (n=1) | — | — | 0.8853 |
| stock anchor (transformer) | 0.7734 | 0.8201 reference | — | 0.8605 |
| d-deepsets-twoview (ablation, negative) | 0.7419 +- 0.0156 | — | — | 0.7804 |

### The result that decided it: an interaction, not an additive gain

The three arms in the middle of that table are **statistically tied on the primary metric**. Every
pairwise gap is 0.0012 against thresholds of 0.0020 to 0.0052. Neither of the two ideas separated
on its own:

- PMA readout alone (`d-pma0-aug`): 0.8162 +- 0.0038 — tied with the baseline.
- MeanPt preprocessing alone (`d-deepsets-aug-meanpt`): 0.8186 +- 0.0007 — tied with the baseline.

Together they give **0.8260 +- 0.0009**, which separates from all three (gap 0.0074 against a
threshold of 0.0015) and is first on the official area and on held-out as well. This was not
predicted from the individual arms and should not be read as an additive stack; it is a genuine
interaction. A plausible reading — untested, and offered as such — is that both changes attack the
same failure from different ends: MeanPt removes the dependence of every surviving candidate's pt
feature on *how many* candidates were lost, and attention pooling with learned seeds re-weights
*which* survivors dominate the summary, so neither is sufficient while the other bottleneck stands.

### What each step was worth

Measured against the stock transformer anchor at 0.7734:

| step | mean_area | delta |
|---|---|---|
| stock anchor | 0.7734 | — |
| + mask fix (frozen weights, no retraining) | 0.7918 | +0.018 |
| Deep Sets, trained with NO degradation at all | 0.8036 | +0.030 over anchor |
| + WP-B's dead-region augmentation | 0.8174 | +0.014 |
| + MeanPt preprocessing and PMA readout | **0.8260** | +0.009 |

The architecture is the single largest contributor, and it is worth restating that
`d-deepsets-clean` never saw a dead region during training.

### Held-out generalisation

All four candidates score about 0.02 below their scoring-family `mean_area` on the held-out shapes
(ellipse, annulus, diagonal). My first reading was that this is entirely an artefact: ellipse and
diagonal drop 100% of candidates at s=1.0, so every arm sits at exactly 0.5 there by construction,
the same dead endpoint as `towers` on the scoring bench. WP-E's recomputation with those endpoints
excluded shows that accounts for **about a quarter** of the gap; the remaining three quarters is a
real generalisation gap to unseen shape families. The ruling uses WP-E's figure, and this section
follows it rather than my original claim.

The submission is first on held-out (0.8021 +- 0.0006) and separable there from
`d-deepsets-aug-meanpt` and `d-deepsets-aug`, though not from `d-pma0-aug`, whose sigma of 0.0048
is too wide to call. It also has the tightest sigma of any candidate on all three metrics.

### The negative result

`d-deepsets-twoview` — two-view consistency on the set encoder — scored 0.7419 +- 0.0156, below the
stock anchor, with clean AUC falling 0.892 -> 0.780. It is an ablation row, not a candidate. The
run was healthy by every training-side measure (per-event invariance real at cos 0.877 against a
shuffled control of 0.086, normalised drift flat at 0.27, validation head sound). The encoder had
satisfied the consistency term by inflating a common latent offset from 2.97 to 2324 while the
spread reached only 3.88 — an offset-to-spread ratio of 599, against 3.9 on the winning arm. Both
consistency terms are blind to a uniform translation by construction. Full mechanism in WP-C's
section; the trajectory data is from this arm.

## Caveats, stated rather than buried

- **All three rows are n=1 on the probe.** They were measured before `bench_eval.py` gained
  `--probe_repeats`. `eval.py` seeds nothing, so the probe is refit differently every run and every
  area carries that noise. `d-deepsets-aug` is being re-benched with repeats as a candidate;
  `d-deepsets-clean` stays n=1 by planner ruling, as a reference row rather than a candidate.
  Until the floor is a measured number, differences in the third decimal here mean nothing.
- The `--full` re-bench moved mean_area by -0.0019 against the 20k run, which is the only direct
  evidence available so far that these areas are stable at all.
- The rect AUC curve is not monotone in severity (0.885, 0.863, 0.830, 0.711, 0.572, 0.600 at
  s = 0 ... 1.0). The `--full` run moved rect the least of any family (-0.005), so the non-monotonicity
  is curve noise at high severity rather than an unstable area.

## Acceptance tests (`tests/test_set_encoders.py`, all pass)

- **Deletion equivalence**, the property the architecture is chosen for: with 50% of candidates
  zeroed, the latent equals the latent of the same batch with those candidates physically removed
  (shorter N) to `1.8e-07` for Deep Sets and `4.2e-07` for PMA(L=2).
- Permutation invariance to `6.3e-08`.
- All-dead event produces a finite latent (guarded; one slot kept alive).
- N=200 and N=400 both run, so nothing depends on the train/eval token-count difference.
