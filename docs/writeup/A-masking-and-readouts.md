# A: Encoder masking, readouts, and deletion equivalence

Every number here is sourced to a `runs/` JSON or a `reports/` file. Branch `wp-a`.

## 1. The mask fix: what it is and what it is worth

`src/embedding/dataloader.py` builds the attention padding mask from `pt == 0` **before**
degradation is applied. Candidates that degradation zeroes therefore entered attention as
identical all-zero tokens rather than being excluded. `TransformerEncoder.forward` now re-derives
the dead rows from the input it actually receives:

```python
dead = (x.abs().sum(dim=-1) == 0)   # rows zeroed by degradation or padding
mask = mask.clone(); mask[:, 1:] |= dead
```

Measured on the **stock reference checkpoint with no retraining**, so this isolates masking alone:

| architecture | clean AUC | mean_area | source |
|---|---|---|---|
| Transformer, CLS readout (anchor) | 0.8605 | 0.7734 | `runs/anchor-stock-baseline_*.json` |
| Transformer, CLS readout + mask fix | 0.8572 | **0.7899** | `runs/a-maskfix-stockckpt_20260903_200704.json` |

**+0.0165 mean_area for free**, clean AUC unchanged within the probe floor. Independently
reproduced by the planner at 0.7918 (`runs/planner-maskfix-stockckpt_*.json`); the 0.0019
difference is inside the probe-refit noise floor established in §4.

## 2. Deletion equivalence: the property the fix buys

A candidate that degradation zeroed must be worth exactly what a candidate that was never there
is worth. Test (`tests/test_deletion_invariance.py`): zero 50% of candidates in place, and
separately rebuild the same events with those candidates physically removed (400 → 235 tokens,
zero-padded), then compare latents. The dataloader mask is built from the **pre-degradation** `pt`
column, as in the real pipeline, so the encoder must re-derive the dead rows itself.

| readout | float64 max abs dz | float32 max abs dz |
|---|---|---|
| cls (stock checkpoint) | 1.24e-14 | 4.17e-06 |
| cls+mean | 1.11e-15 | 4.77e-07 |
| cls+mean+max | 9.16e-16 | 5.96e-07 |
| pma | 1.11e-15 | 4.77e-07 |
| cls + prenorm | 1.22e-15 | — |

Exact to float64 precision; the float32 residuals are summation-order noise. All-dead events
yield finite latents in every readout — this needs an explicit guard for `pma`, which attends only
over particle tokens, so an all-masked row would give `softmax(-inf)` = NaN without one.

**`dead_frac_token` deliberately fails this test** (max abs dz 0.32) and the test reports rather
than asserts it: it feeds the encoder the fraction of dead rows, 0.5 for a zeroed event and 0.0
for the same event with those candidates deleted. See §5 for why it was rejected.

## 3. Readout variants

`TransformerEncoder(readout=...)`, default `"cls"` so `eval.py` and the stock checkpoint are
unaffected. `cls+mean` and `cls+mean+max` concatenate the CLS output with a count-normalised
masked mean (and max) over surviving particle tokens; `pma` replaces the CLS readout with
Set-Transformer pooling by 4 learned seed queries cross-attending over the surviving tokens.

Benchmarking a non-default readout needed no change to the shared ruler: `bench_eval.py` builds
the encoder without a `readout` kwarg but does accept `--encoder_class`, so
`TransformerEncoderClsMean` / `ClsMeanMax` / `PMA` subclass with the readout defaulted. The same
classes are the drop-in default if a variant is shipped, since `eval.py` never passes `readout`
either.

`a-pma` is derived byte-for-byte from `train_config_b_aug_stock.yaml` with only the readout
changed, so the row reads "transformer + aug + PMA readout" against B's "transformer + aug + CLS".
`a-clsmean` was cut at convergence.

### Which comparison is valid, and which is not

**The §1 mask-fix number is the robust claim in this section precisely because it is not a
retrain.** It is one set of frozen weights evaluated two ways, so training budget, seed and data
order are held exactly. Nothing else here has that property.

`a-pma` is a 25-epoch retrain at batch 256. An earlier version of this section claimed the
retrained transformers carried a **protocol handicap** against the organisers' 60-epoch
checkpoint, generalising from `c-twoview-cos1`'s clean AUC of 0.8370 against the anchor's 0.8605.
**That was wrong, and E's clean control falsifies it.**

| row | protocol | mask fix | aug | clean AUC | mean_area |
|---|---|---|---|---|---|
| anchor (organisers' checkpoint) | 60 ep | no | no | 0.8605 | 0.7734 |
| mask fix, same frozen weights | 60 ep | eval only | no | 0.8572 | 0.7899 |
| **E clean control** | 25 ep / bs256 | yes | no | **0.9025** | **0.8089 +- 0.0004** |
| `c-twoview-cos1` (two-view) | 25 ep / bs256 | yes | yes | 0.8370 | 0.7712 |

The 25-epoch schedule is **not** a handicap: E's control at the same protocol reaches clean AUC
0.9025, well *above* the 60-epoch anchor's 0.8605, and mean_area 0.8089. So
`c-twoview-cos1`'s low clean AUC is a property of that run's two-view objective and augmentation,
not of the training budget. I had one confounded row and read a general conclusion off it; the
proper control says the opposite.

**The corrected reading, and it is a bigger result than the handicap would have been:** retraining
with masking *active* is worth **+0.019 mean_area over the frozen mask fix** (0.8089 vs 0.7899)
and **+0.035 over the anchor** (0.7734). Excluding dead candidates from attention matters more
when the encoder can adapt to it than when it is applied post hoc to weights that never saw it.
E's control also beats `d-deepsets-clean` (0.8036), so a masked transformer is competitive with
the set encoders once retrained.

`a-pma` is therefore read against **E's control (0.8089)** and **B's `b-aug-stock`**, both at the
same protocol. If it lands below 0.8089, the augmentation is costing the transformer regardless of
readout.

### The comparison that is actually informative

D's `d-pma0-aug` is a set encoder with the same attention-pooling readout and **no transformer
body** (`num_layers=0`): clean 0.9215, mean_area 0.8162 at R=5. `a-pma` is the same readout on top
of the full 4-layer transformer body. Between them they isolate what the transformer body
contributes once the readout is held fixed:

- `a-pma` close to `d-pma0-aug` -> the body contributes little; the readout was doing the work.
- `a-pma` well below it -> the body actively costs robustness, which is the stronger version of
  D's result and the one that would justify the set encoder as the submission on mechanism rather
  than on score alone.

### The readout question has an answer, on the set-encoder side

While `a-pma` was training, WP-D settled the readout question in the architecture that matters:
**attention pooling (PMA, k=4 seeds) beats masked mean+max pooling by 0.0074**, 0.8260 +- 0.0009
against 0.8186, both with the MeanPt preprocessor. `d-pma0-aug-meanpt` is the separable leader and
the probable submission. So the readout hypothesis this package set out to test — that a pooled
readout degrades more gracefully than a single CLS token — is supported, but the decisive evidence
comes from D's set encoders rather than from my transformer arms.

**On the MeanPt "interaction", stated carefully.** It is tempting to write that MeanPt is worth
+0.0012 on Deep Sets and +0.0098 on PMA, and read an architecture interaction off the difference.
Only one of those is a measured gain:

| pairing | gain | threshold | verdict |
|---|---|---|---|
| Deep Sets `rep10` (R=10) -> `meanpt` (R=5) | +0.0012 | 0.0021 | **not separable** (0.6x) |
| PMA `aug` (R=5) -> `meanpt` (R=5) | +0.0098 | 0.0052 | separable (1.9x) |

The Deep Sets figure is inside the floor and consistent with zero, so quoting the two side by side
implies a quantitative contrast the data does not support. What the data supports is weaker and
should be stated as such: **MeanPt's benefit is demonstrated on PMA and is not demonstrated on
Deep Sets.** Whether that is a genuine architecture interaction or merely that Deep Sets has not
been measured precisely enough is **open** — resolving a gain that size on that pair would need
roughly 3x more repeats than we have. Since the submission is PMA+MeanPt, this matters for the
causal story: if the two changes were independent you would expect similar gains on both bodies,
and tonight's runs cannot tell us. (Caught by E.)

### `a-pma` result and the ruling

`a-pma` = PMA readout on the full 4-layer transformer body, stock preproc, WP-B augmentation.
**mean_area 0.8275 +- 0.0006, clean AUC 0.9228 +- 0.0008** (R=5, `runs/a-pma_20260903_224712.json`).
Held-out families **0.8059 +- 0.0016** (`runs/a-pma-heldout_20260903_225347.json`).

| comparison | delta | threshold | verdict |
|---|---|---|---|
| vs `d-pma0-aug` — **same readout, no body, same preproc** | **+0.0113** | 0.0052 | **separable, 2.2x** |
| vs E clean control (transformer, CLS, no aug) | +0.0186 | 0.0010 | separable, 19.2x |
| vs `d-deepsets-aug-meanpt` | +0.0089 | 0.0012 | separable, 7.2x |
| vs certified winner `d-pma0-aug-meanpt`, scoring set, R=5 | +0.0015 | 0.00145 | tie at the edge, 1.0x |
| vs certified winner, scoring set, **R=20** | +0.0021 | 0.00123 | **separably above, 1.7x** |
| vs certified winner, **held-out** | +0.0038 | 0.0023 | separably above, 1.65x |
| vs certified winner, **clean AUC** | +0.0093 | 0.00096 | separably above |

**The bench and the official metric disagree about these two models.** On the bench a-pma is
separably ahead at R=20 (0.8276 +- 0.0017 against 0.8255 +- 0.0007), on held-out, and on clean AUC.
On the organisers' own `eval.py` it is behind on all three draws: 0.8764, 0.8792, 0.8793, none
clearing the 0.8820 bar. The pre-registered rule made the official number decisive, which is the
right structure — the official metric is the organisers' evaluation and the bench is our proxy for
it — but the record should show a disagreement, not a bench that agreed with the outcome.

**Methodological caution, and it applies to numbers throughout this writeup.** a-pma's per-refit
sigma was **0.0006 at R=5 and 0.0017 at R=20 — 2.8x larger with more repeats**. Five refits
underestimated its variability nearly threefold. The winner's barely moved (0.0009 -> 0.0007), so
the unreliability was not symmetric. Every separability verdict computed at R=5 inherits this,
including the original tie, which was not merely close but computed with a sigma too small on one
side. Any R=5 comparison sitting near its threshold should be treated as unresolved rather than
decided.

**The substantive result is the first row.** `d-pma0-aug` is the identical PMA readout with
`num_layers=0` and the identical stock preprocessor, so the difference isolates the transformer
body with everything else held fixed: **the body is worth +0.0113 at 2.2x the threshold.** That
runs against the reading this project converged on for most of the evening — that set encoders are
intrinsically more robust and the attention body is the liability. On this pair, with the readout
controlled, the body helps.

**The ruling: `a-pma` was not promoted, and the certified set encoder stands.** Under the
pre-registered switch criteria it passed held-out (criterion ii, separably above) but failed the
official-area margin: run 1 scored 0.8793 against an amended bar of 0.8820. The record is:
**indistinguishable on every accuracy measurement, tie broken by a pre-registered rule and a 27x
parameter difference** — 2,445,478 parameters for `a-pma` against 89,606 for the winner, with
O(N^2) attention against O(N) pooling on 400-candidate events. That is the right call: two models
that cannot be told apart on accuracy should not be separated by 0.0015 of mean_area, and the
cheaper one wins on every other axis.

The honest summary of this package's contribution to the winner: **the readout hypothesis is
what the winning architecture is built on**, and it was tested here on the transformer and
decisively by WP-D on the set encoders. `a-pma` is the strongest transformer of the night and
still not worth shipping.

**Use the right threshold, computed per pair.** The 3 sigma = 0.0025 in §5 is the *transformer's*
single-row floor and is the wrong number for a two-sample comparison. The threshold is
`3 * sqrt(s_a^2/R_a + s_b^2/R_b)` with each side's own sigma (transformer 0.0008, Deep Sets 0.0038):

| comparison | threshold |
|---|---|
| transformer vs transformer, R=5 both (readout arms) | **0.0015** |
| `a-pma` vs `d-pma0-aug`, R=5 both (cross-architecture) | **0.0052** |
| Deep Sets vs Deep Sets, R=5 both | 0.0072 |

A cross-architecture gap must be **3.4x** larger to be separable than a readout-vs-readout gap.
An earlier version of this section put the first row at 0.0024, which was three sigma of a single
row with no two-sample combination and no division by R — conservative, but wrong, and it
understated how much resolving power R=5 buys.

**And the Deep Sets claim needs a repeated row.** I previously wrote that Deep Sets beats the
retrained masked transformer citing `d-deepsets-aug` 0.8183 against E's control 0.8089. That row
is n=1, so it carries the 0.0038 architecture floor undivided:

| pairing | delta | threshold | verdict |
|---|---|---|---|
| E control vs `d-deepsets-aug` (n=1) | 0.0094 | 0.0114 | **not separable** (0.8x) |
| E control vs `d-deepsets-aug-rep10` (R=10) | 0.0085 | 0.0020 | separable (4.3x) |
| E control vs `d-deepsets-aug-meanpt` (R=5) | 0.0097 | 0.0011 | separable (9.0x) |

The conclusion holds at 4-9x, but **not via the row I originally cited** — a single unrepeated fit
carries a sigma that swamps a 0.0094 gap. Both corrections are E's.

## 4. The wedge tail: what the mask fix does at s=1.0, and why it is not a regression

From F's curves (`runs/plots/curves_latest.png`): at `wedge` s=1.0 the anchor holds AUC 0.68
while both mask-fix runs **on the same weights** fall to 0.48, below chance, having gained about
+0.04 at s=0.8. Net wedge area for the mask fix is slightly negative. The set encoders hold
0.72-0.76 there.

Below-chance AUC means the probe's ranking **inverts**, not merely degrades. That is a stronger
statement than "the mask fix lost signal", and it should not be read as one, for two reasons.

**First, the latent at that point is degenerate.** My centred-drift measurement on the stock
checkpoint puts `wedge` s=1.0 at cosine-to-clean **0.0012** against a shuffled-pair floor of
**-0.0134** — per-event identity is entirely gone, statistically indistinguishable from pairing
each degraded event with an unrelated one. An AUC on either side of chance there is close to a
coin flip on a degenerate representation. The anchor's 0.68 should not be treated as real signal
that masking destroyed without probe repeats on that specific point; the measured floor is
3 sigma = 0.0025 on *mean_area*, and single points at the tail are far noisier than that.

**Second, "the latents cross the boundary" is in tension with a measurement of mine.** The
degradation displacement is near-orthogonal to the linear discriminant: `along_w` = 0.021-0.026
against a 0.34 random-direction baseline, consistently across all 25 family/severity cells, and
the shared-shift direction's cosine with the discriminant is 0.0003-0.0489 throughout. Motion
orthogonal to the class axis cannot systematically swap classes. Both cannot be true as stated.

**Both outcomes are pre-registered** (the test is `tests/probe_drift_decomposition.py`, scheduled
after the 22:50 ruling as writeup material rather than a candidate decision). It reports
nearest-clean-centroid accuracy on the degraded latents — pure geometry, no fitted probe — and the
component of the rigid shift along the clean class axis:

- **If nearest-centroid accuracy stays at or above 0.5 while EvalMLP sits at 0.4473:** the
  geometry did *not* invert. Events did not change side; the MLP's extrapolated output is
  anti-correlated with the truth in a region where it was never fitted. This is the extrapolation
  picture at its extreme, and the inversion is a property of the grader's nonlinear probe rather
  than of the encoder.
- **If nearest-centroid accuracy also drops below 0.5:** the geometry really did invert, and my
  `along_w` orthogonality result is wrong or misleading. In that case the orthogonality needs
  explaining before anything is concluded about the mask fix.

A second, independent handle on the same question comes free from the same run: a logistic probe
fitted on the same train split, scored on the same degraded latents. If the **linear** probe stays
at or above 0.5 on `wedge` s=1.0 while EvalMLP is at 0.4473, inversion belongs to the nonlinear
probe. If both invert, it belongs to the latents.

Whichever way it falls, the practical conclusion for the mask fix is unchanged: it is worth
**+0.0165 mean_area** across the five families on frozen weights (§1), which is 6.6x the measured
floor, and the wedge tail is one degenerate point inside that average.

## 5. Probe-refit noise floor

The grader retrains its probe on every evaluation, so repeated measurements of the *same*
checkpoint on the *same* events differ. Four independent refits, stock checkpoint:

```
0.8599   0.8604   0.8611   0.8614      spread ~0.0015
```

**Differences in clean AUC below ~0.0015 are not real.** Cross-check against E's
`bench/probe_variance.py`, which measures this directly.

## 6. Considered and rejected

- **`dead_frac_token`** (project the dead-row fraction into the CLS input so the encoder can
  compensate). Rejected as probe-incompatible: the probe is fitted only on clean latents, where
  the dead fraction is ~0, so an encoder that shifts its latent by `f(dead_frac)` at eval time
  moves into a region the probe never saw. It is only coherent paired with a consistency loss that
  forces the shifted latent back onto the clean manifold, which makes it a WP-C-dependent arm
  rather than a standalone one. Kept in the code, defaulted off.
- **Placeholder rectangle augmentation** (used by A/C/D before WP-B landed). Dropped: it removed
  only ~3% of candidates per event on average, too weak to train against. Superseded by WP-B's
  generator.
- **`prenorm`**. Implemented and deletion-equivalent (1.22e-15), but untested at the protocol;
  defaulted off. Not rejected on evidence, only unmeasured.
