# The full training file, and the input-distribution defects it exposed

WP-E. Full detail and method in `reports/e-fulldata.md`; the subsample is
`~/hackathon-shared/data/train_400_balanced_160k.pt` with provenance in its sidecar JSON.

## What the 12 GB file is

Inspected with `torch.load(..., mmap=True)`, never loaded whole.

| | full | small (what everyone trained on) |
| --- | --- | --- |
| shape | `[940048, 400, 8]` | `[80000, 200, 8]` |
| candidates/event | **400** | 200 |
| padding | none -- every slot filled | none |
| class balance | 4.6:1 (QCD 450,979 ... WJets 97,423) | exactly 20,000 each |

The small file is **not** a prefix: it is a class-balanced 20k-per-class subsample of the full
file, truncated to the **top 200 candidates by pt**. All 80,000 leading-candidate `(pt, eta, phi)`
fingerprints occur in the full file, and for 300 matched events all 200 rows are bit-identical to
the full file's first 200 rows. The discarded candidates are the soft tail, and the cut is
arbitrary with respect to physics -- in the worked example candidate 200 has pt 2.707 and
candidate 201 has 2.703.

Cost, since Deep Sets is O(n) in candidates and attention is O(n^2):

| | events | cand/event | tokens | vs small |
| --- | ---: | ---: | ---: | ---: |
| small | 80,000 | 200 | 16.0M | 1.0x |
| full | 940,048 | 400 | 376.0M | 23.5x |

Deep Sets on the full file projects to 4-8 min/epoch (2-3.5 h for 25). The transformer is 47x
(11.75x events x 4x attention): 50-80 min/epoch single-view, 3-4 h/epoch two-view. **Full-file
transformer training was ruled out on this basis.**

## Two input-distribution defects, one root cause

`PFPreProcessor` normalises pt per event by the **sum over surviving candidates**,
`pt -> log(pt / sum_pt)`. That is the encoder's primary continuous input, and the choice of a
*sum* produces two independent problems.

### Defect 1: severity-dependent drift

Degradation removes candidates, so `sum_pt` falls, so every *surviving* candidate's feature
shifts upward -- monotonically with severity. Over 6000 real eval events, in units of the
feature's own clean std (0.425):

```
family    sev  cands/ev   shift/std
rect      0.2     323.7        0.51
rect      0.4     259.2        1.06
rect      0.6     187.0        1.86
rect      0.8     101.8        3.34
towers    0.8      79.2        3.82
```

At severity 0.8 the encoder's main input is displaced by **3.3-3.8 sigma**, in the same direction
for every event. This is a corruption channel entirely separate from attention masking: a
perfectly mask-aware encoder still receives systematically wrong inputs as severity rises.
Augmentation and two-view consistency teach a model to tolerate this; neither removes the cause.

**Fix, measured.** Normalising by *mean* pt instead of *sum* pt (shift under `rect`, in units of
each scheme's own clean std):

```
normalisation        s=0.2   s=0.4   s=0.6   s=0.8
sum_pt (current)      0.51    1.06    1.86    3.34
mean_pt               0.02    0.04    0.08    0.14     <- 24x better at s=0.8
median_pt            -0.01   -0.02   -0.05   -0.08
raw log pt           -0.01   -0.02   -0.04   -0.05
```

`log(pt/mean_pt) = log(pt/sum_pt) + log(n_valid)`: the current feature plus one term. It keeps the
per-event energy-scale invariance the sum normalisation was chosen for. Implemented by WP-B as
`PFPreProcessorMeanPt`.

### Defect 2: a geometry offset at zero severity

The eval set has 400 candidates per event; the small train file has 200. Because the normaliser
is a sum over candidates, that truncation alone offsets the feature between training and scoring,
**before any dead region is applied**:

```
TRAIN small (200 cands)   mean log(pt/sum_pt)  -5.390   std 0.368
TRAIN full  (400 cands)   mean log(pt/sum_pt)  -6.097   std 0.393
EVAL        (400 cands)   mean log(pt/sum_pt)  -6.120   std 0.425
```

A **1.9 sigma** train/eval offset, with `PFPreProcessor`'s BatchNorm running statistics learned at
-5.39 and applied to inputs centred at -6.12. That the full file (400) matches eval almost exactly
while full-truncated-to-200 matches the small file confirms the shift is purely a function of
candidate count.

#### Decomposing that offset, and which normalisation actually fixes it

The +0.730 total splits cleanly into a **geometry** part (same events, truncated to their top 200)
and a **physics** part (train processes vs the eval sample, at matched geometry). Measured on eval
events, so the geometry column contains no sample difference at all:

```
scheme    eval400   evalTrunc200    geometry   physics    total
stock     -6.1201       -5.4269      +0.6932   +0.0368   +0.7300
MeanPt    -0.1287       -0.1286      +0.0001   +0.0368   +0.0369
MaxPt     -2.3011       -2.0174      +0.2837   (unchanged across schemes)
AbsPt     +1.2492       +1.5329      +0.2837
```

The physics term is identical under every scheme, as it must be -- it is a property of the
samples, not of the normalisation. So there is no coincidental cancellation: MeanPt removes the
geometry term outright and what remains is the genuine sample difference.

**Geometry offset removed: MeanPt 100%, MaxPt and AbsPt 59%, stock 0%.**

The mechanism, and the reason the ranking is counter-intuitive. Truncating to the top 200 does two
things: it raises the denominator, and it raises the mean of `log(pt)` itself by +0.2837, because
the surviving candidates are the hard ones. A scheme cancels the offset only if its denominator
moves with the composition. `AbsPt` has no denominator; `MaxPt`'s denominator is the event maximum,
which lies inside the top 200 and is therefore *unchanged* by truncation (verified). Both are
exactly invariant per candidate -- which is what makes them look immune -- and both leave the whole
composition term standing. `MeanPt` is the one whose denominator tracks the composition:

```
sum_pt  : (+0.2837) + log(S400/S200)  = +0.2837 + 0.4095 = +0.6932
mean_pt : (+0.2837) + log(m400/m200)  = +0.2837 - 0.2836 = +0.0001
```

Equivalently, under mean normalisation the per-event feature mean is
`mean[log pt] - log(mean[pt]) = log(GM/AM)` of that event's pt spectrum -- a dimensionless *shape*
statistic, scale-free by construction and nearly invariant under top-k truncation of a heavy-tailed
distribution.

*Cross-checked:* WP-B independently reproduced this decomposition on a separate 4000-event slice
and agreed to four decimals (geometry +0.6934 stock / +0.0003 MeanPt, physics +0.0364 under both),
and contributed the AbsPt/MaxPt result above. Two people measured it separately, from opposite
prior conclusions, before it was written down.

**Exact per-candidate invariance is the wrong target.** What reaches the encoder is the
distribution of the feature over the candidates actually present, since that is what
`PFPreProcessor`'s BatchNorm computes its statistics from. A per-candidate measure (hold the
candidate set fixed, recompute only the denominator) gives 0.0 for AbsPt/MaxPt and 0.2836 for
MeanPt -- the exact reverse of the ranking that matters. This distinction cost a round of
mistaken correction between WP-B and WP-E before it was pinned down; it is recorded in B's
section as a correction history.

**Fix:** train on 400-candidate events. `train_400_balanced_160k.pt` is a class-balanced
40k-per-class subsample of the full file at the eval geometry (verified: loads through
`load_data`, 40k per class, all finite, no zero-pt slots, `out[i] == full[source_indices[i]]`):

```
small train (200)   mean log(pt/sum_pt)  -5.390
400-cand subsample  mean log(pt/sum_pt)  -6.089
eval        (400)   mean log(pt/sum_pt)  -6.120
```

Residual 0.08 sigma, against 1.9. At 64.0M tokens it is 4x the small file, versus 47x for the full
file under a transformer.

### They are complementary, not redundant

`MeanPt` removes the severity-dependent drift but leaves the encoder trained at a token count it
never sees at eval. The 400-candidate file removes the geometry offset and exercises attention and
pooling at the eval token count, but does nothing about drift under degradation. The ablation
table carries both arms separately.

## The honest framing for the writeup

Every model trained before these two fixes was compensating for input-distribution defects, not
only for the masking problem the work packages were scoped around. Gains previously attributed to
augmentation and two-view consistency may partly re-attribute to these fixes once both arms land,
which is why both appear as separate rows rather than being folded into a single "improvements"
number.
