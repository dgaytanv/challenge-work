# WP-E: the full training file, and a preprocessing bug it exposed

`~/hack-data/C9_robust_tagging/train/robust_tagging_train_data.pt`, inspected with
`torch.load(..., mmap=True)` (2.1 s to open; never loaded whole). CPU only, no GPU touched.

**Read the last section first if you read only one.** Inspecting this file surfaced a
severity-dependent bug in `preprocs.py` that is costing us area on every model we have trained,
and the fix is one line.

## 1. What the file is

| | full | small |
| --- | --- | --- |
| shape | `[940048, 400, 8]` | `[80000, 200, 8]` |
| dtype | float32 | float32 |
| size | 12.03 GB | 0.51 GB |
| candidates/event | **400** | 200 |
| padding | **none** -- every slot filled, in all 12000 sampled events | none |
| label column | col 7, integral, constant across all 400 slots | same |
| label values | 0, 1, 2, 3 | 0, 1, 2, 3 |

Per-class counts:

| class | full | small |
| --- | ---: | ---: |
| 0 QCD | 450,979 (48.0%) | 20,000 |
| 1 DY | 98,878 (10.5%) | 20,000 |
| 2 TT | 292,768 (31.1%) | 20,000 |
| 3 WJets | 97,423 (10.4%) | 20,000 |

**The full file is heavily imbalanced (4.6:1 between QCD and WJets); the small file is exactly
balanced.** Training on the full file therefore needs `class_weights: inv_freq` in the data
config or a balanced sampler, otherwise the encoder skews to QCD and TT. `compute_class_weights`
already supports `inv_freq`; nothing else has to change.

## 2. Relationship between the two files

The small file is **not** a prefix of the full file (`small[:2000] != full[:2000,:200,:]`), but
every one of its events is in there: all 80,000 leading-candidate `(pt, eta, phi)` fingerprints
occur in the full file, and for 300 matched events all 200 rows are bit-identical to the full
file's first 200 rows of the matching event.

So: **the small file is a class-balanced 20k-per-class subsample of the full file, truncated to
the top 200 candidates by pt.** Both files are pt-sorted per event. The discarded candidates are
the soft tail -- in the worked example, candidate 200 has pt 2.707 and candidate 201 has 2.703,
so the cut is arbitrary with respect to physics.

Feature ranges agree except where truncation explains the difference (valid candidates only):

```
        pt min   pt mean   eta range      phi range      is_pf   pdgId
full     1.568     3.813   [-5.00, 5.00]  [-3.14, 3.14]     1     [-211, 211]
small    2.080     4.832   [-5.00, 5.00]  [-3.14, 3.14]     1     [-211, 211]
```

The full file's lower pt floor and mean are exactly what keeping 400 rather than 200 candidates
per event produces. `dxy`/`dxysig` ranges are comparable (`dxysig` reaches +-65k in both; worth
knowing it is that heavy-tailed, though `PFPreProcessor` passes it through unscaled).

## 3. Epoch-time estimates

Deep Sets is O(n) in candidates, so cost scales with total candidate-tokens; the transformer's
attention is O(n^2), so it scales with events x candidates^2.

| dataset | events | cand/event | tokens | vs small |
| --- | ---: | ---: | ---: | ---: |
| small | 80,000 | 200 | 16.0M | 1.0x |
| full | 940,048 | 400 | 376.0M | 23.5x |

**DeepSetsEncoder**, from D's measured 20 s/epoch on the small file:

- compute-bound upper estimate: 23.5x -> **~7.8 min/epoch**, 25 epochs ~ **3.3 h**
- step-bound lower estimate (if much of the 20 s is fixed per-step overhead; step count scales
  11.75x, not 23.5x): ~3.9 min/epoch, 25 epochs ~ 1.6 h
- so plan for **2-3.5 h** for a full-file Deep Sets run.

**Transformer**, for contrast -- 11.75x more events x 4x attention cost = **47x**:

- stock single-view (my e-stock-maskfix, ~65-100 s/epoch on small) -> **~50-80 min per epoch**
- C's two-view (~4-5 min/epoch on small) -> **~3-4 h per epoch**

**A full-file transformer run is not feasible today.** Deep Sets is.

Resource notes: `train.py` calls `load_data` without mmap and `clean_data` does an in-place
`nan_to_num_` on the whole tensor, then `make_train_val_split` copies via `index_select`; peak
RAM ~22-25 GB. The box has 464 GB free, so that is fine. On the GPU, 400 candidates at batch 256
gives a `[256, 8, 401, 401]` attention tensor (~1.3 GB per layer) -- the transformer would need
batch 128 or lower, Deep Sets is unaffected.

## 4. The finding that matters: a severity-dependent preprocessing bug

`PFPreProcessor` normalises pt **per event by the sum over surviving candidates**:
`pt -> log(pt / sum_pt)`. This is the encoder's primary continuous input.

Degradation removes candidates, so `sum_pt` falls, so **every surviving candidate's pt feature
shifts upward by an amount that grows monotonically with severity**. Measured on 6000 real eval
events, in units of the feature's own clean standard deviation (0.425):

```
family    sev  cands/ev  mean feat   shift  shift/std
clean     0.0     400.0     -6.120   0.000       0.00
rect      0.2     323.7     -5.902   0.218       0.51
rect      0.4     259.2     -5.671   0.449       1.06
rect      0.6     187.0     -5.328   0.792       1.86
rect      0.8     101.8     -4.701   1.419       3.34
cells     0.8     159.7     -5.197   0.923       2.17
towers    0.8      79.2     -4.498   1.622       3.82
```

At severity 0.8 the encoder's main input is displaced by **3.3-3.8 standard deviations**, in the
same direction for every event. This is a corruption channel entirely separate from attention
masking: even a perfectly mask-aware encoder sees systematically wrong inputs as severity rises.
It is very likely a large part of the AUC decay we are all trying to fix, and no amount of
augmentation or two-view consistency removes the cause -- they only teach the model to tolerate it.

The same mechanism explains a **train/eval mismatch** that exists today with no degradation at
all. Training events have 200 candidates, eval events have 400:

```
TRAIN small (200 cands)      sum_pt  967.1   mean log(pt/sum_pt) -5.390  std 0.368
TRAIN full  (400 cands)      sum_pt 1523.0   mean log(pt/sum_pt) -6.097  std 0.393
EVAL        (400 cands)      sum_pt 1616.9   mean log(pt/sum_pt) -6.120  std 0.425
```

A **1.9 sigma** offset between what the encoder is trained on and what it is scored on, before
any dead region is applied. `PFPreProcessor`'s BatchNorm running statistics are learned at -5.39
and applied to inputs centred at -6.12. Note the full file (400 candidates) matches eval almost
exactly, and full-truncated-to-200 matches the small file -- confirming the shift is purely a
function of candidate count.

### The fix, measured

Normalise by **mean** pt instead of **sum** pt. Shift under `rect` degradation, in units of each
scheme's own clean std:

```
normalisation        s=0.2   s=0.4   s=0.6   s=0.8
sum_pt (current)      0.51    1.06    1.86    3.34
mean_pt               0.02    0.04    0.08    0.14     <-- 24x better at s=0.8
median_pt            -0.01   -0.02   -0.05   -0.08
raw log pt           -0.01   -0.02   -0.04   -0.05
```

`mean_pt` is algebraically the current feature plus `log(n_valid)`:

```
log(pt / mean_pt) = log(pt / (sum_pt / n)) = log(pt / sum_pt) + log(n_valid)
```

so it keeps the per-event energy-scale invariance the sum normalisation was chosen for, costs one
extra term, and **fixes the 200-vs-400 train/eval mismatch for free**, since that offset is also
just `log(n)`. `median_pt` is marginally more stable still and is robust to the pt-ordering, at the
cost of a per-event median.

**I have not implemented this.** `preprocs.py` is outside my package -- and, as far as I can tell
from the assignments, outside everyone's. It needs an owner and a controlled A/B, because it
changes the meaning of `norm_constants` and every existing checkpoint's BatchNorm statistics, so
it cannot be swapped under a trained model; it requires a retrain to evaluate. My
`e-stock-maskfix` control run is the natural A/B partner: same config, one changed line.

## 5. Recommendation

1. **Assign the `preprocs.py` pt-normalisation fix and A/B it against `e-stock-maskfix`.** This is
   the highest-value item I have measured today: it removes a 3.3-3.8 sigma severity-dependent
   input corruption and a 1.9 sigma train/eval offset, for one line and one retrain.
2. **Do not move the transformer to the full file.** At 47x it is 50-80 min per epoch single-view,
   3-4 h per epoch two-view.
3. **A full-file Deep Sets run is affordable** (2-3.5 h) if D's architecture is competitive, but
   use `class_weights: inv_freq` -- the full file is 4.6:1 imbalanced where the small file is exactly
   balanced.
4. **The real prize in the full file is the 400-candidate geometry, not the extra events**, since it
   matches eval exactly. A class-balanced subsample of ~160-240k events at 400 candidates would buy
   that for roughly 2-3x the current transformer cost, rather than 47x. Worth more than raw volume.
   Note that fix (1) removes the *pt-feature* part of the 200-vs-400 mismatch on its own; a
   400-candidate subsample additionally exercises attention and pooling at the eval token count.
