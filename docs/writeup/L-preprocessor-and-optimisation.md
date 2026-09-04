# WP-L: preprocessor and optimisation (campaign 2)

Owner: WP-L. Files: `src/embedding/preprocs.py`, `train.py` (selection and weight averaging), configs.
Every number here is measured; the source of each is named. Results tables fill in as arms bench.

---

## 1. The finding that applies to every campaign-1 checkpoint

**Checkpoint selection in campaign 1 was ~11x noise-dominated, and the noise was the training
augmentation's own randomness.**

`train.py` passes the training `degradation` object to `validate_epoch`, and `Degradation.forward`
dispatches on `severity` (which is `None` in train mode) rather than on `.eval()`. So validation has always
run *under the training generator* — but with a **fresh random corruption drawn every epoch**. Successive
epochs' validation losses are therefore computed on different corruptions, and the best-val-loss rule
compares them as though they were not.

**Measured on the champion's own frozen weights** (`rt_d_pma0_aug_meanpt_encoder_20260903_221125.pth`,
2000 validation events, 8 corruption draws, weights held fixed so only the draw varies):

```
val loss   mean 0.907366   std 0.005896   spread 0.019562
val AUC    mean 0.9157     std 0.0021
```

**Against the champion's own training log**, its last six epochs:

| epoch | 20 | 21 | 22 | 23 | 24 | 25 |
|---|---|---|---|---|---|---|
| val loss | 1.041768 | 1.051330 | 1.042153 | 1.041679 | 1.043764 | **1.041243** ← selected |

The best three epochs span **0.000525**. The corruption draw alone moves the same quantity by **0.005896**,
about **11x further**. Among near-tied epochs the rule therefore selects the epoch that happened to draw the
easiest corruption, not the best model. This is a property of every checkpoint campaign 1 produced, not of
any one arm.

### Confirmed by the fix

`val_degradation_seed` (WP-L L4) seeds the global RNG immediately before the validation pass and restores
the previous state immediately after, so the corruption is identical every epoch and the *training* RNG
stream is untouched. Effect on the shape of the selection criterion, measured over the last 10 epochs of
25-epoch runs on the small file:

| run | mean \|Δ val loss\| | epochs that set a new best |
|---|---|---|
| **L4, seeded validation corruption** | **0.002161** | **25 / 25** |
| L1 arm, champion validation behaviour | 0.007445 | 17 / 25 |
| champion, campaign 1 | 0.006682 | 16 / 25 |

Without the seed, epoch-to-epoch movement (~0.007) sits at the independently measured corruption noise
(0.0059) — the criterion is mostly reading its own draw. With the seed it falls to 0.0022 and becomes
monotone: every epoch improves on the last, which is what a criterion tracking *learning* looks like.

Caveat: the seeded row is one run; the unseeded behaviour is two independent runs that agree. The residual
0.0022 is genuine epoch-to-epoch change in the weights, not noise.

### The larger consequence: the same lottery decides the training BUDGET

Credit to WP-H for this. Early stopping fires on the *same* criterion, so a bad run of draws does not only
change which epoch is kept — it ends the run. Of WP-H's three reference runs, **s22 early-stopped at epoch 23
after its epoch-17 best**, so that seed received a different training budget from the other two, decided by
the same draw shown above to move val loss by 0.0059 against a 0.000525 spread.

Proximity to the patience-5 threshold, measured as the longest run of consecutive non-improving epochs:

| | longest non-improving streak | margin to early stop |
|---|---|---|
| seeded validation (3 runs) | 0, 1, 1 | 5, 4, 4 |
| unseeded (4 runs) | 3, 2, 1, 2 | 2, 3, 4, 3 |
| WP-H reference s22 | **5** | **0 — stopped at epoch 23** |

Unseeded runs sit two or three bad draws from losing budget; one did. A seeded run cannot early-stop while
its criterion is monotone, because every epoch resets the counter.

### A matched-seed demonstration

Two runs early-stopped tonight and **both are seed 22**, across two different configurations — and the same
seed under *seeded* validation ran to completion:

| configuration | seed | validation corruption | outcome |
|---|---|---|---|
| WP-H champion reference | 22 | unseeded (champion) | **early stopped, epoch 23** |
| WP-L `l1-sincos` | 22 | unseeded (champion) | **early stopped, epoch 23** |
| WP-L `l4-valseed` | 22 | **seeded** | ran the full 25 |

Every other run of either kind ran 25/25: seeded 3 of 3, unseeded 4 of 6.

This is close to a controlled pair. Seed 22's draw sequence produces a bad streak reliably enough to
truncate two *different* configurations at the same epoch, and seeding the validation corruption removes it
at that same seed. It also says something about the nature of "seed noise": part of it is a **deterministic
property of the seed's corruption sequence**, reproducible across configs, rather than irreducible training
randomness. Comparing two arms at matched seeds therefore cancels part of it; comparing arms measured at
different seed counts does not.

This is **mechanistic support, not a second independent result**: the streak metric overlaps between the
groups (the `l3-ema` run also reached 1), unlike the smoothness metric, which separates completely.

**The two effects are not independent** (WP-H). Because early stopping and checkpoint selection read the same
criterion, one bad streak both truncates the budget *and* leaves the retained checkpoint older relative to the
run. s22 is the case in point: it stopped at 23 and kept epoch 17 — budget lost at the end, checkpoint kept
from well before it. So "best-vs-last gap" and "budget loss" are overlapping consequences of one draw, not two
separate costs, and must not be added together.

WP-H's s22 was also the highest-scoring of the three seeds (0.829586 against 0.824978 and 0.823891) despite
the shortest budget. That is n=1 and the direction is the opposite of the naive prediction, which is a reason
to distrust it rather than to build on it. It is recorded because it shows the budget variation is not
obviously harmless, and because it sits *inside* the ±0.0030 seed floor the campaign is calibrated on — part
of what is called seed noise is early-stopping-decision noise wearing the same clothes.

**Consequence for campaign 2's measurement standard.** Seed-to-seed spread in any 3-seed comparison carries
this selection noise *on top of* training noise unless the arm sets `val_degradation_seed`. Arms that fix it
and arms that do not are therefore not measured with the same precision, which matters when the separability
test is `Δ > 3·sqrt(s_a²/3 + s_b²/3)` on seed standard deviations.

---

## 2. Arms

Each is the champion with **one** change, three seeds (11/22/33), screened on the small file.

| arm | change | `num_features` |
|---|---|---|
| `l1-sincos` | φ → (sin φ, cos φ) | **15** |
| `l2-dxysig` | dxysig → tanh(dxysig / 20) before BatchNorm | 14 |
| `l3-ema` | EMA of learnable weights, 0.755 per epoch (equivalent to 0.999 per step) | 14 |
| `l4-valseed` | fixed-seed validation corruption | 14 |
| `l6-classw` | per-class inverse-frequency weights renormalised to mean 1.0 — **Phase 2 only** | 14 |

### Why L1
Raw φ is discontinuous at ±π: two candidates a hair apart across the wrap sit at opposite ends of the
feature, and the encoder must learn that the coordinate is circular. (sin, cos) removes the discontinuity
and is what the colleague group uses.

### Why L2
WP-G's input audit makes dxysig the worst-conditioned of the fourteen encoder inputs: after BatchNorm the
encoder sees |max| 65.3 against p99.9 of 13.74, a ratio of **4.8**, costing three integer bits for 0.1% of
candidates. `dxy` is deliberately **not** touched — its ratio is 1.00, so its range is genuinely required.

### Why L6 is Phase 2 only, and what it originally would have measured
L6 was first specified as `class_weights: inv_freq` screened on the small file. **The small file is exactly
balanced** — counts [20000, 20000, 20000, 20000], ratio 1.00, built with `nevents_per_class: 20000`. The
4.63x imbalance is a property of the **full** file alone ([450979, 98878, 292768, 97423]).

Worse, `inv_freq` would not have been a no-op on balanced data. `compute_class_weights` normalises its
output to sum to 1, while the `setting=None` path returns `torch.ones` **unnormalised**:

| setting | small-file weights |
|---|---|
| `None` (champion) | [1.0, 1.0, 1.0, 1.0] |
| `inv_freq` | [0.25, 0.25, 0.25, 0.25] |

Per class identical, but 4x smaller absolutely — so it would have scaled the CE term down 4x against the
contrastive term, shifting the loss balance from 1:0.05 to 1:0.2. A Phase-1 screen would have measured a
CE-versus-contrastive reweighting and attributed it to class balancing: the right number with the wrong
cause. L6 is now an explicit weight **list**, inverse-frequency renormalised to mean 1.0, on the full file,
so it changes relative class weighting and nothing else. The CE-scale question went to WP-K as K6.

---

## 3. Correctness gates

`tests/test_wpl_preprocs.py`, all passing on CPU:

* **The champion preprocessor is bit-exact after the refactor — max diff 0.0.** L1 and L2 are implemented by
  adding two hooks (`phi_features`, `dxysig_feature`) to the shared variant base, whose defaults reproduce
  the champion exactly. Both new arms subclass `PFPreProcessorMeanPt`, so this is the gate that keeps the
  comparison single-variable.
* Each arm changes only its own channels; every other channel is bit-identical to the champion.
* Dead candidates stay exactly zero (the encoder derives its attention mask from all-zero rows).
* **All three preprocessors build and run through `eval.py`'s exact path**: fetched by name from the config,
  constructed positionally, encoder sized by `preproc.num_features`. No constructor kwarg anywhere. This is
  the defence against silent-failure class 11 (ReLU weights strict-loaded into a GELU graph because the
  option was a kwarg the eval harness never passes), and it is what lets L1 change the input width 14 → 15
  safely: the grader resizes the encoder from `preproc_type` alone.

### A defect caught before it ran
The first EMA implementation averaged parameters **and** buffers. BatchNorm running statistics are buffers,
are already an EMA of their own, and — for the *preprocessor's* BatchNorm — describe the **input data** and
depend on no weight at all, so averaging them is pure lag. Measured on a 2-epoch smoke, it moved
`preproc.batch_norm.running_var` by **8.36e4 on a raw magnitude of 9.34e5 (9%)**: a shift in the input
normalisation with none of the averaging effect being tested. Learnable weights moved by ≤5.5e-3. L3 now
averages parameters only and copies buffers from the latest epoch (verified: buffer diff exactly 0.0).

Known approximation: a textbook SWA would *recompute* activation BatchNorm statistics under the averaged
weights (`swa_utils.update_bn`); copying the final epoch's is cheaper and is what L3 does, so the
projector's two BatchNorms hold statistics for the final weights rather than the averaged ones.

---

## 4. Results

Reference: WP-H `h-ref-small-s33` mean_area 0.8239 ± 0.0003, clean 0.9184 (small file).
_Table fills in as arms bench; rows report mean over seeds ± seed std, with the per-seed probe std beside._

---

## 5. Best-vs-last on the champion, and what it does not measure

| reference | best-val | last epoch | gap | threshold | verdict |
|---|---|---|---|---|---|
| s22 | 0.8296 ± 0.0005 (ep 17) | 0.8277 ± 0.0006 (ep 22) | **+0.0019** | 0.00105 | **separable** |
| s11 | 0.8255 ± 0.0009 (ep 23) | 0.8245 ± 0.0013 (ep 24) | +0.0010 | 0.00212 | tie |
| s33 | — | — | — | — | excluded: best-val epoch **is** its last |

**Correction to my own framing.** I proposed this as "what the selection lottery costs". It is not. It
compares **selection against no selection** — the best-val epoch against simply taking the final one. Both
gaps are positive, so the criterion carries real signal *despite* its noise. Isolating the noise's cost would
need one run selected both ways, which cannot exist: a run has a single validation stream, and seeding it
changes which epochs are candidates.

**What it does support:** checkpoints the criterion treats as near-tied differ by **0.001–0.002 in
mean_area** — so where the noise lands moves a run by that order, comparable to the ±0.0030 seed floor and a
fifth to a quarter of the 0.0074 bar arms are judged on.

**And against my own arm:** the evidence is *variance*, not bias — on s22 the noise landed on a separably
**better** checkpoint. Seeding removes variance, and there is no bias for it to remove, so L4 should not be
expected to raise the mean. `l-l4-valseed-s11` at 0.8244 duly sits slightly below the reference. **L4 buys
reproducibility, not accuracy.**

### Free calibration: the bench itself is not the noise

Two checkpoints were each benched twice, independently (WP-M re-ran them while running the suites):

| checkpoint | bench 1 | bench 2 | spread |
|---|---|---|---|
| `h-ref-small-s22-last` | 0.8277 | 0.8274 | 0.0003 |
| `h-ref-small-s11-last` | 0.8245 | 0.8251 | 0.0007 |

**Mean spread 0.00049**, against the reference's ±0.0030 seed std — the bench is about **11× more reproducible than
the seed-to-seed variation it is used to measure**. So the spread seen across seeds is genuine training
variation (of which the selection lottery is one component, worth 0.001–0.002 by the best-vs-last pairs), not
measurement noise. This complements WP-G's campaign-1 figure of 0.0008 for a full train-and-bench repeat:
0.0008 end to end, 0.00049 for the bench alone, so most of even that small number is training.

The repeats also let me re-check the s11 best-vs-last verdict against the mean of both measurements of its
last-epoch checkpoint: gap +0.0007 against a threshold of 0.00145 — still a tie, so that conclusion does not
depend on which of the two measurements is used.
