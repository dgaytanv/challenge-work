# WP-B writeup: degradation generator, symmetry augmentation, curriculum, and the pt-rule family

Owner: WP-B. Covers `src/embedding/degradation.py` and the `preproc_type` variants in
`src/embedding/preprocs.py`. Every number here is measured; the source of each is named.

---

## 1. The problem the generator has to solve

The grader trains a probe on **clean** latents and applies it, frozen, to **degraded** latents of the
same events. So the target is not "classify well under damage" but "put the damaged event's latent where
the clean event's latent was". Training augmentation therefore has to span the space of plausible dead
detector regions without being tuned to any particular one — the grader's corruption is undisclosed by
design, and the organisers' `degradation_eval.py` is marked temporary upstream and stated to change
before judging.

## 2. Generator design

`Degradation(severity)` has two modes keyed on `severity`:

* `severity=None` — **train mode**, random, used by `train.py` as augmentation.
* `severity=s` — **eval mode**, deterministic from a seed, for local sweeps.

### 2.1 Severity semantics
`s` is the **target fraction of the eta-phi plane that is dead**, and every family is parameterised so the
expected fraction of candidates dropped is approximately `s`. This is the single design decision that makes
the families comparable to one another and to the grader's notion of "% of the eta-phi plane".

### 2.2 The five families
Detector failures are not one shape, so the generator samples uniformly among five, each covering a
different correlation structure in (eta, phi):

| family | shape | motivation |
|---|---|---|
| `rect` | K<=4 axis-aligned rectangles | a dead module or group of modules |
| `wedge` | phi bands, full eta | a dead sector / HV trip in phi |
| `strip` | eta bands, full phi | a dead ring or endcap layer |
| `cells` | coarse grid, cell in {0.25, 0.5, 1.0}, fraction dead | patchy readout loss at several granularities |
| `towers` | 0.1 x 0.1 cells, Bernoulli | isolated dead towers / channel masking |

`rect`/`wedge`/`strip` are membership tests on (eta, phi) broadcast over `[B, N, K]`; `cells`/`towers` use a
per-event random field plus a gather. No Python loop over events anywhere.

### 2.3 Milder failure modes
Real detectors degrade partially, not just totally. With small probability a region kills **only charged**
(|pdgId| in {11,13,211}) or **only neutral** candidates, or merely **scales pt** by U(0.3,0.9) instead of
zeroing. The pt-scaling mode is the one that surprises people: those rows stay **alive** with reduced pt,
so any test asserting "surviving candidates are bit-identical between two views" must set `p_pt_scale=0`.
This is recorded in `tests/test_degradation.py` because it broke one of our own tests first.

### 2.4 Deviation from the original spec, and why
The work package prescribed rectangle sizes `d_eta ~ U(0.5,2.5)`, `d_phi ~ U(0.3,1.2)`, `K<=4`. Those cover
at most **~19% of the plane** (max area 4 x 2.5 x 1.2 = 12 of 10 x 2pi = 62.8), so rectangles could never
reach the `s=0.8` the same spec's tests demanded. Resolution: keep the prescribed ranges as an
**aspect-ratio prior** and scale both dimensions by a common factor to hit `s`. Similarly, eta strips were
specified with both a `d_eta` and a narrow `d_phi`, which makes them duplicate rectangles; they are
implemented as full-phi eta bands, the natural complement to phi wedges. Both approved by the planner.

## 3. Symmetry augmentation (free accuracy)

Before any dead region, train mode applies two **exact** symmetries of a pp collision: a global phi rotation
and an eta reflection. These are not approximations — the physics is invariant under both — so they cost
nothing in label noise and enlarge the training distribution for free. Applied only to valid rows, so
zeroed rows stay exactly zero, which the encoder's mask derivation depends on.

Verified exact (`test_symmetries_are_exact`): pt, dxy, dxysig, is_pf and pdgId are bit-identical; |eta| is
preserved; the phi rotation is constant within an event to `9.54e-07`.

For a two-view consistency loop, both views must share **one** rotation. Two independent `Degradation`
instances would draw different angles. The correct pattern is **compose-then-degrade**: one
`dead_regions=False` instance applies the symmetry to `x`, and the degrading instance (with
`rotate_phi=False, reflect_eta=False`) is applied to its output.

## 4. Curriculum

`s_max` ramps linearly from 0.2 to its full value over the first `warmup_calls` train-mode forwards
(600 under the Phase-1 protocol, ~2 epochs at batch 256). The model learns the clean task before being
asked to be invariant to heavy damage.

One bug found and fixed here: the ramp started unconditionally at 0.2 and interpolated **to** `s_max`, so
any `s_max < 0.2` ran the ramp *downwards* and produced **more** severity than configured (`s_max=0`
started at 0.2). Start is now `min(0.2, s_max)`. This mattered because the symmetries-only configuration
is exactly `s_max=0`.

## 5. The pt-rule family — the largest single robustness leak we found

### 5.1 The mechanism
Stock `PFPreProcessor` encodes pt as `log(pt_i / sum_pt)`, with `sum_pt` summed over the **surviving**
candidates. Kill a dead region and `sum_pt` falls, so **every surviving candidate** has its pt feature
shifted by the same constant `-log(f)`, `f` = surviving pt fraction. Candidates nowhere near the dead
region move too. This is in the **input**, so no amount of attention masking, readout choice or
architecture can undo it.

### 5.2 How much of the damage it accounts for (WP-B measurement)
Stock reference checkpoint, eval-mode degradation, 512 events. `|dz|` is clean-to-degraded latent distance
in units of the clean-latent std. "frozen" repeats the degraded event but normalises pt by the **clean**
`sum_pt`, isolating the denominator.

| severity | pt lost | -log f | shift in BN units | \|dz\| stock | \|dz\| frozen | removed by freezing |
|---|---|---|---|---|---|---|
| 0.2 | 0.155 | 0.169 | 0.458 | 0.4280 | 0.1871 | **56.3%** |
| 0.4 | 0.314 | 0.376 | 1.021 | 0.7966 | 0.3449 | **56.7%** |
| 0.6 | 0.475 | 0.644 | 1.746 | 1.0239 | 0.4784 | **53.3%** |
| 0.8 | 0.637 | 1.014 | 2.751 | 1.2270 | 0.6218 | **49.3%** |

The pt feature's BatchNorm running std is 0.369, so at severity 0.8 the shift is **2.75 batch-norm units**.
**Roughly half the latent displacement under degradation is this one term.**

### 5.3 The four rules
Implemented as subclasses of one base that overrides only the pt rule, selected via the existing
`preproc_type` config key. `PFPreProcessor` is untouched, `num_features` stays 14 and the checkpoint layout
is unchanged.

These variants were built behind the config switch as a hedge, because `preprocs.py` was not in the
challenge's listed set of editable files and the work would have had to be droppable if it turned out to be
out of scope. **The organisers have since confirmed that preprocessor changes are allowed**, so the hedge
is no longer load-bearing and `PFPreProcessorMeanPt` ships in the certified submission. The design still
earns its keep for a different reason: leaving `PFPreProcessor` byte-identical is what made the
`b-aug-stock` vs `b2-*` comparison a single-variable one, and it is what let the stock rule be pushed
through the same code path to prove the refactor bit-exact (section 6).

| rule | formula | denominator moves when... |
|---|---|---|
| stock | `log(pt_i / sum_pt)` | *any* candidate is lost |
| `AbsPt` | `log(pt_i)` | never (no denominator) |
| `MaxPt` | `log(pt_i / max_pt)` | the *leading* candidate is lost |
| `MeanPt` (WP-E) | `log(pt_i / mean_pt)` | count and pt stop scaling together |

`MeanPt` is `log(pt_i / sum_pt) + log(n_valid)`: the stock rule plus a count correction. It is the robust
member because a dead region removes candidates **and their pt together**, so `sum_pt` and `n_valid` shrink
by roughly the same factor and the ratio barely moves.

### 5.4 Feature-level shift (WP-B measurement, 2048 events)
Mean |shift| of a surviving candidate's pt feature, in nats:

| severity | stock | AbsPt | MaxPt | MeanPt |
|---|---|---|---|---|
| 0.2 | 0.1730 | 0.0000 | 0.0720 | **0.0171** |
| 0.4 | 0.3810 | 0.0000 | 0.1505 | **0.0282** |
| 0.6 | 0.6469 | 0.0000 | 0.2549 | **0.0395** |
| 0.8 | 1.0123 | 0.0000 | 0.3961 | **0.0517** |

`MeanPt` removes 95% of the stock shift at severity 0.8, ~8x better than `MaxPt`, while keeping the
relative-scale information `AbsPt` discards.

### 5.5 The train/eval geometry offset (second, independent defect)

**The file construction.** Both files are sorted by descending pt. The small train file keeps the top 200
candidates per event (min pt 2.080); the eval file keeps 400 (min pt 1.622). An eval event's top 200 slots
carry **66.5%** of its total pt. So training sees the top-200-by-pt truncation of the kind of event the
grader scores at 400, and the pt feature is offset between training and scoring before any dead region.

**Two different measurements, and only one of them is the one that matters.**

*(a) Per-candidate, paired.* Hold the candidate set fixed and recompute only the denominator: how does one
physical particle's encoded value change? This shift is a constant within an event, `log(D200/D400)`.

*(b) Distributional.* The mean of the feature distribution actually produced by the preprocessor for a
200-slot event vs a 400-slot event. This is what reaches the encoder, because `BatchNorm` at eval time
normalises with **running statistics fixed during training**, so a shift in the location of the feature
distribution is passed straight through to attention.

Measured on the same eval events (4000 events, no physics difference anywhere in this comparison):

| rule | (a) per-candidate | (b) distributional geometry | geometry removed |
|---|---|---|---|
| stock | +0.4093 | **+0.6934** | - |
| AbsPt | 0.0000 | **+0.2841** | 59% |
| MaxPt | 0.0000 | **+0.2841** | 59% |
| MeanPt | -0.2838 | **+0.0003** | **99.96%** |

The two columns disagree because truncation does two things at once. It changes the denominator (column a)
**and** it deletes the soft candidates, which raises `mean log(pt)` by **+0.2841**. Column (a) sees only the
first; the encoder sees both. Under the stock rule the two add (+0.2841 + 0.4093 = +0.6934); under `MeanPt`
they cancel (+0.2841 - 0.2838 = +0.0003).

**The cancellation is structural, not luck.** Under `MeanPt` the mean feature of an event is
`mean[log pt] - log(mean[pt])` = `log(GM/AM)` of that event's pt spectrum: a dimensionless **shape**
statistic, scale-free by construction, and the arithmetic/geometric-mean ratio of a heavy-tailed pt
spectrum barely moves under top-k truncation.

**The counterintuitive part, worth stating explicitly.** `AbsPt` and `MaxPt` are *exactly* invariant in
measure (a) — zero to machine precision — and this makes them look immune. They are not: having no
candidate-set-dependent denominator means they have nothing to absorb the composition term either, so they
leave 59% of the geometry offset standing. Exact per-candidate invariance is the wrong target.

**Decomposing the real train-vs-eval offset** confirms there is no meaningful confound in the file-to-file
comparison (eval400 -> eval truncated to 200 -> real train file):

| rule | geometry | physics (sample difference) | total |
|---|---|---|---|
| stock | +0.6934 | +0.0364 | +0.7298 |
| MeanPt | +0.0003 | +0.0364 | +0.0367 |

The physics term is identical between the two schemes, as it must be — it is a property of the samples, not
of the normalisation — and it is only ~5% of the stock offset. So `MeanPt` removes essentially all of the
geometry artifact and leaves exactly the genuine sample difference, which we do **not** want removed.

**Correction history, recorded deliberately.** An earlier draft reported measurement (a) and concluded
`MeanPt` fixed only 31% of this offset, demoting it below `AbsPt`/`MaxPt`. That was wrong. The error was
mine and instructive: measurement (a) is a real quantity answering a real question, but it is not the
question `BatchNorm` poses, and I trusted it *because* it contradicted a result I had produced and I was
readier to distrust my own favourable number than a plausible-looking unfavourable one. WP-E supplied the
decomposition; the numbers above are an independent reproduction of it, agreeing to four decimals.

**Still a separate row from the severity drift.** The two defects reach the metric through different
channels. The probe is fit on clean **eval-domain** latents and applied to degraded **eval-domain**
latents, so a static train/eval offset does not enter the score directly - it costs encoder quality and
clean AUC. The severity-dependent drift of 5.2/5.4 is what bends the AUC-vs-severity curve. `MeanPt`
happens to address both; WP-E's 400-candidate resampled train file addresses only the geometry offset.
They remain separate ablation rows because they do different jobs.

**Possible third arm.** WP-E measured `median_pt` as marginally better than `mean_pt` on the severity drift
(-0.01/-0.02/-0.05/-0.08 vs -0.02/-0.04/-0.08/-0.14 in clean-std units), at the cost of a per-event median.
Noted as an option, not built.

## 6. Validation

* **Faithfulness.** The stock pt rule pushed through the refactored variant base reproduces
  `PFPreProcessor` **bit-exactly** (max diff `0.0`) on clean and degraded input, so the `b-aug-stock` vs
  `b2-*` comparison differs only in the pt rule.
* **Against the shared bench.** The eval-mode module run through the **original** `eval.py` on the stock
  checkpoint gives area **0.8341**; the bench's independently implemented `cells` family on the same
  checkpoint gives **0.8347** — agreement to `0.0006` across different implementations, 2000 vs 20000
  events and different severity grids. Note this validates the *cells family only*: eval mode is a
  single-family proxy and reads ~0.04 above `mean_area`, which averages five families including the much
  harsher `rect`. It is not a `mean_area` estimate.
* **Cost.** 3.27 ms per `[128, 200, 7]` forward on the A10, against a 5 ms budget. Two things dominated
  before optimisation: a CUDA buffer call counter forcing a device sync every forward, and the three
  coarse-cell sizes being generated separately instead of in one fused flat grid.
* **Coverage.** Dropped-fraction distribution spans 0.000-0.885 with 9 of 10 deciles occupied; all five
  families reach high severity individually.
* **Contract.** Padding rows stay exactly zero; survivors are bit-identical; works unchanged on the
  400-candidate eval tensor; `eval.py` runs end to end.

## 7. Effect on trained models

**Augmentation on the set encoder: clearly positive.** `d-deepsets-clean` **0.8036** ->
`d-deepsets-aug` **0.8183**, i.e. **+0.0147 mean_area** from augmentation alone, at a slightly *higher*
clean AUC (0.8853 -> 0.8920) — robustness bought without costing nominal performance, which is not the
usual trade for corruption-style augmentation. Both are single probe fits (`R=1`), predating
`--probe_repeats`, so they carry unmeasured probe error; the floor is `3 sigma = 0.0025`, well below the gap.

**Augmentation on the transformer: unresolved, and possibly negative.** Under the same Phase-1 protocol:

| transformer arm | clean AUC | mean_area |
|---|---|---|
| stock anchor (frozen, 60ep/bs128) | 0.8605 | 0.7734 |
| mask fix only, frozen weights | 0.8604 | 0.7918 |
| **WP-E control: retrained, mask fix, NO augmentation** | **0.9025** | **0.8089 +- 0.0004** |
| WP-C run 1: retrained, augmentation + two-view 2B batch | 0.8370 | 0.7712 +- 0.0034 |

WP-E's control settles a question I had got wrong. I had argued that the retrained transformer rows were
depressed because the protocol gives 4.8x fewer optimizer steps than the anchor (7,025 vs 33,720). That
arithmetic is right, but it is **not** the explanation: trained under exactly that protocol with no
augmentation, the transformer reaches clean 0.9025 and `mean_area` 0.8089, comfortably *above* both the
anchor and the frozen mask fix. The protocol trains this architecture perfectly well.

So the 0.0655 clean-AUC deficit in WP-C's run is caused by something in that run, and there were only two
candidates: **this generator**, or the two-view 2B-batch construction. `b-aug-stock` — same protocol, same
mask fix, augmentation on, single view — separates them. **It ran, and the answer is unambiguous:**

| transformer arm | clean AUC | mean_area | R |
|---|---|---|---|
| WP-E control: no augmentation | 0.9025 +- 0.0004 | 0.8089 +- 0.0004 | 5 |
| **`b-aug-stock`: augmentation, single view** | **0.9035 +- 0.0003** | **0.8058 +- 0.0043** | 5 |
| WP-C: augmentation + two-view 2B batch | 0.8370 | 0.7712 +- 0.0034 | 5 |

**1. The generator is exonerated on the transformer.** With augmentation on and the two-view construction
removed, clean AUC is **0.9035** — level with the control (+0.0010) and **+0.0665 above** WP-C's run. The
deficit was the two-view 2B-batch construction, not this generator. The earlier suspicion that heavy token
deletion was damaging the CLS readout is not supported.

**2. Augmentation gives no separable `mean_area` gain on the transformer.** The gap is **-0.0031** against
a threshold of 0.0058: not separable, consistent with zero. This is a real asymmetry against the set
encoders, where the identical generator gave `d-deepsets-aug` **+0.0147**.

**3. The selection bias makes conclusion 2 stronger, not weaker** — which is why it was worth stating
before the number existed. `b-aug-stock` had the *advantageous* selection criterion (checkpoint chosen on
degraded validation loss, i.e. on the quantity `mean_area` scores) while the control was selected on clean
validation loss. So the measured gap is an **upper bound** on the augmentation effect. The upper bound
came out at approximately zero, and slightly negative. An effect bounded above by ~0 by a comparison
biased *in its favour* is a firmer null than an unbiased comparison landing on zero would have been.

So the honest summary is: **this augmentation helps symmetric-pooling set encoders and does not measurably
help the transformer**, and it costs nothing on either (clean AUC +0.0010 here, +0.0067 on Deep Sets). Why
the readouts differ is unexplained and is the same open question raised in section 7's pt-rule discussion —
two independent results now point at the readout as the thing that decides whether a robustness
intervention converts into `mean_area`.

Caveat retained: best validation loss landed at **epoch 25 of 25**, monotonically decreasing throughout
with early stopping never triggering — as it did for WP-E's control (epoch 23) and WP-C's run. All three
transformer rows are lower bounds on what the architecture reaches with a longer schedule.

**How that row must be read — written before it landed, deliberately.** It is the augmentation **arm**,
not the augmentation **effect**, and the difference is not pedantic. WP-E's control sets `augment: false`,
which passes `degradation=None` to *both* `train_epoch` and `validate_epoch`. So the two arms differ in
two ways: the intended variable, and the model-selection criterion. `b-aug-stock`'s saved checkpoint is
chosen on **degraded** validation loss; the control's on **clean** validation loss.

**The bias has a direction and it favours my arm** (WP-E's observation, not mine): selecting on degraded
val loss selects for the very quantity `mean_area` measures, an advantage owing nothing to augmentation.
So whatever gap appears, **it is an upper bound on the augmentation effect, not an unbiased estimate** —
the augmentation term alone lies somewhere below it. The selection term is plausibly small but is
**unmeasured and unrecoverable**, because only the best checkpoint was kept. WP-E's clean-val trajectory
shows why it cannot simply be assumed negligible: epochs 22, 23 and 25 sit within 0.007 of each other
(0.7928 / 0.7845 / 0.7911) with ~0.01 noise between adjacent epochs, so a different criterion could
plausibly have selected a different epoch, and nobody can now say how much `mean_area` that would move.

The clean fix is a separate `augment_val` key, or passing `None` to `train_epoch` while keeping the
degradation object for `validate_epoch`, so the selection criterion is held fixed across arms. One line,
identified too late to use. Recorded as future work.

I should record that I argued the wrong way here. My cross-architecture argument — that one generator
cannot cost clean AUC on the transformer while adding it on two set encoders, so the step budget must be
the cause — was too confident. Different architectures need not respond alike to heavy token deletion:
symmetric pooling over surviving elements degrades gracefully by construction, whereas a CLS token must
aggregate through attention over whatever tokens remain, and at the high dead fractions this generator
samples (up to 85%) those are very different situations. Architecture-specific sensitivity was always the
more likely explanation than a protocol artifact, and I reached for the protocol because it was the
explanation that did not implicate my own component.

**The pt rule — measured end to end, and the result is mixed.** At the convergence freeze the planner
ruled the pt-rule question is decided on WP-D's Deep Sets arm. It benched as:

| arm | clean AUC | mean_area | R |
|---|---|---|---|
| `d-deepsets-aug` (stock preproc) | 0.8921 | 0.8174 +- 0.0020 | 10 |
| `d-deepsets-aug-meanpt` | **0.9066** | 0.8186 +- 0.0007 | 5 |

A second arm then landed on the attention-pooled set encoder:

| arm | mean_area | R |
|---|---|---|
| `d-pma0-aug` (stock preproc) | 0.8162 +- 0.0038 | 5 |
| `d-pma0-aug-meanpt` | **0.8260 +- 0.0009** | 5 |

**The gain is demonstrated on PMA and not demonstrated on Deep Sets — which is not the same as saying it
is absent on Deep Sets.** On PMA the gap is 0.0098 against a threshold of 0.0052: **separable at 1.9x**.
On Deep Sets it is 0.0012 against 0.0021: **not separable**. (Thresholds are
`3*sqrt(s_a^2/R_a + s_b^2/R_b)`, reproduced independently.) The Deep Sets pair simply lacks the precision
to resolve a gain of that size — roughly **3x more probe repeats** would be needed, which no one had time
for. Quoting the two gains side by side invites the inference that `MeanPt` helps one architecture and not
the other; **the data does not support that contrast.** What it supports is one demonstrated gain and one
open question. (Correction owed to WP-E, who flagged that my earlier "architecture-dependent" headline
over-claimed exactly this.)

*If* the interaction turns out to be real once the Deep Sets pair is measured properly, there is a
mechanism ready to test. Offered as a hypothesis contingent on that, not as an explanation of the present
data: attention pooling computes its weights from the token
embeddings themselves, and attention logits are bilinear in those embeddings, so a common-mode shift
perturbs the *attention distribution* rather than merely offsetting the pooled output. Symmetric mean/max
pooling has no such pathway — the shift passes through as a shift. If that is right, the readouts that
should benefit most from a stable pt denominator are exactly the attention-based ones (PMA, and the
transformer's CLS), which is testable and was not tested.

I flag one thing about that hypothesis explicitly, because it is adjacent to a claim of mine that was
correctly overturned earlier. I had argued that masked mean pooling would *blunt* the denominator shift;
that was wrong and stays wrong — the mean of uniformly shifted embeddings is shifted just as much, and
nothing cancels. The present hypothesis is a different claim: not that mean pooling neutralises the shift,
but that attention pooling is *additionally* sensitive to it through the weights. Those are compatible, and
the second does not rehabilitate the first.

Three things are nonetheless real and point the other way:

* **Clean AUC +0.0145** (0.8921 -> 0.9066), far outside probe noise. Consistent with section 5.5: `MeanPt`
  also removes the train/eval geometry offset, which costs representation quality rather than bending the
  severity curve. That is the channel where it visibly paid.
* **Official metric, both runs in.** On the organisers' `eval.py` at full settings, 0.8708 and 0.8701,
  **mean 0.8705** (spread 0.0007), against **0.8598** for the stock-preproc arm — a gain of **+0.0107**,
  an order of magnitude above the run-to-run spread. This is the clearest evidence for the pt rule, and it
  is on the organisers' own corruption rather than our development bench.
* **The trade is structural, not noise.** On Deep Sets, `MeanPt` gives up `rect` (0.7538 vs 0.7754) to
  gain `wedge`/`towers`/`cells`. `rect` is also precisely the family my eval-mode sweep does **not**
  represent (section 6), so it is the family I had least visibility into throughout.

**Correction on the `rect` inversion — it is not a property of the pt rule.** I earlier reported that
`MeanPt` "holds the only below-chance cell among the finalists" (`rect` at s=1.0, 0.451) and flagged it as
a `MeanPt` failure mode. That was true of the finalist set at the time and misleading as a claim. Measured
across arms, `rect` at s=1.0:

| arm | preproc | rect @ s=1.0 |
|---|---|---|
| `c-twoview-cos1` (transformer) | **stock** | **0.3901** |
| `e-stock-maskfix` (transformer) | **stock** | **0.4710** |
| `d-deepsets-aug-meanpt` | MeanPt | 0.4514 |
| `d-deepsets-aug` | stock | 0.5491 |
| **`d-pma0-aug-meanpt`** (leader) | **MeanPt** | **0.5563** — no inversion |
| `d-pma0-aug` | stock | 0.5793 |

The two **worst** inversions are stock-preproc arms, and the current leader — which *uses* `MeanPt` — does
not invert at all. Three arms with three different preprocessor configurations invert in the same family,
which is what a corruption-regime effect looks like and not what a preprocessor effect looks like. Credit
to WP-F for catching that my original phrasing invited exactly the wrong reading. The mechanism is the
severity-distribution gap in section 9: training signal runs out near s=1.0 for *every* arm at once.

**The two metrics disagree about this arm, and that is itself a finding.** Our five-family development
bench says the pt rule is *not separable* (0.0012 against a 0.0021 threshold). The organisers' own sweep
says it is clearly separable (+0.0107 against a 0.0007 spread, ~15x). Both are measured correctly; they are
simply different corruption distributions, and we steered all evening on the one that shows no effect here.
The decision rule handles this case correctly by design — arms tied within the probe floor are broken by
official area — but it is worth recording that a development bench built to avoid over-fitting to any one
dead-region shape can, for that same reason, be insensitive to a real improvement that the grader's
distribution rewards. The per-family split in this arm shows why: `MeanPt` trades `rect` for
`wedge`/`towers`/`cells`, so a five-family unweighted mean averages the gain away, while a corruption that
does not look like `rect` sees it.

**Held-out shapes: the generalisation gap my generator did not close.** WP-E's held-out check
(ellipse / annulus / diagonal — shapes in neither this generator nor the scoring bench) puts every
candidate about **0.02 lower** than on the five scoring families. Roughly a quarter of that is mechanical:
two of the three held-out families drop 100% of candidates at s=1.0, forcing AUC to 0.5, against one of
five on the scoring bench. The remaining **~0.015 is a genuine generalisation gap**. So the honest headline
for the submission is a pair, not a number: **~0.826 on shapes we trained and measured with, ~0.80 on
shapes from neither our generator nor our bench.**

This is the most direct measurement of the limit of section 2's design. The five families were chosen to
span plausible detector failures and to avoid over-fitting to any one shape, and they do transfer — 0.80 on
never-seen geometries is well above the 0.7734 anchor. But a curved or diagonal dead region is still
measurably harder than the rectangles, bands and grids the model trained on, and no amount of breadth
*within* axis-aligned shapes fixes that. Adding curved and rotated regions to the generator is the obvious
next step and was never tried.

**What I take from this.** The section 5 measurements are correct as measurements and were verified twice,
including once against my own error. What they do not license is the inference I was implicitly making:
that removing a large input-level distortion converts proportionally into `mean_area`. It did not. The
encoder evidently compensates for a systematic common-mode shift during training more effectively than the
raw displacement figures suggest, and the benefit surfaced in clean AUC and on the official sweep instead.
A large, well-measured input-level defect is a reason to investigate, not a promise of a metric gain.

## 8. Considered and rejected

* **Tuning any family toward the organisers' 10x10 grid.** Explicitly avoided. The coarse-cell family
  already covers that shape generically, and the upstream grid is marked temporary and stated to change
  before judging.
* **Eval-mode compensation for drop probability.** Train mode compensates so effective drop ~= `s`; eval
  mode deliberately does not (it follows the spec's literal "fraction `s` of cells x drop prob 0.8"), so
  its effective drop is `0.8s`. Kept as specified because the bench, not this sweep, is the ruler.
* **Reseeding the eval generator per forward.** Would make the corruption depend on an event's position
  within its batch. The generator is seeded once and advances across batches instead.
* **`AbsPt` and `MaxPt` as robustness fixes.** Cut at convergence, and section 5.5 explains why they
  deserved to be: both are *exactly* invariant per-candidate, which is what made them attractive, but
  distributionally they leave **59%** of the geometry offset standing because they have no
  candidate-set-dependent denominator to absorb the composition term. Exact per-candidate invariance is
  the wrong target.
* **`median_pt`.** Marginally better than `mean_pt` on the severity drift in WP-E's measurement, at the
  cost of a per-event median. Not built; the obvious next variant if the pt rule proves decisive.

## 9. What this section does not establish

Stated explicitly so the final table is not over-read:

1. **No WP-B transformer arm completed.** `b-aug-stock` and `b2-meanpt` were queued behind four other
   sessions on a single shared A10 and were demoted to ablation rows at the freeze. Every number in
   sections 2-6 is generator behaviour, timing, or a measurement on a frozen checkpoint.
2. **The augmentation and pt-rule results are on Deep Sets, not the transformer.** Whether the same gains
   transfer to `TransformerEncoder` is untested. There is a specific reason to expect a *different* size
   of effect rather than an identical one: the two architectures differ in how a common-mode input shift
   reaches the latent, and an early WP-B prediction that masked mean pooling would blunt it was wrong and
   was withdrawn (mean pooling normalises for token *count*, not for a common-mode shift in feature values).
3. **The severity-drift and geometry-offset tables use my eval-mode degradation**, a single coarse-cell
   family. It reproduces the bench's `cells` family to 0.0006, but it is not the five-family mean, and the
   harshest family (`rect`) is the one it does not represent.
4. **The step-budget gap is a limitation, not an explanation — retracted as a cause.** The Phase-1
   protocol gives **7,025 optimizer steps** against the anchor's **33,720** (4.8x fewer, 2.4x fewer samples).
   I originally offered this as the reason the retrained transformer rows underperformed. WP-E's
   no-augmentation control refutes that: under the identical protocol the transformer reaches clean 0.9025
   and `mean_area` 0.8089, above both the anchor and the frozen mask fix. The step gap remains a real
   limitation on comparing these rows to the 60-epoch anchor, but it does not explain WP-C's deficit, and
   it should not be cited as though it does.

5. **The generator's severity distribution is mismatched to what the sweep scores.** Measured on 4000
   real events (WP-F's histogram, reproduced independently here):

   * **Only 9.0% of training events exceed a 0.6 dropped fraction, and 0.8% exceed 0.8.** The bench and the
     grader both sweep severity to 1.0, so the model is trained almost entirely on corruptions milder than
     the ones it is scored hardest on. This follows from sampling `s ~ U(0, s_max)`: uniform in `s` puts
     most mass at low severity, and the region families undershoot their target at high `s` because regions
     overlap, thinning the tail further. Sampling `s` with weight toward the top of the range — or uniformly
     in *realised* dropped fraction rather than in target area — is the obvious fix and was never tried.
   * **The effective clean fraction is 23.6%, not the 15% `p_clean` implies** (design arithmetic:
     `0.15 + 0.85 x 0.10 = 23.5%`; measured 23.6%, WP-F 23.9%). The extra 8.5% is the `p_pt_scale` mode,
     which dims rows rather than removing them, so those events contribute *zero* to a dropped-fraction
     histogram while still being degraded. Nothing is wrong with the mode — it teaches a different
     invariance — but the zero bin of any dropped-fraction plot conflates two unlike things, and roughly a
     quarter of training events show the encoder no missing candidates at all.

   **Measured per family (WP-F, 8000 events, train mode, curriculum off):**

   | family | mean dropped | >0.6 | >0.8 | max ever seen | bench s=1.0 applies | beyond training max |
   |---|---|---|---|---|---|---|
   | rect | 0.207 | 0.78% | **0.00%** | 0.798 | 0.940 | +0.142 |
   | wedge | 0.228 | 1.01% | **0.00%** | 0.700 | 0.900 | +0.200 |
   | strip | 0.222 | 2.25% | 0.03% | 0.848 | 0.863 | +0.015 |
   | cells | 0.291 | 19.64% | 0.00% | 0.755 | 0.762 | +0.007 |
   | towers | 0.311 | 21.06% | 4.15% | 0.890 | 1.000 | +0.110 |
   | *aggregate* | 0.250 | 9.11% | 0.86% | — | — | — |

   **Extrapolation at s=1.0 is real and universal — this part is now measured rather than inferred.**
   Every family's s=1.0 bench point lies beyond the maximum dropped fraction that family ever produced in
   training. Note also that the aggregate "0.86% above 0.8" is carried almost entirely by `towers` (4.15%);
   **not one of 8000 `rect` events exceeded 0.8**, and `rect` and `cells` never reach it at all. So the
   aggregate figure should not be quoted about any individual family, and I have stopped doing so.

   **What this does NOT explain: why `rect` specifically inverts.** An earlier draft of this section claimed
   the severity gap explained the `rect` inversion. WP-F's per-family measurement refutes that, and I am
   retracting it. `rect` and `wedge` have near-identical training tails (0.78% and 1.01% above 0.6, neither
   ever reaching 0.8), and by distance beyond its own training maximum `wedge` is *further* out than `rect`
   (+0.200 vs +0.142). If thin coverage caused inversion, `wedge` should invert at least as readily as
   `rect`. It does not: the masking-trained arms invert at `rect`, while the frozen mask-fix arms invert at
   `wedge`.

   The honest form is therefore two claims, not one:

   1. **Measured.** Extrapolation at s=1.0 is universal across families and is the condition under which a
      below-chance AUC becomes possible at all.
   2. **Observed, unexplained.** *Which* family a given arm inverts in tracks its training setup rather
      than the generator's coverage. Nobody has a mechanism for this and it should be recorded as an open
      question, not folded into (1).

   **The process lesson, which is the transferable part.** Neither of us could have reached the per-family
   answer from what we already had. My coverage test asserted that dropped fractions spanned 0.000-0.885
   with 9 of 10 deciles occupied; WP-F's first histogram reported the aggregate tail. **Both were fully
   consistent with `rect` having a fat tail or no tail at all.** Getting the answer took conditioning on
   the family — about twenty minutes of work that either of us could have done at any point in the
   preceding three hours. The failure was not that the assertion or the aggregate was wrong; both were
   correct. It was that we each held a number incapable of answering the question, and neither noticed
   until the question was put directly. A summary statistic that cannot distinguish the hypotheses you
   care about is not weak evidence, it is no evidence, and it reads as reassurance either way.

   Both are plausible contributors to any weak augmentation result on the transformer, and both were found
   by plotting the generator's output rather than by reading its code. I had validated coverage
   (0.000-0.885, 9 of 10 deciles occupied) and treated that as sufficient; a decile histogram with an
   occupied tail hides the fact that the tail holds 9% of the mass.

6. **Probe error.** The `d-deepsets-clean` vs `d-deepsets-aug` comparison quoted above is `R=1` on both
   sides. The gap is ~6x the measured 3-sigma floor, so the sign is safe, but the magnitude is not precise.
