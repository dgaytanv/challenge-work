# The probe noise floor: how small a difference is meaningless

WP-E. Source: `bench/probe_variance.py`, run on the organisers' stock checkpoint
(`robust_tagging_encoder_20260902_212357.pth`) through `~/rt-e` at 20k events, 10 repeats.

## The defect

`eval.py`'s `train_linear_probe` constructs an `EvalMLP` with a random initialisation and trains
it with a shuffling DataLoader. **Nothing in `eval.py` or `bench_eval.py` calls `manual_seed`.**
So the probe is refit differently on every run, and every area anyone quotes -- the organisers'
official number and our own five-family `mean_area` alike -- inherits that variance.

This was found while chasing a related point: WP-D observed that the organisers' new
`degradation_eval.py` draws its Bernoulli drop mask from the unseeded global RNG, and noted the
probe as a second unseeded source not specific to their degradation. It is not specific to the
grader at all -- it is in the ruler we steer by.

Before this was measured, the table ranked candidates to four decimal places with no idea whether
the third decimal meant anything.

## Isolating it

Two sources move together in a naive end-to-end rerun (corruption sampling and probe refit), so a
rerun cannot attribute the spread. `probe_variance.py` separates them: `BenchDegradation` is
seeded, so the script embeds every family and severity **once** and then refits **only** the probe
over those frozen latents. Any spread in the resulting areas is the probe alone.

## Result

Stock checkpoint, `TransformerEncoder`, 20k events, five scoring families, 10 refits:

```
  clean AUC    mean 0.8601  std 0.0006  min 0.8593  max 0.8608  spread 0.0016
  rect         mean 0.7565  std 0.0006
  wedge        mean 0.7960  std 0.0014
  strip        mean 0.7860  std 0.0007
  towers       mean 0.7846  std 0.0012
  cells        mean 0.8336  std 0.0010
  mean_area    mean 0.7913  std 0.0008  min 0.7901  max 0.7925  spread 0.0023
```

**3 sigma on `mean_area` = 0.0025.** Two tags differing by less than that are not distinguishable.

Sanity check that the pipeline reproduces: repeat 1 gave `mean_area` 0.7917 against the
independently recorded 0.7918 for the same checkpoint under the same mask-fixed code
(`runs/planner-maskfix-stockckpt_20260903_195138.json`).

## What survives the floor

Each pair tested with its own sigmas and repeat counts, `3*sqrt(s_a^2/R_a + s_b^2/R_b)`:

| comparison | delta | threshold | verdict |
| --- | ---: | ---: | --- |
| mask fix vs stock anchor (both transformer, n=1) | +0.0183 | 0.0034 | real, 5.4x |
| `d-deepsets-aug` vs anchor (cross-arch, n=1) | +0.0449 | 0.0116 | real, 3.9x |
| Deep Sets aug vs clean (both Deep Sets, n=1) | +0.0147 | 0.0161 | **not separable, 0.9x** |
| Deep Sets 20k vs `--full` (both Deep Sets, n=1) | -0.0019 | 0.0161 | inside the noise |

The third row is the cautionary one, and it is the error described below: when first reported it
was certified "18.4 sigma, real" by dividing the gap by the **transformer's** 0.0008. Against the
Deep Sets floor of 0.0038 at `n=1` it does not clear the threshold at all. The underlying claim --
that augmentation helps Deep Sets -- is likely true and would be resolvable with repeats, but the
`n=1` rows cannot establish it.

The fourth row is the reverse case and the reason to measure rather than guess: it had been called
"inside the noise" from a +-0.005 estimate, and it is, but a repeated-row comparison would have
needed only 0.0021 to resolve it -- so the original verdict was right by luck rather than by
evidence.

## The errors are correlated -- do not propagate them as independent

One probe is fit on the clean latents and applied **frozen** to every severity, so a refit that
happens to fit the clean split well looks good everywhere. The five family areas within a repeat
share that same probe and are correlated with each other too.

The size of the effect, since it is easy to talk yourself out of:

```
mean per-family sigma                       0.00098
if the five families were independent:      0.00098 / sqrt(5) = 0.00044
measured mean_area sigma:                   0.0008   -> 1.83x larger
```

That factor is the correct behaviour of a correlated average, not a bug. Anyone "fixing" it by
dividing by `sqrt(5)` would halve the floor and start ruling on noise. Threshold on the
`mean_area` sigma; never propagate per-severity or per-family sigmas as independent draws.

## What was changed as a result

- `bench_eval.py` gained `--probe_repeats` (default 5): embed once, refit the probe N times,
  report mean +- std for clean AUC, every family area and `mean_area`. The JSON keeps the
  per-repeat values alongside mean and std.
- `ablation_table.py` prints `mean_area ±s`, and flags rows written before the flag existed as
  `_(n=1)_` -- single fits with an unmeasured error, rather than pretending they have none.
- Standing rule adopted by the planner: **no ruling between two arms whose `mean_area` differs by
  less than the measured 3 sigma floor for that checkpoint family.**

## A separability verdict belongs to a PAIR, not to a number

The floor was introduced to stop people ruling on noise, and then three of us misapplied it within
the hour, each in a different way:

- The floor was measured on the transformer and then used to certify a comparison between two
  **Deep Sets** rows, whose per-refit sigma is 0.0038 -- nearly five times larger. The certified
  gain did not survive its own architecture's floor. (WP-E.)
- A threshold was written as `3 * sigma` of a single row rather than
  `3 * sqrt(s_a^2/R_a + s_b^2/R_b)` over the pair, understating the resolving power that repeats
  buy: 0.0024 where the correct figure was 0.0015. (WP-A.)
- A claim was supported by citing an `(n=1)` row, whose 0.0038 floor swamped the 0.0094 gap under
  discussion; the same claim held at 4-9x against the repeated rows. (WP-A.)
- Two **official** areas were compared by eye, by the same person who had spent the evening
  insisting nobody compare bench areas by eye -- and the official column carries a second unseeded
  source on top of the probe, so its floor is *wider*, not narrower. (WP-E.)

WP-A stated the general form, and it is the useful takeaway:

> A separability verdict is not a property of a number, it is a property of a **pair**. It does not
> travel with the number when that number is reused in a different pairing.

`0.8183` is a perfectly good measurement. It simply cannot support a 0.0094 gap at `n=1`. The
corollary is that every number entering a comparison needs the test applied *to that comparison*,
including numbers received from someone who already applied it correctly to their own use of them.

WP-A also named the mechanism behind the repetition: **a corrected claim feels checked.** Having
just fixed something adjacent, the rest of the sentence gets treated as inherited rather than as
new material. The errors above each moved one step away from wherever the author had just been
looking -- from the arms to the threshold, from the threshold to the cited row, from the bench
column to the official column.

This is why `ablation_table.py` computes the "sep. from leader" column itself, per pair, with each
side's own sigma and repeat count, rather than publishing a single floor for readers to apply by
hand. A number in a table invites reuse; a computed verdict does not.

## Two floors, and the probe floor is the smaller one

The probe-refit sigma is a floor on **one** source of variation. A comparison between two
independently *trained* models has a second, larger floor: the training pipeline's own
reproducibility. WP-G supplied the measurement that makes this concrete — `gB-q-b1e-6-l1t` and
`gD-bits10-l1t` are **the same configuration trained and benched twice**, and they differ by
**0.0008** (0.8079 vs 0.8087).

That has a direct consequence for reading the quantization sweep. 6 bits (0.8080 ±0.0001) against
10 bits (0.8087 ±0.0003) is a gap of 0.0007 against a probe-only threshold of
`3*sqrt(0.0001²/5 + 0.0003²/5)` = 0.0004, so the probe floor calls it separable. But 0.0007 is
*inside* the 0.0008 pipeline reproducibility. The honest phrasing is **"separable on probe noise,
inside pipeline reproducibility"** — the difference is real as a property of these two trained
artefacts and is not evidence that 10 bits is better than 6 as a configuration.

### The 0.0008 is a lower bound, and the practical reading is 0.002

That pair were **12-epoch distillation runs on 10k events per epoch from a shared seed
checkpoint**. Sharing a seed checkpoint removes most of the initialisation variance, and a short
distillation schedule on a tenth of the data explores far less of the loss surface than the runs
the ruling rests on. So 0.0008 is a **lower bound** on the training-seed variance of a full
25-epoch contrastive run over 72k events, not an estimate of it.

**Practical reading: treat any bench separability below about 0.002 between two independently
trained full models as unproven.** Not wrong — unproven. The two comparisons the ruling depends on
clear it: the winner-versus-runner-up gap is 0.0074 and the a-pma R=20 gap is 0.0022. Everything
finer than that in this table is a property of the particular checkpoints measured, not an
established difference between the configurations that produced them.

The general form: `--probe_repeats` measures the noise in *scoring a fixed checkpoint*. It says
nothing about the noise in *producing* a checkpoint. Any comparison between separately trained runs
needs the second floor, and the only way to get it is to train the same configuration twice — which
tonight happened once, by accident, on a schedule that understates it, and is the sole reason we
can state it at all.

## Scope

This floor is for the transformer at 20k on the five scoring families. A different architecture or
`latent_dim` can have a different floor -- a 32-dimensional latent gives the probe more to fit and
may well be noisier -- which is why the rule is written per checkpoint family rather than as a
single global constant. WP-D is measuring the same quantity for `DeepSetsEncoder`.

The organisers' official area has an *additional* unseeded source on top of this one (their
Bernoulli mask), so its spread is expected to be larger; WP-D is measuring that with repeated
end-to-end runs. Subtracting the two in quadrature gives the Bernoulli contribution, though only
indicatively: the official runs use a 10x10 grid at 10 severities on 69k events while this uses
five families at 6 severities on 20k, so the two probe sigmas are not identical.


## Appendix: two drift statistics, and why neither explains the ranking

`bench/tsne_latents.py` reports, for the `cells` s=0.8 displacement over all 6000 events, both
computed on RAW LATENTS (not on the t-SNE embedding, which preserves neighbourhoods rather than
directions):

| model | coherence | along-probe fraction | along-probe absolute (clean-spread units) | AUC clean -> s=0.8 |
| --- | ---: | ---: | ---: | --- |
| winner (PMA + MeanPt) | 0.442 | 0.348 | 0.219 | 0.915 -> 0.842 |
| anchor (organisers' stock) | 0.500 | 0.022 | 0.021 | 0.861 -> 0.772 |

**Neither statistic explains which model wins.** The anchor is *more* coherent and moves far *less*
along its own discriminative direction, yet loses more AUC. The hypothesis that only the
along-probe component costs AUC is not supported by these numbers.

The absolute column rules out the obvious confound: it is scale-free (each model's own clean-cloud
radius) but magnitude-preserving, and the anchor is ~10x smaller there too, so the small fraction
is not an artefact of normalising by a large displacement.

**The remaining explanation is untested.** The probe direction here comes from a logistic
regression on the clean latents, a linear stand-in for `eval.py`'s four-layer MLP, which has no
single normal. If the anchor's latent is less linearly separable, that direction is a poor proxy
for what the MLP actually uses and 0.022 would be a property of the proxy rather than of the
encoder. Testing it means comparing each model's linear-probe clean AUC against the MLP's, which
was not run. Both numbers are therefore **descriptive only**; WP-A's drift decomposition rests on
WP-A's own method, not on these.
