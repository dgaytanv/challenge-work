# Generator coverage: the severity a family is asked for is not the severity it delivers

WP-J, campaign 2. Sources: `reports/j-0345.md`, `reports/j-0420.md`, `reports/j-0515.md`,
`bench/plot_augmentation_j.py` and its two figures, and `~/c2-j/src/embedding/degradation.py`
(branch `c2-j` @ `dc45142`).

The package had two questions: does the training generator cover enough kinds of dead region
(coverage), and does it cover enough severity (curriculum and severity mixture). Both were answered,
both answers were negative for the score, and the interesting part turned out to be a third question
nobody had asked — whether the generator delivers the severity it thinks it does.

## The claim the generator was making

`Degradation`'s module docstring states the contract plainly:

> `severity` / the internally sampled `s` is the *target fraction of the eta-phi plane that is
> dead*, and every family is parameterised so the expected fraction of candidates dropped is
> approximately `s`.

Two different quantities are joined there by an "and". The first — fraction of the **plane** — is
what every family actually computes. The second — fraction of the **candidates** — is what the
grader's severity means, what the bench sweeps, and what the training signal is. They coincide only
if the candidate density is uniform over the plane.

It is not.

## Coverage: borrowing another group's geometry

The champion draws from five dead-region families that were all written by one author, so "robust to
dead regions" risked meaning "robust to WP-B's five shapes". Group 3 (`colleague-group3`, branch
`dorian` @ `37168c1`) had independently written a generator with its own five training families, and
arm J1 ported them verbatim behind our interface as `c_rect`, `c_eta_band`, `c_phi_wedge`,
`c_multi_patch` and `c_cand_loss` — an independently designed source of geometry, not a variation on
ours.

Three things about the port are worth recording.

**Their held-out three are not ported and must never be.** `ellipse`, `cell_dropout` and
`edge_truncation` are the families `bench/colleague_degradation.py --families colleague` scores with.
Training on them would destroy the only independent generalisation check we have. Verified: the only
mentions of those names in `src/embedding/degradation.py` are four comment lines saying so.

**One deliberate deviation.** Their families hardcode `eta in [-3, 3]`, an L1T/PUPPI acceptance.
Ported unchanged onto our eta ±5 data they would only ever touch the core, and severity would stop
meaning fraction-of-plane — campaign 1 measured that distortion at up to 0.17 absolute. The shapes
are theirs verbatim; the eta extent comes from `self.eta_span`.

**Every ported family carries a `c_` prefix**, because their suite and ours both contain a family
called `ellipse` and both contain rectangles and wedges. Campaign 1 lost time to that collision and
campaign 2 lost a bench row to it.

The ported families turned out to be *better calibrated than three of our own*, which is what sent
this document in the direction it went.

## The candidate density is edge-peaked, and the edge is where the plane calibration cannot reach

Measured over the eval file, valid candidates only (`pt > 0`), 4000 events:

```
  frac |eta| < 1: 0.197    < 2: 0.342    < 3: 0.454    < 4: 0.699    < 4.5: 0.849    < 5: 0.999
  uniform on [-5, 5]:      0.200         0.400         0.600         0.800           0.900
  max |eta| = 5.000 exactly -- the distribution is clipped hard at the acceptance edge
```

30% of all candidates sit in `|eta| ∈ [4, 5]` and 15% in `[4.5, 5.0]`, against 20% and 10% for a
uniform plane. The median `|eta|` is 3.16. Nothing about a dead-region family placed by plane area
knows this.

The consequence is not a small bias. Fraction of candidates actually dropped against the requested
severity, over 4000 events:

```
family        s=0.2  s=0.4  s=0.6  s=0.8  s=1.0    mean |err|
rect          0.164  0.292  0.395  0.469  0.528      0.231
wedge         0.180  0.319  0.428  0.512  0.578      0.197
strip         0.179  0.311  0.413  0.491  0.548      0.212
cells         0.200  0.401  0.600  0.651  0.650      0.100   <- designed ceiling, see below
towers        0.200  0.401  0.600  0.800  1.000      0.000
```

`rect` asked for a fully dead plane and removed 53% of the event.

## Three mechanisms, not one, and each was measured before it was fixed

The temptation was to find the density argument, fix the placement, and declare victory. The fix
recovered less than half the error, which is how the other two turned up.

**1. Placement.** `rect` and `strip` size their dead regions by plane area, so a region placed
uniformly in eta systematically misses the forward band where the candidates are. Direct evidence —
drop rate by `|eta|` bin, which a family calibrated on candidates would make flat. This is Group 3's
`c_eta_band`, whose centre is constrained to keep the band strictly inside the acceptance:

```
  s=0.4   0-1:0.669  1-2:0.590  2-3:0.414  3-4:0.251  4-4.5:0.127  4.5-5:0.042
  s=0.8   0-1:1.000  1-2:1.000  2-3:1.000  3-4:0.767  4-4.5:0.380  4.5-5:0.125
```

At `s=0.8` it erases the entire core and leaves seven eighths of the forward region alive. The dead
region is *anti-correlated* with the candidate density.

Fixed by placing the eta extent in the batch's **empirical eta-CDF coordinate**: a coordinate of
span 1 in which the candidate density is uniform by construction, so a dead band of width `w` covers
`w` of the candidates. Computed per batch from the data rather than from a stored table, so the same
code is correct on the PF file (eta ±5, edge-peaked) and on the L1T file (eta ±3) with no constant
to keep in sync — the class of bug `--eta_max` existed to paper over in campaign 1.

**2. Overlap.** `rect` and `_drop_bands` scaled their shapes so the *sum* of the areas was `s` of the
plane, then placed the shapes independently, so the union was smaller than the sum. Fixed by
inverting the union instead: `m` shapes each covering `a`, with `1-(1-a)^m = s`, i.e.
`a = 1-(1-s)^(1/m)`. Identical to the old formula at `m=1`, which is why it went unnoticed.

For `rect` the closed form is not enough: its boxes are deliberately heterogeneous (wide aspect
priors), and at fixed total area a heterogeneous set has a *smaller* union than an equal-area set.
It also gets clipped, because eta does not wrap — a full-width box placed off-centre still misses
part of the plane, which is why `rect` sat at 0.870 rather than 1.000 at `s=1`. So the centres are
drawn *before* the scale is solved, and a 24-step bisection finds the scale at which the clipped
extents give the requested union. It remains an approximation: the product treats the boxes as
independent and ignores overlap between the particular boxes drawn.

**3. The `p` ceiling.** A family that kills candidates with probability `p < 1` inside its region can
never remove more than `E[p]` of an event, however wide the region. `strip` sat at exactly 0.746 and
`wedge` at 0.798 at `s=1.0` — their `E[p]` to three decimals. Fixed by ramping `p` to 1 with
severity, which keeps the partial-kill character at low `s` and lets the family reach a fully dead
region at `s=1`.

Result, mean absolute error against the requested severity:

```
  rect   0.231 -> 0.028      wedge  0.197 -> 0.011      strip  0.212 -> 0.031
  cells  0.100 (unchanged)   towers 0.000 (unchanged)
```

## What that cost the champion

The champion's *nominal* severity distribution, simulated over 7025 steps of a 25-epoch run: mean
`s` 0.349, 13.9% of mass above 0.7, 0.0% above `s_max = 0.85`. Its *effective* distribution, from
running the real `_forward_train` and counting rows actually zeroed:

```
  mean effective dropped fraction   0.248    (nominal 0.349)
  mass above 0.7                     2.7%    (nominal 13.9%)
  mass above 0.85                    0.1%
  events losing nothing at all      23.4%    (configured p_clean is 0.15)
```

Two things follow. The champion has essentially never seen an event lose more than 70% of its
candidates, while the grader sweeps severity to 1.0 and the bench families remove 0.94-1.00 at
`s=1.0`. And the 23.4%-against-15% gap is the `p_pt_scale = 0.10` branch, which applies the dead
region as a pt rescale rather than by zeroing rows — so a `p_clean = 0.0` arm still leaves about 10%
of events with nothing removed.

The calibration fix alone, changing nothing about the sampler, moves the effective mean from 0.248
to 0.295 and the mass above 0.7 from 2.7% to 8.6%. **About a third of the effective-severity deficit
was calibration error rather than the severity sampler.**

## The curriculum, and an arm that was dropped rather than run

The package brief included a curriculum arm: remove or double the `s_max` warm-up. It was dropped
before it was run, on a measurement rather than on a hunch. The warm-up ramps `s_max` from 0.2 to
0.85 over `warmup_calls = 600` forward passes, and a 25-epoch run at batch 256 over 72k events is
about **7025** steps — so the curriculum governs the severity seen by **8.5%** of training and is at
its final value for the other 91.5%. Removing or doubling it changes what fewer than one step in
eleven sees.

The prediction was that this would land at or below the pipeline reproducibility floor (~0.002
between independently trained full models, `E-probe-noise-floor.md`) and so could not be resolved by
a single pair of runs. The planner accepted it and spent the slot on J5 instead. Given how the night
ended — J1 needing three seeds to show +0.0014, and J2 and J5 both failing outright — that was the
right trade, though it means the curriculum claim remains a prediction and not a result.

The general point is worth separating from this particular arm: a hyperparameter that is only active
for the first 8.5% of training cannot be ruled on by a pair of runs whose floor is wider than any
effect it could plausibly have, and the cheap way to find that out is to count the steps it touches
before spending the GPU time.

## The denominator, checked rather than assumed

WP-M raised the right objection: `bench_degradation`'s `wedge` has `p_drop = 0.9` and its `cells`
`p_drop ~ U(0.5, 1.0)`, so for those the design target is `p_drop * s` and an error computed against
`s` would be measuring the design. Campaign 1 made exactly that mistake and published a 0.154
"defect" that was 0.007 against the family's own target.

It does not apply to the three families above, and the reason is worth stating because it is the
opposite convention in code that looks the same. The *training* generator absorbs `p` into the width
solve instead of leaving it in the target:

```python
p = self._u((B,), coord, p_lo, p_hi, gen)
m = torch.clamp(torch.round(s * span / w0), min=1, max=m_max)
w = torch.clamp(s * span / (m * p), min=0.0, max=span)      # divides by p
```

`_drop_rect` does the same by folding `p` into the area it scales against. So `p` is a shape prior
the family widens to absorb, and the target is `s`. `cells` is the exception *because it clamps*:
its `frac = s / p_mean` saturates at 1, and above `s = 0.65` the compensation runs out.

The finding survives the other denominator regardless. Mean absolute error against both candidate
targets:

```
  family   p prior       vs s     vs p_mean*s      short of p_mean at s=1.0
  rect     U(0.5,1.0)   0.230        0.086              0.222
  wedge    U(0.6,1.0)   0.197        0.085              0.222
  strip    U(0.5,1.0)   0.212        0.078              0.202
```

Smaller against `p_mean * s`, but still large and still entirely one-sided at the top of the range.

The two readings are the same fact. The `p` compensation is real, which makes the target `s`; it
**fails exactly where `w` hits its `max=span` clamp**, and at that point the family saturates at
`E[p]`. That is why `strip` sat at 0.746 and `wedge` at 0.798 at `s = 1.0`. `p` was involved — as
the ceiling, not as the target.

## `cells` is not miscalibrated, and the reason is a name collision

`cells`' 0.100 above is its *design*, not an error: `_drop_cells_multi` uses `p_lo=0.3, p_hi=1.0`,
so `frac = s/0.65` saturates at 1 and the family's ceiling is `p_mean = 0.65`. It hits
`min(s, 0.65)` to three decimals.

The trap is that **the bench has a family called `cells` too, and it is different code with a
different `p_drop`** — `U(0.5, 1.0)`, so its target is `0.75s`, not `min(s, 0.65)`. Checking one
against the other's target produces a confident, wrong error report. Campaign 1 made exactly that
mistake in the opposite direction and published it before catching it, which is where register class
9 comes from: **validate the reference before reporting an error against it.** The same discipline
applied here is what confirmed that Group 3's five ported families have `p_drop = 1.0` — their
`_spatial_mask` returns a plain boolean and `apply` does `masked_fill`, with no per-candidate
probability anywhere — so their target is `s` itself. That was read out of their source, not assumed
from the comment I had just written saying so.

Group 3's suite and ours also both contain a family called `ellipse`, from different code. Every
ported family carries a `c_` prefix for this reason.

## Mechanism note: a prediction about the colleague families that held

Before the suites were run, WP-J told WP-M in advance where any J1 effect on Group 3's held-out
three should appear: on `c_ellipse` — a filled convex region, closest in kind to the axis-aligned
boxes J1 adds through `c_rect` and `c_multi_patch` — rather than on `c_cell_dropout` or
`c_edge_truncation`. Measured, against the champion at matched seed 11:

```
  family              champion            j1-s11            delta
  c_ellipse           0.8296 / 0.8293     0.8313 ±0.0009   +0.0017 / +0.0020
  c_cell_dropout      0.8052 / 0.8048     0.8054 ±0.0010   +0.0002 / +0.0006
  c_edge_truncation   0.7633 / 0.7638     0.7617 ±0.0009   -0.0016 / -0.0021
```

The prediction held, and the `c_ellipse` gap clears probe noise
(`3*sqrt(0.0009²/5 + 0.0005²/5)` = 0.0014 against a gap of 0.0017).

**It is still not a claim.** These are two independently trained checkpoints, so the floor that
applies is the seed floor (0.0030 std, a 0.0074 bar), not the probe floor, and 0.0017 is nowhere
near it. J1's own three-seed scoring row confirms the reading: +0.0014 against a 0.0059 threshold.
So this is a correct prediction about a difference that is not established — worth recording because
the direction was called in advance, worth nothing as evidence.

Two caveats travel with it, both WP-M's. `c_cand_loss` is independent Bernoulli per candidate, the
fine-granularity limit of their `cell_dropout`, so that row is a weakened check rather than a clean
one — though the champion already trains on `towers`, a Bernoulli field at cell 0.1, so J1 adds
weight there rather than a new capability. And the colleague suite is quoted per family, never as a
suite mean: over our acceptance `edge_truncation`'s two branches cross above `s = 0.5` and
`cell_dropout` saturates to AUC 0.5000 at `s = 1.0`, so a mean averages one self-overlapping family,
one saturating family and one real curve.

## What generalises

The failure is not "someone got a formula wrong". Every one of these families is internally
consistent and each was checked when it was written — against the quantity its own author had in
mind. Two independently developed generators, ours and Group 3's, both wrote code that hits
*fraction of the plane* while the docstring, the bench and the grader all mean *fraction of the
candidates*, and nothing anywhere compared the two until they were measured side by side.

The general form: **when two quantities are equal under an assumption, name the assumption in the
test, not just in the comment.** The assumption here was a uniform candidate density, stated nowhere
and false by a factor of 1.5 at the edge of the acceptance where a third of the data lives. A test
that had ever plotted dropped-fraction against requested severity would have shown it on the first
run. `bench_eval.py` has recorded `dropped_fraction` per family since campaign 1 and the bench
families were calibrated against it then; the *training* generator never was, and the check
costs four lines.

The corollary for anyone reading a robustness number: a training generator's `severity` and an
evaluation sweep's `severity` are two different measurements that happen to share a name and a
range. Whether they agree is an empirical question about the data, and on this data they did not.

## What happened when the miscalibration was fixed: nothing

This is the part that matters most, and it is a negative result.

`on_target_placement` was trained as arm J5 -- one change from the campaign-2 champion, one seed
under the deadline -- and benched **0.8251** against a reference seed mean of **0.8262 +- 0.0030**.
Inside the floor, and on the wrong side of it.

The companion arm went the other way and lost. J2 used the effective-severity measurement above to
push the fraction of training events losing more than 0.7 of their candidates from 2.7% to 30.7%.
It benched **0.8240** with clean AUC **0.9142** against the reference's **0.9220** -- every scoring
family fell, and it gave up 0.0086 of clean AUC to do it. The measurement said the champion trains
at a much lower effective severity than the grader scores at, and that was true. The inference --
that closing the gap would help -- was not.

J1, which added Group 3's five training families, came in at **0.8277 +- 0.0017** over three seeds
against 0.8262 +- 0.0030: delta +0.0014 against a threshold of 0.0059. Its seed 11 had looked like a
+0.0044 gain concentrated in `rect`, exactly the family the mechanism predicts; the matched-seed
deltas across the three seeds turned out to be +0.0044, -0.0030, +0.0029.

So the honest summary of this document is:

**The generator was miscalibrated by 0.20-0.23 mean absolute error in three of its five families,
the diagnosis is reproducible and the mechanisms are understood, and correcting it did not move the
score.** A correct diagnosis of a real defect is not the same as a defect that mattered.

Three things are worth keeping anyway. The measurement itself, because `severity` will keep being
quoted as though it were the fraction of candidates removed, and on this data it is not. The
negative result on J2, because "train on harder corruptions" is the obvious next idea for anyone
reading the robustness brief and it costs clean AUC here. And the failure mode that nearly carried
J1: a single-seed per-family breakdown that lands on the family your mechanism predicts feels like
the mechanism confirming itself, when a single seed produces some largest-delta family whatever the
truth is, and a story for whichever one it is costs nothing to write.
