# WP-L Phase 1 table

**Reference** `h-ref-small` mean_area **0.8262 ± 0.0030** (seed std, n=3), clean 0.9220 ± 0.0033.
Bar for a 3-seed arm ≈ 0.0074. Screening rule: seed-11 must reach ≥ 0.8263 to earn further seeds.

## mean_area rows — FINAL

| arm | seeds | mean over seeds ± seed std | per-seed probe std | clean | verdict |
|---|---|---|---|---|---|
| **reference** `h-ref-small` | 3 | **0.8263 ± 0.0024** | — | 0.9220 | — |
| **`l1-sincos`** | **3** | **0.8270 ± 0.0021** | 0.0006 / 0.0009 / 0.0007 | 0.9202 | **Δ +0.0007, threshold 0.0055, ratio 0.12 → NOT SEPARABLE, tie** |
| `l2-dxysig` | 1 | 0.8278 (s11 only) | 0.0008 | 0.9266 | seeds queued, will not make the stop |
| `l3-ema` | 1 | 0.8260 (s11 only) | 0.0007 | 0.9228 | below the 0.8263 screen, no seeds |
| `l4-valseed` | 1 | 0.8244 (s11 only) | 0.0008 | 0.9224 | below the screen, no seeds |
| `l6-classw` | — | not run | — | — | Phase 2 only (small file exactly balanced) |

**L1 is the only completed three-seed arm and it ties the champion.** No WP-L arm is promotable.

### The screen was noise, demonstrated on my own arm
L1's seeds: **0.8289, 0.8280, 0.8241** — spread 0.0048, comparable to the reference's own 0.0057.

Seed 11 alone cleared the screen by +0.0026 and looked like the best arm of the night. The completed
three-seed mean is **+0.0007**, ratio 0.12 against the threshold — twelve percent of the way to separable.
Before s33 landed I projected that it would need to score ≈0.8373 for L1 to separate; it came in at **0.8241**.

This is the cleanest illustration available of why one seed proves nothing, and it happens to be my own arm:
had the campaign promoted on the seed-11 screen, it would have shipped a change that is statistically
indistinguishable from the champion.

### What the standard is really saying
With a reference seed std of 0.0024–0.0030 on n=3, the separability threshold is ≈0.0055 **whatever the arm
does** — even an arm with zero seed variance needs to beat the champion by 0.0052. Nothing tonight is within
a factor of seven of that. **Measurement precision, not arm quality, is the binding constraint**, and no
amount of further arms fixes it at n=3.

## L4's primary result (not a mean_area row)

Per the 04:14 ruling, L4 is measured on the selection criterion, not on a 3-seed variance test.

| | mean \|Δ val loss\| (last 10 ep) | new best / 25 | early stopped |
|---|---|---|---|
| seeded validation (3 runs) | 0.001893 / 0.001966 / 0.002161 | 25, 23, 24 | 0 of 3 |
| unseeded (5 WP-L runs + champion) | 0.005497 – 0.007445 | 16 – 18 | 1 of 6 (+1 of 3 refs) |

Groups do not overlap on either metric; exact permutation **p = 1/35 = 0.029** one-sided; ratio 3.17×.
Unseeded epoch-to-epoch movement (~0.0064) sits at the independently measured corruption noise (**0.0059**,
8 draws on frozen champion weights): the criterion was largely reading its own draw.

Four different arms make up the unseeded group, so the val seed is the operative variable, not anything
arm-specific.

**Why not the 3-seed std** (F(2,2)): it needs a **6.2×** std ratio at α=0.05, and a 3-seed std has a 95% CI
of [0.52s, 6.28s] — the reference's 0.0030 is consistent with [0.0016, 0.0189].

## The budget effect, with a matched-seed pair

| configuration | seed | validation | outcome |
|---|---|---|---|
| WP-H champion reference | 22 | unseeded | early stopped, epoch 23 |
| WP-L `l1-sincos` | 22 | unseeded | early stopped, epoch 23 |
| WP-L `l4-valseed` | 22 | **seeded** | ran full 25 |

Both early stops tonight are seed 22, in two different configurations; the same seed under seeded validation
ran to completion. Part of "seed noise" is therefore a **deterministic property of the seed's corruption
sequence**, reproducible across configs — matched-seed comparisons cancel part of it, unmatched ones do not.

Not additive with the best-vs-last cost (WP-H): both read one criterion, so one bad streak truncates the
budget *and* leaves the kept checkpoint older. s22 stopped at 23 and kept epoch 17.

## Best-vs-last on the champion — and a correction to what it measures

| reference | best-val | last epoch | gap | threshold | verdict |
|---|---|---|---|---|---|
| s22 | 0.8296 ± 0.0005 (ep 17) | 0.8277 ± 0.0006 (ep 22) | **+0.0019** | 0.00105 | **separable** |
| s11 | 0.8255 ± 0.0009 (ep 23) | 0.8245 ± 0.0013 (ep 24) | +0.0010 | 0.00212 | tie |
| s33 | — | — | — | — | excluded: best-val epoch **is** its last |

Mean gap over the two measurable pairs **+0.0015**.

**Correction to my own framing.** I proposed this measurement as "what the selection lottery costs", and it
is not that. It compares **selection against no selection** — best-val epoch against simply taking the final
epoch. Both gaps are positive, so the criterion carries real signal *despite* its noise; taking the last
epoch instead would be worse. Isolating the *noise's* cost would require the same run selected both ways,
which is impossible: a run has one validation stream, and seeding it changes which epochs exist to choose
between.

**What the pair does support**, and this is the number worth carrying: checkpoints that the criterion treats
as near-tied differ by **0.001–0.002 in mean_area**. So which of them the noise lands on moves a run's
result by that order — comparable to the ±0.0030 seed floor the campaign is calibrated on, and a fifth to a
quarter of the 0.0074 bar arms are judged against.

**Consequence for L4, stated against my own arm.** The evidence shows *variance*, not bias: on s22 the noise
landed on a checkpoint that was separably **better**. Seeding removes the variance and there is no bias for
it to remove, so L4 should not be expected to raise the mean — and `l-l4-valseed-s11` at 0.8244 duly sits
slightly below the reference. **L4 buys reproducibility, not accuracy.** It is worth having for the
measurement standard; it is not a champion-beating change.

## Health warning for the Phase 1 table as a whole

`rc` is not a completion signal. Three of my benches reported `rc=0` having never run (two drained, one
killed to break a scheduler deadlock). **The only reliable check is whether the JSON appeared in `runs/`.**
Any table assembled from wrapper logs will contain rows that silently do not exist. Now in the README.
