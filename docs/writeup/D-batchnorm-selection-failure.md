# A validation-time failure that shipped an epoch-1 encoder (WP-D)

A bug worth recording because it produced a run that looked healthy, trained a perfectly good
encoder, and then selected the wrong checkpoint — with nothing in the training log flagging it
except a number that is easy to read past.

## Symptom

`d-deepsets-twoview` early-stopped at epoch 6 of 25 and saved its **epoch 1** checkpoint. In the
log, training accuracy rose normally to 0.65 while validation accuracy sat at exactly **0.2500** —
chance for four balanced classes — on every epoch, and validation cross-entropy rose monotonically
(1.42 -> 2.09) while training CE fell (1.24 -> 0.84). That pattern reads like overfitting. It was
not.

## Diagnosis

Loading the checkpoint and evaluating the validation split directly, in eval mode and with
BatchNorm on batch statistics:

| arm | eval-mode acc | train-mode acc | eval proj-emb std |
|---|---|---|---|
| aug (single view) | 0.7356 | 0.7288 | 0.1469 |
| clean (single view) | 0.7710 | 0.7720 | 0.1643 |
| **two-view** | **0.2361** | 0.6641 | 0.0288 |

In eval mode the two-view classifier predicted **one class for all 4096 events**. The latent's
per-dimension standard deviation was identical between the two modes, so the encoder was fine: the
collapse was entirely in the projector-and-classifier head, via the `Projector`'s `BatchNorm1d`
running statistics.

## Mechanism, and why it is not "two-view" as such

The trigger is not the size of the offset. It is the **running-mean error measured in units of the
running standard deviation**, at the projector's first BatchNorm:

| | latent offset | latent spread | BN input std | mean error / running_std |
|---|---|---|---|---|
| two-view (broken) | 4.783 | 0.326 | 0.078 | **8.42** mean, 31.67 max |
| aug single-view (healthy) | 3.364 | 1.861 | 0.484 | 0.30 mean, 0.74 max |

The offsets are comparable. The *spreads* differ six-fold, so the same absolute lag in
`running_mean` is worth twenty-eight times more in sigma. Normalising by statistics that are eight
sigma wrong saturates the downstream GELU stack, every event lands on the same embedding direction,
and `argmax` collapses.

The two-view consistency term is what compresses the latent — pulling `z_d` onto `z_c` shrinks the
per-event spread — on an encoder whose latent was already compact. WP-C's transformer, whose latent
spread is 3.03, never hit this, and a synthetic reproduction that shifted the latent by +90 without
compressing it did not reproduce it either. **The prediction this yields is uncomfortable: the
failure gets worse as the consistency term succeeds.** Better invariance means a more compact
latent, means fewer absolute units per sigma. The arms most likely to break are the ones working
best.

## Fix and confirmation

WP-C's `bn_batch_stats()` runs the projector's BatchNorm on batch statistics during two-view
validation, with momentum forced to 0 so a validation pass cannot write into the running stats.
Confirmed on the failing checkpoint before committing to a rerun, because the synthetic case did
not exhibit the bug and so could not demonstrate the repair:

```
eval (running stats)    emb std 0.0285   acc 0.2361   preds [0, 0, 8192, 0]
batch stats (the fix)   emb std 0.1976   acc 0.6517   preds [1499, 2952, 2155, 1586]
running_mean unchanged by the pass: True
```

Ruled default-on for all runs from the Phase-2 merge.

## What it did and did not affect

**The corrupted signal never touched the shipped representation.** The grader embeds with the
encoder and fits its own probe on the latent; it never calls the projector or the classifier. Every
benchmark number reported by WP-D stands. What the bug corrupted was validation loss, and therefore
early stopping and checkpoint selection — a model-selection failure, not a measurement failure.

## Generalisation

This belongs with two other bugs found the same evening, all of which **fail by looking like they
work**: a checkpoint glob (`sorted(glob("*.pth"))[-1]`) that silently selects `_last.pth` over the
best-validation file, and an acceptance test that compared an implementation against itself and
passed vacuously. In each case the failing component produced a plausible number rather than an
error. The common defence is to check the thing you are relying on against an independent
measurement — a shuffled-pair control, a batch-statistics comparison, an explicit path audit —
rather than trusting that a green run is a correct one.

---

# Did the silent-failure register actually work? (WP-M, campaign 2)

WP-N declined to put this claim in the register itself, on the grounds that the register asserting
its own value is not evidence. It belongs here instead, with the instances named so a reader can
check them.

**Four things I caught in campaign 2 because I had read the register first and recognised the
shape.** In each case the register did not contain the bug — it contained a bug of the same
*family*, which is what made the new one visible.

1. **The seed table read `n=1` for every configuration.** `latest_per_tag()` collapsed rows before
   the seed section saw them, so three seeds filed under one tag became one row. I went looking for
   it because register class 1 is "never group on one field; treat ambiguity as do-not-group", and
   grouping was exactly what I had just written.
2. **`compare_arms --seeds` silently mixed data regimes**, comparing a full-file candidate against
   a small-file reference and printing a confident ratio. Same family as 1, found by the same
   reflex: I asked what my grouping key did *not* contain.
3. **My suite watcher would have re-triggered on its own output forever** — an `-l1t` row has the
   same five scoring families as a dev row, so a families-only test could not tell them apart. The
   register's "two things with one name" entries are why I checked the `data` field rather than
   trusting the family set.
4. **Our held-out `ellipse` had been silently replaced by Group 3's `ellipse`.** This one I did not
   find; the planner did, from the shape of the anomaly. But I recognised the *class* immediately
   and knew to compare `dropped_fraction` — a field recorded by every run and read by nobody —
   which is what identified the substituted corruption to four decimal places.

The honest scoring is three-and-a-half out of four: the register did not find any of these. It
made me suspicious of the right things, in code I had written minutes earlier, which is a
different and more modest claim than "it catches bugs".

**What it did not protect against.** I published two numbers from memory in my first status
message, and the register has an entry for that only *because* I did it. I also stated a root cause
as pinned when my evidence was consistent with two hypotheses, having had a cheap disambiguating
test available and skipped it under time pressure. Reading a register does not make you careful
when you are in a hurry; it makes certain shapes recognisable when you happen to look.
