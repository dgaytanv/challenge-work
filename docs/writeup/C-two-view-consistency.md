# WP-C writeup: two-view consistency training, and the latent-offset pathology

Owner: WP-C (`jovyan-c2`). Covers `src/embedding/training.py`, `src/embedding/loss.py`,
`train.py`, `configs/train_config*.yaml`. Every number is measured and its source named.
Claims that were made and later withdrawn are recorded as such rather than deleted.

---

## 1. Why a second view at all

The grader fits a probe on **clean** latents and applies it, frozen, to **degraded** latents of the same
events. Supervised contrastive learning with class positives cannot express that target: it only asks a
degraded event to land somewhere in its class cluster, not where its own clean latent landed. The stock
objective is therefore blind to the thing being scored.

The two-view loop makes the target explicit. For each batch, build `x_c = x` and `x_d = degradation(x)`,
concatenate on the batch dimension, take **one** forward pass through preproc and encoder so BatchNorm
sees both views together, then split `z_c, z_d`. CE and SupCon run on the full 2B batch; a consistency
term acts on the **latent** (pre-projector), because that is the tensor the grader's probe reads.

Symmetries (phi rotation, eta reflection) are applied **once, before the split**, so both views share
them. Consistency then targets dead-region invariance alone while CE and SupCon still see a rotated event.
(Planner ruling; the alternative — symmetries on the degraded view only — makes the consistency term
solve two problems at once.)

### 1.1 Acceptance
`two_view: false` reproduces stock training **bit-for-bit**: max |Delta CE| = 0.000e+00 over 19 steps
against `git show origin/integration:src/embedding/training.py` under a fixed seed
(`tests/test_two_view_equivalence.py`). This was re-run and still passed after every subsequent change.

## 2. The offset pathology

This is the main finding of the work package, and it began as a reporting error of mine.

### 2.1 The artefact
Run 1 logged `cos(z_c, z_d) = 0.9999` and I reported it as evidence of per-event invariance. WP-D
(`jovyan-bc`) predicted it might be an artefact of an uncentred latent and specified the control. It was.
Measured on run 1's epoch-8 checkpoint, 512 train events:

| | matched `cos(z_c[i], z_d[i])` | shuffled `cos(z_c[i], z_d[perm(i)])` |
|---|---|---|
| raw latent | 0.999933 | **0.999557** |
| centred | 0.8537 | -0.0026 |

Two latents from **different events of different classes** sit at 0.9995. The matched figure is barely
distinguishable from it. True per-event invariance was 0.854, not 0.9999.

### 2.2 The cause
The latent carries a common offset far larger than the per-event spread. Frozen epoch-8 checkpoint:

| sample | `\|\|E[z]\|\|` | spread `mean_i \|\|z_i - mu\|\|` | ratio |
|---|---|---|---|
| train, n=512 | 92.2164 | 3.0273 | 30.46x |
| train, n=4096 | 92.1752 | 3.0784 | 29.94x |
| eval, n=4096 | 91.0261 | 2.6122 | 34.85x |

Sample size accounts for +1.7%; the train-vs-eval difference at matched n=4096 is -15.1%, i.e. genuinely a
dataset effect. **A train-measured ratio flatters by 16%** relative to eval, which is what the grader scores.
Stock, for calibration, is 0.17x on train (WP-A) — the healthy regime is well under 1.

### 2.3 Which term caused it
Only one term in run 1's objective is coupled to a common offset at all. Sweeping an offset added
identically to both views, everything else fixed:

| `\|\|mu\|\|` | uncentred cos loss | centred cos loss | raw MSE | population NMSE |
|---|---|---|---|---|
| 0 | 0.04600739 | 0.04632383 | 0.77048 | 0.08188 |
| 20 | 0.00452682 | 0.04632382 | 0.77048 | 0.08188 |
| 90 | **0.00023913** | 0.04632383 | 0.77048 | 0.08188 |

Both MSE forms and the centred cosine are invariant to eight significant figures. The uncentred cosine
falls by a factor of 190. CE and SupCon are *structurally* blind: the Projector is
`Linear -> BatchNorm1d -> GELU`, and `Linear(z + c)` adds a batch-constant `Wc` that BatchNorm subtracts
exactly — measured max |proj(z+mu) - proj(z)| in train mode is 4.8e-07 at offset 20 and 3.1e-06 at offset 155.

So the uncentred cosine both **rewarded** offset inflation and was **rendered meaningless** by it: one term,
two symptoms. Run 1 weighted it 1.0 against the MSE's 0.1.

### 2.4 Where the offset lives (WP-A's audit, credited)
The offset grew 59x relative to stock, but `||W||_F` grew only 2.0x and `||E[h]||` 5.5x — a product of 11.2x.
The missing factor is **alignment**: `||W E[h]|| / (||W||_F ||E[h]||)` went 0.174 -> 0.949. The bottleneck
rotated to map a common pre-bottleneck direction almost maximally into the latent. That direction exists
upstream (`h`'s own offset ratio 7.56x vs stock 0.50x), so a bottleneck-only fix cannot work. The fix
belongs in the loss.

### 2.5 Run 1's offset trajectory (three post-hoc points)

Run 1 predates the per-epoch geometry logging, and `train.py` overwrites its single best-val checkpoint on
every improvement, so only three frozen checkpoints survive. **This is a limitation of run 1, not a result:**
three post-hoc points cannot distinguish a plateau from slow growth plus noise.

| epoch | `\|\|E[z]\|\|` | spread | ratio |
|---|---|---|---|
| 8 | 92.216 | 3.027 | 30.46x |
| 17 | 154.880 | 4.125 | 37.54x |
| 25 | 153.919 | 4.236 | 36.31x |

Inflation happened between epochs 8 and 17 and then **stopped**: the offset is flat from 17 to 25 while
the spread kept growing. Two readings, and the second is the stronger one:

* The plateau is *consistent with* the uncentred-cosine mechanism saturating — the cosine loss falls as
  roughly `1/||mu||^2` and is flat well before 90, so past ~36x there is no gradient left to extract. But a
  prediction that also matches "nothing further happened" is weak evidence, and I am not presenting it as
  confirmation.
* The run improved on val loss for **eight more epochs while the offset sat still**. If offset inflation
  were producing the val-loss improvement, the two would move together. They did not, which decouples them
  more cleanly than continued growth would have.

Run 2's per-epoch curve is what would settle it, with a falsifiable prediction: the centred cosine is
exactly offset-invariant (section 2.3), so `lat_offset` should sit near stock's scale from epoch 1 rather
than rising and plateauing. **If it rises anyway, the uncentred cosine was not the driver.**

> **This prediction is UNTESTED.** Run 2 was cut at convergence — it could not affect the ruling and the
> card was needed for the submission check. Everything needed to run it is pinned and on the branch:
>
> ```
> cd ~/rt-c && ~/hackathon-shared/gpu_run.sh python -u train.py \
>   --data_cfg configs/data_config_collide1m_small.yaml \
>   --train_cfg configs/train_config_c_twoview_ccos1_nmse.yaml \
>   --data ~/hack-data/C9_robust_tagging/train/robust_tagging_train_data_small.pt \
>   --outdir checkpoints
> ```
>
> That config has `consistency_cos_centered: true`, `consistency_mse_normalized: true`,
> `val_bn_batch_stats: true`, 25 epochs at batch 256. The run logs `lat_offset` and `lat_spread` every
> epoch, so the prediction can be read directly off the training log without any extra tooling — compare
> against run 1's three post-hoc points in the table above (30.46x, 37.54x, 36.31x) and against WP-A's
> stock reference of 0.17x on the same convention. **This is the first thing to run next.**

## 3. The fixes

| change | what it does | commit |
|---|---|---|
| centred cosine | centre both views by the **clean** batch mean (detached) before the cosine | `62a5c8e` |
| population-normalised MSE | measure displacement in units of the clean batch's per-dimension variance | `b362571` |
| shuffled-pair control | `cos_shuf` logged every epoch; a matched figure is void unless shuffled sits near 0 | `62a5c8e` |
| geometry logging | `lat_spread`, `lat_offset` per epoch | `b9c78d0` |
| JSD logit consistency | penalise displacement where the probe can see it; default off | `b9c78d0` |

Centring by the **clean** mean (not each view's own mean) is deliberate: a between-view shift `t` survives
in the degraded branch and is still penalised. Per-view centring would remove `t` exactly and is a trap.

The population-normalised MSE removes a second, unplanned problem: raw MSE scales with `||z||^2`, so as the
latent norm grew, that term's *effective* weight crept up through the run — 22% of the val-loss rise that
was accumulating early-stopping patience between epochs 10 and 12 came from it.

### 3.1 The validation BatchNorm failure (`val_bn_batch_stats`)

WP-D found that a two-view run had early-stopped at epoch 6 and shipped an **epoch-1 encoder**. The
encoder was fine; the *validation head* was broken. In eval mode the classifier predicted one class for
every validation event (`preds [0, 0, 8192, 0]`, acc 0.2361, projected-embedding std 0.0285).

The failure quantity is **(running_mean error) / (running_std)**, not the offset magnitude:

| | latent offset | latent spread | BN input std | mean error in units of running_std |
|---|---|---|---|---|
| two-view (broken) | 4.783 | **0.326** | 0.078 | 8.42 mean, 31.67 max |
| aug single-view (fine) | 3.364 | 1.861 | 0.484 | 0.30 mean, 0.74 max |

The offsets are similar. What differs is the **spread**: the two-view latent is compact, so the same
absolute lag in BatchNorm's running average is worth 8.4 sigma instead of 0.3. Normalising by statistics
that far off saturates the downstream GELU stack, every event lands on the same embedding direction, and
argmax collapses. Val loss, val AUC, early stopping and checkpoint selection are all corrupted together.

`bn_batch_stats()` runs the projector's BatchNorm on **batch** statistics during validation, with momentum
forced to 0 so a validation pass cannot write into the running statistics. Verified on WP-D's failing
checkpoint: embedding std 0.0285 -> 0.1976, acc 0.2361 -> 0.6517, predictions
`[1499, 2952, 2155, 1586]`, running stats bit-identical across the pass.

**It is default-on for every run, not gated on `two_view`.** Gating it would switch the fix off exactly
when the method starts working: the better the invariance, the smaller the spread, the more sigmas any lag
is worth. A fix that disengages as the method succeeds fails silently at the moment you would most trust
the result. Cost on a healthy model (run 1, spread 4.236): val acc 0.7910 -> 0.7832, embedding std
0.2887 -> 0.2884. `mean_area` is untouched in all cases — the bench scores **encoder latents** and never
runs the projector, so this changes which checkpoint is *selected*, never how one is *measured*.

Run 1 was checked for the signature and is clean: val Acc 0.5366 -> 0.7468 across 25 epochs, never within
0.02 of 0.25, val AUC 0.7635 -> 0.9205 monotone. Its latent spread **grew** (3.03 -> 4.24) where WP-D's
shrank to 0.326, which is why the transformer never hit this.

*Credit: diagnosis and the sigma argument are WP-D's; the fix and the healthy-model cost measurement are mine.*

## 4. Two translations, which are not the same thing

A shared offset present in **both** views (the model's own) and a shared shift **between** views (what
degradation does to the whole cloud) are different quantities, and the loss terms treat them oppositely:

| | raw MSE | population NMSE | centred cos | uncentred cos |
|---|---|---|---|---|
| offset in both views | invariant | invariant | invariant | **rewards it** |
| shift between views, `\|\|t\|\|=2` | 0.66667 | 0.07775 | 0.044795 | 0.044676 |

"Translation-invariant" refers to the first row only. A between-view shift is the *easiest* thing an MSE
can penalise — it contributes `||t||^2` with no cancellation.

### 4.1 A rationale that did not survive contact with the data

I proposed a reweight arm (`consistency_weight` 0.1, `consistency_mse_weight` 1.0, config
`train_config_c_twoview_mse1_ccos0.1.yaml`, commit `a39d3b2`) on the argument that the degradation
displacement is "51-99% rigid shift", so the MSE is the term matched to the dominant failure mode. That
argument is **not supported** and the arm has been relabelled.

* The rigid fraction is not one number. It is measured per (family, severity) and rises steeply with
  severity: 0.18 at s=0.2 to 0.99 at s=1.0 on `d-deepsets-clean` (WP-A). "Mostly rigid" describes high
  severity, not the model. Since the bench area is a **uniform** average over the severity grid, a term
  helping mainly at high severity helps only in those cells.
* At s=0.5 the rigid component is the **less harmful** half — 3.6x less, with the per-event residual alone
  worse than the full degraded latent (WP-D). Weighting up a term *because* it penalises the rigid
  component would weight up a penalty on the component that costs less AUC. Whether this still holds at
  s=1.0, where the rigid part is 99% of the drift, is **not measured** and must not be assumed.
* Comparing a grid-averaged rigid fraction between two models is only fair when their AUC-vs-severity
  curves are similarly shaped, because the curve determines which cells dominate the average.
* Most importantly: the reweight was a **repair for a vacuous cosine**, and run 2 already repairs that by
  centring. With both terms carrying real gradient it is no longer a fix but an ordinary hyperparameter
  question — displacement magnitude versus per-event direction.

What survives is the narrower claim that never depended on the rigid fraction: **run 1 put weight 1.0 on a
term with no gradient and 0.1 on the term doing the work.** The arm is retained on WP-D's lock-free chain
as a hyperparameter comparison; the transformer twin is held pending WP-A's decomposition of run 1 and is
dropped outright if run 1's rigid fraction falls below the threshold WP-A pre-committed to (0.45).

## 5. Results

Run 1 (`c-twoview-cos1`) is the **raw / uncentred** ablation row: it is the run that exposed the offset
pathology, and it is not a submission candidate. Its config is pinned (`consistency_mse_normalized: false`,
`consistency_cos_centered: false`) so the row stays reproducible.

| | value | source |
|---|---|---|
| epochs | 25/25, best at the final epoch | training log |
| val loss / val AUC | 0.992114 / 0.9205 | training log, epoch 25 |
| bench `mean_area` (20k, R=5) | **0.7712 +- 0.0034** | `runs/c-twoview-cos1_20260903_220316.json` |
| bench clean AUC | 0.8370 +- 0.0036 | same |
| per family | rect 0.6824, wedge 0.8176, strip 0.7481, towers 0.7876, cells 0.8204 | same |

**This row is a regression and is reported as one.**

| comparison | delta | reading |
|---|---|---|
| vs stock anchor 0.7734 | **-0.0022** | inside probe noise — a tie, not a gain |
| vs mask fix on frozen weights 0.7918 | **-0.0206** | ~6 sigma worse |
| clean AUC vs 0.8605 anchor | **-0.0235** | outside the 0.02 tolerance of standing ruling 3 |

A full retraining with WP-B's generator and the two-view loop came out **worse than applying the mask fix
to the stock checkpoint and not training at all**, and it fails standing ruling 3 on both halves: clean AUC
fell more than 0.02 and `mean_area` did not gain to pay for it.

What it does and does not show. Run 1 is the arm whose dominant consistency term was vacuous — the
uncentred cosine contributed ~4e-5 of loss with essentially no gradient from epoch 7 on, while the MSE
carried weight 0.1. So this row is close to "stock training + WP-B's augmentation + a 2B batch through
BatchNorm", with the invariance objective largely absent. It is evidence that *that* combination does not
beat the free mask fix. It is **not** evidence about two-view consistency as a method, because the method
was not effectively running. Run 2 is the row that would test the method; it is an ablation row queued
behind three jobs and may land after the ruling.

Run 2 (`c-twoview-ccos1-nmse`) and run 3 (the reweight arm, section 4.1) were both **cut** at convergence.
Run 2 was the row that would have tested whether the centred cosine fixes the offset pathology; it could
not affect the ruling and the card was needed for the submission check. The exact command to run it is in
section 2.5, and it is the first thing to run next. **So this work package ships one measured row — a
negative one — plus the fixes that WP-D's candidate arms depend on.**

### 5.1 What this work package contributed to the candidates
The loop itself is the deliverable that matters. WP-D's two-view Deep Sets arms run on this code: the
two-view loop, the centred cosine, the population-normalised MSE and `val_bn_batch_stats`. The last of
those turned a run that shipped an epoch-1 encoder into one that trains to completion, so it is load-bearing
for any two-view candidate in the final table rather than for a row of my own.

### 5.2 The regression is one family, and it is a probe inversion

Reading `mean_area` alone hides what happened. Per-family areas, verified from the JSONs:

| family | run 1 | vs anchor | vs mask fix |
|---|---|---|---|
| rect | 0.6824 | **-0.0458** | **-0.0745** |
| wedge | 0.8176 | +0.0194 | +0.0215 |
| strip | 0.7481 | -0.0003 | -0.0385 |
| towers | 0.7876 | +0.0139 | +0.0031 |
| cells | 0.8204 | +0.0020 | -0.0143 |

Run 1 starts 0.0234 **below** both baselines at s=0 (clean AUC 0.8370 vs 0.8604/0.8605) — that deficit
applies to every family equally — and then loses *less* with severity than the anchor in four of five.
Against the anchor it ends ahead at s=1.0 in wedge (+0.094), cells (+0.073), strip (+0.045) and level in
towers. So the loop did buy robustness; it paid for it out of clean AUC.

The exception is **rect**, and it is not a gentle loss: `rect` at s=1.0 reads **AUC 0.3901**, which is
*below chance*. The probe is not merely uninformed there, it is **inverted**. Excluding rect, run 1's
mean over the other four families is 0.7934 against the anchor's 0.7847 (+0.0087) and the mask fix's
0.8005 (-0.0070).

Two honest qualifications on that:

* Excluding a family because it went badly is not a legitimate score. The 0.7934 figure is diagnostic
  only — it says *where* the loss is concentrated, not what the run is worth.
* Which baseline you use changes the story. Against the **anchor** run 1 wins three families and ties one.
  Against the **mask fix**, which is the stronger baseline and free, it wins only wedge and towers and
  still loses overall. The mask fix remains the thing to beat, and run 1 does not beat it.

The `rect` inversion is a specific, diagnosable failure rather than a diffuse one, and it is the single
largest contributor to the gap. It is not explained by anything in this section and is left as an open
item.

### 5.3 Incidental: evidence on the wedge collapse

The mask fix applied to the untrained-for-it stock checkpoint collapses at `wedge` s=1.0 (AUC 0.4829).
Run 1, which was **trained with** the mask fix in `models.py`, reads 0.7763 at the same point — no
collapse. That supports WP-E's hypothesis that the wedge collapse is a train/eval mismatch rather than
anything intrinsic to masking. It is **not** the controlled test: run 1 also changed the objective and the
generator, so `e-stock-maskfix` remains the clean comparison. *(Observation credit: WP-F.)*

### 5.4 A protocol explanation I proposed, and WP-E's control refuted

**Retracted.** I argued that part of run 1's clean-AUC deficit was the 25-epoch schedule, on the evidence
that run 1 set its best val loss on **epoch 25 of 25** and never triggered patience-5, with train AUC still
rising. That fact is true and still stated here. The inference from it was wrong.

WP-E's control ran the *identical* protocol — 25 epochs, batch 256, mask fix, **no augmentation**:

| row | mean_area | clean AUC |
|---|---|---|
| **E control (25 ep, mask fix, no aug)** | **0.8089 +- 0.0004** | **0.9025** |
| d-deepsets-clean (full) | 0.8017 | 0.8887 |
| mask fix on frozen 60-epoch weights | 0.7918 | 0.8604 |
| organisers' 60-epoch stock anchor | 0.7734 | 0.8605 |
| C run 1 (25 ep, aug, two-view) | 0.7712 | 0.8370 |

A 25-epoch transformer beats the organisers' 60-epoch checkpoint on both metrics. The schedule therefore
does not handicap this architecture, and it cannot account for run 1's 0.0655 clean-AUC deficit against a
control that ran the same schedule.

The error is worth naming because it was seductive: I read a **within-run** trajectory (still improving at
the end) as a **between-run** explanation (therefore behind the others). Those are different claims, and
the second needs a control, which E supplied. The corollary I also offered — that the transformer's larger
late-epoch gain (+0.0135 against the set encoders' +0.006) showed it was the more handicapped architecture
— fails identically: E's control gained +0.0138 after epoch 15, the same as run 1, while finishing 0.0655
higher. **Late-epoch gain measures nothing about how converged or how handicapped a run is.**

What remains is that run 1's deficit is a cost of the augmentation, the two-view 2B batch, or their
combination. `b-aug-stock` (single view + generator) is the row that separates those two.

### 5.5 The prediction was tested and FAILED — and that is the best result here

WP-D's corrected two-view Deep Sets run (v2: centred cosine, population-normalised MSE,
`val_bn_batch_stats`, 25 epochs, no early stop) is a test of section 2.5's prediction. I predicted that
because the centred cosine is *exactly* offset-invariant, `lat_offset` would stay near stock's scale.
**It did the opposite.** From `~/rt-d/logs/training_20260903_221016.log`:

| epoch | offset | spread | ratio | cos_val | cos_shuf_val | drift/spread | val AUC |
|---|---|---|---|---|---|---|---|
| 1 | 2.97 | 0.195 | 15x | 0.9038 | +0.011 | 0.233 | 0.8425 |
| 5 | 480.1 | 2.492 | 193x | 0.8674 | +0.065 | 0.253 | 0.8501 |
| 15 | 1424.1 | 4.688 | 304x | 0.8799 | +0.102 | 0.264 | 0.8692 |
| 25 | **2323.96** | 3.883 | **599x** | 0.8766 | +0.086 | 0.269 | 0.8670 |

The offset grew **780-fold** to a ratio of 599x — **sixteen times worse than run 1's 36x**, with the term
I had identified as the driver removed. So:

**The section 2.3 conclusion is wrong.** The uncentred cosine has a real gradient incentive to inflate the
offset (the 190x sweep stands as a measurement), but it is **not necessary** for the inflation. Removing it
made things worse, not better. What actually drives it remains unidentified.

**What is confirmed is the weaker, correct account — WP-A's original one, which I talked them out of.**
A said: an MSE consistency term is invariant to a common translation, so *nothing penalises* the offset.
I objected that indifference explains why nothing stops it, not why it grows, and went looking for a term
that actively rewards it. I found one, and it turned out not to be the cause. Indifference was sufficient:
both MSE forms and the centred cosine are blind to a uniform translation **by construction**, so no
gradient opposes it and any drift compounds freely. A had it right first.

**Why this is the most useful negative result in the package: the failure is invisible to every
training-side metric.** At epoch 25 everything the loop can see is healthy — cosine 0.8766 against a
shuffled control of +0.086 (so the invariance is real, not an artefact), drift/spread flat at 0.27 across
the whole run, val AUC 0.8670 and rising. Only the raw `lat_offset` shows anything wrong. A run in this
state looks converged and well-behaved from inside.

The damage appears downstream, at the probe. `EvalMLP` fitted on latents with `||E[z]||` = 2324 and a
spread of 3.9 is numerically ill-conditioned — it sees near-constant inputs carrying all the signal in the
sixth significant figure. That is the mechanism behind the row's bench numbers: clean AUC 0.7804 and
`mean_area` 0.7419 +- **0.0156**, a probe refit sigma five times any other Deep Sets row. The offset
inflation *is* the degenerate-latent failure.

**The fix this argues for**, untested: a direct penalty on `||mean(z)||` over the batch, or a non-affine
latent normalisation, i.e. something that is *not* blind to a uniform translation. Every consistency term
we ran tonight was translation-invariant by design — that property was chosen deliberately (section 4) to
avoid measuring the model's own offset, and the unintended consequence is that nothing in the objective
constrains it at all.

**Two negative rows, two architectures, bugs removed.** Transformer `mean_area` 0.7712 against 0.8089 for
the same architecture trained single-view; Deep Sets 0.7419 +- 0.0156 against 0.8174 for the identical
encoder and generator single-view. The two-view objective as run tonight hurts on both.

## 6. Claims made and withdrawn

Recorded rather than deleted, because several appeared in messages and reports others were reading.

1. **"cos_val 0.9999 shows per-event invariance"** — void. Uncentred cosine on an offset-dominated latent.
   Every `cos` figure I reported before commit `62a5c8e` is void. Caught by WP-D's shuffled-pair control.
2. **"relative drift 4.24%"** (`||z_d - z_c|| / ||z_c||`) — withdrawn. Offset-deflated denominator; the
   probe-relevant figure is drift over population *spread*, 0.3000. Metric replaced by `drift_spread`.
3. **"an MSE consistency term can be satisfied by inflating a shared offset"** — wrong. MSE is exactly
   translation-invariant. Corrected by the planner; the offset incentive belongs to the uncentred cosine.
4. **"the offset is the robust quantity, the spread is what moves"** — false in general. True of this model
   because its offset is dominant; stock's offset is a small residual varying 2.8x between datasets.
   Stability is a symptom of the pathology, not a property of the metric. Caught by WP-A.
5. **ep8 -> ep17 prediction** — half wrong. Offset rose 68% as predicted (92.216 -> 154.880); the spread did
   **not** collapse, it grew 36% (3.027 -> 4.125). Between those epochs the pathology is purely offset
   inflation.
6. **"the drift is mostly rigid, so weight the MSE up"** — rationale withdrawn (section 4.1). The rigid
   fraction is severity-dependent, and at mid severity the rigid component is the *less* harmful one.
7. **"part of run 1's clean-AUC deficit is the 25-epoch schedule"** — refuted by WP-E's control, which ran
   the same schedule and reached clean AUC 0.9025 against run 1's 0.8370 (section 5.4). I read a within-run
   trajectory as a between-run explanation. The related claim that the transformer is the more
   schedule-handicapped architecture fails the same way.
8. **"the uncentred cosine drives the offset inflation"** — refuted (section 5.5). Removing it made the
   inflation 16x worse. The 190x incentive sweep is still a valid measurement; the causal claim is not.
9. **A gradient probe on a shared bias parameter** — discarded, not quoted. Badly constructed: the detached
   clean branch still carried a zero-valued gradient path, so it could not distinguish the variants.

## 7. Two failures that pass

Two defects found in this package share a shape worth naming, because neither produces an error:

* The **acceptance test** compared `train_epoch` against `origin/integration`'s. Once wp-c merged there,
  that reference would contain the two-view loop and the test would compare the implementation against
  **itself** and pass forever. It now raises if the reference already contains `two_view`.
* The **checkpoint glob**: `_bestauc.pth` and `_last.pth` matched `<name>_encoder_*.pth`, and `_last` is
  rewritten every epoch, so `ls -t | head -1` always selected it. Worse, the organisers' notebook uses
  `sorted(glob("checkpoints/*.pth"))[-1]`, and `"_last.pth"` sorts after `".pth"` — at judging time it
  would have scored our last epoch. Auxiliary checkpoints now live in `<outdir>/aux/`.

A test that compares an implementation against itself and a glob that selects the wrong file both fail by
**passing**. Neither would have appeared in any log. *(Framing credit: WP-D.)*

## 8. Sources

- Acceptance, sweeps, controls: `tests/test_two_view_equivalence.py`, and the measurements above are
  reproducible from the frozen checkpoints named in `reports/c-*.md`.
- Run 1 training log: `~/rt-c/logs/`, per-epoch table via `tools_cos_table.py` (which prints a warning when
  a log predates the centred-cosine fix).
- Bench JSONs: `~/hackathon-shared/runs/`.
- Reports: `~/hackathon-shared/reports/c-2005.md`, `c-2010.md`, `c-2016.md`.
