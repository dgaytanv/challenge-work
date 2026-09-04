# WP-K: the consistency objective with the offset fix, and what the probe actually needs

Owner: WP-K (campaign 2). Covers the consistency terms in `src/embedding/loss.py` and
`src/embedding/training.py`, the latent-geometry diagnostics, and `PMAEncoderNormLatent`.
Every number is measured and sourced. Claims made and withdrawn are recorded as withdrawn.

**Headline: no arm in this package is promotable. Two of the three measured arms are separable
regressions, and the most useful thing the package produced came out of one of them.**

---

## 1. The premise, and why campaign 1 left it open

The grader fits a probe on **clean** latents and applies it, frozen, to **degraded** latents of the
same events. A loss that pulls `z(degraded)` toward `z(clean)` targets that directly, which is why
campaign 1 tried it. It failed for a reason campaign 1 diagnosed but never fixed: the latent's
common offset inflated while every consistency term in use was invariant to a uniform translation —
the centred cosine and population-NMSE exactly, the raw MSE because it only ever sees `z_d - z_c`.
Nothing in the objective opposed the offset. The fix — penalise `||mean z||` directly, or normalise
the latent — was written down and never run. This package ran it.

## 2. The offset penalty works, and that is not the same as helping

`offset_penalty_weight` adds `λ·||mean over the 2B batch z||²`. Default 0.0, so every prior config
is bit-unchanged. λ chosen from a 10-point scan (`reports/k-0343.md`), `--test_mode`, 8 epochs:

| λ | val AUC | ratio | offset | spread |
|---|---|---|---|---|
| 0 | 0.8750 | 64.27× | 14.191 | 0.221 |
| 1e-3 | 0.8749 | 1.86× | 0.513 | 0.275 |
| **1e-2** | **0.8759** | **0.27×** | 0.078 | 0.289 |
| 1e-1 | 0.8690 | 0.06× | 0.014 | 0.249 |

The offset falls 182× while the spread rises 31%, so roughly 95% of the log-scale ratio improvement
is offset reduction. **Both terms are quoted deliberately**: a ratio is extremal for two opposite
reasons and cannot distinguish them (register class 24).

Two caveats on the bottom of that table. A per-batch offset has a sampling floor of about
`spread/sqrt(256)` ≈ 0.011 — a cloud with true offset zero reports 0.027 — so the last two rows mean
"below what this measurement resolves", not "0.06×". And the whole scan is `--test_mode`; the
transfer to full data is not automatic, as section 6 shows.

## 3. The result: the probe uses latent magnitude, not only direction

`PMAEncoderNormLatent` L2-normalises the latent, so the offset is removed *by construction* rather
than by penalty. It is the cleanest possible version of the fix, and it is a **separable regression**:

| arm | mean_area | clean | vs reference 0.826314 |
|---|---|---|---|
| k2_normlatent s33 | **0.799879 ± 0.003195** | 0.8781 | −0.0264, separably worse |

Its **pre-normalisation** offset ratio is 22× and rising. So the arm does not remove the pathology,
it hides it from the probe — and hiding it costs 0.026 of mean_area.

**The probe reads latent magnitude, not just direction.** Projecting onto the unit sphere discards
information the frozen MLP probe is using. That constrains every future attempt to normalise or
whiten this latent, and it is the one finding here that should survive the campaign.

*Credit: WP-I predicted before the arm ran that normalising would make the pathology "unmeasurable
rather than impossible", and required the pre-normalisation logging that measured it. Without that
the arm would have shown a healthy bounded ratio and a bad score, with no way to connect them.*

## 4. K5: a consistency term with a degenerate minimiser

Measuring the displacement in the clean batch's full covariance (squared Mahalanobis,
`diff·cov⁻¹·diff`) instead of per-dimension. Motivated by campaign 1's finding that ~89% of an
isotropic MSE budget is spent on probe-invisible directions. It collapses:

| epoch | val AUC | ratio | cond | **effective rank** | mse |
|---|---|---|---|---|---|
| 1 | 0.7216 | 244× | 787 | 1.18 | 0.45 |
| 3 | 0.7294 | 514× | 3,470 | 1.03 | 104.9 |
| 6 | 0.7325 | 73× | 37,820 | **1.00** | 1.10 |

Benched at **0.701247**, a 0.125 regression. The mechanism: a displacement becomes free if the model
inflates the variance along the direction it drifts in, and the cheapest way to do that is to put
all the variance in ONE direction. The term is satisfiable by reshaping the latent instead of by
being invariant — the same shape as the pathology the package exists to fix.

*This was only visible because effective rank was in the logging, and effective rank was only there
because WP-I raised it as a caution against a different claim of mine. Without it, cond 37,820 reads
as "ill-conditioned" and a rank-1 collapse is invisible: the two produce identical condition numbers.*

## 5. Scale-invariant geometry improves; attribution is unresolved

Measured by WP-I on one instrument (4000 clean eval events, same code, seed 11):

| | offset | spread | ratio | cond | eff_rank |
|---|---|---|---|---|---|
| champion s11 | 9.5075 | 3.3766 | 2.816 | 78.2 | 1.68 |
| k1_offpen s11 | 0.2747 | 0.1049 | 2.619 | **32.4** | **1.94** |

`cond` and `eff_rank` are scale-invariant and cannot be produced by a uniform rescale, so those
improvements are real — the first thing in this campaign measured to move effective rank at all.

But the offset falls 34.6× and the spread falls 32.2×, so the ratio moves 7%: between the champion
and K1 the **whole latent shrinks ~32×**. That comparison spans three changes (one-view → two-view,
plus the MSE term, plus the penalty) and cannot attribute the shrink to any one. WP-D measured the
two-view consistency term alone compressing the latent 5.7× in campaign 1, which makes the MSE term
the leading candidate; and within the two-view arm the penalty *expands* the spread (§2).

### 5.1 Attribution, resolved from K3 without the extra run

K3 is the control nobody planned as one: **two-view with no latent consistency term at all**
(logit-JSD only, no MSE, no penalty). It isolates the two-view configuration — the 2B concatenated
batch through BatchNorm, degradation on half the batch — from the MSE term.

| arm | instrument | offset | spread | cond | eff_rank |
|---|---|---|---|---|---|
| champion (one-view, no consistency) | eval events (WP-I) | 9.508 | 3.377 | 78.2 | 1.68 |
| **k3_jsd (two-view, JSD only, no MSE)** | train batches | 6.537 | **2.732** | 72.5 | 1.73 |
| k1_offpen (two-view + MSE + penalty) | train batches | 0.245 | **0.106** | 32.9 | 1.94 |

**The two-view configuration alone does not shrink the latent.** K3 sits at spread 2.732 and
cond 72.5, beside the champion's 3.377 and 78.2 — same order on both, on different instruments.
The 26× collapse appears only once the MSE term is present.

Combined with §2, where raising λ *increases* the spread within the two-view+MSE arm, the
attribution is: **the MSE consistency term is the shrinking agent; the offset penalty mildly
opposes it.** That is what WP-D's campaign-1 measurement (5.7× compression from the consistency
term) predicted, now confirmed on this architecture.

*Caveat: K1 and K3 differ in two things (the MSE term and the penalty), so this is not a
single-variable comparison. It is decisive against "two-view causes the shrink" and, with the λ
scan's direction, against "the penalty causes the shrink". A λ=0 two-view+MSE row would make it
single-variable and was not run.*

## 6. Withdrawn claims

Recorded rather than deleted, because each was circulated and acted on.

| claim | why it was wrong |
|---|---|
| "the offset is a property of the champion's architecture-plus-objective" | the one-view champion plateaus at ~2.36× with a flat offset and growing spread. Inflation is specific to the two-view configuration; which part of it is unidentified. The half that survives: the uncentred cosine is *not* the driver — a two-view run with no cosine inflates identically. |
| "champion conditioning 6,000–18,000 vs 33" | a `--test_mode` per-batch measurement against a full-data one. The champion on eval events is **78**. A comparison of data regimes wearing the clothes of a comparison of objectives. |
| "the spread is flat across the λ scan" | the summary regex `spread_tr` matched inside `drift_spread_tr`, so a different quantity was reported as the spread for every row. Found by WP-N dividing three printed numbers and getting 75.5 against a printed 64.27. |
| "the penalty removes offset rather than inflating the denominator" | true within the arm, but the guard tested only one failure mode. Its mirror — numerator and denominator shrinking together, leaving the ratio flat — is what §5 shows between champion and K1. |

The common shape: in each case the number was real and the reading of **what produced it** was
wrong. The defence is to check what else differs between two rows before attributing a difference
to the thing you changed.

## 7. Method notes worth keeping

- **Log both terms of any ratio.** Not decoration — disambiguation (class 24).
- **`cond` is meaningless without `eff_rank`.** Every arm here sits at effective rank 1.3–2.1, so a
  condition number is dominated by directions carrying ~1% of the variance.
- **A diagnostic must run on the baseline too.** The geometry logging originally lived inside the
  `two_view` branch, which made "does the champion have this disease?" unanswerable.
- **rc is not a completion signal.** Wrappers here report `JSON-OK` / `NO-JSON`, judged on the row
  appearing in `runs/`. Logs for progress, JSONs for facts — that separation is why this table could
  not be fabricated while two of my wrappers were reporting completions for jobs that never ran.
- **Anchor regexes on field names**: `(?<!drift_)spread_tr`, or you reproduce §6's bug silently.
