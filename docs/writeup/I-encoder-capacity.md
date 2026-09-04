# I — Encoder capacity

WP-I, campaign 2. Branch `c2-i` @939ac19 off `integration-2` @8dfe03a. Four arms, each the campaign-2
champion config with **one behavioural change**, three seeds each (11/22/33), small file, 25 epochs.
**All four arms reached n=3. Nothing was promoted.** Every number below names the JSON it comes from.

## 1. The question, and the hole in the ladder

Campaign 1 left one gap: `PMAEncoder(num_layers=0)` at 89,606 parameters scored 0.8255 (R=20) and `a-pma`
— a 4-layer transformer body with the same PMA readout, 2,445,478 parameters — scored 0.8277: separable on the
bench, tied on the official grader, **27× the size**. Nothing in between had been measured. The question is
therefore not "does much more capacity help" but **"how little extra buys that +0.0022, if anything does"**.

Measured parameter counts of the specified arms (`tools/check_arms.py`, which builds each from the grader's
fixed signature):

| arm | change | params | × champion |
|---|---|---:|---:|
| I4 | 8 seed queries (`PMAEncoder8`) | 94,214 | 1.05 |
| I6 | `latent_dim: 32` | 102,944 | 1.15 |
| I3 | `embed_size: 256` | 343,046 | 3.83 |
| I1 (specified) | `num_layers: 1` | 682,798 | **7.62** |
| I2 (specified) | `num_layers: 2` | 1,275,990 | 14.24 |

**The specified ladder could not answer the question.** `TransformerEncoderBlock` hard-codes
`dim_feedforward=2048` — 16× the 128-wide embedding — and `PMAEncoder` never passed it, so one block costs
593k parameters and the ladder jumps 1.15 → 3.83 → **7.62** with nothing between 1.15× and 3.83×. Narrowing
the FFN reaches exactly the missing region: 1 layer at ff=128 is 189,358 (2.11×), at ff=256 is 222,254
(2.48×), at ff=512 is 288,046 (3.21×) — all cheaper than I3, which was already on the list.

The planner replaced both layer arms with **I1′ = one block at `dim_feedforward: 256`** (2.48×). `dim_feedforward`
is now a `PMAEncoder` kwarg defaulting to 2048, so every existing model is bit-identical.

## 2. Results

Reference `h-ref-small`, seeds 0.825465 / 0.829586 / 0.823891 → **seed mean 0.826314, seed std 0.002941**
(`h-ref-small-s11_20260904_041301.json`, `h-ref-small-s22_20260904_035504.json`,
`h-ref-small-s33_20260904_034839.json`).
Separability: |Δ| > 3·√(s_a²/n_a + s_ref²/n_ref). Non-separable promotion floor: 0.828314.

| arm | n | mean_area | seed sd | clean AUC | Δ | threshold | separable | params | ×champ |
|---|---:|---|---|---:|---:|---:|---|---:|---:|
| **I6** latent 32 | 3 | **0.8292** | 0.0003 | 0.9270 | +0.0028 | 0.0051 | no | 102,944 | 1.15 |
| **I1′** 1 block ff256 | 3 | 0.8288 | 0.0016 | 0.9253 | +0.0025 | 0.0058 | no | 222,254 | 2.48 |
| I4 8 seeds | 3 | 0.8269 | 0.0014 | 0.9235 | +0.0005 | 0.0057 | no | 94,214 | 1.05 |
| I3 embed 256 | 3 | 0.8257 | 0.0028 | 0.9216 | **−0.0007** | 0.0071 | no | 343,046 | 3.83 |
| reference | 3 | 0.8263 | 0.0029 | 0.9220 | — | — | — | 89,606 | 1.00 |

Per-seed sources — I1′: `i-ff256l1-s11_20260904_045308.json` (0.828634),
`i-ff256l1-s22_20260904_045856.json` (0.830310), `i-ff256l1-s33_20260904_050919.json` (0.827263).
I3: `i-embed256-s11_20260904_045418.json` (0.826200), `i-embed256-s22_20260904_052215.json` (0.822644),
`i-embed256-s33_20260904_051030.json` (0.828196).
I4: `i-seeds8-s11_20260904_045524.json` (0.828403), `i-seeds8-s22_20260904_052340.json` (0.826574),
`i-seeds8-s33_20260904_050439.json` (0.825579).
I6: `i-lat32-s11_20260904_045631.json` (0.829141), `i-lat32-s22_20260904_050544.json` (0.829435),
`i-lat32-s33_20260904_050652.json` (0.828877). Table: `i_phase1_table.json`.

### The capacity conclusion
**Encoder capacity is not the binding constraint on this metric.** The largest arm, I3 at **3.83×** the
champion's parameters, lands at **0.8257 — below the reference seed mean** and is dropped by the rule outright.
I1′ at 2.48× fails to separate. The two cheapest arms (1.05×, 1.15×) are indistinguishable from the reference
and from the expensive ones. Across a 3.6× span of parameter count, nothing moves beyond seed noise.

### Nothing is promotable
I6 (0.829151) and I1′ (0.828763) both clear the non-separable floor of 0.828314 by seed mean. Neither is
promotable: **criterion (ii)** — no separable regression on the held-out and colleague suites — is
**unmeasured**, I ran neither; and **criterion (iii)**, an independent diagnostic supporting the mechanism,
I cannot supply honestly, because my only diagnostic for I6 predicted it would *not* help and was falsified by
its own bench (§4). "Ties go to the smaller model" also removes I1′ at 2.48×.

## 3. What adding seeds did — and a claim of mine that was wrong

| arm | seeds | n=1 | n=2 | n=3 | range |
|---|---|---|---|---|---:|
| I1′ | 0.8286 / 0.8303 / 0.8273 | 0.8286 | 0.8295 ↑ | 0.8288 ↓ | 0.0031 |
| I3 | 0.8262 / 0.8226 / 0.8282 | 0.8262 | 0.8244 ↓ | 0.8257 ↑ | **0.0055** |
| I4 | 0.8284 / 0.8266 / 0.8256 | 0.8284 | 0.8270 ↓ | 0.8269 ↓ | 0.0028 |
| I6 | 0.8291 / 0.8294 / 0.8289 | 0.8291 | 0.8293 ↑ | 0.8292 ↓ | **0.0006** |

I reported to the planner that "every seed added moved an arm DOWN, except I6's", and asked for it to be
recorded. **It was false**, and false already when written — two arms went up on their second seed, and I had
described I1′ peaking at n=2 in the same message before generalising past my own counter-example. I
pattern-matched on the two arms I had been advocating for, which happen to be the two that fell.

**The accurate statement:** adding a seed moved an arm's mean by up to 0.0014 in **either** direction, with no
systematic drift. Single-seed point estimates move by **0.0028–0.0055** between seeds — comparable to or
larger than any arm's entire margin over the reference. That is the multiplicity point, and it needs no
direction. Three of four arms sat above the reference on their first seed; that is what a null field produces.

## 4. Latent geometry (CPU, no GPU slot) — `i_latent_geometry_all.json`

`ratio_mon` is campaign 1's mean-of-norms convention; `eff_rank` is the participation ratio of the latent
covariance; `cond` is its condition number. 3000 clean eval events.

| arm | dim | ratio_mon | cond | eff_rank |
|---|---:|---|---|---|
| I1′ | 6 | 1.24–1.62 | 382–1951 | 1.40–1.56 |
| I3 | 6 | 2.22–2.50 | 97–396 | 1.63–1.75 |
| I4 | 6 | 2.61–3.30 | 74–219 | 1.56–1.79 |
| I6 | **32** | 2.70–3.95 | **90,626–151,121** | **1.82–2.07** |
| champion (s11) | 6 | 2.82 | 78.2 | 1.68 (`i_latent_geometry_s11.json`) |

**I6's 32-dimensional latent has an effective rank of ~2 on all three seeds** — its top 6 eigenvalues carry
96.9% of the variance, its top 2 carry 78.8%. Widening 6 → 32 bought about a third of an effective direction,
and its condition number is unpopulated directions rather than poor conditioning.

**A prediction of mine, recorded in advance and falsified.** I predicted before any bench that I6 would not
help *because* of that rank. I6 then produced the **highest single-seed row in the package** (0.829141). The
measurement stands; the inference does not. **Effective rank describes the latent and does not predict the
metric**, defeated by a caveat campaign 1 had already measured and I had already written down: *variance is
not discriminability* — a low-variance direction can be exactly the one the probe needs. A condition number on
a wide latent measures emptiness unless effective rank is reported beside it.

**Cross-package result at no cost** (`i_latent_geometry_xpkg.json`): the champion's condition number on the
real data regime is **78.2**, not the 6,000–18,000 WP-K had measured under `--test_mode`. That retired their
conditioning lead and saved two GPU slots they had queued to measure it.

## 5. The noise budget — the most transferable result here

**Bench reproducibility, from a duplicate I failed to drain.** The same checkpoint benched twice, independently,
five probe refits each: **0.830310** (`i-ff256l1-s22_20260904_045856.json`) and **0.830395**
(`i-ff256l1-s22_20260904_050334.json`) — **difference 8.5e-5**. That is a quarter of one row's probe sd and
**35× smaller than the seed std of 0.0030**. Re-benching a checkpoint buys nothing; re-seeding buys everything.
A table quoting probe-refit error bars (±0.0008) looks ~4× more decisive than the seed test allows.

**The detection floor is set by the reference, not by the candidates.** With s_ref = 0.0029 at n_ref = 3, the
reference term alone contributes **0.0051** to every threshold:

| arm seed sd | threshold |
|---|---|
| 0.0002 (a perfectly reproducible arm) | 0.0051 |
| 0.0016 (I1′) | 0.0058 |
| 0.0029 (reference-like) | 0.0072 |

**Phase 1 could not have detected a genuine +0.003 improvement from any arm, however reproducible.**
"No arm separated" is correct; "nothing helped" does not follow, and the first is read as the second unless the
floor is stated beside it. **Prescription: spend spare runs on REFERENCE seeds, not arm seeds** — the floor
falls as 1/√(n_ref), so ten reference seeds would take it from 0.0051 to ~0.0028, at the cost of runs of one
configuration rather than three per arm.

## 6. The class-not-kwarg rule, and `check_arms.py`

`num_seeds: 8` trains fine — `train.py` reads an `encoder_kwargs` config key. But **`bench_eval.py:156` and
`eval.py:34` both construct the encoder from the grader's fixed signature and pass no keyword arguments**, so
an 8-seed checkpoint would be untestable by our own bench and ungradable by `eval.py`. It would have failed
loudly (`pma.seeds` is [1,8,E] against [1,4,E]) — but relying on a failure being loud is what the register says
not to do.

Both new arms are therefore **classes whose default IS the final choice**, selected by `encoder_class:`:
`PMAEncoder8` (num_seeds 8) and `PMAEncoderFF256` (dim_feedforward 256). If either were promoted the shipped
branch aliases `TransformerEncoder` to it and the grader gets the option from the signature alone. **The
planner made this a campaign rule at 03:38** and corrected WP-K, whose latent-normalisation flag had the same
defect.

`tools/check_arms.py` asserts the property directly, per arm: build via the train path (`**encoder_kwargs`),
rebuild via the bench/grader path (no kwargs), compare every parameter name and shape, strict-load, forward.
All arms pass. Run it before starting any arm.

*Note on I1′ and the one-change audit:* it carries two config lines (`num_layers: 1` and
`encoder_class: PMAEncoderFF256`) and is still one behavioural change — at `num_layers: 0` the class is
bit-identical to `PMAEncoder`, since `dim_feedforward` is only read when a block is built. Neither line alone
expresses "add one transformer block of FFN width 256".

## 7. Durable checkpoints

Six Phase-2 candidates, sha256-verified against the source each run's `Saved best encoder to:` line names,
digests matching the planner's independently-written sidecars:
`~/hackathon-shared/checkpoints/c2/i-ff256l1-s{11,22,33}.pth` and `i-lat32-s{11,22,33}.pth` (+ `.sha256`).

## 8. Future work — one hypothesis, explicitly not a finding

**Does a wider latent buy seed STABILITY rather than mean performance?** I6's three seeds span **0.0006**
against every other arm's 0.0028–0.0055 and the reference's 0.0057 — roughly ten times tighter. If real, that
matters more than a mean shift, because seed noise is this campaign's binding constraint and the detection
floor is made of it. **Not claimed:** an sd from n=3 carries ~40% relative uncertainty (95% interval on 0.0003
runs roughly 0.00013–0.00073), and three seeds cannot distinguish a stable arm from a lucky draw. It pairs
naturally with the reference-seed prescription in §5 — both spend runs on variance rather than on means.

## 9. What this package cost, and the errors behind it

All 12 runs trained and **all 12 seeds benched**, every arm at n=3. The gaps and delays were:
1. **The campaign-wide bench deadlock 04:13–04:31** (WP-H's token-before-slot bug). Not mine.
2. **Duplicate wrappers**, twice — a superseded driver's queued jobs survive the driver being killed (correct
   under the scheduler change), so two runs of one arm wrote to one outdir, leaving orphan checkpoints. Fixed
   by selecting the checkpoint the arm's own log names, never a glob.
3. **I drained my own active bench** with a loop that did not distinguish my new job from the duplicate I meant
   to kill.

Each of (2) and (3) is the same shape: acting on a set without verifying membership. Two claims I made and
had to retract:
- I asserted I had destroyed two *running* benches at 04:29. `gpu.log` shows my first bench acquisition was
  **04:50:44** — they had never started. I inferred causation from an empty `nvidia-smi` that the deadlock
  explained, with the log that would have settled it open in the same session.
- I asserted `i3_embed256_s22` never completed and that the geometry file was contaminated. **Both false.**
  Two writers had left that log full of NUL bytes (63,831 of 84,281), so `grep` silently treated it as binary
  and printed *nothing* — not "0" — and I read that emptiness as absence. `grep -a` shows 25 epochs and
  `released rc=0`. My own Python guard had read it correctly; I concluded the tool was wrong when it was right.
  Recorded as register classes 35 and 36.

**The defence that generalises: an empty result is not a finding until the tool has been shown to return a
non-empty one under conditions known to be true.**

Three completion signals were used in this package and all three were wrong at first: "a checkpoint exists"
(train.py writes one every improving epoch, so a run 9 epochs in looks finished), "the scheduler logged
released" (a crashed run gets one, with rc≠0), and "the wrapper returned rc=0" (true for a job that never
acquired a slot). The working test is: `released` **with rc=0** in the arm's own log for training, and the
**presence of a JSON in `runs/`** for a bench.
