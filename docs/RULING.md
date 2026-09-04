# FINAL RULING (planner, 22:38 by the wall clock)

WINNER: d-pma0-aug-meanpt -> branch submission-pma0-meanpt @41d45a9 (README-only commit on top of 3ac262f, which passed the
full fresh-clone acceptance; code, config and checkpoint byte-identical), checkpoint rt_d_pma0_aug_meanpt_encoder_20260903_221125.pth,
PFPreProcessorMeanPt + PMAEncoder (num_layers 0, 4 seed queries, mean+max not used), trained 25 ep/bs256 with WP-B's generator.
Basis: separable above every other row on the development bench (0.8260 +- 0.0009 vs 0.8186 +- 0.0007, gap 0.0074, thr 0.0015)
and on the organisers' eval.py at full settings (0.8790, 0.8786 vs 0.8703, 0.8701, 0.8701), and on the held-out shapes
(0.8021 +- 0.0006 vs 0.7977 +- 0.0009). Leads on all three metrics; no tie-break needed.
RUNNER-UP: d-deepsets-aug-meanpt (submission-deepsets-meanpt @791c450), which would have won the tie-break chain.
Fresh-clone acceptance (accept.sh --submission with class and sha256 assertions), organisers' eval.py, full 70k set, 10 severities:
area 0.8780 on 3ac262f (AUC 0.921 clean -> 0.823 at s=1.0; three independent runs 0.8790/0.8786/0.8780, spread 0.0010).
PASS (E, 22:47): accept: PASS branch=submission-pma0-meanpt commit=3ac262f ckpt=rt_d_pma0_aug_meanpt_encoder_20260903_221125.pth
eval.py area=0.8780 (fresh clone, full run to completion, all guards: class assertions, single shipped checkpoint, sha256 match).
PASS (E, 22:52): accept: PASS branch=submission-pma0-meanpt commit=41d45a9 ckpt=rt_d_pma0_aug_meanpt_encoder_20260903_221125.pth
eval.py area=0.8795 (--fast, fresh clone of the shipped tip, all guards). Four independent official measurements of this
checkpoint: 0.8790, 0.8786, 0.8780, 0.8795 (spread 0.0015) vs the organisers' 0.8201: +0.059 like-for-like.
Provenance (E, git object level): between 3ac262f and 41d45a9 the only change is README.md (+57); the blobs of models.py,
preprocs.py, configs/train_config.yaml, configs/data_config_eval.yaml, eval.py and the single shipped checkpoint are identical.


Candidates (bench mean_area over probe refits, 20k events, five families):
  d-deepsets-aug-meanpt  0.8186 +- 0.0007 (R=5)   clean 0.9066   official: 0.8708 / 0.8701 / 0.8701 (mean 0.8703, range 0.0007)   held-out: 0.7977 +- 0.0009
  d-deepsets-aug         0.8174 +- 0.0020 (R=10)  clean 0.8921   official: 0.8586 / 0.8610   held-out: 0.7962 +- 0.0011
  d-pma0-aug             0.8162 +- 0.0038 (R=5)   clean 0.9215   official: 0.8693 / 0.8613 / 0.8702 (mean 0.8669, range 0.0089)   held-out: 0.7983 +- 0.0048
  d-pma0-aug-meanpt      0.8260 +- 0.0009 (R=5)   clean 0.9147   official: 0.8790 / 0.8786 (mean 0.8788, range 0.0004)   held-out: 0.8021 +- 0.0006 (best; separable vs meanpt 0.7977, thr 0.0015)   <- SEPARABLE LEADER (22:41)
  d-deepsets-twoview-v2  0.7419 +- 0.0156 (R=5)   clean 0.7804   <- negative row; two-view hurts on Deep Sets too
Separability: all pairwise gaps below 3*sqrt(s_a^2/R_a + s_b^2/R_b) -> tie on mean_area.
Tie-break 1, official area (organisers' eval.py, full settings, n=3 each): meanpt 0.8703 (range 0.0007) vs pma0 0.8669 (range 0.0089)
  vs aug 0.8598 (n=2, 0.0024): aug separably below; meanpt vs pma0 gap 0.0034 inside pma0's range -> not called.
Tie-break 2, held-out families (ellipse/annulus/diagonal, R=5): TIED (0.7977 / 0.7962 / 0.7983)
Tie-break 3, clean AUC: pma0 0.9215 > meanpt 0.9066 > aug 0.8921

PROVISIONAL WINNER (pending its officials/held-out): d-pma0-aug-meanpt  branch submission-pma0-meanpt @3ac262f
  preproc PFPreProcessorMeanPt   encoder PMAEncoder (alias, num_layers 0)   checkpoint rt_d_pma0_aug_meanpt_encoder_20260903_221125.pth
  Why: separable on the bench from every other row (0.8260 +- 0.0009 vs 0.8186 +- 0.0007, gap 0.0074 vs thr 0.0015); no tie-break
  needed. Interaction: PMA readout alone (0.8162) and MeanPt alone (0.8186) did not separate from 0.8174; together they do.
RUNNER-UP (previous provisional winner): d-deepsets-aug-meanpt
  branch submission-deepsets-meanpt @791c450   preproc PFPreProcessorMeanPt   encoder DeepSetsEncoder (alias)
  checkpoint checkpoints/rt_d_deepsets_aug_meanpt_encoder_20260903_214738.pth
Why: tied with d-pma0-aug on the bench mean (0.8186 vs 0.8162), on official area (0.8703 vs 0.8669, n=3 each, gap 0.0034 inside pma0's
0.0089 range) and on held-out shapes (0.7977 vs 0.7983); tie-break 3 (pre-registered 22:38): lower variance of the scored
quantity on BOTH metrics (official range 0.0007 vs 0.0089; bench per-refit sigma 0.0007 vs 0.0038). Clean AUC (tie-break 4)
would have favoured pma0 (0.9215 vs 0.9066) and is reported, not used. The submission-pma0 hedge branch (99b7d2e) exists.
Missing row (B): b-aug-stock (transformer, same protocol, augmentation only) did not run because the main lock was needed for
submission and certification work, not on merit; the transformer augmentation question is OPEN, bracketed by E's clean control
(0.8089) and C's augmented two-view run (0.7712), which do not separate augmentation from the two-view batch.
Ablation rows for the table: anchor 0.7734; mask fix frozen 0.7918 (+0.0165 A); E control (transformer, mask fix, 25 ep, no aug) 0.8089 +- 0.0004, clean 0.9025; c-twoview-cos1
0.7712 (transformer, inert cosine); a-pma __ (if landed); b-aug-stock __ (if landed); deepsets-clean 0.8036 (n=1).
Rect s=1.0 (corrected, B/F): all arms degrade sharply and several fall below chance, including two stock-preproc arms
(c-twoview 0.390, E control 0.471; meanpt Deep Sets 0.451; the leader d-pma0-aug-meanpt 0.556 does NOT invert). Two claims, not
one (F, per-family): extrapolation at s=1.0 is universal (every family's s=1.0 bench point lies beyond the highest dropped fraction
any training event of that family reached; rect 0.94 vs 0.80 max, no rect event above 0.8), which is what makes a below-chance
AUC possible; but WHICH family an arm inverts in is not explained by coverage (rect and wedge have near-identical coverage and
wedge extrapolates further, yet masking-trained arms invert at rect and frozen mask-fix arms at wedge). Unexplained; not the
preprocessor.
meanpt Deep Sets trades rect (0.7538 vs 0.7754) for wedge/towers/cells. The WINNER has no below-chance cell in any scoring family
and the best rect s=1.0 of any arm (0.556 +- 0.003); its two held-out s=1.0 cells at 0.49995 are the degenerate all-dropped
point (dropped fraction 1.000, refit std 0), not inversions (F).
Inversions (F): rect s=1.0 below chance for c-twoview-cos1 (0.39), meanpt (0.45), E control (0.47); wedge s=1.0 below chance
for both frozen mask-fix rows; the aug (R=10) and pma0 finalists invert nowhere. Which family inverts tracks the training setup.
E control wedge s=1.0 = 0.74 (collapse gone): the frozen-weights wedge inversion was a train/eval mismatch.
Held-out (ellipse/annulus/diagonal, R=5): meanpt 0.7977 +- 0.0009 (.786/.809/.799), aug 0.7962 +- 0.0011 (.786/.806/.797),
pma0 0.7983 +- 0.0048 (.788/.809/.798): tied; all ~0.02 below scoring areas because ellipse and diagonal drop 100% at s=1.0
(AUC 0.5 by construction), so the demotion rule is read candidate-vs-candidate: none demoted; meanpt inverts on no held-out family.
Headline pair (E, final): 0.8260 +- 0.0009 on our five shape families, 0.8021 +- 0.0006 on shapes from neither our generator nor
our bench (both R=5). The winner's held-out gap (0.0239) is the largest of the four finalists but the differences are not
separable, and ex-endpoint it equals the runner-up's (0.0176). Quote the pair, never 0.8260 alone; the ~0.02 gap is
uniform across candidates (differences not separable). Recomputing areas without the s=1.0 point (E): about a quarter of the gap
is the degenerate endpoint (two of three held-out families lose every candidate at s=1.0 vs one of five scoring families);
the remaining ~0.015 is a genuine generalisation gap to unseen shapes. Blunt estimate; direction and rough size are solid.
Per-architecture probe floors per refit: transformer 0.0008, Deep Sets 0.0020, PMA 0.0038.
Limitations: transformer rows at 7,025 steps vs 33,720 for the anchor; official grid is temporary and unseeded; 20k-event bench.

## Closing record (D, 22:47)
Submission: PMAEncoder(num_layers=0) + PFPreProcessorMeanPt, 0.8260 +- 0.0009 vs 0.7734 anchor, 0.090M params (26x fewer than
the transformer), 11.9 ms/step. Findings to carry forward: architecture was the largest single contributor (+0.030, measured
with no degradation in training); the winner is an interaction (neither PMA readout nor MeanPt separated alone); most
degradation drift is harmless (null-space component 3.8x larger than the probe-relevant one, costing 0.0008 vs 0.0312 AUC);
two-view consistency failed by inflating the latent offset 3 -> 2324, invisible to every logged quantity.
Seven silent failures caught tonight, each producing a plausible answer rather than an error: the checkpoint glob, --fast's
buffered grep, identical preprocessor state_dicts, the vacuous acceptance test, the stale-checkpoint pick, the branch-checkout
deletion, the dangling training-branch reference. Common defence: verify against an independent measurement, never trust a
green run.

## Late result (22:48, after the ruling): a-pma
Transformer body (4 layers) + PMA readout, stock preproc, WP-B augmentation, 25 ep/bs256: bench 0.8275 +- 0.0006 (R=5), clean 0.9228.
Against the certified winner (0.8260 +- 0.0009): gap 0.0015 vs threshold 0.00145 = 3.10 sigma against a 3.00 bar; a tie at the
edge, not an ordering that would survive re-measurement (E). Outside the pre-registered candidate set (benched after the cutoff).
The certified submission (41d45a9) stands. Held-out and two official runs requested from A as the independent measurements;
escalated to the user as an option. Note it reaches this with the STOCK preprocessor, a different route from the winner's.
The substantive finding (A): a-pma minus d-pma0-aug (same PMA readout, same stock preproc, no body) = +0.0113 at 2.2x the
threshold, so the transformer body HELPS separably once the readout is attention pooling; this reverses the evening's reading
that the body is the liability. Cautions: no MeanPt (untested combination), O(N^2) attention vs O(N) for the set encoder.

## Certification note (E, 23:00)
submission-a-pma @d450bcf pre-flighted: one shipped checkpoint, preproc PFPreProcessor, a real TransformerEncoder whose readout
kwarg defaults to 'pma' (eval.py passes none), strict load OK. Asymmetry worth stating: a readout mismatch fails loudly (strict
load), a preprocessor mismatch loads silently (shared state_dict keys) and yields a plausible wrong number; the dangerous silent
failures cluster where two objects share a schema, which is why --expect-preproc earns its place.

## a-pma measurements (as they land)

> Planner, 01:08 (4 Sep): this section and "FINAL (user decision 23:13)" are superseded by "Margin amendment: settled" and the "R=20 addendum" below, which carry the three official draws and the R=20 bench numbers. Kept for the record; read the later sections for the final numbers.
- held-out (R=5): 0.8059 +- 0.0016 vs winner 0.8021 +- 0.0006: gap 0.0038 vs thr 0.0023 -> criterion (ii) PASSES (a-pma is
  separably BETTER on unseen shapes; per family ellipse .794 annulus .817 diagonal .807). Scoring-to-held-out drops comparable
  (a-pma -0.0216, winner -0.0239); a-pma's held-out sigma is ~3x the winner's, already in the threshold (A). Officials (rank
  criterion vs 0.8790) and R=20 pending. F: a-pma's rect s=1.0 cell (0.4986) is 0.0014 below chance on a 94%-dropped cell,
  not an inversion in the sense of the 0.39-0.47 cases; neither arm has a true below-chance held-out cell.
- official run 1 (probe, full settings, classes resolved PFPreProcessor/TransformerEncoder, d450bcf): 0.8793 vs winner's best
  0.8790: above by 0.0003, inside the official spread. Criterion (i) needs the WORSE of two runs above 0.8790; run 2 pending.

## 23:08 organisers' answer: preprocessor changes ARE allowed. The winner's PFPreProcessorMeanPt dependency is legal; the
Deep Sets stock-preproc fallback is no longer needed for that reason. Hand-over format still to confirm.

## FINAL (user decision 23:13): amendment adopted; a-pma not promoted
Criterion: both a-pma official runs must exceed 0.8790 + 0.003 = 0.8820. Run 1 = 0.8793 fails it. The two models are
indistinguishable on the bench (tie at the threshold), on held-out (a-pma +0.004 at 1.6x, R=5) and on official area (0.8793 vs
0.8790 best-of-four, inside every observed repeat spread). The certified submission stands: submission-pma0-meanpt @41d45a9,
89,606 parameters (1.17 MB) against a-pma's 2,445,478 (29.5 MB), 27.3x smaller at equal measured robustness, with the lower
variance on the scored quantity. a-pma is the top ablation row and its finding stands: a transformer body with an
attention-pooled readout and the stock preprocessor reaches the same place as a set encoder with the MeanPt rule, by a
different route, and the body adds +0.011 over the bodiless attention-pooled encoder with readout and preprocessor fixed.
Record (F, A, 23:16): against a-pma the certified submission leads on none of the three metrics (bench tie at 1.03x, held-out
behind at 1.63x, official tie at n=1); the decision rests on indistinguishability, the pre-registered margin rule, end-to-end
certification and the 27.3x parameter difference. Credit (A): the readout mechanism was decided by D's set-encoder arms; A's
package contributed the mask-fix isolation, deletion equivalence, the drift diagnostics and the body finding.
Stated plainly for the writeup (E, 23:20): our best predictor of the undisclosed grading, the held-out check, favoured a-pma
separably (1.7x at R=5, with a-pma's sigma 2.7x the winner's); the certified model was retained not because it measured better
but under the pre-registered margin rule, with certification complete on the shipped tip, at 27.3x fewer parameters and
statistically indistinguishable robustness. The user decided with that held-out edge on the table.
- official run 2 (a-pma): 0.8764. So a-pma's officials are 0.8793 / 0.8764 (mean 0.8779, range 0.0029) vs the winner's
  0.8790 / 0.8786 / 0.8780 / 0.8795. The original rank rule (worse run > 0.8790) would ALSO have failed; the decision is the same
  under both rules. Official areas are indistinguishable within the observed spreads.

## Cross-evaluation note (F, 23:27)
The colleague's suite and our held-out set both contain a family named 'ellipse' with different code; they are written with
c_ prefixes in the JSON and labelled by suite in every figure and table; never compared panel to panel. Scoring figures use an
allow-list on the five scoring families so no external suite can enter the ranking or move the accent.
Pooled by checkpoint identity (F, 23:30): a-pma officials n=2 mean 0.8779 (spread 0.0029); winner officials n=4 mean 0.8788
(spread 0.0015); difference -0.0009 inside both spreads: official area is a tie. The n=1 reading of a-pma 'ahead' is stale.
Official-column pooling key (E): (checkpoint basename, branch), since commits differ across README-only changes; winner n=4
spread 0.0015 (not 0.0004 as the per-tag table showed), a-pma n=2 spread 0.0029, meanpt Deep Sets n=3.
Colleague suite over OUR data (eta [-5,5] vs their [-3,3]): dropped fraction at nominal s=1.0 is c_cell_dropout 1.00,
c_phi_wedge 1.00, c_candidate_loss 1.00, c_edge_truncation 0.73, c_eta_band 0.45, c_multi_patch 0.45, c_rectangle 0.41,
c_ellipse 0.34; their held-out mean over our data is dominated by c_cell_dropout. The L1T-file benches are the fair comparison.

## Colleague-suite cross-evaluation (landing)
- winner on OUR eval data, colleague suite (c_ellipse, c_cell_dropout, c_edge_truncation): 0.7990 +- 0.0003 (R=5), clean 0.9137.

<!-- ===== WP-E sections, appended 00:28. These were written to hackathon-shared/RULING.md
     (a stray file E created) instead of here, and are merged in unchanged. Some may overlap
     the planner's own entries above; nothing above this line was modified. ===== -->

## Note on the colleague suite (WP-E, added after the ruling)

An observation about the code at `colleague-group3` branch `dorian` (@37168c1), not a statement
about their models, which we have not seen.

**Quote their per-family areas, not the suite mean.** Two of the three held-out families stop
measuring what they are named for in the upper half of our severity grid:

- **`edge_truncation` self-overlaps above s = 0.5.** It drops `eta <= -3 + 6s` OR `eta >= 3 - 6s`,
  one branch per event by coin flip. Those bounds cross when `-3 + 6s >= 3 - 6s`, i.e. `s >= 0.5`:

  ```
    s=0.2  low=-1.8  high=+1.8   disjoint edges
    s=0.4  low=-0.6  high=+0.6   disjoint edges
    s=0.5  low=+0.0  high=+0.0   OVERLAP begins
    s=1.0  low=+3.0  high=-3.0   each branch removes nearly the whole plane
  ```

  So above s = 0.5 it is no longer "truncate one edge" but "remove almost everything, on a side
  chosen at random". The measured sweep follows exactly that, inverting below chance:

  ```
    c_edge_truncation  s=0.2 0.8864   s=0.4 0.8332   s=0.6 0.7199   s=0.8 0.5889   s=1.0 0.4668
  ```

- **`cell_dropout` saturates**: AUC exactly 0.5000 +- 0.0000 at s = 1.0, every candidate removed,
  the same degenerate endpoint as our own `towers`.

- **`ellipse` is bounded by their eta convention.** Their families assume `eta in [-3, 3]`; our PF
  data spans `[-5, 5]`, so at nominal severity 1.0 `c_ellipse` removes only 0.339 of candidates.
  Measured dropped fractions at s = 1.0 on our data: `c_cell_dropout` 1.000, `c_phi_wedge` 1.000,
  `c_candidate_loss` 1.000, `c_edge_truncation` 0.727, `c_eta_band` 0.454, `c_multi_patch` 0.450,
  `c_rectangle` 0.412, `c_ellipse` 0.339.

Their own grid runs 0.2-0.8, so the top of ours is outside the range these families were built for,
and none of this is a defect in their work at the severities they use. It does mean a **mean over
their suite mixes one saturating family, one self-overlapping family and one bounded family**, so
the suite mean is not a robustness summary.

**The eta bound is confirmed by measurement, not inferred.** Running the same families over the L1T
file (`eta in [-3, 3]` by construction, 100% of candidates inside) moves the dropped fractions at
s = 1.0 exactly as the diagnosis predicts:

```
                     our PF [-5,5]   ->   L1T [-3,3]
  c_ellipse               0.34               0.70
  c_edge_truncation       0.73               1.00
  c_cell_dropout          1.00               1.00
```

So on their own acceptance their families reach the plane they were designed for, and the bounded
behaviour over our data is a property of the acceptance mismatch rather than of their design. Note
the cost, though: with the convention corrected, TWO of the three held-out families saturate to AUC
0.5000 at s = 1.0 (`c_cell_dropout` and `c_edge_truncation`) where over our data only one did, so
the L1T suite mean averages two constants and one real curve at its endpoint. Per-family areas
remain the right thing to quote on either file.

## Cross-condition check: winner vs runner-up in four independent conditions (WP-E)

The certified submission (`d-pma0-aug-meanpt`, PMA + MeanPt) against the runner-up
(`d-deepsets-aug-meanpt`, DeepSets + MeanPt), measured on our families and on Group 3's, over our
PF eval file and over the L1T file. Each row is R=5 with its own two-sample threshold
`3*sqrt(s_a^2/5 + s_b^2/5)`:

| condition | winner | runner-up | delta | threshold | ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| PF, our scoring families | 0.8260 ±0.0009 | 0.8186 ±0.0007 | 0.0074 | 0.0015 | 4.8x |
| PF, colleague suite | 0.7990 ±0.0003 | 0.7883 ±0.0014 | 0.0107 | 0.0019 | 5.6x |
| L1T, our scoring families | 0.8046 ±0.0008 | 0.8021 ±0.0016 | 0.0025 | 0.0024 | 1.0x |
| L1T, colleague suite | 0.7794 ±0.0007 | 0.7775 ±0.0008 | 0.0019 | 0.0014 | 1.3x |

**Same sign in four independent conditions; magnitude only on PF.** The colleague families were
written without knowledge of our models, and the L1T file is a detector acceptance neither model
trained on, so the consistency is not an artefact of our own benchmark design. But the two L1T
margins are 1.0x and 1.3x — at the edge of the floor — so on the unfamiliar acceptance the two
models are close to indistinguishable, and only the PF margins support a claim about size.

Both models fall by about 0.02 moving PF -> L1T (winner 0.8260 -> 0.8046, runner-up 0.8186 ->
0.8021). That is the same model meeting a different acceptance it was never trained for, **not a
defect**, and the L1T rows are kept in their own block for that reason.

## Margin amendment: settled (WP-E, verified from the JSONs)

All official runs of `a-pma`, pooled by (checkpoint basename, branch) on
`robust_tagging_encoder_20260903_220600.pth` / `submission-a-pma`:

| time | tag | official area |
| --- | --- | ---: |
| 23:05:41 | a-pma-probe | 0.8793 |
| 23:23:28 | a-pma | 0.8764 |
| 23:46:08 | a-pma | 0.8792 |

n=3, mean 0.8783, spread 0.0029, **maximum 0.8793 against the required 0.8820**. No run exceeds the
threshold, so the pre-registered margin amendment is settled and the certified set encoder
(`submission-pma0-meanpt` @`41d45a9`) stands. **Final.**

Note the spread: 0.0029 across three runs of one checkpoint, more than the entire 0.0009 gap
between a-pma and the winner on pooled official area. The amendment's requirement of a clear margin
was the right instrument for exactly this reason.

## R=20 addendum: the bench and the official metric disagree about this pair

At R=20 the bench separates the pair in a-pma's favour: a-pma 0.8277 ± 0.0015 (two R=20 runs
pooled, n=40; A's single-run figure 0.8276 ± 0.0017) vs the certified winner 0.8255 ± 0.0007
(R=20, fresh clone @41d45a9, shipped checkpoint), Δ0.0022, threshold 0.0008 pooled (0.0012 single
run), separable. a-pma is also separably ahead on clean AUC (+0.0093) and held-out (+0.0038,
1.65x). The official area does not separate them (0.8783 n=3 vs 0.8788 n=4). The bench and the
official metric disagree about this pair; the pre-registered rule made the official metric
decisive, and the margin amendment failed on all three official draws. The ruling stands on that
rule, the completed certification chain and the 27.3x size difference, and is recorded as a
disagreement between proxy and official, not as the bench agreeing with the outcome.

*Thresholds verified independently: pooled 3·sqrt(0.0015²/40 + 0.0007²/20) = 0.00085; single-run
3·sqrt(0.0017²/20 + 0.0007²/20) = 0.00123. Both clear.*

### Methods note: R=5 is not enough for transformer rows near their threshold

Going from R=5 to R=20 inflated a-pma's probe-refit sigma **2.8x** (0.0006 -> 0.0017) while the
winner's barely moved (0.0009 -> 0.0007, 0.8x). A sigma estimated from five refits is itself a
noisy estimate, and it was biased low for exactly the row whose separability verdict was marginal.
**Any R=5 separability verdict on a transformer row sitting near its threshold inherits that**, and
should be re-run at higher R before being relied on. The set-encoder rows were less affected.

### Augmentation ablation (WP-B's b-aug-stock against WP-E's clean control)

  b-aug-stock       0.8058 ± 0.0043   (transformer, mask fix, WP-B generator, R=5)
  e-stock-maskfix   0.8089 ± 0.0004   (transformer, mask fix, NO augmentation, R=5)

Δ = **-0.0031** against a threshold of 0.0058 — not separable, and the point estimate favours the
*clean* control. So: augmentation helps symmetric-pooling set encoders (+0.0147 on Deep Sets), has
no separable effect on the transformer, and costs nothing on either. Note the two arms also differ
in model-selection criterion (the clean control selects on clean validation loss, the augmented arm
on degraded), so this is the augmentation ARM, not an isolated augmentation effect.

## ADDENDUM (planner jovyan-2a, 04:33, 4 Sep): the training-seed floor, measured, re-reads every single-run comparison

Campaign 2 retrained the certified recipe three times on the small file with seeds 11/22/33 (runs/h-ref-small-s{11,22,33}_*.json):

    seed 11  mean_area 0.8250  clean 0.9228
    seed 22  mean_area 0.8296  clean 0.9249
    seed 33  mean_area 0.8239  clean 0.9184
    seed mean 0.8262, seed std 0.0030, range 0.0057 (probe-refit std per seed 0.0003-0.0006)

So the training-seed spread of ONE configuration is about ten times its probe-refit noise, and 3.7x the 0.0008
"pipeline floor" campaign 1 inferred from a single same-configuration pair of 12-epoch distillation runs. The
separability bar between two configurations measured with three seeds each is 3·sqrt(2·0.0030²/3) ≈ 0.0074.

Consequences for the campaign-1 record, stated as binding readings:
1. Every campaign-1 bench "separability" between two independently trained single runs was computed against probe
   noise only. Any such gap under roughly 0.006 is UNPROVEN (not refuted). That includes the R=20 addendum's
   a-pma-vs-winner gap of 0.0022: the bench does NOT separate a-pma from the certified winner once training-seed
   variance is counted, and the sentence "the bench and the official metric disagree about this pair" is withdrawn
   in favour of "neither metric separates this pair at the noise floors now measured".
2. The winner-vs-runner-up gap of 0.0074 (0.8260 vs 0.8186, single runs) sits AT the three-seed bar and was measured
   with one seed per side; it is likewise unproven as an ordering between configurations. The certified submission
   stands on the pre-registered rule, the completed certification chain and the fresh-clone acceptance, not on a
   proven ordering over its runner-up.
3. The winner-vs-anchor gap (0.8260 vs 0.7734, 0.0526) and the mask-fix gain (+0.018 on frozen weights) are far
   above any floor and stand.
4. Campaign 2 makes no ordering claim below three seeds on both sides (prompts/c2/00-common-v2.md).
