# Campaign 2 ruling (planner jovyan-2a) — RULED 05:15, 4 Sep 2026 (ten minutes ahead of the 05:25 deadline; every Phase 1 table was in)

## User instruction (04:25)
"If no improvement is found within an hour, drop the search and get the champion ready for submission." Ruling at 05:25.

## RULING
1. No improvement was found. Eighteen configurations were benched; seven reached two or three seeds; none separates from the champion
   (best three-seed gap +0.0030 against a bar of 0.0052). Two arms exceed the one-run bar and both are regressions (normalised latent −0.026,
   whitened consistency −0.125).
2. The certified champion ships unchanged: `submission-pma0-meanpt` @41d45a9. Its campaign-1 certification stands. Today's fresh-clone
   re-verification on the current tooling passed every assertion (classes PFPreProcessorMeanPt / PMAEncoder, single checkpoint, sha256
   2f34628ccd78... byte-identical) and wrote four official records on the shipped commit: 0.8794 (05:14), 0.8788 (05:32, patched harness), 0.8774 (05:43, patched),
   0.8805 (05:45, the --submission rerun of record on the patched harness) — runs/official_submission-cert-c2_*.json,
   official_m-accept-regression_*.json, official_m-accept-fixed3_*.json. With campaign 1's four (0.8790 / 0.8786 / 0.8780 / 0.8795) the
   eight-run official spread is 0.8774-0.8805
   against the organisers' reference 0.8201; the spread is the grader's unseeded Bernoulli draw. Two harness defects were found and fixed in
   bench/accept.sh during the re-verification (a multi-GPU SIGPIPE race on the free-memory check; a fast-exit whose SIGTERM eval.py ignored while inside sklearn's t-SNE C extension, fixed by an escalating
   TERM/KILL with confirmation, plus a wrong-pid record fixed with $BASHPID); neither touches the shipped branch. The hand-over note names the
   six full-protocol official measurements of record (campaign 1's four, today's 0.8794 and 0.8805; spread 0.0025); the 0.8788 and 0.8774
   runs were harness regression checks on the same checkpoint and are kept in runs/.
3. Phase 2 candidates by number only (seed mean ≥ 0.8283, not separable, no supporting mechanism, suites unmeasured): I6 latent 32 (0.8292 ± 0.0003,
   1.15x) and I1' one block with ff 256 (0.8287 ± 0.0015, 2.48x). Neither is promoted; both are named for a future campaign that measures them
   with enough seeds on the full file. Their checkpoints (all seeds) are in ~/hackathon-shared/checkpoints/c2/i-lat32-s*.pth and
   i-ff256l1-s*.pth with sha256 sidecars, digests matching the bench JSONs; configs on c2-i @e1c7aa8 (configs/c2/).
4. Data scale, complete at three seeds (06:02): the champion recipe on the full 940k-event × 400-candidate file, 25 epochs, seeds 11/22/33 =
   0.8225 / 0.8290 / 0.8273, seed mean 0.8263, seed std 0.0033, range 0.0064 (runs/h-ref-full-s{11,22,33}_*.json), against the small-file
   reference 0.8263 ± 0.0029: Δ −0.00001, zero to four decimals. Twelve times the data has NO effect on the robustness metric; clean AUC rises
   0.0024 (0.9242 vs 0.9218). The in-regime seed std (0.0033) is the first measured full-file floor: the seed floor is a property of the
   training procedure, not of dataset size (+14% on 12x the data), and the bar borrowed from the small file was slightly PERMISSIVE
   (in-regime 3v3 bar 0.0082 vs 0.0074 borrowed); no call flips. The −0.0037 first-seed reading recorded at the stop was a seed draw
   (s11 is the lowest of the three). The generator removes the same fraction on both files, so the
   regime comparison is not confounded. Data-scale decision for the user: not a lever, on this evidence.
5. Group 3 did not beat us under any measured condition (below).

## Outcome detail
- Certified submission: `submission-pma0-meanpt` @41d45a9 (PMAEncoder num_layers 0 + PFPreProcessorMeanPt, 89,606 params), unchanged.
- Promotion rule (a) — three full-data seeds separably above the full-data reference — is UNATTAINABLE under a one-hour stop: a full-file run
  takes ~75 min and no arm has one. The first full-file reference seed landed at 05:08: h-ref-full-s11 = 0.8224 (clean 0.9248 full-set; 0.9259 was the probe test split) on 940k events × 400 candidates, 25 epochs,
  against the small-file reference 0.8262 ± 0.0030. One seed, inside the 1-vs-3 noise, but the direction is DOWN on the robustness metric while
  clean AUC rose 0.002 like-for-like (auc_clean_full): twelve times the data did not buy robustness on its own.
  Generator calibration is identical on both files (median dropped fraction 0.055 on each), so the comparison is not confounded by the generator. Seeds 22/33 land after the deadline and go into the writeup.
  The data-scale decision for the user is therefore "not on this evidence". Therefore no arm can be promoted at 05:25 by the pre-registered rule; small-file three-seed rows can only name Phase 2 candidates.
- Arm table at 05:25 (dev bench, small file, R=5; reference h-ref-small 0.8262 ± 0.0030, n=3; three-seed bar ≈ 0.0052-0.0074 depending on the arm's own spread; non-separable route needs seed mean ≥ 0.8283 AND a supporting diagnostic):

| arm | change | n | mean_area (seed std) | Δ vs ref | verdict | params |
|---|---|---|---|---|---|---|
| I6 latent 32 | latent_dim 6 → 32 | 3 | 0.8292 (0.0003) | +0.0030 | not separable (bar 0.0052); above the 0.8283 floor by number but its only diagnostic (effective rank ~2/32) does not support a mechanism → not promoted | 102,944 (1.15x) |
| I1' one block, ff 256 | +1 transformer block, FFN 256 | 3 | 0.8287 (0.0015) | +0.0025 | not separable (bar 0.0058); above the floor by number, no supporting diagnostic, loses the size clause | 222,254 (2.48x) |
| L1 sin/cos φ | φ → (sin φ, cos φ) | 3 | 0.8270 (0.0026) | +0.0008 | not separable (bar 0.0068); below the floor; suites clean (held-out 0.8067, colleague 0.8019) | 89,734 |
| I4 seeds 8 | 4 → 8 seed queries | 3 | 0.8269 (0.0014) | +0.0007 | not separable (bar 0.0058); below the floor | 94,214 (1.05x) |
| L2 dxysig squash | tanh(dxysig/20) | 1 | 0.8278 | +0.0016 | n=1, no verdict (seeds never got slots) | 89,606 |
| I3 embed 256 | embed_size 128 → 256 | 3 | 0.8257 (0.0028) | −0.0005 | at the reference at 3.83x: capacity is not binding | 343,046 (3.83x) |
| L3 EMA | per-epoch EMA of weights | 1 | 0.8260 | −0.0003 | below the screen | 89,606 |
| K3 logit-JSD | two-view, JSD on logits only | 1 | 0.8249 | −0.0014 | below the screen; leaves latent geometry at the champion's | 89,606 |
| L4 seeded val corruption | seed the validation corruption | 1 | 0.8244 | −0.0019 | below the screen; buys reproducibility, not accuracy → campaign-3 standard | 89,606 |
| K1 offset penalty | λ‖mean z‖², λ 0.01, two-view + MSE | 1 | 0.8237 | −0.0026 | below the screen | 89,606 |
| K2 normalised latent | L2-normalise the latent | 1 | 0.7999 | −0.0264 | separably worse: the probe uses latent magnitude | 89,606 |
| K5 whitened consistency | Mahalanobis consistency | 1 | 0.7012 | −0.1251 | collapsed (rank-1 degenerate minimiser) | 89,606 |
| J1 colleague train families | +5 Group 3 train families | 3 | 0.8277 (0.0017) | +0.0014 | not separable (bar 0.0059); matched-seed deltas +0.0044 / −0.0030 / +0.0029; below the floor; suites clean (held-out 0.8053, colleague .8313/.8054/.7617) | 89,606 |
| J2 effective-tail mixture | 30% of events lose >0.7 (on J1) | 1 | 0.8240 | −0.0022 | below the screen; clean −0.0078 and every family down: the champion's severity distribution is not obviously too soft | 89,606 |
| J3 no clean events | p_clean 0.15 → 0 | 1 | 0.8264 | +0.0001 | cleared the screen by 0.0001 (noise); seeds drained in favour of J1 | 89,606 |
| J5 on-target placement | edge-aware, union-inverted, p→1 | 1 | 0.8251 | −0.0011 | below the screen (calibration fix alone does not move the score on one seed) | 89,606 |
| Group 3 best (their checkpoint) | — | 1 | dev bench not run | — | see the Group 3 section | 2.4M |

Reading: 7 of 11 single-seed arms cleared the reference mean, the null expectation at seed std 0.0030. Adding a seed moved an arm's mean by up to 0.0014 in EITHER direction (L1 +0.0026 → +0.0008; I4 +0.0021 → +0.0007 → +0.0007; I1' +0.0023 → +0.0032 → +0.0025;
I3 0.0000 → +0.0009 → −0.0005; I6 flat at +0.0030): no systematic drift, but single-seed point estimates move by 0.002-0.003 between seeds, comparable to or
larger than any arm's entire margin over the reference. The design's detection floor was ≈0.005, set by
the reference's own seed spread (0.0051 from s_ref alone at n=3): "no arm separated" is correct; "nothing helped" does not follow.
The only arm to clear the promotion floor (I6, latent 32) did so with a seed spread ten times tighter than the reference's, and could still not separate,
because the detection floor was set by the reference rather than by the candidates; its suites were not measured and its own diagnostic did not
support a mechanism, so it is not promoted.

## Reference and noise floor (measured)
h-ref-small seeds 11/22/33 = 0.8250 (0.8255 on a repeat bench of the same checkpoint) / 0.8296 / 0.8239; seed mean 0.8262, seed std 0.0030,
range 0.0057; probe-refit std per seed 0.0003-0.0009. Three-seed separability bar ≈ 0.0074; one-run-vs-one-run bar ≈ 0.0125.
Seed noise is ~8x probe noise. Campaign-1 single-run separabilities below ~0.006 are unproven (see writeup/RULING.md addendum 04:33).

## Group 3 comparison (user's #1 priority; complete 04:56)
One instrument (our bench, one probe implementation) for every row. Group 3 did not beat us under any measured condition:
- Their suite, their L1T eval file, like-for-like (our recipe trained on the same L1T file): ours 0.7862 / 0.8270 (areas 0-1.0 / their 0-0.8
  grid), families .8128/.7956/.7503 vs their best (paired consistency) 0.7706 / 0.8107, .7954/.7732/.7432; their control 0.7390 / 0.7852.
- Their suite, their file, our PF-trained champion: 0.7792 / 0.8195 (never trained on that acceptance).
- Their suite, PF file: their best 0.7702 vs our champion 0.7990; their families are milder than labelled there (their code paints on ±3).
- Their published 0.8224 for their best used a different probe; our probe scores that checkpoint 0.8107 on their grid. Never compare across probes.
- Their robust training works: best beats control on every family on both files.

## Findings that stand regardless of promotion
1. Training-seed spread is the binding noise (0.0030), not probe refits; every future ordering needs seeds on both sides.
2. Checkpoint selection under an unseeded validation corruption was ~11x noise-dominated; seeding it removes the noise (criterion smoothness
   3x, p = 1/35) and buys reproducibility, not accuracy (best-vs-last on the champion: +0.0019 and +0.0010, positive both times).
3. The champion's generator effectively never removed >70% of an event's candidates (2.7% of steps); a third of that deficit was calibration
   error in our own families (edge-peaked eta density; area-sum vs union; p<1 kill ceiling). Fixed in WP-J's on-target placement.
4. Capacity is not the binding constraint: embed 256 (3.83x) lands on the reference mean; latent 32 uses ~2 effective dimensions.
5. The probe uses latent magnitude, not only direction: normalising the latent costs 0.026 while hiding a 22x offset.
6. Two-view latent shrink is caused by the MSE consistency term, not by the two-view batch; the offset penalty mildly opposes it.
7. Whitened consistency has a degenerate minimiser (rank-1 collapse).
8. Class imbalance (4.6x) is unweighted in the champion; only meaningful on the full file.

## Infrastructure record
Durability: four checkpoints named by runs/ JSONs had vanished with session scratchpads within two hours of their rows being published;
28 surviving checkpoints (every campaign-2 arm, both Phase 2 candidates, the G6a variants) were rescued to checkpoints/c2/rescued/ with sha256
and .origin sidecars, and the two campaign-1 checkpoints thought lost (a-pma, the Deep Sets runner-up) were recovered from their submission
branches, where accept.sh had asserted byte-identity with the scored files. The shipped submission was never at risk (three independent copies).
Rule adopted: a result is only as reproducible as the least durable artefact it depends on; every checkpoint a published row names goes to
durable storage before publication.
Scheduler faults found and fixed tonight (queued-job abandonment, two vanishing-registry races, bench-token deadlock, drain exit code,
priority authorisation), the held-out ellipse name collision (resolver order), accept.sh's multi-GPU SIGPIPE race. Register classes 13-2x in runs/plots/README.md.

## FPGA line (WP-G)
Folded token loop (G6b, ReuseFactor 400, mask emitted once per token): the N=400 design converts in 6 s, compiles in 183 s (campaign 1: killed
unfinished at 2,710 s) and its C-simulation is bit-exact against Keras at the full 400-candidate budget (max|d| 0, 100% of elements, 16 events,
xcvu13p-flga2577-2-e, 200 MHz). io_stream is unreachable for any HGQ2 model in hls4ml 1.3.0 (three independent blockers). Per-channel vs
per-tensor activation quantization ties at three seeds (0.8086 ± 0.0013 vs 0.8087 ± 0.0009) at +4.6% EBOPs for homogeneity. Vitis HLS
C-synthesis at N=400 hit its 3 h cap (08:08) inside the Unroll/Inline phase with no utilisation, latency or II; an N=1 calibration run
showed the phase is N-independent (383,350 LLVM instructions at N=1; the two 512-wide tail quantizers cost ~152 instructions per ap_fixed
conversion), i.e. this architecture through hls4ml's io_parallel path in Vitis 2024.2, not the token count. The N=400 run continues to a
11:10 cap as the deliverable; restructuring the tail quantizers (quantize before the flatten, or a per-tensor tail) is the next change.
Fit proxy from the emitted firmware (multiplier counts exact, LUT figures a proxy): the folded design instantiates 107,776 multipliers
(per-candidate datapath 38,656 iterated 400x; out_proj 65,536 because hls4ml unrolls its four seed positions; combine 512; bottleneck 3,072)
against 15.7M for the unfolded design. On xcvu13p (12,288 DSP48E2, 1.73M LUTs) most multipliers are LUT logic; at 25-60 LUTs each the
configured design needs 2.7-6.5M LUTs and probably does NOT fit as is. Folding out_proj (one line) gives 58,624 multipliers (1.5-3.5M LUTs,
straddling the budget); ReuseFactor 4 on the two 128x128 per-candidate layers gives 34,048 (measured from the emitted firmware of the committed
configuration quant/hls_config/g6b_N400_fit: LUT proxy 0.85-2.04M vs 1.73M, i.e. straddling rather than over; C-simulation bit-exact at N=400),
at a latency floor of 1,600 cycles (8 µs at 200 MHz); ReuseFactor 8 halves it again at 16 µs. Measured utilisation, latency and II were NOT obtained: the pf=1 N=400 C-synthesis was stopped at its 11:10 cap after about
6 h 10 min, still inside Vitis's Unroll/Inline phase (partial logs kept). On this box, this architecture through hls4ml's io_parallel path
does not reach a synthesis report; the fit answer rests on the exact multiplier counts and the LUT proxy above, and the first change for
anyone continuing is the tail quantizers. Vivado out-of-context synthesis is blocked by the
container's missing /run/udev (licence-library crash); a no-op libudev stand-in was not built pending the user's decision, so timing closure at
200 MHz is unproven. HLS-only deliverable until then.
