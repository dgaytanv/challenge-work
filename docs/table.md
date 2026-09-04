# Robust tagging ablation table

> **The floor that matters is the TRAINING SEED, not the probe.** Campaign-2 reference `h-ref-small` (champion config, small file, three seeds): 0.8250 / 0.8296 / 0.8239 (`runs/h-ref-small-s{11,22,33}_*.json`), **seed mean 0.8262, seed std 0.0030, range 0.0057**, clean 0.9220 ± 0.0033. The three-seed separability bar against another n=3 configuration is therefore **about 0.0074** -- an order of magnitude above the per-refit probe sigma (0.0008 transformer, 0.0020-0.0038 set encoders) that campaign 1 thresholded on.

> **Campaign-1 reading note** (planner, binding; also in `writeup/RULING.md` addendum 04:33). Campaign-1 bench separabilities between single training runs were computed against probe noise only; **gaps under about 0.006 are unproven**, including `a-pma` vs the winner (0.0022) and the winner vs the runner-up (0.0074, at the bar). The winner-vs-anchor gap (0.0526) stands. Unproven means not established by the evidence available, not refuted -- none of those pairs was measured with seeds.

Generated from `/home/jovyan/hackathon-shared/runs` by `bench/ablation_table.py`. 115 bench run(s), 19 official run(s). Delta is vs `anchor-stock-baseline` (mean_area 0.7734).

`mean_area` is our five-family development bench (`bench_eval.py`) and is the number we steer on. `official area` comes from `accept.sh` running the organisers' `eval.py`, which now imports their `degradation_eval.py`. The two are **not comparable to each other**: different corruptions, different sweeps. The official grid is marked TEMPORARY upstream and is stated to change before judging, so nothing is tuned to it. Its Bernoulli drop also uses the global RNG unseeded, so repeated runs of the same checkpoint differ; where a tag has more than one official run the spread is shown.

> **The top row by `mean_area` is NOT the submission.** The shipped model is the row marked **[CERTIFIED SUBMISSION]**. The two are statistically indistinguishable; the smaller one was kept under a margin rule fixed before the deciding number existed, and it is 27.3x smaller. Details below. The separability column compares each row to the top-`mean_area` row, which is a ranking aid, not a verdict on what shipped.

**FINAL RULING.** `submission-pma0-meanpt` @`41d45a9` is the certified submission (certified tip = shipped tip; two fresh-clone PASS runs, official 0.8780 full / 0.8795 on the shipped hash). It is retained, but **not because it measured better.** Against `a-pma` (transformer body + PMA readout + *stock* preprocessor), **neither metric separates this pair at the measured floors** (planner, campaign 2, 04:33). Bench 0.8260 vs 0.8275 is a gap of 0.0022 against a three-seed bar of about 0.0074; official 0.8788 (n=2) vs 0.8793 (n=1) is inside the observed run-to-run spread. The earlier reading -- that the bench and the official metric disagreed, with a-pma separably ahead on bench and held-out -- is **withdrawn**: those separabilities were computed against probe noise only, and campaign 2 measured the training-seed floor that dominates it. The retention rests on three things that are not the metric: the **pre-registered margin rule** adopted by the user (both a-pma officials had to exceed 0.8820; run 1 at 0.8793 fails it), the fact that the certified chain is **already complete** on a shipped tip, and a **27.3x size difference** (89,606 encoder parameters against 2,445,478) at statistically indistinguishable robustness -- which for a Fast ML challenge is the more decision-relevant axis once the metric is exhausted. Stated plainly: the two models reach the same place by different routes, a-pma has a slight and partly separable edge on the measurements, and the smaller model was kept under a rule fixed before the deciding number existed.

**Tie-break order** (planner ruling, amended 22:2x before held-out landed): **1.** `mean_area` over probe refits. **2.** official area. **3.** lower variance of the *scored* quantity -- official spread first, then bench sigma. **4.** clean AUC. Tie-break 3 precedes clean AUC because the grading is a single unseeded draw, so a candidate whose own runs scatter widely is a worse bet at equal mean; and because clean AUC is one point of an eleven-point sweep and is not the quantity being scored. A tie created by one candidate's variance must not then be resolved in that candidate's favour by an unscored number. Every comparison at every step uses the 3-sigma rule below; a step only passes to the next when the candidates are **not** separable at that step.

**Two floors.** The `±s` below is the PROBE floor -- noise in scoring a fixed checkpoint. Comparisons between two independently TRAINED models carry a second, larger floor: WP-G's `gB-q-b1e-6-l1t` and `gD-bits10-l1t` are the same configuration trained twice and differ by 0.0008, and that pair used a short distillation schedule from a shared seed checkpoint, so it is a lower bound. **Practical reading: a bench gap below ~0.002 between two independently trained full models is unproven, not established.** The ruling's comparisons clear it (0.0074 winner-vs-runner-up; 0.0022 a-pma at R=20).

`eval.py` seeds nothing, so its linear probe is refit differently on every run and every area inherits that noise. `mean_area ±s` is the standard deviation over `--probe_repeats` refits of one frozen set of embeddings; rows marked _(n=1)_ predate that flag and are single fits with an unmeasured error. **Do not rule between two arms whose mean_area differs by less than ~3s.** The per-severity errors are correlated (one probe is applied frozen across all severities), so threshold on the mean_area s rather than propagating per-point errors, which would double-count.



> **Checkpoint availability.** A published row is only as reproducible as the checkpoint it names. Four checkpoints named by rows below vanished within about two hours of being written, from clones and session scratchpads rather than the durable store; the surviving 28 were copied into `checkpoints/c2/rescued/` with digests. Note the limitation of the list that follows: every row in it predates the `ckpt_sha256` field, so it is matched by FILENAME, and some of these bytes do exist in the store under a different name (recovered from the submission branches). 'Absent' here means 'not resolvable from this row', which is the property that matters for re-benching it, not 'the bytes are destroyed'.
> 
> **Not re-benchable — checkpoint absent and no file of that name in the durable store:**
> - `a-pma` — `robust_tagging_encoder_20260903_220600.pth` (no recorded sha256, so recovery cannot even be checked)
> - `a-pma-heldout` — `robust_tagging_encoder_20260903_220600.pth` (no recorded sha256, so recovery cannot even be checked)
> - `a-pma-rep20` — `robust_tagging_encoder_20260903_220600.pth` (no recorded sha256, so recovery cannot even be checked)
> - `c-deepsets-meanpt` — `rt_d_deepsets_aug_meanpt_encoder_20260903_214738.pth` (no recorded sha256, so recovery cannot even be checked)
> - `c-deepsets-meanpt-l1t` — `rt_d_deepsets_aug_meanpt_encoder_20260903_214738.pth` (no recorded sha256, so recovery cannot even be checked)
> - `c-deepsets-meanpt-l1t-eta3` — `rt_d_deepsets_aug_meanpt_encoder_20260903_214738.pth` (no recorded sha256, so recovery cannot even be checked)
> - `c-winner` — `rt_d_pma0_aug_meanpt_encoder_20260903_221125.pth` (no recorded sha256, so recovery cannot even be checked)
> - `c-winner-l1t` — `rt_d_pma0_aug_meanpt_encoder_20260903_221125.pth` (no recorded sha256, so recovery cannot even be checked)
> - `c-winner-l1t-eta3` — `rt_d_pma0_aug_meanpt_encoder_20260903_221125.pth` (no recorded sha256, so recovery cannot even be checked)
> - `d-pma0-aug-meanpt-rep20` — `rt_d_pma0_aug_meanpt_encoder_20260903_221125.pth` (no recorded sha256, so recovery cannot even be checked)
> - `deepsets-meanpt-l1t` — `rt_d_deepsets_aug_meanpt_encoder_20260903_214738.pth` (no recorded sha256, so recovery cannot even be checked)
> - `deepsets-meanpt-l1t-eta3` — `rt_d_deepsets_aug_meanpt_encoder_20260903_214738.pth` (no recorded sha256, so recovery cannot even be checked)
> - `winner-l1t` — `rt_d_pma0_aug_meanpt_encoder_20260903_221125.pth` (no recorded sha256, so recovery cannot even be checked)
> - `winner-l1t-eta3` — `rt_d_pma0_aug_meanpt_encoder_20260903_221125.pth` (no recorded sha256, so recovery cannot even be checked)



_Corruption fingerprints: no mismatches. 205 family-row(s) predate the fingerprint field and are unverified rather than verified-clean._



## Per-configuration (campaign 2: 3 seeds per configuration)

Rows are grouped by configuration (tag without its `-s<seed>` suffix) and by data regime (`train_data`) and suite; seeds are never pooled across regimes or suites. **The ± here is the SEED std, not the probe std** -- a different and larger quantity, shown beside it. A configuration with one seed has an *unmeasured* seed std, printed `—`, and the seed test then reports `unmeasurable` rather than a verdict: n=1 is an anecdote, not a tie.

Separability is the pre-registered seed test, `|dmean| > 3*sqrt(sa^2/na + sb^2/nb)` with s the seed std. At n=3 that std carries 2 degrees of freedom and is itself noisy, so a verdict near the threshold should be read as 'not resolved by three seeds', not as a fact about the models. **The error runs both ways: a seed std that happens to be tiny at n=3 does not license a claim either** -- three accidentally close runs shrink the threshold and can separate a pair that a fourth seed would re-tie. Campaign 1 quoted an n=2 range as if it were a variance and had to retract the ordering it implied.

| configuration | regime | suite | seeds | mean_area (seed mean ± seed std) | probe std | clean | params | vs reference |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `i-lat32` | robust_tagging_train_data_small.pt | pf/scoring | 3 (11,22,33) | 0.8292 ± 0.0003 | 0.0017 | 0.9270 | 102,944 | +0.0029 vs 0.0052 — **not separable** |
| `i-ff256l1` | robust_tagging_train_data_small.pt | pf/scoring | 3 (11,22,33) | 0.8287 ± 0.0015 | 0.0006 | 0.9253 | 222,254 | +0.0025 vs 0.0058 — **not separable** |
| `l-l2-dxysig` | robust_tagging_train_data_small.pt | pf/scoring | 1 (11) | 0.8278 ± — | 0.0008 | 0.9266 | 89,606 | +0.0015 — unmeasurable (n=1 somewhere) |
| `a-pma-rep20` | unrecorded | pf/scoring | 1 (None) | 0.8277 ± — | 0.0017 | 0.9230 | — | no reference |
| `j1` | robust_tagging_train_data_small.pt | pf/scoring | 3 (11,22,33) | 0.8277 ± 0.0017 | 0.0007 | 0.9219 | 89,606 | +0.0014 vs 0.0059 — **not separable** |
| `h-ref-small-s22-last` | robust_tagging_train_data_small.pt | pf/scoring | 1 (22) | 0.8276 ± — | 0.0006 | 0.9214 | 89,606 | +0.0013 — unmeasurable (n=1 somewhere) |
| `a-pma` | unrecorded | pf/scoring | 1 (None) | 0.8275 ± — | 0.0006 | 0.9228 | — | no reference |
| `l-l1-sincos` | robust_tagging_train_data_small.pt | pf/scoring | 3 (11,22,33) | 0.8270 ± 0.0026 | 0.0007 | 0.9202 | 89,734 | +0.0008 vs 0.0068 — **not separable** |
| `i-seeds8` | robust_tagging_train_data_small.pt | pf/scoring | 3 (11,22,33) | 0.8269 ± 0.0014 | 0.0007 | 0.9235 | 94,214 | +0.0006 vs 0.0057 — **not separable** |
| `j3` | robust_tagging_train_data_small.pt | pf/scoring | 1 (11) | 0.8264 ± — | 0.0006 | 0.9246 | 89,606 | +0.0002 — unmeasurable (n=1 somewhere) |
| `h-ref-small` | robust_tagging_train_data_small.pt | pf/scoring | 3 (11,22,33) | 0.8262 ± 0.0030 | 0.0006 | 0.9219 | 89,606 | *(reference)* |
| `l-l3-ema` | robust_tagging_train_data_small.pt | pf/scoring | 1 (11) | 0.8260 ± — | 0.0007 | 0.9228 | 89,606 | -0.0002 — unmeasurable (n=1 somewhere) |
| `d-pma0-aug-meanpt` | unrecorded | pf/scoring | 1 (None) | 0.8260 ± — | 0.0009 | 0.9147 | — | no reference |
| `i-embed256` | robust_tagging_train_data_small.pt | pf/scoring | 3 (11,22,33) | 0.8257 ± 0.0028 | 0.0009 | 0.9216 | 343,046 | -0.0006 vs 0.0071 — **not separable** |
| `d-pma0-aug-meanpt-rep20` | unrecorded | pf/scoring | 1 (None) | 0.8255 ± — | 0.0007 | 0.9137 | — | no reference |
| `j5` | robust_tagging_train_data_small.pt | pf/scoring | 1 (11) | 0.8251 ± — | 0.0006 | 0.9197 | 89,606 | -0.0011 — unmeasurable (n=1 somewhere) |
| `k-k3_jsd` | robust_tagging_train_data_small.pt | pf/scoring | 1 (11) | 0.8251 ± — | 0.0009 | 0.9224 | 89,606 | -0.0011 — unmeasurable (n=1 somewhere) |
| `h-ref-small-s11-last` | robust_tagging_train_data_small.pt | pf/scoring | 1 (11) | 0.8248 ± — | 0.0006 | 0.9229 | 89,606 | -0.0014 — unmeasurable (n=1 somewhere) |
| `l-l4-valseed` | robust_tagging_train_data_small.pt | pf/scoring | 1 (11) | 0.8244 ± — | 0.0008 | 0.9224 | 89,606 | -0.0018 — unmeasurable (n=1 somewhere) |
| `n-audit-h-ref-small` | unrecorded | pf/scoring | 1 (33) | 0.8240 ± — | 0.0012 | 0.9190 | 89,606 | no reference |
| `j2` | robust_tagging_train_data_small.pt | pf/scoring | 1 (11) | 0.8240 ± — | 0.0008 | 0.9142 | 89,606 | -0.0023 — unmeasurable (n=1 somewhere) |
| `k-k1_offpen` | robust_tagging_train_data_small.pt | pf/scoring | 1 (11) | 0.8237 ± — | 0.0006 | 0.9198 | 89,606 | -0.0025 — unmeasurable (n=1 somewhere) |
| `h-ref-full` | robust_tagging_train_data.pt | pf/scoring | 1 (11) | 0.8225 ± — | 0.0017 | 0.9254 | 89,606 | no reference |
| `d-deepsets-aug-meanpt` | unrecorded | pf/scoring | 1 (None) | 0.8186 ± — | 0.0007 | 0.9066 | — | no reference |
| `d-deepsets-aug` | unrecorded | pf/scoring | 1 (None) | 0.8183 ± — | — | 0.8920 | — | no reference |
| `d-deepsets-aug-rep10` | unrecorded | pf/scoring | 1 (None) | 0.8174 ± — | 0.0020 | 0.8921 | — | no reference |
| `d-pma0-aug` | unrecorded | pf/scoring | 1 (None) | 0.8162 ± — | 0.0038 | 0.9215 | — | no reference |
| `h-ref-small-s22-l1t` | robust_tagging_train_data_small.pt | l1t/scoring | 1 (22) | 0.8118 ± — | 0.0004 | 0.9086 | 89,606 | no reference |
| `d-pma0-aug-meanpt-l1t` | robust_tagging_train_data_small_l1t.pt | l1t/scoring | 1 (None) | 0.8117 ± — | 0.0003 | 0.9067 | — | no reference |
| `h-ref-small-s22-last-l1t` | robust_tagging_train_data_small.pt | l1t/scoring | 1 (22) | 0.8111 ± — | 0.0005 | 0.9060 | 89,606 | no reference |
| `e-stock-maskfix` | unrecorded | pf/scoring | 1 (None) | 0.8089 ± — | 0.0004 | 0.9025 | — | no reference |
| `gF-bits6-relu-l1t` | robust_tagging_train_data_small_l1t.pt | l1t/scoring | 1 (None) | 0.8089 ± — | 0.0005 | 0.8991 | — | no reference |
| `h-ref-small-s33-l1t` | robust_tagging_train_data_small.pt | l1t/scoring | 1 (33) | 0.8088 ± — | 0.0004 | 0.9035 | 89,606 | no reference |
| `gD-bits10-l1t` | robust_tagging_train_data_small_l1t.pt | l1t/scoring | 1 (None) | 0.8087 ± — | 0.0003 | 0.9025 | — | no reference |
| `g6a-hom-bits6-relu-l1t` | robust_tagging_train_data_small_l1t.pt | l1t/scoring | 3 (11,22,33) | 0.8087 ± 0.0009 | 0.0004 | 0.8977 | 0 | no reference |
| `g6a-het-bits6-relu-l1t` | robust_tagging_train_data_small_l1t.pt | l1t/scoring | 3 (11,22,33) | 0.8086 ± 0.0013 | 0.0006 | 0.8980 | 0 | no reference |
| `h-ref-small-s11-l1t` | robust_tagging_train_data_small.pt | l1t/scoring | 1 (11) | 0.8084 ± — | 0.0009 | 0.9044 | 89,606 | no reference |
| `h-ref-small-s11-last-l1t` | robust_tagging_train_data_small.pt | l1t/scoring | 1 (11) | 0.8083 ± — | 0.0007 | 0.9043 | 89,606 | no reference |
| `gD-bits6-l1t` | robust_tagging_train_data_small_l1t.pt | l1t/scoring | 1 (None) | 0.8080 ± — | 0.0001 | 0.8978 | — | no reference |
| `gB-q-b1e-6-l1t` | robust_tagging_train_data_small_l1t.pt | l1t/scoring | 1 (None) | 0.8079 ± — | 0.0011 | 0.9019 | — | no reference |
| `g-stageA-l1t` | robust_tagging_train_data_small_l1t.pt | l1t/scoring | 1 (None) | 0.8078 ± — | 0.0007 | 0.8976 | — | no reference |
| `gD-relu-l1t` | robust_tagging_train_data_small_l1t.pt | l1t/scoring | 1 (None) | 0.8078 ± — | 0.0009 | 0.9014 | — | no reference |
| `l-l1-sincos-s11-heldout` | robust_tagging_train_data_small.pt | pf/heldout | 1 (11) | 0.8067 ± — | 0.0005 | 0.9249 | 89,734 | no reference |
| `winner-l1t-eta3` | unrecorded | l1t/scoring | 1 (None) | 0.8063 ± — | 0.0006 | 0.8986 | — | no reference |
| `a-pma-heldout` | unrecorded | pf/heldout | 1 (None) | 0.8059 ± — | 0.0016 | 0.9226 | — | no reference |
| `b-aug-stock` | unrecorded | pf/scoring | 1 (None) | 0.8058 ± — | 0.0043 | 0.9035 | — | no reference |
| `j1-s11-heldout` | robust_tagging_train_data_small.pt | pf/heldout | 1 (11) | 0.8053 ± — | 0.0003 | 0.9218 | 89,606 | no reference |
| `h-ref-small-s22-heldout` | robust_tagging_train_data_small.pt | pf/heldout | 1 (22) | 0.8050 ± — | 0.0010 | 0.9249 | 89,606 | no reference |
| `winner-l1t` | unrecorded | l1t/scoring | 1 (None) | 0.8046 ± — | 0.0008 | 0.8973 | — | no reference |
| `gE-stageA-maskedbn-l1t` | robust_tagging_train_data_small_l1t.pt | l1t/scoring | 1 (None) | 0.8044 ± — | 0.0007 | 0.8924 | — | no reference |
| `h-ref-small-s11-last-heldout` | robust_tagging_train_data_small.pt | pf/heldout | 1 (11) | 0.8043 ± — | 0.0005 | 0.9231 | 89,606 | no reference |
| `h-ref-small-s11-heldout` | robust_tagging_train_data_small.pt | pf/heldout | 1 (11) | 0.8042 ± — | 0.0010 | 0.9227 | 89,606 | no reference |
| `deepsets-meanpt-l1t-eta3` | unrecorded | l1t/scoring | 1 (None) | 0.8042 ± — | 0.0019 | 0.8909 | — | no reference |
| `d-deepsets-clean` | unrecorded | pf/scoring | 1 (None) | 0.8036 ± — | — | 0.8853 | — | no reference |
| `h-ref-small-s22-last-heldout` | robust_tagging_train_data_small.pt | pf/heldout | 1 (22) | 0.8035 ± — | 0.0006 | 0.9216 | 89,606 | no reference |
| `d-pma0-aug-meanpt-heldout` | unrecorded | pf/heldout | 1 (None) | 0.8021 ± — | 0.0006 | 0.9134 | — | no reference |
| `deepsets-meanpt-l1t` | unrecorded | l1t/scoring | 1 (None) | 0.8021 ± — | 0.0016 | 0.8908 | — | no reference |
| `l-l1-sincos-s11-colleague` | robust_tagging_train_data_small.pt | pf/colleague | 1 (11) | 0.8019 ± — | 0.0006 | 0.9253 | 89,734 | no reference |
| `d-deepsets-clean-full` | unrecorded | pf/scoring/full | 1 (None) | 0.8017 ± — | — | 0.8887 | — | no reference |
| `h-ref-small-s33-heldout` | robust_tagging_train_data_small.pt | pf/heldout | 1 (33) | 0.8014 ± — | 0.0011 | 0.9185 | 89,606 | no reference |
| `k-k2_normlatent` | robust_tagging_train_data_small.pt | pf/scoring | 1 (33) | 0.7999 ± — | 0.0032 | 0.8781 | 89,606 | -0.0264 — unmeasurable (n=1 somewhere) |
| `h-ref-full-s11-heldout` | robust_tagging_train_data.pt | pf/heldout | 1 (11) | 0.7996 ± — | 0.0020 | 0.9257 | 89,606 | no reference |
| `j1-s11-colleague` | robust_tagging_train_data_small.pt | pf/colleague | 1 (11) | 0.7994 ± — | 0.0009 | 0.9214 | 89,606 | no reference |
| `h-ref-small-s11-last-colleague` | robust_tagging_train_data_small.pt | pf/colleague | 1 (11) | 0.7994 ± — | 0.0003 | 0.9230 | 89,606 | no reference |
| `h-ref-small-s11-colleague` | robust_tagging_train_data_small.pt | pf/colleague | 1 (11) | 0.7993 ± — | 0.0007 | 0.9227 | 89,606 | no reference |
| `c-winner` | unrecorded | pf/colleague | 1 (None) | 0.7990 ± — | 0.0003 | 0.9137 | — | no reference |
| `d-pma0-aug-heldout` | unrecorded | pf/heldout | 1 (None) | 0.7983 ± — | 0.0048 | 0.9202 | — | no reference |
| `h-ref-small-s22-colleague` | robust_tagging_train_data_small.pt | pf/colleague | 1 (22) | 0.7982 ± — | 0.0010 | 0.9247 | 89,606 | no reference |
| `h-ref-full-s11-colleague` | robust_tagging_train_data.pt | pf/colleague | 1 (11) | 0.7979 ± — | 0.0018 | 0.9258 | 89,606 | no reference |
| `d-deepsets-aug-meanpt-heldout` | unrecorded | pf/heldout | 1 (None) | 0.7977 ± — | 0.0009 | 0.9067 | — | no reference |
| `h-ref-small-s33-colleague` | robust_tagging_train_data_small.pt | pf/colleague | 1 (33) | 0.7974 ± — | 0.0004 | 0.9182 | 89,606 | no reference |
| `h-ref-small-s22-last-colleague` | robust_tagging_train_data_small.pt | pf/colleague | 1 (22) | 0.7973 ± — | 0.0007 | 0.9217 | 89,606 | no reference |
| `d-deepsets-aug-heldout` | unrecorded | pf/heldout | 1 (None) | 0.7962 ± — | 0.0011 | 0.8917 | — | no reference |
| `planner-maskfix-stockckpt` | unrecorded | pf/scoring | 1 (None) | 0.7918 ± — | — | 0.8604 | — | no reference |
| `a-maskfix-stockckpt` | unrecorded | pf/scoring | 1 (None) | 0.7899 ± — | — | 0.8572 | — | no reference |
| `c-deepsets-meanpt` | unrecorded | pf/colleague | 1 (None) | 0.7883 ± — | 0.0014 | 0.9068 | — | no reference |
| `d-pma0-aug-meanpt-l1t-colleague` | robust_tagging_train_data_small_l1t.pt | l1t/colleague | 1 (None) | 0.7862 ± — | 0.0006 | 0.9063 | 89,606 | no reference |
| `h-ref-small-s22-l1t-colleague` | robust_tagging_train_data_small.pt | l1t/colleague | 1 (22) | 0.7847 ± — | 0.0004 | 0.9090 | 89,606 | no reference |
| `h-ref-small-s22-last-l1t-colleague` | robust_tagging_train_data_small.pt | l1t/colleague | 1 (22) | 0.7835 ± — | 0.0007 | 0.9060 | 89,606 | no reference |
| `h-ref-small-s33-l1t-colleague` | robust_tagging_train_data_small.pt | l1t/colleague | 1 (33) | 0.7816 ± — | 0.0003 | 0.9037 | 89,606 | no reference |
| `h-ref-small-s11-last-l1t-colleague` | robust_tagging_train_data_small.pt | l1t/colleague | 1 (11) | 0.7807 ± — | 0.0004 | 0.9045 | 89,606 | no reference |
| `h-ref-small-s11-l1t-colleague` | robust_tagging_train_data_small.pt | l1t/colleague | 1 (11) | 0.7807 ± — | 0.0004 | 0.9046 | 89,606 | no reference |
| `gD-bits4-l1t` | robust_tagging_train_data_small_l1t.pt | l1t/scoring | 1 (None) | 0.7797 ± — | 0.0018 | 0.8547 | — | no reference |
| `c-winner-l1t` | unrecorded | l1t/colleague | 1 (None) | 0.7794 ± — | 0.0007 | 0.8970 | — | no reference |
| `c-winner-l1t-eta3` | unrecorded | l1t/colleague | 1 (None) | 0.7792 ± — | 0.0004 | 0.8970 | — | no reference |
| `c-deepsets-meanpt-l1t` | unrecorded | l1t/colleague | 1 (None) | 0.7775 ± — | 0.0008 | 0.8915 | — | no reference |
| `c-deepsets-meanpt-l1t-eta3` | unrecorded | l1t/colleague | 1 (None) | 0.7770 ± — | 0.0012 | 0.8913 | — | no reference |
| `anchor-stock-baseline` | unrecorded | pf/scoring | 1 (None) | 0.7734 ± — | — | 0.8605 | — | no reference |
| `c-twoview-cos1` | unrecorded | pf/scoring | 1 (None) | 0.7712 ± — | 0.0034 | 0.8370 | — | no reference |
| `c3-best-colleague` | robust_tagging_train_data_small_l1t.pt | pf/colleague | 1 (None) | 0.7706 ± — | 0.0051 | 0.8990 | 2,375,846 | no reference |
| `c3-best-colleague-l1t` | robust_tagging_train_data_small_l1t.pt | l1t/colleague | 1 (None) | 0.7706 ± — | 0.0091 | 0.8905 | 2,375,846 | no reference |
| `d-deepsets-twoview-v2` | unrecorded | pf/scoring | 1 (None) | 0.7419 ± — | 0.0156 | 0.7804 | — | no reference |
| `c3-control-colleague-l1t` | robust_tagging_train_data_small_l1t.pt | l1t/colleague | 1 (None) | 0.7390 ± — | 0.0006 | 0.8729 | 2,375,846 | no reference |
| `c3-control-colleague` | robust_tagging_train_data_small_l1t.pt | pf/colleague | 1 (None) | 0.7185 ± — | 0.0015 | 0.8619 | 2,375,846 | no reference |
| `k-k5_whiten` | robust_tagging_train_data_small.pt | pf/scoring | 1 (11) | 0.7012 ± — | 0.0039 | 0.7682 | 89,606 | -0.1250 — unmeasurable (n=1 somewhere) |
| `gD-bits3-l1t` | robust_tagging_train_data_small_l1t.pt | l1t/scoring | 1 (None) | 0.5000 ± — | 0.0000 | 0.5000 | — | no reference |


| tag | commit | encoder | data | trained on | events | clean AUC | rect | wedge | strip | towers | cells | mean_area | delta | sep. from top mean_area | official area | official sep. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| i-ff256l1-s22 | 939ac19 | PMAEncoderFF256 | PF | PF | 20k | 0.9217 | 0.7740 | 0.8616 | 0.8093 | 0.8306 | 0.8765 | **0.8304** ±0.0003 | +0.0570 | **leader** | - | - |
| h-ref-small-s22 | 8dfe03a-dirty | PMAEncoder | PF | PF | 20k | 0.9249 | 0.7702 | 0.8608 | 0.8092 | 0.8311 | 0.8767 | **0.8296** ±0.0005 | +0.0562 | **no** (Δ0.0008 vs 0.0008) | - | - |
| j1-s11 | dc45142 | PMAEncoder | PF | PF | 20k | 0.9215 | 0.7734 | 0.8600 | 0.8092 | 0.8296 | 0.8757 | **0.8296** ±0.0006 | +0.0561 | **no** (Δ0.0008 vs 0.0009) | - | - |
| i-lat32-s22 | 939ac19 | PMAEncoder | PF | PF | 20k | 0.9266 | 0.7704 | 0.8603 | 0.8082 | 0.8309 | 0.8773 | **0.8294** ±0.0020 | +0.0560 | **no** (Δ0.0010 vs 0.0027) | - | - |
| i-lat32-s11 | 31b6219 | PMAEncoder | PF | PF | 20k | 0.9276 | 0.7639 | 0.8623 | 0.8087 | 0.8327 | 0.8781 | **0.8291** ±0.0015 | +0.0557 | **no** (Δ0.0013 vs 0.0021) | - | - |
| l-l1-sincos-s11 | f522f49 | PMAEncoder | PF | PF | 20k | 0.9247 | 0.7662 | 0.8609 | 0.8094 | 0.8314 | 0.8766 | **0.8289** ±0.0006 | +0.0555 | yes (Δ0.0015 vs 0.0009) | - | - |
| i-lat32-s33 | 939ac19 | PMAEncoder | PF | PF | 20k | 0.9269 | 0.7695 | 0.8592 | 0.8089 | 0.8302 | 0.8765 | **0.8289** ±0.0015 | +0.0555 | **no** (Δ0.0015 vs 0.0021) | - | - |
| i-ff256l1-s11 | 31b6219 | PMAEncoderFF256 | PF | PF | 20k | 0.9265 | 0.7671 | 0.8616 | 0.8051 | 0.8319 | 0.8775 | **0.8286** ±0.0010 | +0.0552 | yes (Δ0.0018 vs 0.0014) | - | - |
| i-seeds8-s11 | 31b6219 | PMAEncoder8 | PF | PF | 20k | 0.9247 | 0.7686 | 0.8601 | 0.8067 | 0.8305 | 0.8762 | **0.8284** ±0.0008 | +0.0550 | yes (Δ0.0020 vs 0.0012) | - | - |
| i-embed256-s33 | 939ac19 | PMAEncoder | PF | PF | 20k | 0.9238 | 0.7685 | 0.8596 | 0.8079 | 0.8293 | 0.8756 | **0.8282** ±0.0007 | +0.0548 | yes (Δ0.0022 vs 0.0011) | - | - |
| l-l1-sincos-s22 | f522f49 | PMAEncoder | PF | PF | 20k | 0.9209 | 0.7709 | 0.8583 | 0.8086 | 0.8281 | 0.8742 | **0.8280** ±0.0009 | +0.0546 | yes (Δ0.0024 vs 0.0012) | - | - |
| l-l2-dxysig-s11 | f522f49 | PMAEncoder | PF | PF | 20k | 0.9266 | 0.7669 | 0.8594 | 0.8067 | 0.8303 | 0.8755 | **0.8278** ±0.0008 | +0.0543 | yes (Δ0.0026 vs 0.0011) | - | - |
| a-pma-rep20 | d450bcf | TransformerEncoderPMA | PF | — | 20k | 0.9230 | 0.7686 | 0.8587 | 0.8072 | 0.8288 | 0.8748 | **0.8276** ±0.0017 | +0.0542 | yes (Δ0.0028 vs 0.0012) | 0.8783 (n=3, spread 0.0029) | **no** (Δ0.0006 vs 0.0030) |
| a-pma | d8cf58d | TransformerEncoderPMA | PF | — | 20k | 0.9228 | 0.7679 | 0.8589 | 0.8071 | 0.8288 | 0.8746 | **0.8275** ±0.0006 | +0.0540 | yes (Δ0.0029 vs 0.0009) | 0.8783 (n=3, spread 0.0029) | **no** (Δ0.0006 vs 0.0030) |
| h-ref-small-s22-last | f522f49 | PMAEncoder | PF | PF | 20k | 0.9216 | 0.7676 | 0.8588 | 0.8070 | 0.8291 | 0.8745 | **0.8274** ±0.0006 | +0.0540 | yes (Δ0.0030 vs 0.0009) | - | - |
| i-ff256l1-s33 | 939ac19 | PMAEncoderFF256 | PF | PF | 20k | 0.9278 | 0.7598 | 0.8617 | 0.8053 | 0.8317 | 0.8778 | **0.8273** ±0.0004 | +0.0538 | yes (Δ0.0031 vs 0.0007) | - | - |
| j1-s33 | dc45142 | PMAEncoder | PF | PF | 20k | 0.9244 | 0.7614 | 0.8599 | 0.8067 | 0.8304 | 0.8756 | **0.8268** ±0.0008 | +0.0534 | yes (Δ0.0036 vs 0.0011) | - | - |
| j1-s22 | dc45142 | PMAEncoder | PF | PF | 20k | 0.9196 | 0.7665 | 0.8580 | 0.8066 | 0.8277 | 0.8742 | **0.8266** ±0.0005 | +0.0532 | yes (Δ0.0038 vs 0.0008) | - | - |
| i-seeds8-s22 | 939ac19-dirty | PMAEncoder8 | PF | PF | 20k | 0.9241 | 0.7665 | 0.8574 | 0.8069 | 0.8277 | 0.8743 | **0.8266** ±0.0008 | +0.0531 | yes (Δ0.0038 vs 0.0011) | - | - |
| j3-s11 | dc45142 | PMAEncoder | PF | PF | 20k | 0.9246 | 0.7629 | 0.8589 | 0.8059 | 0.8294 | 0.8750 | **0.8264** ±0.0006 | +0.0530 | yes (Δ0.0040 vs 0.0009) | - | - |
| i-embed256-s11 | 31b6219 | PMAEncoder | PF | PF | 20k | 0.9244 | 0.7620 | 0.8582 | 0.8077 | 0.8288 | 0.8744 | **0.8262** ±0.0015 | +0.0528 | yes (Δ0.0042 vs 0.0021) | - | - |
| l-l3-ema-s11 | f522f49 | PMAEncoder | PF | PF | 20k | 0.9228 | 0.7625 | 0.8587 | 0.8055 | 0.8291 | 0.8743 | **0.8260** ±0.0007 | +0.0526 | yes (Δ0.0044 vs 0.0011) | - | - |
| d-pma0-aug-meanpt **[CERTIFIED SUBMISSION]** | 9801bdd | PMAEncoder | PF | — | 20k | 0.9147 | 0.7728 | 0.8546 | 0.8072 | 0.8252 | 0.8701 | **0.8260** ±0.0009 | +0.0525 | yes (Δ0.0044 vs 0.0013) | 0.8789 (n=8, spread 0.0031) _[d-pma0-aug-meanpt+m-accept-fixed3+m-accept-regression+submission-cert-c2+submission-final]_ | **leader** |
| i-seeds8-s33 | 939ac19 | PMAEncoder8 | PF | PF | 20k | 0.9216 | 0.7686 | 0.8555 | 0.8055 | 0.8264 | 0.8719 | **0.8256** ±0.0006 | +0.0522 | yes (Δ0.0048 vs 0.0010) | - | - |
| d-pma0-aug-meanpt-rep20 | 41d45a9 | PMAEncoder | PF | — | 20k | 0.9137 | 0.7724 | 0.8540 | 0.8070 | 0.8245 | 0.8696 | **0.8255** ±0.0007 | +0.0521 | yes (Δ0.0049 vs 0.0006) | 0.8789 (n=8, spread 0.0031) _[d-pma0-aug-meanpt+m-accept-fixed3+m-accept-regression+submission-cert-c2+submission-final]_ | **leader** |
| h-ref-small-s11 | f437155 | PMAEncoder | PF | PF | 20k | 0.9222 | 0.7616 | 0.8582 | 0.8051 | 0.8287 | 0.8738 | **0.8255** ±0.0009 | +0.0520 | yes (Δ0.0049 vs 0.0013) | - | - |
| k-k3_jsd-s11 | f386372 | PMAEncoder | PF | PF | 20k | 0.9225 | 0.7630 | 0.8581 | 0.8025 | 0.8288 | 0.8742 | **0.8253** ±0.0009 | +0.0519 | yes (Δ0.0051 vs 0.0012) | - | - |
| h-ref-small-s11-last | f522f49 | PMAEncoder | PF | PF | 20k | 0.9228 | 0.7607 | 0.8580 | 0.8047 | 0.8285 | 0.8738 | **0.8251** ±0.0006 | +0.0517 | yes (Δ0.0052 vs 0.0009) | - | - |
| j5-s11 | dc45142 | PMAEncoder | PF | PF | 20k | 0.9197 | 0.7630 | 0.8561 | 0.8067 | 0.8266 | 0.8732 | **0.8251** ±0.0006 | +0.0517 | yes (Δ0.0053 vs 0.0009) | - | - |
| l-l4-valseed-s11 | f522f49 | PMAEncoder | PF | PF | 20k | 0.9224 | 0.7587 | 0.8579 | 0.8025 | 0.8283 | 0.8746 | **0.8244** ±0.0008 | +0.0510 | yes (Δ0.0060 vs 0.0012) | - | - |
| l-l1-sincos-s33 | f522f49 | PMAEncoder | PF | PF | 20k | 0.9149 | 0.7624 | 0.8549 | 0.8062 | 0.8258 | 0.8712 | **0.8241** ±0.0007 | +0.0507 | yes (Δ0.0063 vs 0.0010) | - | - |
| n-audit-h-ref-small-s33 | 8dfe03a | PMAEncoder | PF | — | 20k | 0.9190 | 0.7635 | 0.8562 | 0.8028 | 0.8255 | 0.8720 | **0.8240** ±0.0012 | +0.0506 | yes (Δ0.0064 vs 0.0016) | - | - |
| j2-s11 | dc45142 | PMAEncoder | PF | PF | 20k | 0.9142 | 0.7646 | 0.8547 | 0.8045 | 0.8248 | 0.8714 | **0.8240** ±0.0008 | +0.0506 | yes (Δ0.0064 vs 0.0011) | - | - |
| h-ref-small-s33 | 8dfe03a-dirty | PMAEncoder | PF | PF | 20k | 0.9184 | 0.7641 | 0.8561 | 0.8026 | 0.8250 | 0.8717 | **0.8239** ±0.0003 | +0.0505 | yes (Δ0.0065 vs 0.0006) | - | - |
| k-k1_offpen-s11 | eb64f35 | PMAEncoder | PF | PF | 20k | 0.9198 | 0.7561 | 0.8575 | 0.8024 | 0.8285 | 0.8740 | **0.8237** ±0.0006 | +0.0503 | yes (Δ0.0067 vs 0.0009) | - | - |
| i-embed256-s22 | 939ac19-dirty | PMAEncoder | PF | PF | 20k | 0.9166 | 0.7617 | 0.8544 | 0.8022 | 0.8246 | 0.8700 | **0.8226** ±0.0006 | +0.0492 | yes (Δ0.0078 vs 0.0009) | - | - |
| h-ref-full-s11 | f437155 | PMAEncoder | PF | robust_tag | 20k | 0.9248 | 0.7637 | 0.8544 | 0.7974 | 0.8256 | 0.8718 | **0.8226** ±0.0017 | +0.0491 | yes (Δ0.0078 vs 0.0023) | - | - |
| d-deepsets-aug-meanpt | 55f0534 | DeepSetsEncoder | PF | — | 20k | 0.9066 | 0.7538 | 0.8520 | 0.7985 | 0.8223 | 0.8664 | **0.8186** ±0.0007 | +0.0452 | yes (Δ0.0118 vs 0.0010) | 0.8703 (n=3, spread 0.0007) | yes (Δ0.0086 vs 0.0012) |
| d-deepsets-aug | 975fd5d | DeepSetsEncoder | PF | — | 20k | 0.8920 | 0.7751 | 0.8424 | 0.8011 | 0.8141 | 0.8588 | **0.8183** _(n=1)_ | +0.0449 | yes (Δ0.0121 vs 0.0114) | 0.8598 (n=2, spread 0.0024) | yes (Δ0.0191 vs 0.0037) |
| d-deepsets-aug-rep10 | 55f0534 | DeepSetsEncoder | PF | — | 20k | 0.8921 | 0.7754 | 0.8401 | 0.8009 | 0.8127 | 0.8577 | **0.8174** ±0.0020 | +0.0440 | yes (Δ0.0130 vs 0.0020) | 0.8598 (n=2, spread 0.0024) | yes (Δ0.0191 vs 0.0037) |
| d-pma0-aug | 6c027fd | PMAEncoder | PF | — | 20k | 0.9215 | 0.7627 | 0.8443 | 0.7944 | 0.8177 | 0.8619 | **0.8162** ±0.0038 | +0.0428 | yes (Δ0.0142 vs 0.0051) | 0.8669 (n=3, spread 0.0089) | yes (Δ0.0120 vs 0.0085) |
| e-stock-maskfix | 2d2e3b6 | TransformerEncoder | PF | — | 20k | 0.9025 | 0.7462 | 0.8414 | 0.7875 | 0.8115 | 0.8578 | **0.8089** ±0.0004 | +0.0355 | yes (Δ0.0215 vs 0.0007) | - | - |
| b-aug-stock | 8ee2655 | TransformerEncoder | PF | — | 20k | 0.9035 | 0.7479 | 0.8346 | 0.7859 | 0.8066 | 0.8542 | **0.8058** ±0.0043 | +0.0324 | yes (Δ0.0246 vs 0.0059) | - | - |
| d-deepsets-clean | 8235247 | DeepSetsEncoder | PF | — | 20k | 0.8853 | 0.7438 | 0.8328 | 0.7820 | 0.8084 | 0.8510 | **0.8036** _(n=1)_ | +0.0302 | yes (Δ0.0268 vs 0.0114) | - | - |
| d-deepsets-clean-full | 41623cb | DeepSetsEncoder | PF | — | full | 0.8887 | 0.7391 | 0.8316 | 0.7789 | 0.8080 | 0.8506 | **0.8017** _(n=1)_ | +0.0282 | yes (Δ0.0287 vs 0.0114) | - | - |
| k-k2_normlatent-s33 | f386372 | PMAEncoderNormLatent | PF | PF | 20k | 0.8781 | 0.7428 | 0.8307 | 0.7810 | 0.8012 | 0.8438 | **0.7999** ±0.0032 | +0.0265 | yes (Δ0.0305 vs 0.0043) | - | - |
| planner-maskfix-stockckpt | 74a4064 | TransformerEncoder | PF | — | 20k | 0.8604 | 0.7570 | 0.7961 | 0.7866 | 0.7845 | 0.8347 | **0.7918** _(n=1)_ | +0.0183 | yes (Δ0.0386 vs 0.0024) | - | - |
| a-maskfix-stockckpt | 36c1cd0 | TransformerEncoder | PF | — | 20k | 0.8572 | 0.7545 | 0.7947 | 0.7848 | 0.7835 | 0.8321 | **0.7899** _(n=1)_ | +0.0165 | yes (Δ0.0405 vs 0.0024) | - | - |
| anchor-stock-baseline | 25ff4d1 | TransformerEncoder | PF | — | 20k | 0.8605 | 0.7283 | 0.7982 | 0.7485 | 0.7737 | 0.8185 | **0.7734** _(n=1)_ | +0.0000 | yes (Δ0.0570 vs 0.0024) | - | - |
| c-twoview-cos1 | d6dd0fe | TransformerEncoder | PF | — | 20k | 0.8370 | 0.6824 | 0.8176 | 0.7481 | 0.7876 | 0.8204 | **0.7712** ±0.0034 | -0.0022 | yes (Δ0.0591 vs 0.0046) | - | - |
| d-deepsets-twoview-v2 | 9801bdd | DeepSetsEncoder | PF | — | 20k | 0.7804 | 0.6909 | 0.7696 | 0.7280 | 0.7432 | 0.7780 | **0.7419** ±0.0156 | -0.0315 | yes (Δ0.0885 vs 0.0209) | - | - |
| k-k5_whiten-s11 | f386372 | PMAEncoder | PF | PF | 20k | 0.7682 | 0.6550 | 0.7282 | 0.6831 | 0.6997 | 0.7402 | **0.7012** ±0.0039 | -0.0722 | yes (Δ0.1291 vs 0.0052) | - | - |

### L1T eval file (different acceptance, not trained on)

These models were trained and selected on the PF-candidate acceptance (`eta in [-5, 5]`). The L1T/PUPPI file is `eta in [-3, 3]` with a higher pt floor, so these numbers are **the same models meeting a different detector**, not a better or worse score on the same task. Do not rank against the table above. The official-area columns are omitted here: every official run was measured on the PF file, so an official number has no meaning beside an L1T row. **Check the `trained on` column**: rows trained on PF and merely evaluated on L1T answer a different question from rows trained on L1T, and only the latter are a like-for-like comparison with each other.

| tag | commit | encoder | data | trained on | events | clean AUC | rect | wedge | strip | towers | cells | c_ellipse | c_cell_dropout | c_edge_truncation | mean_area | delta | sep. from top mean_area |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| d-pma0-aug-meanpt-l1t | 143e0b2 | PMAEncoder | L1T | L1T | 20k | 0.9067 | 0.7540 | 0.8447 | 0.7745 | 0.8194 | 0.8661 | - | - | - | **0.8117** ±0.0003 | - | **leader** |
| h-ref-small-s22-l1t | f437155 | PMAEncoder | L1T | PF | 20k | 0.9085 | 0.7581 | 0.8410 | 0.7789 | 0.8172 | 0.8626 | - | - | - | **0.8116** ±0.0004 | - | **no** (Δ0.0002 vs 0.0007) |
| h-ref-small-s22-last-l1t | f522f49 | PMAEncoder | L1T | PF | 20k | 0.9058 | 0.7575 | 0.8415 | 0.7779 | 0.8178 | 0.8626 | - | - | - | **0.8114** ±0.0005 | - | **no** (Δ0.0003 vs 0.0008) |
| g6a-het-bits6-relu-l1t-s33 | 98e7671 | QuantizedPMAEncoder | L1T | L1T | 20k | 0.8982 | 0.7535 | 0.8412 | 0.7750 | 0.8165 | 0.8618 | - | - | - | **0.8096** ±0.0007 | - | yes (Δ0.0021 vs 0.0010) |
| g6a-hom-bits6-relu-l1t-s33 | 98e7671 | QuantizedPMAEncoder | L1T | L1T | 20k | 0.8978 | 0.7539 | 0.8410 | 0.7753 | 0.8161 | 0.8611 | - | - | - | **0.8095** ±0.0003 | - | yes (Δ0.0023 vs 0.0007) |
| g6a-het-bits6-relu-l1t-s22 | 98e7671 | QuantizedPMAEncoder | L1T | L1T | 20k | 0.9001 | 0.7528 | 0.8409 | 0.7746 | 0.8156 | 0.8612 | - | - | - | **0.8090** ±0.0006 | - | yes (Δ0.0027 vs 0.0009) |
| g6a-hom-bits6-relu-l1t-s22 | 682eccf-dirty | QuantizedPMAEncoder | L1T | L1T | 20k | 0.9000 | 0.7526 | 0.8409 | 0.7745 | 0.8156 | 0.8609 | - | - | - | **0.8089** ±0.0004 | - | yes (Δ0.0028 vs 0.0007) |
| gF-bits6-relu-l1t | aaaafc2 | QuantizedPMAEncoder | L1T | L1T | 20k | 0.8991 | 0.7533 | 0.8404 | 0.7739 | 0.8157 | 0.8611 | - | - | - | **0.8089** ±0.0005 | - | yes (Δ0.0029 vs 0.0009) |
| gD-bits10-l1t | 3a061e8-dirty | QuantizedPMAEncoder | L1T | L1T | 20k | 0.9025 | 0.7529 | 0.8400 | 0.7729 | 0.8154 | 0.8626 | - | - | - | **0.8087** ±0.0003 | - | yes (Δ0.0030 vs 0.0006) |
| h-ref-small-s33-l1t | f437155 | PMAEncoder | L1T | PF | 20k | 0.9029 | 0.7559 | 0.8375 | 0.7751 | 0.8143 | 0.8606 | - | - | - | **0.8087** ±0.0004 | - | yes (Δ0.0031 vs 0.0007) |
| h-ref-small-s11-l1t | f437155 | PMAEncoder | L1T | PF | 20k | 0.9044 | 0.7549 | 0.8366 | 0.7767 | 0.8138 | 0.8597 | - | - | - | **0.8083** ±0.0009 | - | yes (Δ0.0034 vs 0.0012) |
| h-ref-small-s11-last-l1t | f522f49 | PMAEncoder | L1T | PF | 20k | 0.9041 | 0.7546 | 0.8365 | 0.7765 | 0.8136 | 0.8593 | - | - | - | **0.8081** ±0.0007 | - | yes (Δ0.0037 vs 0.0011) |
| gD-bits6-l1t | 92982cc-dirty | QuantizedPMAEncoder | L1T | L1T | 20k | 0.8978 | 0.7534 | 0.8398 | 0.7721 | 0.8149 | 0.8599 | - | - | - | **0.8080** ±0.0001 | - | yes (Δ0.0037 vs 0.0005) |
| gB-q-b1e-6-l1t | 4fff865 | QuantizedPMAEncoder | L1T | L1T | 20k | 0.9019 | 0.7518 | 0.8393 | 0.7717 | 0.8149 | 0.8620 | - | - | - | **0.8079** ±0.0011 | - | yes (Δ0.0038 vs 0.0016) |
| g-stageA-l1t | bd7e635-dirty | FloatBNPMAEncoder | L1T | L1T | 20k | 0.8976 | 0.7535 | 0.8393 | 0.7730 | 0.8139 | 0.8595 | - | - | - | **0.8078** ±0.0007 | - | yes (Δ0.0039 vs 0.0010) |
| gD-relu-l1t | 3b5c238 | QuantizedPMAEncoder | L1T | L1T | 20k | 0.9014 | 0.7512 | 0.8392 | 0.7715 | 0.8153 | 0.8620 | - | - | - | **0.8078** ±0.0009 | - | yes (Δ0.0039 vs 0.0013) |
| g6a-hom-bits6-relu-l1t-s11 | 682eccf | QuantizedPMAEncoder | L1T | L1T | 20k | 0.8954 | 0.7513 | 0.8399 | 0.7720 | 0.8147 | 0.8604 | - | - | - | **0.8077** ±0.0004 | - | yes (Δ0.0041 vs 0.0007) |
| g6a-het-bits6-relu-l1t-s11 | 682eccf-dirty | QuantizedPMAEncoder | L1T | L1T | 20k | 0.8957 | 0.7509 | 0.8392 | 0.7718 | 0.8140 | 0.8598 | - | - | - | **0.8071** ±0.0004 | - | yes (Δ0.0046 vs 0.0007) |
| winner-l1t-eta3 | 3ac262f | PMAEncoder | L1T | — | 20k | 0.8986 | 0.7534 | 0.8353 | 0.7740 | 0.8116 | 0.8570 | - | - | - | **0.8063** ±0.0006 | - | yes (Δ0.0055 vs 0.0009) |
| winner-l1t _(superseded: eta_max 5)_ | 3ac262f | PMAEncoder | L1T | — | 20k | 0.8973 | 0.7451 | 0.8358 | 0.7804 | 0.8092 | 0.8526 | - | - | - | **0.8046** ±0.0008 | - | yes (Δ0.0071 vs 0.0011) |
| gE-stageA-maskedbn-l1t | aaaafc2 | FloatBNPMAEncoder | L1T | L1T | 20k | 0.8924 | 0.7515 | 0.8339 | 0.7712 | 0.8100 | 0.8553 | - | - | - | **0.8044** ±0.0007 | - | yes (Δ0.0074 vs 0.0010) |
| deepsets-meanpt-l1t-eta3 | 791c450 | DeepSetsEncoder | L1T | — | 20k | 0.8909 | 0.7499 | 0.8343 | 0.7709 | 0.8104 | 0.8553 | - | - | - | **0.8042** ±0.0019 | - | yes (Δ0.0076 vs 0.0027) |
| deepsets-meanpt-l1t _(superseded: eta_max 5)_ | 791c450 | DeepSetsEncoder | L1T | — | 20k | 0.8908 | 0.7435 | 0.8329 | 0.7786 | 0.8066 | 0.8487 | - | - | - | **0.8021** ±0.0016 | - | yes (Δ0.0097 vs 0.0021) |
| d-pma0-aug-meanpt-l1t-colleague _(colleague)_ | 143e0b2 | PMAEncoder | L1T | L1T | 20k | 0.9063 | - | - | - | - | - | 0.8128 | 0.7956 | 0.7503 | **0.7862** ±0.0006 | - | - |
| h-ref-small-s22-l1t-colleague _(colleague)_ | f437155 | PMAEncoder | L1T | PF | 20k | 0.9088 | - | - | - | - | - | 0.8125 | 0.7924 | 0.7491 | **0.7847** ±0.0004 | - | - |
| h-ref-small-s22-last-l1t-colleague _(colleague)_ | f522f49 | PMAEncoder | L1T | PF | 20k | 0.9059 | - | - | - | - | - | 0.8113 | 0.7914 | 0.7476 | **0.7835** ±0.0007 | - | - |
| h-ref-small-s33-l1t-colleague _(colleague)_ | f437155 | PMAEncoder | L1T | PF | 20k | 0.9040 | - | - | - | - | - | 0.8090 | 0.7891 | 0.7470 | **0.7817** ±0.0003 | - | - |
| h-ref-small-s11-l1t-colleague _(colleague)_ | f437155 | PMAEncoder | L1T | PF | 20k | 0.9047 | - | - | - | - | - | 0.8095 | 0.7878 | 0.7454 | **0.7809** ±0.0004 | - | - |
| h-ref-small-s11-last-l1t-colleague _(colleague)_ | f522f49 | PMAEncoder | L1T | PF | 20k | 0.9044 | - | - | - | - | - | 0.8093 | 0.7879 | 0.7453 | **0.7808** ±0.0004 | - | - |
| gD-bits4-l1t | 92982cc | QuantizedPMAEncoder | L1T | L1T | 20k | 0.8547 | 0.7321 | 0.8095 | 0.7482 | 0.7849 | 0.8237 | - | - | - | **0.7797** ±0.0018 | - | yes (Δ0.0320 vs 0.0025) |
| c-winner-l1t _(colleague)_ | 3ac262f | PMAEncoder | L1T | — | 20k | 0.8970 | - | - | - | - | - | 0.8084 | 0.7863 | 0.7436 | **0.7794** ±0.0007 | - | - |
| c-winner-l1t-eta3 _(colleague)_ | 3ac262f | PMAEncoder | L1T | — | 20k | 0.8970 | - | - | - | - | - | 0.8081 | 0.7861 | 0.7435 | **0.7792** ±0.0004 | - | - |
| c-deepsets-meanpt-l1t _(colleague)_ | 791c450 | DeepSetsEncoder | L1T | — | 20k | 0.8915 | - | - | - | - | - | 0.8051 | 0.7847 | 0.7426 | **0.7775** ±0.0008 | - | - |
| c-deepsets-meanpt-l1t-eta3 _(colleague)_ | 791c450 | DeepSetsEncoder | L1T | — | 20k | 0.8913 | - | - | - | - | - | 0.8047 | 0.7844 | 0.7419 | **0.7770** ±0.0012 | - | - |
| c3-best-colleague-l1t _(colleague)_ | 3601a46 | TransformerEncoder | L1T | L1T | 20k | 0.8905 | - | - | - | - | - | 0.7954 | 0.7732 | 0.7432 | **0.7706** ±0.0091 | - | - |
| c3-control-colleague-l1t _(colleague)_ | 3601a46 | TransformerEncoder | L1T | L1T | 20k | 0.8729 | - | - | - | - | - | 0.7743 | 0.7246 | 0.7180 | **0.7390** ±0.0006 | - | - |
| gB-q-b1e-5-relu-l1t **[RETRACTED]** | 4fff865-dirty | QuantizedPMAEncoder | L1T | L1T | 20k | 0.6250 | 0.5265 | 0.5191 | 0.5206 | 0.5202 | 0.5266 | - | - | - | **0.5226** ±0.0026 | - | yes (Δ0.2891 vs 0.0035) |
| gD-bits3-l1t | 2c99ed3-dirty | QuantizedPMAEncoder | L1T | L1T | 20k | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | - | - | - | **0.5000** ±0.0000 | - | yes (Δ0.3117 vs 0.0005) |

**Row descriptions** (supplied by their authors; not inferred from tag names):

- `gF-bits6-relu-l1t` — ReLU, hard cap 6 bits (proposed operating point) *(per WP-G)*
- `gD-bits10-l1t` — HGQ2 quantized from stage A, hard cap 10 bits (cap never binds; same configuration as gB-q-b1e-6-l1t, trained twice) *(per WP-G)*
- `gD-bits6-l1t` — hard cap 6 bits (weights: total bits; activations: fractional bits, integer part free) *(per WP-G)*
- `gB-q-b1e-6-l1t` — HGQ2 quantized from stage A, learned widths (about 8 bits), EBOPs regulariser beta0 1e-6 (beta later shown inert) *(per WP-G)*
- `g-stageA-l1t` — stage A, LayerNorm replaced by BatchNorm, float, distilled from the L1T float reference (swap cost only, no quantization) *(per WP-G)*
- `gD-relu-l1t` — ReLU instead of GELU, hard cap 8 bits *(per WP-G)*
- `gE-stageA-maskedbn-l1t` — stage A variant, BatchNorm statistics over live slots only *(per WP-G)*
- `gD-bits4-l1t` — hard cap 4 bits *(per WP-G)*
- **`gB-q-b1e-5-relu-l1t`** — RETRACTED by WP-G: harness bug (GELU emulator strict-loaded ReLU weights). Not a measurement.
- `gD-bits3-l1t` — hard cap 3 bits, collapsed network (one distinct latent for all events), not an operating point *(per WP-G)*

### Diagnostic runs (non-scoring families)

Held-out families (`ellipse`/`annulus`/`diagonal`, ours) and the colleague suite (`c_` prefix, Group 3's code). **These `mean_area` values are not comparable to the scoring table above, nor to each other across suites** -- different shapes, different severity semantics. See the colleague caveats in the notes.

Official-area columns are omitted for the same reason as the L1T block.

| tag | commit | encoder | data | trained on | events | clean AUC | ellipse | annulus | diagonal | c_ellipse | c_cell_dropout | c_edge_truncation | mean_area | delta | sep. from top mean_area |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| l-l1-sincos-s11-heldout _(heldout)_ | f522f49 | PMAEncoder | PF | PF | 20k | 0.9249 | 0.7951 | 0.8166 | 0.8085 | - | - | - | **0.8067** ±0.0005 | - | - |
| a-pma-heldout _(heldout)_ | d450bcf | TransformerEncoderPMA | PF | — | 20k | 0.9226 | 0.7943 | 0.8165 | 0.8068 | - | - | - | **0.8059** ±0.0016 | - | - |
| j1-s11-heldout _(heldout)_ | dc45142 | PMAEncoder | PF | PF | 20k | 0.9218 | 0.7940 | 0.8153 | 0.8066 | - | - | - | **0.8053** ±0.0003 | - | - |
| h-ref-small-s22-heldout _(heldout)_ | f437155 | PMAEncoder | PF | PF | 20k | 0.9249 | 0.7929 | 0.8152 | 0.8068 | - | - | - | **0.8050** ±0.0010 | - | - |
| h-ref-small-s11-last-heldout _(heldout)_ | f522f49 | PMAEncoder | PF | PF | 20k | 0.9232 | 0.7943 | 0.8137 | 0.8059 | - | - | - | **0.8047** ±0.0005 | - | - |
| h-ref-small-s11-heldout _(heldout)_ | f437155 | PMAEncoder | PF | PF | 20k | 0.9229 | 0.7940 | 0.8132 | 0.8056 | - | - | - | **0.8043** ±0.0010 | - | - |
| h-ref-small-s22-last-heldout _(heldout)_ | f522f49 | PMAEncoder | PF | PF | 20k | 0.9219 | 0.7909 | 0.8130 | 0.8053 | - | - | - | **0.8031** ±0.0006 | - | - |
| d-pma0-aug-meanpt-heldout _(heldout)_ | 9801bdd | PMAEncoder | PF | — | 20k | 0.9134 | 0.7910 | 0.8120 | 0.8033 | - | - | - | **0.8021** ±0.0006 | - | - |
| l-l1-sincos-s11-colleague _(colleague)_ | f522f49 | PMAEncoder | PF | PF | 20k | 0.9253 | - | - | - | 0.8330 | 0.8071 | 0.7657 | **0.8019** ±0.0006 | - | - |
| h-ref-small-s33-heldout _(heldout)_ | f437155 | PMAEncoder | PF | PF | 20k | 0.9185 | 0.7907 | 0.8102 | 0.8032 | - | - | - | **0.8014** ±0.0011 | - | - |
| h-ref-full-s11-heldout _(heldout)_ | f437155 | PMAEncoder | PF | robust_tag | 20k | 0.9257 | 0.7867 | 0.8090 | 0.8031 | - | - | - | **0.7996** ±0.0020 | - | - |
| h-ref-small-s11-last-colleague _(colleague)_ | f522f49 | PMAEncoder | PF | PF | 20k | 0.9229 | - | - | - | 0.8297 | 0.8049 | 0.7640 | **0.7996** ±0.0003 | - | - |
| j1-s11-colleague _(colleague)_ | dc45142 | PMAEncoder | PF | PF | 20k | 0.9214 | - | - | - | 0.8313 | 0.8054 | 0.7617 | **0.7994** ±0.0009 | - | - |
| h-ref-small-s11-colleague _(colleague)_ | f437155 | PMAEncoder | PF | PF | 20k | 0.9230 | - | - | - | 0.8293 | 0.8051 | 0.7633 | **0.7993** ±0.0007 | - | - |
| c-winner _(colleague)_ | 3ac262f | PMAEncoder | PF | — | 20k | 0.9137 | - | - | - | 0.8297 | 0.8017 | 0.7655 | **0.7990** ±0.0003 | - | - |
| d-pma0-aug-heldout _(heldout)_ | 9801bdd | PMAEncoder | PF | — | 20k | 0.9202 | 0.7879 | 0.8093 | 0.7975 | - | - | - | **0.7983** ±0.0048 | - | - |
| h-ref-small-s22-colleague _(colleague)_ | f437155 | PMAEncoder | PF | PF | 20k | 0.9247 | - | - | - | 0.8288 | 0.8056 | 0.7602 | **0.7982** ±0.0010 | - | - |
| h-ref-full-s11-colleague _(colleague)_ | f437155 | PMAEncoder | PF | robust_tag | 20k | 0.9258 | - | - | - | 0.8315 | 0.8033 | 0.7589 | **0.7979** ±0.0018 | - | - |
| d-deepsets-aug-meanpt-heldout _(heldout)_ | 9801bdd | DeepSetsEncoder | PF | — | 20k | 0.9067 | 0.7857 | 0.8085 | 0.7990 | - | - | - | **0.7977** ±0.0009 | - | - |
| h-ref-small-s33-colleague _(colleague)_ | f437155 | PMAEncoder | PF | PF | 20k | 0.9182 | - | - | - | 0.8282 | 0.8027 | 0.7617 | **0.7975** ±0.0004 | - | - |
| h-ref-small-s22-last-colleague _(colleague)_ | f522f49 | PMAEncoder | PF | PF | 20k | 0.9217 | - | - | - | 0.8275 | 0.8047 | 0.7602 | **0.7975** ±0.0007 | - | - |
| d-deepsets-aug-heldout _(heldout)_ | 9801bdd | DeepSetsEncoder | PF | — | 20k | 0.8917 | 0.7856 | 0.8061 | 0.7970 | - | - | - | **0.7962** ±0.0011 | - | - |
| c-deepsets-meanpt _(colleague)_ | 791c450 | DeepSetsEncoder | PF | — | 20k | 0.9068 | - | - | - | 0.8220 | 0.7997 | 0.7430 | **0.7883** ±0.0014 | - | - |
| c3-best-colleague _(colleague)_ | 3601a46 | TransformerEncoder | PF | L1T | 20k | 0.8991 | - | - | - | 0.7971 | 0.7862 | 0.7300 | **0.7711** ±0.0051 | - | - |
| c3-control-colleague _(colleague)_ | 3601a46 | TransformerEncoder | PF | L1T | 20k | 0.8620 | - | - | - | 0.7606 | 0.7308 | 0.6646 | **0.7187** ±0.0015 | - | - |
