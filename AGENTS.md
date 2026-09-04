# AGENTS.md — for an agent continuing this work

You are reading the hand-over of a two-campaign hackathon effort on Fast ML 2026 Challenge C9 "Robust Tagging". This file tells you what
exists, what is established, what is unproven, what failed silently and how to not repeat it. Read it fully before running anything.

## 1. The task, in one paragraph
An encoder maps a set of particle-flow candidates (pt, η, φ, dxy, dxy_sig, is_pf, pdgId; 200 per event in the small train file, 400 in
eval and in the full train file) to a 6-d latent. The grader (`eval.py`) embeds the clean eval set, fits an MLP probe on those latents,
re-embeds the same events with candidates in random dead η-φ regions zeroed out (`src/embedding/degradation_eval.py`), and scores them with
the frozen probe. Score = area under AUC-vs-severity. The probe is unseeded, so one grader run varies by ~±0.002.

## 2. What is shipped and certified
Branch `submission-pma0-meanpt` @41d45a9: `PMAEncoder(num_layers=0)` (4 seed queries, 8 heads, 89,606 params) + `PFPreProcessorMeanPt`,
one checkpoint `checkpoints/rt_d_pma0_aug_meanpt_encoder_20260903_221125.pth` (sha256 `2f34628ccd78...`; full digest in
`docs/SUBMISSION-HANDOVER.md`). Eight fresh-clone official runs: 0.8774–0.8805 (organisers' reference 0.8201). Certification = `accept.sh
--submission` from a fresh clone with class assertions, single-checkpoint check and byte-identity of the scored file. Do not change this
branch. If you improve the model, you ship a NEW submission branch through the same certification.

**Grader contract you must not break:** `eval.py` constructs `TransformerEncoder(num_features, embed_size, latent_dim, num_heads,
num_layers, linear_dim, num_tokens=None, pairwise)` from the shipped train config and passes no other kwargs. Any encoder variant must be a
CLASS selectable by `encoder_class:` in the train config whose defaults ARE the variant; the submission branch aliases the grader's name to
that class. Checkpoint keys `preproc`, `encoder`, `projector`, `classifier`, `norm_constants` stay. Nothing may depend on a fixed token count.

## 2b. Models colleagues can evaluate
`models/MODELS.md` is the index: per model, the best checkpoint (digest-verified) or a pointer to the branch shipping it, its train config,
`BUILD.md` (commit, classes, alias) and `evaluate.sh`. Use the model's OWN train config with `bench_eval.py --train_cfg`; building a
checkpoint against another config strict-loads silently in some cases (14-vs-15 input features) and produces a plausible wrong curve.

## 3. The measurement standard (binding; everything below was decided under it)
- The ruler is `tooling/bench/bench_eval.py`: five seeded families (rect, wedge, strip, towers, cells) on 20k eval events, `--probe_repeats 5`,
  JSON to `runs/`. Suites: `--families heldout` (ellipse/annulus/diagonal, never trained on), `--families colleague` (Group 3's three,
  `c_` prefix, read per family, never as a mean), the L1T file with `--eta_max 3.0`. Record `--train_data` and `--seed`; the JSON carries
  `ckpt_sha256`, `eta_max_applies`, per-family corruption fingerprints, `degradation_class`.
- **Three training seeds per configuration.** Separable iff Δ(seed means) > 3·sqrt(s_a²/3 + s_b²/3). Measured floors: training-seed std
  0.0030 (small file) / 0.0033 (full file); probe-refit std 0.0003–0.0009; same-checkpoint bench reproducibility 0.00009. The floor is a
  property of the training procedure, not of data size. The detection floor for a 3-vs-3 comparison is ≈0.005–0.007 and is dominated by
  the REFERENCE's seed spread: to see a 0.003 effect, add reference seeds (ten halve the floor), not arm seeds.
- One-seed screens are triage ("worth three seeds"), not evidence: at seed std 0.0030 about half of null arms clear the reference mean on
  one seed, and every arm we ran did exactly that.
- Every number in a report is read from its `runs/*.json` in the same command that writes the text, filename beside it.
- A run is complete only when the scheduler's `released rc=0` line is in its log AND the JSON exists; a checkpoint's existence proves
  nothing (`train.py` writes best-so-far every improving epoch; `aux/` holds `_bestauc` and `_last`). Select the checkpoint named by the
  log's "Saved best encoder to:" line, never the newest file.
- Any checkpoint a published row names goes to durable storage with `.sha256` and `.origin` sidecars BEFORE publication
  (`docs/CHECKPOINTS.md`). Session scratchpads vanish.

## 4. What is established (with where the evidence is)
| finding | evidence |
|---|---|
| Mask fix (derive the attention mask from zeroed rows): +0.018 frozen, +0.035 retrained | docs/writeup/A-*, docs/RULING.md |
| Set encoder + MeanPt is the champion; a-pma (transformer body + PMA readout, 27x params) does NOT separate at the seed floor | RULING.md addendum 04:33, RULING-2 |
| MeanPt removes 95% of the pt-denominator latent drift and all of the 200-vs-400 candidate offset | docs/writeup/B-*, D-* |
| Data scale: full file (940k×400) vs small (80k×200): Δ 0.0000 at three seeds each | RULING-2 item 4, H-* |
| Capacity not binding: embed 256 (3.83x) at the reference; latent 32 uses ~2 effective dims; +1 block ff256 (2.48x) +0.0025, not separable | I-encoder-capacity.md |
| Probe uses latent magnitude: L2-normalising the latent costs 0.026 | K-consistency-and-latent-geometry.md §3 |
| Whitened (Mahalanobis) consistency has a rank-1 degenerate minimiser | K §4 |
| Two-view latent shrink (26x) is caused by the MSE consistency term, not the two-view batch; offset penalty λ=0.01 controls the offset | K §5 |
| Generator effectively removed >70% of candidates in only 2.7% of steps; a third of that is calibration error (edge-peaked η density, area-sum vs union, p<1 ceiling); on-target placement and a 30%-heavy mixture did NOT move the score | J-generator-coverage.md |
| Validation-corruption selection was ~11x noise-dominated; seeding it removes the noise (criterion smoothness 3x, p=1/35) and buys reproducibility, not accuracy | L-preprocessor-and-optimisation.md |
| Group 3 did not beat us: like for like 0.8270 vs their best 0.8107; their published 0.8224 used another probe | RULING-2 Group 3 section; docs/plots/l1t_colleague_latest.png |
| FPGA: 6-bit cap + ReLU operating point 0.8089 ± 0.0005 on L1T (float 0.8117; the only cost is LayerNorm→BatchNorm −0.0039); folded N=400 design compiles (183 s) and is bit-exact; fitting config 34,048 multipliers (straddles xcvu13p on a LUT proxy, 8 µs floor); Vitis csynth did not finish in 6 h (tail quantizers); Vivado blocked by missing /run/udev | docs/writeup/G-quantization.md |

## 5. What is unproven or open (do not treat as known)
- Phase-2 candidates by number only: `i-lat32` (latent 32, 0.8292 ± 0.0003, 1.15x) and `i-ff256l1` (0.8287 ± 0.0015, 2.48x). Neither separates;
  neither has suites or a mechanism; both need many seeds on the full file. Checkpoints in `checkpoints/candidates/`.
- Whether a wider latent buys seed STABILITY (i-lat32's three seeds span 0.0006 vs 0.0057 for the reference): hypothesis, needs >3 seeds.
- Group 3's rows on the PF file are milder than labelled (their families paint on η ±3); compare on the L1T file only.
- FPGA: no measured utilisation/latency/II; next change = tail quantization (quantize before the flatten, or per-tensor tail), then
  re-synthesise; timing closure needs Vivado, which needs a libudev stand-in in this container (not built; user decision).
- Campaign-1 single-run bench "separabilities" below ~0.006 are unproven (computed against probe noise only).

## 6. Silent failures we hit (41 classes in `docs/plots/README.md`); the ones most likely to bite you
1. Two things with one name: our `ellipse` vs Group 3's `ellipse` (resolver order silently swapped the corruption for hours); generator
   `cells` vs bench `cells` (different p_drop targets). Use prefixes; make bare ambiguous names raise; fingerprint corruptions.
2. A number typed from memory next to sourced numbers is indistinguishable in form. Read from the JSON in the same command.
3. An empty tool result is not a finding: `fuser` is not installed here (prints nothing); `grep` on a log with NUL bytes prints nothing
   (`grep -a`); `find -newermt` returned zero on 54 rows. Show the tool returns non-empty under a known-true condition first.
4. Completion signals that lie: wrapper `rc=0` after a drain, a live child that is a `sleep`, a checkpoint that exists at epoch 9/25.
5. Ratios without both terms (offset/spread, condition number without effective rank) hide which term moved.
6. A diagnostic read as evidence about the objective when it was evidence about the measurement (test_mode vs full data; two-view vs
   one-view; per-token vs per-channel BatchNorm stats). Before attributing, list everything that differs between the two rows.
7. A real, reproducible defect whose fix does not move the metric (generator calibration). Relevance is established only by benching the fix.
8. One-seed per-family breakdowns "confirming" a mechanism: some family is always largest.
9. Generated files edited by hand (`docs/plots/README.md` is generated by `plot_readme.py`); edit the generator.
10. Multi-GPU: `nvidia-smi | head -1` under pipefail dies of SIGPIPE; `kill -0 $$` in a subshell is the parent; SIGTERM is ignored inside
    sklearn's t-SNE (C extension) — escalate to KILL with confirmation. Never edit a running bash script in place (atomic `mv`).
11. Scheduler: never abandon queued jobs when their launcher exits; fair share by package; never take a second resource (bench token)
    before the first (GPU slot) — that deadlocked the bench queue for 18 minutes.

## 7. How to continue
- Environment: `pip install -e .` in a clone; PYTHONPATH must point at YOUR clone's `src/` (editable installs shadowed each other for us).
  HGQ2/hls4ml work uses the `hgq2` env (see `docs/writeup/G-quantization.md` §0); Vitis/Vivado under `/tools/Xilinx` if present.
- Start from `integration-2` (campaign-2 tooling: `degradation_kwargs`, `seed`, `val_bn_batch_stats`, `PMAEncoder` variants,
  `tools/check_arms.py`); configs under `configs/c2/`. Use `tooling/gpu_slot.sh` for any GPU job.
- If you want to try an improvement: pre-register the rule, train three seeds of the REFERENCE first (or reuse `docs/CHECKPOINTS.md`),
  then three seeds of the arm, bench all, run `ablation_table.py`, and only then read the verdict. Budget seeds, not arms.
- If you want the FPGA line: `quant/hls_config/` has both hls4ml configurations with emitted `parameters.h`; change the tail quantization
  first; rerun with the command in `quant/g7_csynth_outcome.json`.
- Rulings and the hour-by-hour record: `docs/prompts/10-phases.md`. Read the retractions; there are many, and each says why.
