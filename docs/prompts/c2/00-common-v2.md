# Campaign 2 common brief (read fully, then read `prompts/00-common.md` for the challenge, code map and grader contract)

Campaign 1 (3 Sep) produced the certified submission: `submission-pma0-meanpt` @41d45a9, PMAEncoder(num_layers=0, 4 seed
queries, 8 heads) + PFPreProcessorMeanPt + WP-B's dead-region generator, trained 25 epochs on the 80k small file.
Campaign 2 (4 Sep, from 03:30) has one goal: **improve that champion as much as the evidence allows, on every benchmark we
have**, and certify the result the same way. Nothing ships without a fresh-clone acceptance run.

**Planner session:** `jovyan-2a [a7128f]`. Use ListAgents, then SendMessage. Rulings are logged in `prompts/10-phases.md`.

## Hardware (new) and the rules that follow from it
- 5 × A10 (24 GB each), 503 GB RAM, **8 CPU cores for everyone**. The CPU is the bottleneck, not the GPU.
- GPU access ONLY through the slot scheduler `~/hackathon-shared/gpu_slot.sh` (WP-H delivers it in Phase 0; until the
  planner announces it, no GPU job at all; develop with `--test_mode` on CPU: `CUDA_VISIBLE_DEVICES= python train.py ... --test_mode`).
  Usage: `~/hackathon-shared/gpu_slot.sh [--slots 1] python train.py ...` takes the first free GPU, sets CUDA_VISIBLE_DEVICES,
  exports PYTHONPATH=$PWD/src, RT_NUM_THREADS=1, logs to `gpu.log`. Benches go through the same scheduler (`--kind bench`).
- Thread cap: training RT_NUM_THREADS=1, benches 1. Never override upward. `nvidia-smi` and `gpu.log` before starting anything.
- Data: the FULL train file `~/hack-data/C9_robust_tagging/train/robust_tagging_train_data.pt` is [940048, 400, 8]
  (QCD 451k, DY 99k, TT 293k, WJets 97k; 400 candidates per event like eval; eta ±5). Load with `mmap=True`
  (2.6 s). Phase 1 screens on the small file; Phase 2 trains on the full file. Cost model: champion ≈ 22 s/epoch on
  80k×200 exclusive; expect ≈ 9 min/epoch on the full file. Say your expected wall time before every full-data run.

## Code base
- Branch `integration-2` on `~/hackathon-shared/repo.git` (WP-H creates it from `wp-d` @143e0b2, which carries PMAEncoder,
  MeanPt, the generator with `degradation_eta_max`, `val_bn_batch_stats`, `seed`). Clone to `~/c2-<x>`, branch `c2-<x>`, push often.
- Champion config: `configs/train_config_d_pma0_aug_meanpt.yaml`. Every arm starts from a COPY of it and changes ONE thing.
  Combinations are their own measured rows (campaign 1 lesson: the champion was an interaction; neither half separated alone).
- The grader reads `latent_dim`, `embed_size`, `num_heads`, `num_layers`, `linear_dim` from the shipped train config and
  constructs `TransformerEncoder(...)` (aliased to PMAEncoder on the submission branch). Anything else must have a default
  that is the final choice.

## Measurement standard (binding)
1. **Three seeds per configuration**, `seed: 11 / 22 / 33` in the config, tags `<x>-<desc>-s11` etc. A single training run is an
   anecdote. Campaign 1 measured the training-pipeline floor once (0.0008 on a same-config pair) and could not use it; we fix that.
2. Bench every checkpoint: `python ~/hackathon-shared/bench/bench_eval.py --repo ~/c2-<x> --ckpt <ckpt> --tag <tag>
   --encoder_class PMAEncoder --train_cfg <cfg> --probe_repeats 5 --train_data <train file basename>` (Phase 1 screen, 20k
   events). Finalists: `--probe_repeats 20 --full`. JSON fields `data`, `eta_max`, `train_data`, `seed` must be present.
3. Report a configuration as **mean over seeds ± std over seeds** (n=3) with the per-seed probe std beside it. Separability of
   two configurations: Δ of seed means > 3·sqrt(s_a²/3 + s_b²/3) with s the seed std. Below that, "not separable" and no
   ordering claim. Never call a gap real that is under 0.002 unless the seed test says so.
4. Suites, and what they are for: dev bench (5 families; the ruler); held-out (ellipse/annulus/diagonal, `--families heldout`;
   generalisation, never trained on); colleague suite (`--families colleague`, `c_` prefix; independent check, per family, never a
   mean); organisers' `eval.py` (official; 3 runs, unseeded probe); L1T file (`--data ..._l1t.pt --eta_max 3.0`; different
   acceptance, own block). **Never train on held-out or colleague held-out families.** The colleague's TRAIN families
   (rectangle, eta_band, phi_wedge, multi_patch, candidate_loss) may be added to the generator (WP-J only).
5. Reference rows are WP-H's: champion × 3 seeds on the small file (Phase 1 reference) and on the full file (Phase 2 reference).
   Compare only against the reference trained in the same data regime.
6. A regression is a result. Retractions are made in place with the reason; nothing is silently deleted (see
   `runs/plots/README.md`, the register of twelve silent-failure classes, and read it before your first run).

## Promotion rule (pre-registered; the planner applies it, nobody else)
A candidate replaces the champion only if ALL hold, on the full-data regime, 3 seeds each:
(a) dev bench: separably above the full-data reference by the seed test;
(b) held-out AND colleague suite (PF): not separably below the reference on any family;
(c) official `eval.py`: mean of 3 runs ≥ reference mean − 0.001;
(d) L1T block: reported, not gating;
(e) fresh-clone `accept.sh --submission` PASS on the shipped tip, class assertions, single checkpoint, sha256 match.
Ties go to the smaller model. Encoder parameter count is reported on every row.

## Reporting
`~/hackathon-shared/reports/<x>-<HHMM>.md` at every milestone and at least every 90 min; one line per row in the form
`<tag>  mean_area <m> ± <probe std>  seeds <n>  clean <c>  params <p>`; then SendMessage the planner with the path and the
one-sentence result. Ask the planner before: any run > 3 h, any change outside your package boundary, any change to config
keys the grader reads. Timestamps: run `date` before writing one.

## Rule added 03:38 (from WP-I): encoder options are classes, not kwargs
`bench_eval.py` and `eval.py` construct the encoder from the grader's fixed signature and pass no kwargs. Any encoder variant
(more seed queries, a body layer, a normalised latent, ...) must therefore be a class selectable by `encoder_class:` whose defaults ARE
the variant, so that the fixed signature alone rebuilds it. `encoder_kwargs` are training-time only and a checkpoint that needs them
is untestable and ungradable. WP-I's `tools/check_arms.py` asserts rebuildability; run it on every new arm.

## Rule (from campaign 1, restated 03:40): checkpoint selection
train.py writes the best-val-loss checkpoint AND a best-AUC file under checkpoints/aux/. Bench and ship only the path the log names in
"Saved best encoder to:"; never a newest-mtime glob.

## Rule added 04:00: durable checkpoints
Any checkpoint that is a reference row or reaches Phase 2 is copied to `~/hackathon-shared/checkpoints/c2/<tag>.pth` with a `<tag>.sha256`
beside it, and is benched from that path. Session scratchpads are ephemeral. `bench_eval.py` records the checkpoint sha256 (WP-M).

## Rule added 04:08 (from WP-I): a run is complete only when the scheduler's `] released` line is in its log
`train.py` writes a best-so-far checkpoint after every improving epoch, so "a checkpoint exists" does not mean "the run finished".
Bench drivers must key completion on the `[slot k] released rc=0` line for that job and refuse to bench without it. Assert exactly one
primary checkpoint (non-recursive glob; `aux/` holds `_bestauc` and `_last`).

## Early evidence on the seed floor (04:08)
Two reference seeds on the small file differ by 0.0057 (s22 0.8296, s33 0.8239). Until three seeds are in for BOTH the reference and an
arm, no ordering claim of any kind is made in reports; write "n<3, no verdict".

## Rules added 04:12
- Every number in a report or message is read from its `runs/*.json` in the same command that composes the text, with the filename beside
  it. A number without a filename is unsourced and is treated as such (from WP-M).
- Scheduler: benches on slots 2-4 (fallback to any slot when no training job is queued), at most 3 at once; `--priority` exists for WP-H
  reference rows and WP-M certification suites only, and every priority acquisition is logged with its package.

## Rule added 04:30: how a Phase 1 arm reaches Phase 2 when nothing separates
The reference seed spread (two seeds so far differ by 0.0057) implies a separability bar near 0.007 on the small file, above every
within-campaign margin argued last night. Most Phase 1 arms will therefore read "not separable". Pre-registered Phase 2 selection:
- Separable above the reference: goes to Phase 2.
- Not separable: goes to Phase 2 only if (i) its seed mean is above the reference seed mean by at least 0.002, (ii) no suite (held-out,
  colleague) shows a separable regression, and (iii) it carries an independent diagnostic supporting its mechanism (offset, condition
  number, effective severity, selection noise, ...). At most ONE such arm per package; the planner picks by seed-mean rank.
- Below the reference seed mean: dropped, whatever the point estimates of individual seeds say.
Full-data Phase 2 uses the same bar against h-ref-full; the seed spread on 12x the data may be smaller, and that is itself a result.

## Rule added 04:45 (from WP-N and WP-K): never report a ratio without both of its terms
A ratio is extremal for two opposite reasons (numerator grew, denominator collapsed) and cannot tell them apart. Offset/spread, condition
numbers and similar diagnostics are reported with both terms beside them; a condition number is reported with the effective rank.

## DEADLINE (user, 04:25): if no improvement is found by 05:25, the search stops and the champion is prepared for submission
- 05:25 is the Phase 1 ruling. An arm counts as an improvement only by the pre-registered rule (three seeds, separable above 0.8263 by
  ≈0.0074, no suite regression) or the non-separable route (seed mean ≥ 0.8283 with a supporting diagnostic).
- From now: no new training launches except seeds 22/33 of arms whose seed 11 is ≥ 0.8263 (the reference mean). Arms with seed 11 below
  the reference mean stop at one seed and are reported as such. Bench everything already trained.
- Bench slots: benches may use any free slot (H). Training queue entries not covered above are drained by their owners.
- WP-M starts the submission-readiness checklist on the certified champion (submission-pma0-meanpt @41d45a9) now, in parallel.
- The full-file reference (h-ref-full, three seeds) continues; it lands after the deadline and is reported to the user as a separate
  decision (data scale is the one lever that cannot be measured within the hour).

## Rule added 05:55 (from WP-M's audit): durable storage covers EVERY checkpoint a published row names
Four checkpoints named by runs/ JSONs had vanished with their session scratchpads within two hours. Any checkpoint that a row in runs/ names is
copied to ~/hackathon-shared/checkpoints/c2/ (or c2/rescued/) with .sha256 and .origin sidecars before the row is published; the table flags
rows whose named checkpoint is absent. A result is only as reproducible as the least durable artefact it depends on.
