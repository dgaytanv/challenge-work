# WP-H: infrastructure, code base, reference runs (session jovyan-b4)
Read `prompts/c2/00-common-v2.md`, then `prompts/00-common.md`, then `README.md`. You are on the critical path for the first 45 minutes; everyone else waits on H1 and H2.

## H1 (now, ≤ 30 min): `~/hackathon-shared/gpu_slot.sh`
Replace the single-lock wrappers with a slot scheduler for 5 GPUs:
- Locks `~/hackathon-shared/locks/gpu{0..4}.lock`. `gpu_slot.sh [--slots N=1] [--kind train|bench] cmd...` tries the slots in order with
  non-blocking `flock -n`, takes the first free, sets `CUDA_VISIBLE_DEVICES=<idx>`, exports `PYTHONPATH=$PWD/src` when `src/embedding`
  exists, `RT_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, logs `[slot k] acquired/released rc=` with cwd and command to `gpu.log`.
  If none is free, waits (poll 5 s) with a queued line in the log. Keep campaign 1's watchdog semantics: a queued child dies with its
  launcher; an acquired job never does. `--kind bench` may only use slots 3-4 (so three GPUs are always training-only) and at most 2 run at once.
- Keep `gpu_run.sh`/`gpu_small.sh` as thin aliases onto slot 0 / kind bench so old scripts still work.
- Test: launch 7 sleeps through it, show the log, kill one queued launcher, prove the queued child dies and the running ones do not.
  Report the test output in your first report. Announce completion to the planner AND to all six other sessions by name.

## H2 (with H1): branch `integration-2`
From `wp-d` @143e0b2 on `~/hackathon-shared/repo.git`. Verify it carries PMAEncoder, PFPreProcessorMeanPt, `degradation_eta_max`, `val_bn_batch_stats`, `seed`,
and that `configs/train_config_d_pma0_aug_meanpt.yaml` trains in `--test_mode` on CPU. Add a `configs/c2/` directory with the champion config
copied as `champion.yaml` and three seed variants `champion_s11/22/33.yaml` (only `seed` and `model_name` differ). Push. Announce.

## H3: full-data loader
`load_data` does `torch.load(path)` without mmap; on the 12 GB file with five processes that is 60 GB and slow. Add `mmap=True`, keep the
tensor on CPU, index per batch. Verify: peak RSS per process, epoch time of the champion on the full file with 1 CPU thread (batch 256 → 3672
steps/epoch). If an epoch exceeds 12 min, profile (data indexing vs GPU) and report before optimising. Also check class balance handling
(`compute_class_weights`) on the full label distribution and report what it does.

## H4: reference runs (the rows every other package compares against)
- Small file, champion × 3 seeds (tags `h-ref-small-s11/22/33`), bench each (R=5). This is the Phase 1 reference; deliver by 05:00.
- Full file, champion × 3 seeds (`h-ref-full-s*`), epochs chosen from H3 timing so a run is ≤ 3 h (state the number and why; fewer epochs
  on 12x the data is expected). These are the Phase 2 reference. Bench each (R=5), then WP-M runs the finalist protocol.
- Report seed std of mean_area for both regimes: that number is campaign 2's noise floor and goes in `prompts/10-phases.md` via the planner.

## Boundary
`gpu_slot.sh`, `README.md` GPU section, branch `integration-2`, `data_utils.load_data`, `configs/c2/champion*.yaml`, reference runs. Nothing else
without asking. Report every 60 min (`reports/h-<HHMM>.md`).
