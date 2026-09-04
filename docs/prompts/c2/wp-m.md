# WP-M: measurement, table, certification (session jovyan-cb)
Read `prompts/c2/00-common-v2.md`, then `prompts/00-common.md`, then `writeup/E-bench-design.md`, `writeup/E-probe-noise-floor.md`,
`writeup/E-acceptance-tooling.md`, `writeup/RULING.md` and `runs/plots/README.md` (the register). You inherit WP-E's tooling in `bench/`.

## M1 (Phase 0-1): protocol v2 in the tooling
- `bench_eval.py`: add `--seed` passthrough recorded in the JSON (read from the train config if present), keep every campaign-1 flag.
- `ablation_table.py`: group rows by configuration (tag without `-s<seed>`), print mean over seeds ± seed std (n), the mean per-seed probe std,
  params, and the seed-test separability against the reference configuration of the same data regime (`train_data` field decides the regime).
  Keep the two-floor reading at the top of the notes. Keep `[RETRACTED]` handling.
- `compare_arms.py`: seed-test mode.
- A `suite_run.sh <repo> <ckpt> <tag> <cfg>` that runs dev bench, held-out, colleague (PF), and L1T+colleague L1T (`--eta_max 3.0`) for one
  checkpoint through `gpu_slot.sh --kind bench`, writing all JSONs; the finalist protocol adds `--probe_repeats 20 --full` and three
  `accept.sh --fast` official runs.

## M2 (Phase 1): run the suites
Screening benches are run by the packages themselves (R=5, dev bench). YOU run held-out + colleague for every Phase 1 configuration's best
seed on request, and the full suite for WP-H's references. Regenerate `runs/table.md` on every new JSON. Flag any row whose gain is
inside the seed floor.

## M3 (Phase 2-3): finalists and certification
Full protocol for every full-data configuration; officials ×3 per finalist; apply nothing yourself: send the planner the numbers and
the promotion-rule checklist (a)-(e) filled in. Then, on the planner's ruling, build the submission branch exactly as campaign 1 did
(`writeup/D-alias-mechanism.md`), run `accept.sh --submission` from a fresh clone, and write `writeup/RULING-2.md`.

## Rules
Never train. Never edit another package's code. Every number you publish traces to a `runs/*.json`. Reports `reports/m-<HHMM>.md` every 60 min.
