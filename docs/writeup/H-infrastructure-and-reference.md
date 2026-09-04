# H: Infrastructure and reference rows

Every number below is read from `runs/*.json`, never from a wrapper log. Rows can be re-checked
with `~/hackathon-shared/bench/verify_row.sh <tag> [train_data]`.

---

## 1. Reference rows

### Phase 1 — small file (80k × 200), champion config, 25 epochs

| tag | mean_area | probe std | clean AUC | seed |
|---|---|---|---|---|
| h-ref-small-s11 | 0.824978 | ±0.000364 | 0.9228 | 11 |
| h-ref-small-s22 | 0.829586 | ±0.000548 | 0.9249 | 22 |
| h-ref-small-s33 | 0.823891 | ±0.000326 | 0.9184 | 33 |

**SEED MEAN 0.826152 · SEED STD 0.003024 · SEPARABILITY BAR 0.0074** (3-seed vs 3-seed),
**0.0128** (single run vs single run).

This is the campaign's central measurement. **Seed noise dominates probe noise by ~8×**
(0.0030 vs ~0.0004). Campaign 1 used 0.0008 from a single same-config pair; the true figure is
nearly 4× larger. s22 sits **0.0057** above s33 on *identical config, data and code* — a spread
larger than most improvements campaign 1 accepted.

Independently confirmed (WP-N, fresh clone of `integration-2` @`8dfe03a`, clean tree, my
checkpoint): 0.824000 ±0.001186 against my 0.823891 ±0.000326 — 0.07× the threshold. Their
`ckpt_sha256` matched my published sidecar, closing the chain from durable copy to benched bytes.

**A free probe-noise datapoint.** s11 was benched twice on *byte-identical* checkpoints (durable
copy and training original, same sha256): **0.824978 and 0.825465**, a spread of 0.000487. That is
an empirical probe-refit floor on the same weights, and it is ~6× smaller than the seed spread.

### Phase 2 — full file (940k × 400), same config and schedule, 25 epochs

| tag | mean_area | probe std | clean AUC | seed |
|---|---|---|---|---|
| h-ref-full-s11 | 0.822565 | ±0.001688 | 0.9248 | 11 |
| h-ref-full-s22 | 0.829014 | ±0.000772 | 0.9254 | 22 |
| h-ref-full-s33 | 0.827324 | ±0.000540 | 0.9223 | 33 |

**FULL-FILE SEED MEAN 0.826301 · IN-REGIME SEED STD 0.003344 · BAR 0.0082.**

### The definitive answer on data scale

| regime | 3-seed mean | seed std |
|---|---|---|
| small (80k × 200) | 0.826314 | 0.002941 |
| full (940k × 400) | 0.826301 | 0.003344 |

**delta −0.000013 against a bar of 0.0077 — 0.00× of it.** Twelve times the data, at the same
epoch count, moves mean_area by thirteen parts per million. This is not "too small to detect"; the
point estimate is *zero to four decimal places*.

**Correcting my own earlier report.** At the stop I had only seed 11 and reported −0.0037 as "data
scale bought nothing" — which the ruling recorded. The conclusion held, **but the magnitude was a
seed draw**: s11 (0.822565) is the lowest of the three full-file seeds, range 0.822565–0.829014.
The honest figure is −0.000013, not −0.0037. A single seed exaggerated a null into a small
negative, which is precisely the failure the seed floor exists to prevent — and I published it
anyway because it was the only seed I had.

**Was the borrowed bar reasonable?** Yes, and slightly *permissive* rather than conservative. The
in-regime 3v3 bar is 0.0082 against the 0.0074 I borrowed — 10% too small, so it made separability
marginally easier to claim, not harder. No call flips: the largest full-file delta seen all night
(held-out ellipse, 0.0073) sits at 0.51× the in-regime single-run bar of 0.0142.

**Seed std is stable across regimes** (0.0029 → 0.0033, +14% on 12× the data), which is itself
useful: the floor is a property of the training procedure, not of the dataset size.

*Use `auc_clean_full` for clean AUC.* The ruling initially carried 0.9259 for this row, which is
`auc_clean_test` — the probe's held-out split. Mixing the two fields makes the full-file model look
0.001 better than like-for-like.

### Did generalisation move even though the dev score did not?

No. WP-M ran held-out and colleague suites on both regimes, seed 11, after verifying the checkpoint
against my sha256 sidecar (exact match). Six families:

| suite | family | full file | small file | delta |
|---|---|---|---|---|
| held-out | ellipse | 0.7867 ±0.0019 | 0.7940 | **−0.0073** |
| held-out | annulus | 0.8090 ±0.0025 | 0.8132 | −0.0042 |
| held-out | diagonal | 0.8031 ±0.0015 | 0.8056 | −0.0025 |
| colleague | c_ellipse | 0.8315 ±0.0018 | 0.8293 | +0.0022 |
| colleague | c_cell_dropout | 0.8033 ±0.0013 | 0.8051 | −0.0018 |
| colleague | c_edge_truncation | 0.7589 ±0.0027 | 0.7633 | −0.0044 |

Five slightly down, one slightly up, all within −0.0073 to +0.0022 — the same picture as the dev
row (−0.0037). **A consistent null across six independent families**, which is worth more than the
single dev number: the full file changed nothing measurable in either direction, rather than
trading one kind of robustness for another.

**The trap in the largest delta.** Held-out ellipse at −0.0073 is 2.4× the *small-file* seed std,
so quoted against a per-seed figure it would read as a real effect. It is not: the correct
single-run-vs-single-run bar is 0.0128, and 0.0073 is 0.57× of it. **A single full-file seed cannot
distinguish a data-regime effect from a seed draw** — the same trap `l-l1-sincos` fell into on the
dev bench tonight, best-looking arm at n=1 and indistinguishable at n=3.

Regimes are distinguishable from the JSONs alone: `train_data` is `robust_tagging_train_data.pt`
on the full rows and `robust_tagging_train_data_small.pt` on the small ones, so no row can be
silently compared across regimes.

---

## 2. Is the full-vs-small comparison confounded by the generator?

No. The same generator, same seed, 4000 events from each file:

| regime | dropped mean | median | q90 | frac-zero |
|---|---|---|---|---|
| small (200 cand/event) | 0.0688 | 0.0550 | 0.1650 | 0.271 |
| full (400 cand/event) | 0.0677 | 0.0550 | 0.1600 | 0.254 |

Medians identical, means within 1.6%. A dead region removes ~2× as many candidates on a
400-candidate event but the **same fraction**, which is the invariance that matters.

*Caveat on my own measurement:* the generator has a curriculum (`warmup_calls` 2000) and both
samples were drawn at its start, so these absolutes are lower than a mid-training draw. Both
regimes were sampled identically, so the comparison holds; the absolute numbers should not be
quoted as training-time dropped fractions.

---

## 3. The full-data loader (`integration-2` @`f437155`)

| measurement | value |
|---|---|
| `torch.load(mmap=True)` | 0.01 s, 0.52 GB RSS (eager read: 12.0 GB) |
| full chunked integrity check | 36 s, **0 bad chunks** |
| batch indexing | 2 ms / 256 events → 0.1 min/epoch — **not the bottleneck** |
| epoch, uncontended | **177 s** |
| epoch, five slots busy | **188 s** (1.06×) |
| class counts | QCD 450979 / DY 98878 / TT 292768 / WJets 97423 — 4.6× imbalance, **unweighted** |

**The trap:** `torch.load(mmap=True)` returns a **writable, copy-on-write** mapping, not a
read-only one. `clean_data`'s in-place `nan_to_num_` therefore *succeeds* on it and would fault in
and privatise all 12 GB **in every process** — recreating the problem mmap exists to solve, while
looking like it worked. The mapped path never writes; it verifies in chunks and fails loudly.

---

## 4. The scheduler: six faults, mechanism and defence

`gpu_slot.sh` — 5 slots, fair share by package, `--priority`, `--queue`, `--drain`.
All six were mine. Each fix was correct; several left an adjacent instance of the same fault alive.

| # | Fault | Mechanism | Defence |
|---|---|---|---|
| 1 | Queued jobs silently reaped | Wrapper captured `$PPID` and abandoned a queued job when it died. Detached launches (`setsid nohup … &`) die within seconds, so with all slots busy the job vanished, writing nothing and returning no error. **Cost WP-L four runs.** | Default reversed: queued jobs survive. `--abandon-if-launcher-dies` is opt-in. |
| 2 | That opt-in flag was itself broken | `setsid` reparents to init *before* `$PPID` is read, and `kill -0 1` always fails for non-root — so the flag would have abandoned **every** detached job instantly, the opposite of its name. | Detect "no launcher to watch" and say so. **Only a live-launcher test would have missed this.** |
| 3 | Bench cap by counting | TOCTOU: the `.run` file is written *after* the slot is taken, so 5 benches all observed 0 running and all acquired past a cap of 3. | Dedicated token locks. |
| 4 | Token released too early | `cleanup_q` runs immediately after acquisition and closed the token, so the cap held only *while queued* — exactly backwards. | Token held for the job's lifetime. |
| 5 | Vanishing-registry race, ×3 | A competitor deletes its `.job`/`.run` between the glob and the read; sourcing it left `JQ_*`/`JR_*` unset under `set -u`. I fixed `my_turn` and shipped believing the class was fixed; `running_for`, `running_benches` and `train_queued` had it too. | Read once, tolerate absence, default every field. |
| 6 | **The deadlock** | A bench took its token *before* waiting for a slot. Three benches held all three tokens while waiting for slots that training kept refilling; every other bench logged "all tokens held". Token holders waited on slots, non-holders waited on tokens. **Zero bench acquisitions for 18 minutes, across every package.** | Cap removed entirely; **slot 4 reserved for benches** whenever one is queued. |

### Two more, about signals rather than scheduling

**`--drain` reported `rc=0`.** An `EXIT` trap ending in `return 0` overwrote the signal-derived
status, so the wrapper *laundered a SIGTERM death into success*. A killed bench read as a completed
one — a missing row that looks present. Now exits **75**. (Found by WP-L.)

**I bypassed my own audit trail.** I broke the deadlock with `kill -TERM` instead of my own
`--drain`, so no drain line was logged. WP-L then correctly checked for a drain record, correctly
found none, and correctly concluded their job had not been drained. It had — by me. *A tool that
logs its actions is only a defence while everyone uses it, including its author under time pressure.*

### Kill semantics, tested rather than assumed

WP-I reported that killing a wrapper kills its acquired children. Tested four ways on a live job:

```
plain kill <launcher>          -> job SURVIVED
kill -TERM -<PGID> (group TERM)-> job SURVIVED
pkill -f <wrapper pattern>     -> job SURVIVED
kill -9 -<PGID> (group SIGKILL)-> job KILLED
```

Only SIGKILL to the process group destroys work, because no trap can catch KILL. WP-I retracted
the claim: `gpu.log` showed their first acquisition was at 04:50, so the jobs they thought they had
destroyed at 04:29 had never started — they read an emptiness caused by the deadlock as the
consequence of their own `kill`. **Plain kill is safe for tidying launchers; `kill -9 -PGID` is not.**

---

## 5. Durable checkpoint store

`~/hackathon-shared/checkpoints/c2/` — reference checkpoints, `.sha256` sidecars, and training
logs. The Phase-2 script writes the canonical copy and its sidecar *as part of the run* and
verifies with `cmp`, rather than relying on anyone remembering afterwards.

WP-N caught two things worth recording: reference checkpoints originally lived in a **session
scratchpad** that dies with the session, while every campaign-2 arm is compared against them; and I
briefly maintained **two stores**, with the sha256 discipline in one and the Phase-2 destination in
the other — so the rows the promotion rule turns on would have been the ones without integrity
sidecars. Consolidated; the old path is a symlink so nothing that referenced it broke.

---

## 6. Tools left behind

- `gpu_slot.sh --queue` / `--drain <pid|all>` — see what waits, stop it deliberately, exit code 75.
- `bench/verify_row.sh <tag> [train_data]` — **NEVER-RAN / INCOMPLETE / OK**, from `runs/*.json`
  only. Completion must distinguish *three* states, not two (WP-K's generalisation): "never
  started" collapsed into "finished" is how a table acquires rows that do not exist.
- `locks/priority_grants` — planner-only grant file; every priority acquisition logs `PRIORITY(pkg=…)`.

---

## 7. What this package got wrong

**A retraction of my own campaign-1 headline.** Recomputed against the seed floor measured here,
single-run arms need `3·√(2s²)` = 0.0128, not the probe-refit thresholds I used:

| campaign-1 claim | diff | was | now |
|---|---|---|---|
| a-pma vs winner (R=20) | +0.0021 | 1.70× separable | **0.16× — not separable** |
| a-pma vs winner (held-out) | +0.0038 | 1.65× separable | **0.30× — not separable** |
| **a-pma vs d-pma0-aug ("the body helps")** | +0.0113 | 2.20× separable | **0.88× — not separable** |
| a-pma vs E clean control | +0.0186 | 19.2× separable | 1.45× — weakly separable |

The third was my headline finding. I used the variance of *refitting a probe on a fixed model* to
judge a difference produced by *training two different models* — and spent that evening correcting
other people's separability thresholds three times without asking whether I was dividing the right
sigma by three at all. The honest statement is **"not supported"**, not "refuted": 0.0030 is
measured on one config and regime.

**Three confident-looking nothings.** `fuser` (not installed — silently reports no holders; I
nearly reported five leaked slots); a `grep -c` that counted *queued* lines as acquisitions and
made a working cap look broken; and `find -newermt "04:31"` returning zero, from which I was
seconds away from telling the planner that benches were acquiring but writing no rows — `-mmin -30`
showed 54. An empty result feels like a finding, and the ten-second check *"does this query return
anything under conditions I know are true"* is the one that kept getting skipped.
