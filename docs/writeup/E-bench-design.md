# The measurement backbone: what we steer on, and why

WP-E. Every number here is sourced to a JSON in `~/hackathon-shared/runs/` or a file in
`~/hackathon-shared/reports/`.

---

## JSON schema reference (for anyone consuming `runs/*.json`)

`runs/` holds **two different record types plus strays**. Discriminate on `kind`, never on the
filename, and skip anything you do not recognise -- `auc_vs_severity.json` and other artefacts get
copied in there too.

```python
d = json.load(open(path))
if not isinstance(d, dict):                          continue     # stray
if d.get("kind") == "official":                      ...          # accept.sh record
elif all(k in d for k in ("tag","families","mean_area")):  ...    # bench_eval.py record
else:                                                continue     # stray
```

### 1. Bench record -- written by `bench_eval.py`, filename `<tag>_<YYYYmmdd_HHMMSS>.json`

Has no `kind` field (it predates the split). Top level:

| key | type | notes |
| --- | --- | --- |
| `tag` | str | the run label; **not unique** -- see de-duplication below |
| `timestamp` | str | `"%Y-%m-%d %H:%M:%S"` |
| `repo`, `commit`, `ckpt` | str | provenance; `commit` may carry a `-dirty` suffix |
| `encoder_class` | str | `TransformerEncoder`, `DeepSetsEncoder`, ... |
| `events` | int | events benchmarked |
| `full` | bool | true if `--full` (whole eval set) |
| `auc_clean_test` | float | probe's own 20% test split |
| `auc_clean_full` | float | **use this one** -- clean AUC over all embedded events |
| `families` | dict | family name -> family block, below |
| `mean_area` | float | unweighted mean of the family areas: the headline metric |

Family block: `severities` (list, always `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`), `aucs` (list, same
length, `aucs[0] == auc_clean_full`), `dropped_fraction` (list, fraction of real candidates
actually removed), `area` (float, `numpy.trapz(aucs, severities)` normalised by the severity
range).

**Added by `--probe_repeats`** (older JSONs lack these -- treat their absence as `n=1` with an
*unmeasured* error, not as zero error): `probe_repeats` (int), `auc_clean_full_std`,
`auc_clean_test_std`, `auc_clean_full_repeats` (list), `mean_area_std`, `mean_area_repeats`
(list), and per family `area_std`, `area_repeats` (list), `aucs_std` (list).

Family names: the five **scoring** families are `rect`, `wedge`, `strip`, `towers`, `cells`. The
three **held-out** families are `ellipse`, `annulus`, `diagonal` -- a record containing only those
is a held-out run, and its `mean_area` is **not comparable** to a scoring run's. Never plot them on
the same axis as a scoring `mean_area` without saying so.

**De-duplicate on `(tag, full, frozenset(families))`, not on `tag`.** A 20k run, a `--full` run and
a held-out run can all share a tag; keying on tag alone silently drops two of them. Also exclude
held-out records when resolving an anchor, or a held-out run sharing the anchor's tag hijacks the
reference. Both of these were real bugs in `ablation_table.py`.

### 2. Official record -- written by `accept.sh`, filename `official_<tag>_<ts>.json`

| key | type | notes |
| --- | --- | --- |
| `kind` | str | always `"official"` |
| `tag`, `branch`, `commit`, `ckpt` | str | provenance |
| `official_area` | float | the number `eval.py` printed |
| `severities`, `aucs` | list | merged from `auc_vs_severity.json`; **10 severities**, not 6, and they are `np.linspace(0, 1, 10)` with a duplicated leading 0 |
| `data` | str | eval file used |
| `stopped_before_tsne` | bool | `--fast` was used; does not affect the area |
| `smoke_test` | bool | **`true` means the area is not a result** -- a small slice used to test the script. Filter these out of any plot |
| `degradation`, `note` | str | provenance and caveats |

`official_area` and `mean_area` are **different measurements on different corruptions with
different severity grids** and must never be plotted on one axis as if commensurable, averaged, or
subtracted. The official grid is marked TEMPORARY upstream and stated to change before judging.

Both the organisers' Bernoulli mask and the probe refit are unseeded, so repeated official runs of
one checkpoint differ; several records may legitimately share a tag, and the spread is the useful
quantity.

---

## The problem the bench solves

The grader degrades the eval set with a method we do not see, embeds the clean events, trains an
MLP probe on those latents, then scores the *same frozen probe* on re-embedded degraded events.
The score is the area under the AUC-vs-severity curve. So the quantity that matters is not
accuracy but **stability of the latent under deletion of a spatial region**, for events the
encoder never trained on.

Steering on a single corruption shape would be the obvious mistake: it is trivial to become
robust to one dead rectangle and no more. So the development ruler,
`bench/bench_eval.py`, mirrors the grader's *mechanics* exactly -- clean latents, probe fit on
80% of them, that probe frozen and applied to degraded latents, grace period dropped -- while
sweeping **five** dead-region families (`rect`, `wedge`, `strip`, `towers`, `cells`) at
severities 0.0-1.0 and reporting `mean_area` across them.

Anchors, from `runs/anchor-stock-baseline_20260903_194212.json`:

| | clean AUC | mean_area |
| --- | ---: | ---: |
| organisers' stock checkpoint | 0.8605 | 0.7734 |

## Three different numbers, none interchangeable

A recurring source of confusion, so it is worth stating flatly:

| number | produced by | corruption | use |
| --- | --- | --- | --- |
| `mean_area` | `bench_eval.py` | our five families, seeded | **what we steer on** |
| official area | `accept.sh` -> organisers' `eval.py` | their `degradation_eval.py`, 10x10 grid | contract check + tie-break |
| held-out area | `bench_eval.py --families heldout` | ellipse/annulus/diagonal, seeded | honesty check |

They are **not comparable to each other** -- different shapes, different severity grids,
different numbers of sweep points. The official number additionally comes from a module the
organisers marked TEMPORARY and stated will change before judging, so nothing is tuned to it.

## Reproducibility problems found and fixed along the way

**The src-layout import collision.** The repo is a src-layout (`package-dir = {"": "src"}`), so
running `python train.py` from a repo root does not put `embedding` on `sys.path`; the single
editable-install `.pth` in the shared environment resolves to whichever clone installed last.
With five clones sharing one Python environment, every instance was training and evaluating
another instance's code. Measured directly: `cd ~/rt-e && python -c "import embedding"` resolved
to `/home/jovyan/rt-d/src/embedding/__init__.py`. Fixed with `PYTHONPATH=<repo>/src`, now
exported automatically by both GPU wrappers. `bench_eval.py` was never affected (it inserts
`repo/src` itself), so benchmark numbers taken before the fix stand.

**`eval.py` memory.** `eval.py` embeds at a hard-coded batch of 1024. At 400 candidates that is a
`[1024, 8, 401, 401]` attention tensor, ~5 GB, so it cannot overlap a training run on one A10.
The requirement is architecture-dependent rather than a property of `eval.py`: attention is
O(batch x heads x N^2) and dominates, while a per-token MLP is O(batch x N x embed) and two
orders of magnitude smaller -- WP-D measured `DeepSetsEncoder` at 448 MiB allocated / 514 MiB
reserved at eval.py's exact shape under `no_grad`. `bench_eval.py`'s default was lowered to 512
for the same reason; batching does not change the latents, since the preprocessor's BatchNorm is
in eval mode and uses running statistics.

**The probe is unseeded.** See `E-probe-noise-floor.md`. This one changes how every row of the
table must be read.

## The ablation table

`bench/ablation_table.py` regenerates `runs/table.md` from every JSON in `runs/`, sorted by
`mean_area`, with a delta-vs-anchor column, the official-area column, and `--plot` for per-tag
AUC-vs-severity figures on `eval.py`'s axes. Nothing in the table is typed by hand.

Two correctness details that took a bug each to get right:

- Runs are de-duplicated on `(tag, full, family-set)`, not on tag. A held-out run and a scoring
  run of the same tag are different measurements; keying on tag alone let one silently evict the
  other.
- The delta-vs-anchor column resolves its anchor **before** filtering, and excludes held-out runs
  from anchor selection. Otherwise a held-out run sharing the anchor's tag hijacked the reference
  and every delta in the table was wrong -- observed, then fixed.

Held-out rows are marked `_(heldout)_` and their delta column is blanked, since a delta against a
five-scoring-family anchor is not meaningful.

## `--eta_max`: severity must be a fraction of the acceptance that has data in it

The five scoring families paint a plane `[-eta_max, eta_max] x (-pi, pi]` and `severity` is the
fraction of that plane made dead. The default 5.0 is the PF-candidate acceptance. Run against the
L1T/PUPPI file, whose data is `eta in [-3, 3]`, two thirds of a `[-5, 5]` plane contains no
candidates at all, so a family painting "60% of the plane" removes far more than 60% of the events.
Measured, `rect` on the L1T file -- fraction of candidates removed against the design target `s`:

```
    s     eta_max=5   eta_max=3
  0.2       0.284       0.234
  0.4       0.523       0.407
  0.6       0.769       0.609
  0.8       0.932       0.807
  1.0       1.000       0.960
```

At 5.0 the overshoot reaches 0.17 and saturates a severity early; at 3.0 the dropped fraction
tracks `s`. `--eta_max` (default 5.0, recorded in the JSON) rescales only the PLANE -- shape priors
stay in absolute eta units, so a rectangle is the same physical size on either acceptance and only
its share of the plane changes, which is what severity is defined against. The default is
bit-identical to the previous behaviour, verified, so every PF row remains reproducible. L1T rows
are labelled by which `eta_max` produced them and are never ranked against PF rows.

## Known property: the commit stamp is read at write time

`bench_eval.py` calls `git_hash(repo)` inside the dict it writes when the run **finishes**, not when
it starts. So the `commit` field records the working tree as it was at the END of the bench, and
`git_hash` appends `-dirty` on any uncommitted diff. **Never change a repo's working tree while a
bench is running against it** -- a merge, a checkout, or even a moment of staging mid-run silently
relabels the result with a commit that did not produce it, or marks it dirty for unrelated reasons.

This bit during convergence: a merge into `wp-e` was deferred until the reference-row bench exited,
precisely to avoid stamping the control row with a commit it was not produced by. It belongs to the
same family as the other silent failures recorded in `E-acceptance-tooling.md` -- the output looks
entirely normal and is simply attributed to the wrong code.
