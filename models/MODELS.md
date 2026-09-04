# The eight models, what builds each, and what each scores

Every number in this file was read out of a JSON in `~/hackathon-shared/runs/` at the time this file
was generated; the sources are listed at the bottom. Nothing here is quoted from memory.

## Read this before you read the table

**The binding noise floor is the training seed, not the probe.** Three seeds of the champion
configuration measure 0.8252 / 0.8296 / 0.8239
(std 0.0030). The separability bar between two three-seed configurations is about
**0.0074**. Any two rows below closer than that are **not distinguishable** by this evidence --
"unproven", which means not established, not refuted.

**Suites are never mixed.** The dev bench (rect/wedge/strip/towers/cells), the held-out suite
(ellipse/annulus/diagonal), and the colleague Group 3 suite are different rulers. A dev-bench number
and a held-out number are not comparable, and neither is an official `eval.py` area.

**Only the champion is a certified submission.** The other seven are research models. Four of them
need a one-line alias before the organisers' `eval.py` can build them at all -- each model's
`BUILD.md` says which, and its `evaluate.sh` applies it.

## The table

| model | dev bench (PF) | held-out | colleague G3 | encoder params | eval.py works as-is? |
|---|---|---|---|---|---|
| **champion-pma0-meanpt** | **0.8262** ± 0.0030 (n=3) | 0.8021 | **0.7983** ± 0.0010 (n=3) | 89,606 | yes |
| deepsets-meanpt | 0.8186 | 0.7977 | -- | 53,126 | yes |
| a-pma-transformer | 0.8275 | 0.8059 | -- | 2,445,478 | yes |
| pma0-stock-preproc | 0.8162 | 0.7983 | -- | 89,606 | yes |
| i-lat32-s22 | **0.8292** ± 0.0003 (n=3) | -- | -- | 102,944 | no -- needs alias |
| i-ff256l1-s22 | **0.8287** ± 0.0015 (n=3) | -- | -- | 222,254 | no -- needs alias |
| l1t-champion | *L1T only* | -- | -- | 89,606 | no -- needs alias |
| gF-bits6-relu-l1t | *L1T only* | -- | -- | 67,075 buffer elts | no -- wrong alias on branch |
| *stock anchor, for scale* | 0.7734 | -- | -- | -- | -- |

### The L1T models, on the L1T file (`--eta_max 3.0`)

These three are only comparable to each other. A PF number for a quantized L1T model is meaningless.

| model | L1T dev bench | L1T colleague G3 |
|---|---|---|
| l1t-champion (float reference) | 0.8117 | 0.7862 |
| gF-bits6-relu-l1t (quantized) | 0.8089 | -- |
| champion, PF-trained, scored on L1T | **0.8097** ± 0.0019 (n=3) | **0.7823** ± 0.0021 (n=3) |

Quantization costs **0.0029** against its own float
reference (0.8117 -> 0.8089). Both are single
runs, so read that against the 0.0074 bar: it is **within noise**, i.e. this evidence does not show
6-bit + ReLU costing anything measurable.

### Official `eval.py` areas -- champion only

The champion is the only model with organisers'-`eval.py` numbers: **0.8790, 0.8786, 0.8780, 0.8795,
0.8794, 0.8805** across six independent runs of the same checkpoint. The 0.0025 spread is the
organisers' unseeded Bernoulli degradation plus an unseeded probe refit; it is why a single official
number is never quoted as exact. Sources are in `docs/writeup/SUBMISSION-HANDOVER.md`.

## What the table does and does not establish

- **The champion's margin over the stock anchor is real**: 0.8262 vs 0.7734,
  about 0.053, far outside any floor here.
- **`a-pma` ties the champion on the dev bench at 27x the parameters** (0.8275 vs
  0.8262) and is *ahead* on held-out (0.8059 vs
  0.8021). Both gaps are inside the bar. The champion was chosen on
  size and on the fact that it is the certified branch, not on a proven accuracy win.
- **Neither Phase-2 candidate is a proven improvement.** i-lat32 is +0.0029
  and i-ff256l1 +0.0025 over the champion's three-seed mean, against a
  0.0074 bar. Campaign 2 found zero separable improvements across 18 configurations.
- **Deep Sets is 0.0076 behind the champion** at 60% of the
  parameters -- the one gap in this table wide enough to call.
- **MeanPt alone is not the win.** `pma0-stock-preproc` is the champion minus MeanPt:
  0.8162 vs 0.8262. The PMA readout and the MeanPt preprocessor each
  tie the baseline on their own; together they separate. The win is an interaction -- see
  `docs/RULING.md`.

## Evaluating any of them

```bash
cd models/<name>
./evaluate.sh                      # clones the right branch+commit, applies any alias, runs both
```

Each directory holds the checkpoint (or a pointer, if over 5 MB), its `.sha256`, the exact
`train_config.yaml` it was trained with, a `BUILD.md`, and `evaluate.sh`. `evaluate.sh` runs the
organisers' `eval.py` from a **fresh clone** of the named branch, then our `bench_eval.py` with the
right `--encoder_class`, `--train_cfg` and eval file.

## Provenance

Every number above came from these files in `~/hackathon-shared/runs/`:

- `a-pma-heldout_20260903_225347.json`
- `a-pma_20260903_224712.json`
- `anchor-stock-baseline_20260903_194212.json`
- `d-deepsets-aug-meanpt-heldout_20260903_222704.json`
- `d-deepsets-aug-meanpt_20260903_220604.json`
- `d-pma0-aug-heldout_20260903_222509.json`
- `d-pma0-aug-meanpt-heldout_20260903_223745.json`
- `d-pma0-aug-meanpt-l1t-colleague_20260904_044836.json`
- `d-pma0-aug-meanpt-l1t_20260904_000722.json`
- `d-pma0-aug_20260903_214732.json`
- `gF-bits6-relu-l1t_20260904_011158.json`
- `h-ref-small-s11-colleague_20260904_044635.json`
- `h-ref-small-s11-colleague_20260904_051311.json`
- `h-ref-small-s11-heldout_20260904_044546.json`
- `h-ref-small-s11-heldout_20260904_051219.json`
- `h-ref-small-s11-l1t-colleague_20260904_044829.json`
- `h-ref-small-s11-l1t-colleague_20260904_051459.json`
- `h-ref-small-s11-l1t_20260904_044738.json`
- `h-ref-small-s11-l1t_20260904_051410.json`
- `h-ref-small-s11_20260904_041158.json`
- `h-ref-small-s11_20260904_041301.json`
- `h-ref-small-s22-colleague_20260904_051128.json`
- `h-ref-small-s22-colleague_20260904_051658.json`
- `h-ref-small-s22-heldout_20260904_051548.json`
- `h-ref-small-s22-l1t-colleague_20260904_043326.json`
- `h-ref-small-s22-l1t-colleague_20260904_051940.json`
- `h-ref-small-s22-l1t_20260904_043234.json`
- `h-ref-small-s22-l1t_20260904_051826.json`
- `h-ref-small-s22_20260904_035504.json`
- `h-ref-small-s33-colleague_20260904_035023.json`
- `h-ref-small-s33-colleague_20260904_040116.json`
- `h-ref-small-s33-colleague_20260904_043512.json`
- `h-ref-small-s33-heldout_20260904_043421.json`
- `h-ref-small-s33-l1t-colleague_20260904_035249.json`
- `h-ref-small-s33-l1t-colleague_20260904_040342.json`
- `h-ref-small-s33-l1t-colleague_20260904_043717.json`
- `h-ref-small-s33-l1t_20260904_035151.json`
- `h-ref-small-s33-l1t_20260904_040248.json`
- `h-ref-small-s33-l1t_20260904_043618.json`
- `h-ref-small-s33_20260904_034839.json`
- `i-ff256l1-s11_20260904_045308.json`
- `i-ff256l1-s22_20260904_045856.json`
- `i-ff256l1-s22_20260904_050334.json`
- `i-ff256l1-s33_20260904_050919.json`
- `i-lat32-s11_20260904_045631.json`
- `i-lat32-s22_20260904_050544.json`
- `i-lat32-s33_20260904_050652.json`

Checkpoint digests for every model are in `docs/CHECKPOINTS.md` and in each model's `.sha256`.
