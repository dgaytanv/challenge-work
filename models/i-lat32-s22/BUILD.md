# i-lat32-s22 -- how to build and evaluate it

Phase-2 candidate: the champion with `latent_dim` 32 instead of 6. Highest three-seed mean measured in campaign 2, but NOT separable from the champion at the seed floor.

## What builds it

| | |
|---|---|
| code branch | `c2-i` |
| code commit | `e1c7aa8` |
| train config | `configs/c2/i6_lat32_s22.yaml` (copied here as `train_config.yaml`) |
| checkpoint | `rt_i6_lat32_s22_encoder_20260904_041601.pth` |
| sha256 | `715a4bfe6af32f56d136ca9d126d442620f92dd004c1f6364065a86a8badc006` |
| encoder class | `PMAEncoder` |
| preprocessor class | `PFPreProcessorMeanPt` |
| encoder parameters | 102,944 |
| eval file | PF (`--eta_max 5.0`) |

## The two config keys that decide what gets built

- `data.preproc_type: PFPreProcessorMeanPt` -- **this one fails silently if wrong.** `PFPreProcessor` and
  `PFPreProcessorMeanPt` have *identical* `state_dict` keys, so a mismatched value loads cleanly and
  then computes a different pt feature. There is no error to catch; the only symptom is a worse number.
- `hyperparameters.encoder_class: PMAEncoder` -- read by `train.py` and by `bench_eval.py --encoder_class`.
  **`eval.py` never reads it.** `eval.py` constructs the name `TransformerEncoder` and nothing else, so
  for `eval.py` the class is decided entirely by the alias at the bottom of `src/embedding/models.py`.

## Alias needed for eval.py

ABSENT on c2-i -- must be appended.

`eval.py` builds `TransformerEncoder(...)` unconditionally. On `c2-i` that name does not
resolve to `PMAEncoder`, so `eval.py` cannot evaluate this checkpoint as the branch stands. `evaluate.sh`
appends one line to the fresh clone before running:

```python
TransformerEncoder = PMAEncoder
```

This was verified: without the line `load_state_dict` raises `RuntimeError: Missing key(s) in state_dict`;
with it the encoder builds and loads with 102,944 parameters. The failure is **loud**, not silent --
a wrong alias cannot quietly score a different model.

## Evaluate it

```bash
./evaluate.sh                 # uses ~/hackathon-shared/repo.git
./evaluate.sh <repo-url>      # or any clone/mirror of it
```

Step (a) runs the organisers' `eval.py` from the fresh clone and writes its plots and
`auc_vs_severity.json`. Step (b) runs `bench_eval.py` and writes a row into
`~/hackathon-shared/runs/reproduce-i-lat32-s22_<timestamp>.json`, which is directly comparable to the
numbers for this model in `docs/table.md` and `models/MODELS.md`.
