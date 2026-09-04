# i-ff256l1-s22 -- how to build and evaluate it

Phase-2 candidate: one transformer layer plus a 256-wide feed-forward before the PMA readout. Best single seed measured (0.8304), also NOT separable from the champion.

## What builds it

| | |
|---|---|
| code branch | `c2-i` |
| code commit | `e1c7aa8` |
| train config | `configs/c2/i1p_ff256l1_s22.yaml` (copied here as `train_config.yaml`) |
| checkpoint | `rt_i1p_ff256l1_s22_encoder_20260904_035107.pth` |
| sha256 | `b1409dfa49f35b35af2c485902b7380e78bc4ea97ca6b7fde90c59f04a086880` |
| encoder class | `PMAEncoderFF256` |
| preprocessor class | `PFPreProcessorMeanPt` |
| encoder parameters | 222,254 |
| eval file | PF (`--eta_max 5.0`) |

## The two config keys that decide what gets built

- `data.preproc_type: PFPreProcessorMeanPt` -- **this one fails silently if wrong.** `PFPreProcessor` and
  `PFPreProcessorMeanPt` have *identical* `state_dict` keys, so a mismatched value loads cleanly and
  then computes a different pt feature. There is no error to catch; the only symptom is a worse number.
- `hyperparameters.encoder_class: PMAEncoderFF256` -- read by `train.py` and by `bench_eval.py --encoder_class`.
  **`eval.py` never reads it.** `eval.py` constructs the name `TransformerEncoder` and nothing else, so
  for `eval.py` the class is decided entirely by the alias at the bottom of `src/embedding/models.py`.

## Alias needed for eval.py

ABSENT on c2-i -- must be appended.

`eval.py` builds `TransformerEncoder(...)` unconditionally. On `c2-i` that name does not
resolve to `PMAEncoderFF256`, so `eval.py` cannot evaluate this checkpoint as the branch stands. `evaluate.sh`
appends one line to the fresh clone before running:

```python
TransformerEncoder = PMAEncoderFF256
```

This was verified: without the line `load_state_dict` raises `RuntimeError: Missing key(s) in state_dict`;
with it the encoder builds and loads with 222,254 parameters. The failure is **loud**, not silent --
a wrong alias cannot quietly score a different model.

## Evaluate it

```bash
./evaluate.sh                 # uses ~/hackathon-shared/repo.git
./evaluate.sh <repo-url>      # or any clone/mirror of it
```

Step (a) runs the organisers' `eval.py` from the fresh clone and writes its plots and
`auc_vs_severity.json`. Step (b) runs `bench_eval.py` and writes a row into
`~/hackathon-shared/runs/reproduce-i-ff256l1-s22_<timestamp>.json`, which is directly comparable to the
numbers for this model in `docs/table.md` and `models/MODELS.md`.
