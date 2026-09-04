# gF-bits6-relu-l1t -- how to build and evaluate it

The recommended quantized operating point: 6-bit cap AND ReLU together, QAT-trained from the L1T float reference. Evaluated on the L1T file -- a PF number for this model would be meaningless.

## What builds it

| | |
|---|---|
| code branch | `wp-g` |
| code commit | `e9939b0` |
| train config | `configs/train_config.yaml` (copied here as `train_config.yaml`) |
| checkpoint | `gF-bits6-relu-l1t.pth` |
| sha256 | `7fa297ddcc03c3af98256c93d7d92e957a700d841ea15c4bd706aa4e76c3af6c` |
| encoder class | `QuantizedPMAEncoder` |
| preprocessor class | `PFPreProcessorMeanPt` |
| encoder parameters | 0 parameters / 67,075 buffer elements |
| eval file | L1T (`--eta_max 3.0`) |

## The two config keys that decide what gets built

- `data.preproc_type: PFPreProcessorMeanPt` -- **this one fails silently if wrong.** `PFPreProcessor` and
  `PFPreProcessorMeanPt` have *identical* `state_dict` keys, so a mismatched value loads cleanly and
  then computes a different pt feature. There is no error to catch; the only symptom is a worse number.
- `hyperparameters.encoder_class: QuantizedPMAEncoder` -- read by `train.py` and by `bench_eval.py --encoder_class`.
  **`eval.py` never reads it.** `eval.py` constructs the name `TransformerEncoder` and nothing else, so
  for `eval.py` the class is decided entirely by the alias at the bottom of `src/embedding/models.py`.

## Alias needed for eval.py

WRONG on wp-g -- the branch aliases `TransformerEncoder = PMAEncoder`, which will not load this checkpoint.

`eval.py` builds `TransformerEncoder(...)` unconditionally. On `wp-g` that name does not
resolve to `QuantizedPMAEncoder`, so `eval.py` cannot evaluate this checkpoint as the branch stands. `evaluate.sh`
appends one line to the fresh clone before running:

```python
TransformerEncoder = QuantizedPMAEncoder
```

This was verified: without the line `load_state_dict` raises `RuntimeError: Missing key(s) in state_dict`;
with it the encoder builds and loads with 0 parameters / 67,075 buffer elements parameters. The failure is **loud**, not silent --
a wrong alias cannot quietly score a different model.

## Evaluate it

```bash
./evaluate.sh                 # uses ~/hackathon-shared/repo.git
./evaluate.sh <repo-url>      # or any clone/mirror of it
```

Step (a) runs the organisers' `eval.py` from the fresh clone and writes its plots and
`auc_vs_severity.json`. Step (b) runs `bench_eval.py` and writes a row into
`~/hackathon-shared/runs/reproduce-gF_<timestamp>.json`, which is directly comparable to the
numbers for this model in `docs/table.md` and `models/MODELS.md`.
