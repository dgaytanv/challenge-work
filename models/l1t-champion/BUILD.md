# l1t-champion -- how to build and evaluate it

The champion recipe retrained on the L1T inputs. This is the float reference the quantized operating point is measured against, so the two must be evaluated on the same L1T file.

## What builds it

| | |
|---|---|
| code branch | `wp-d` |
| code commit | `143e0b2` |
| train config | `configs/train_config_d_pma0_aug_meanpt_l1t.yaml` (copied here as `train_config.yaml`) |
| checkpoint | `rt_d_pma0_aug_meanpt_l1t_encoder_20260903_235356.pth` |
| sha256 | `d8a9fef8dc343c347a17517a982a8d7383215e41bc2e3d1c608b8c87d977ac53` |
| encoder class | `PMAEncoder` |
| preprocessor class | `PFPreProcessorMeanPt` |
| encoder parameters | 89,606 |
| eval file | L1T (`--eta_max 3.0`) |

## The two config keys that decide what gets built

- `data.preproc_type: PFPreProcessorMeanPt` -- **this one fails silently if wrong.** `PFPreProcessor` and
  `PFPreProcessorMeanPt` have *identical* `state_dict` keys, so a mismatched value loads cleanly and
  then computes a different pt feature. There is no error to catch; the only symptom is a worse number.
- `hyperparameters.encoder_class: PMAEncoder` -- read by `train.py` and by `bench_eval.py --encoder_class`.
  **`eval.py` never reads it.** `eval.py` constructs the name `TransformerEncoder` and nothing else, so
  for `eval.py` the class is decided entirely by the alias at the bottom of `src/embedding/models.py`.

## Alias needed for eval.py

ABSENT on wp-d -- must be appended.

`eval.py` builds `TransformerEncoder(...)` unconditionally. On `wp-d` that name does not
resolve to `PMAEncoder`, so `eval.py` cannot evaluate this checkpoint as the branch stands. `evaluate.sh`
appends one line to the fresh clone before running:

```python
TransformerEncoder = PMAEncoder
```

This was verified: without the line `load_state_dict` raises `RuntimeError: Missing key(s) in state_dict`;
with it the encoder builds and loads with 89,606 parameters. The failure is **loud**, not silent --
a wrong alias cannot quietly score a different model.

## Evaluate it

```bash
./evaluate.sh                 # uses ~/hackathon-shared/repo.git
./evaluate.sh <repo-url>      # or any clone/mirror of it
```

Step (a) runs the organisers' `eval.py` from the fresh clone and writes its plots and
`auc_vs_severity.json`. Step (b) runs `bench_eval.py` and writes a row into
`~/hackathon-shared/runs/reproduce-l1t-champion_<timestamp>.json`, which is directly comparable to the
numbers for this model in `docs/table.md` and `models/MODELS.md`.
