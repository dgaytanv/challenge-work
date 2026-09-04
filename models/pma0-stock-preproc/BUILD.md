# pma0-stock-preproc -- how to build and evaluate it

The champion architecture on the STOCK preprocessor. Ships so the MeanPt half of the interaction can be isolated: this is the champion minus MeanPt.

## What builds it

| | |
|---|---|
| code branch | `submission-pma0` |
| code commit | `99b7d2e` |
| train config | `configs/train_config.yaml` (copied here as `train_config.yaml`) |
| checkpoint | `rt_d_pma0_aug_encoder_20260903_204550.pth` |
| sha256 | `f54d4d8293fd3728568a5f289ef00c5c952eee7778d20b3f8a59a60eb8a4a445` |
| encoder class | `PMAEncoder` |
| preprocessor class | `PFPreProcessor` |
| encoder parameters | 89,606 |
| eval file | PF (`--eta_max 5.0`) |

## The two config keys that decide what gets built

- `data.preproc_type: PFPreProcessor` -- **this one fails silently if wrong.** `PFPreProcessor` and
  `PFPreProcessorMeanPt` have *identical* `state_dict` keys, so a mismatched value loads cleanly and
  then computes a different pt feature. There is no error to catch; the only symptom is a worse number.
- `hyperparameters.encoder_class: PMAEncoder` -- read by `train.py` and by `bench_eval.py --encoder_class`.
  **`eval.py` never reads it.** `eval.py` constructs the name `TransformerEncoder` and nothing else, so
  for `eval.py` the class is decided entirely by the alias at the bottom of `src/embedding/models.py`.

## Alias needed for eval.py

already on the branch (`TransformerEncoder = PMAEncoder`).

`eval.py` builds `TransformerEncoder(...)` unconditionally, and on this branch that name already
resolves to `PMAEncoder`. Nothing needs to be patched: a fresh clone evaluates correctly as it stands.
Verified by building the encoder exactly as `eval.py` does and loading this checkpoint into it
(89,606 parameters, no missing or unexpected keys).

## Evaluate it

```bash
./evaluate.sh                 # uses ~/hackathon-shared/repo.git
./evaluate.sh <repo-url>      # or any clone/mirror of it
```

Step (a) runs the organisers' `eval.py` from the fresh clone and writes its plots and
`auc_vs_severity.json`. Step (b) runs `bench_eval.py` and writes a row into
`~/hackathon-shared/runs/reproduce-pma0-stock_<timestamp>.json`, which is directly comparable to the
numbers for this model in `docs/table.md` and `models/MODELS.md`.
