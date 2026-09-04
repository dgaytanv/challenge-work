# a-pma-transformer -- how to build and evaluate it

WP-A's tuned attention encoder (CLS token, 4 layers, pairwise bias MLP). Despite the tag, this is NOT a PMA readout: it is the original architecture, tuned. It scores as well as the champion on the dev bench at 27x the parameters.

## What builds it

| | |
|---|---|
| code branch | `submission-a-pma` |
| code commit | `d450bcf` |
| train config | `configs/train_config.yaml` (copied here as `train_config.yaml`) |
| checkpoint | **not in this branch** -- see `CHECKPOINT-POINTER.txt` (28.1 MB, over the cap) |
| sha256 | `f64d0154289321bbfcccad02f949fd6c5fa78a009d33a9e770c546d8cee642bf` |
| encoder class | `TransformerEncoder` |
| preprocessor class | `PFPreProcessor` |
| encoder parameters | 2,445,478 |
| eval file | PF (`--eta_max 5.0`) |

## The two config keys that decide what gets built

- `data.preproc_type: PFPreProcessor` -- **this one fails silently if wrong.** `PFPreProcessor` and
  `PFPreProcessorMeanPt` have *identical* `state_dict` keys, so a mismatched value loads cleanly and
  then computes a different pt feature. There is no error to catch; the only symptom is a worse number.
- `hyperparameters.encoder_class: TransformerEncoder` -- read by `train.py` and by `bench_eval.py --encoder_class`.
  **`eval.py` never reads it.** `eval.py` constructs the name `TransformerEncoder` and nothing else, so
  for `eval.py` the class is decided entirely by the alias at the bottom of `src/embedding/models.py`.

## Alias needed for eval.py

none needed -- this model IS the stock `TransformerEncoder`.

`eval.py` builds `TransformerEncoder(...)` unconditionally, and on this branch that name already
resolves to `TransformerEncoder`. Nothing needs to be patched: a fresh clone evaluates correctly as it stands.
Verified by building the encoder exactly as `eval.py` does and loading this checkpoint into it
(2,445,478 parameters, no missing or unexpected keys).

## Evaluate it

```bash
./evaluate.sh                 # uses ~/hackathon-shared/repo.git
./evaluate.sh <repo-url>      # or any clone/mirror of it
```

Step (a) runs the organisers' `eval.py` from the fresh clone and writes its plots and
`auc_vs_severity.json`. Step (b) runs `bench_eval.py` and writes a row into
`~/hackathon-shared/runs/reproduce-a-pma_<timestamp>.json`, which is directly comparable to the
numbers for this model in `docs/table.md` and `models/MODELS.md`.
