# deepsets-meanpt -- how to build and evaluate it

Runner-up set encoder: masked mean+max pooling, no attention. The cheapest model here and the clearest demonstration that the win comes from the set readout, not from attention.

## What builds it

| | |
|---|---|
| code branch | `submission-deepsets-meanpt` |
| code commit | `791c450` |
| train config | `configs/train_config.yaml` (copied here as `train_config.yaml`) |
| checkpoint | `rt_d_deepsets_aug_meanpt_encoder_20260903_214738.pth` |
| sha256 | `ad18f158beb5e3361f01969e64a8d7accedd730d3849fdfeeaad7417445fa40b` |
| encoder class | `DeepSetsEncoder` |
| preprocessor class | `PFPreProcessorMeanPt` |
| encoder parameters | 53,126 |
| eval file | PF (`--eta_max 5.0`) |

## The two config keys that decide what gets built

- `data.preproc_type: PFPreProcessorMeanPt` -- **this one fails silently if wrong.** `PFPreProcessor` and
  `PFPreProcessorMeanPt` have *identical* `state_dict` keys, so a mismatched value loads cleanly and
  then computes a different pt feature. There is no error to catch; the only symptom is a worse number.
- `hyperparameters.encoder_class: DeepSetsEncoder` -- read by `train.py` and by `bench_eval.py --encoder_class`.
  **`eval.py` never reads it.** `eval.py` constructs the name `TransformerEncoder` and nothing else, so
  for `eval.py` the class is decided entirely by the alias at the bottom of `src/embedding/models.py`.

## Alias needed for eval.py

already on the branch (`TransformerEncoder = DeepSetsEncoder`).

`eval.py` builds `TransformerEncoder(...)` unconditionally, and on this branch that name already
resolves to `DeepSetsEncoder`. Nothing needs to be patched: a fresh clone evaluates correctly as it stands.
Verified by building the encoder exactly as `eval.py` does and loading this checkpoint into it
(53,126 parameters, no missing or unexpected keys).

## Evaluate it

```bash
./evaluate.sh                 # uses ~/hackathon-shared/repo.git
./evaluate.sh <repo-url>      # or any clone/mirror of it
```

Step (a) runs the organisers' `eval.py` from the fresh clone and writes its plots and
`auc_vs_severity.json`. Step (b) runs `bench_eval.py` and writes a row into
`~/hackathon-shared/runs/reproduce-deepsets_<timestamp>.json`, which is directly comparable to the
numbers for this model in `docs/table.md` and `models/MODELS.md`.
