# champion-pma0-meanpt -- how to build and evaluate it

The graded submission. PMA readout with `num_layers=0` on the MeanPt preprocessor. This is the only model whose branch is a submission branch certified end to end.

## What builds it

| | |
|---|---|
| code branch | `submission-pma0-meanpt` |
| code commit | `41d45a9` |
| train config | `configs/train_config.yaml` (copied here as `train_config.yaml`) |
| checkpoint | `rt_d_pma0_aug_meanpt_encoder_20260903_221125.pth` |
| sha256 | `2f34628ccd788a80860e57d63e3cc050e9e5ba1ce0e395a9ff73f67f8fb3c53c` |
| encoder class | `PMAEncoder` |
| preprocessor class | `PFPreProcessorMeanPt` |
| encoder parameters | 89,606 |
| eval file | PF (`--eta_max 5.0`) |

## The two config keys that decide what gets built

- `data.preproc_type: PFPreProcessorMeanPt` -- **this one fails silently if wrong.** `PFPreProcessor` and
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
`~/hackathon-shared/runs/reproduce-champion_<timestamp>.json`, which is directly comparable to the
numbers for this model in `docs/table.md` and `models/MODELS.md`.
