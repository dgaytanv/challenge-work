# Submission hand-over

One page. Everything needed to grade the submission, and nothing else.

## What ships

| | |
|---|---|
| **Branch** | `submission-pma0-meanpt` on `~/hackathon-shared/repo.git` |
| **Commit (tip)** | `41d45a9f61c71b8e084e10084e64279726f388df` (short `41d45a9`) |
| **Checkpoint** | `checkpoints/rt_d_pma0_aug_meanpt_encoder_20260903_221125.pth` |
| **Checkpoint sha256** | `2f34628ccd788a80860e57d63e3cc050e9e5ba1ce0e395a9ff73f67f8fb3c53c` |
| **Config the grader reads** | `configs/train_config.yaml` |
| **Encoder** | `PMAEncoder(num_layers=0)`, aliased to `TransformerEncoder` |
| **Preprocessor** | `PFPreProcessorMeanPt` |
| **Encoder parameters** | 89,606 |

## How to run the grader's evaluation

```bash
git clone --branch submission-pma0-meanpt <repo> submission && cd submission
PYTHONPATH=$PWD/src python eval.py \
  --train_cfg configs/train_config.yaml \
  --data_cfg  configs/data_config_eval.yaml \
  --encoder   checkpoints/rt_d_pma0_aug_meanpt_encoder_20260903_221125.pth \
  --data      <eval .pt file> \
  --outdir    evalPlots
```

`eval.py` needs no flags beyond these and no changes. It imports `TransformerEncoder` from
`src/embedding/models.py` and constructs it from the fixed signature
`(num_features, embed_size, latent_dim, num_heads, num_layers, linear_dim, num_tokens, pairwise)`,
reading the dimensions out of `configs/train_config.yaml`.

## The three things that make it work, and how each was verified

1. **The alias.** `src/embedding/models.py` ends with
   `AttentionTransformerEncoder = TransformerEncoder` then `TransformerEncoder = PMAEncoder`.
   The grader gets the set encoder without knowing it exists; the original attention encoder stays
   reachable under its own name for ablations.
2. **Every option is a constructor DEFAULT.** `eval.py` passes no keyword arguments, so anything
   that is not in the fixed signature must default correctly. `num_layers` is the one architectural
   value it *does* pass, so the config says `0` **and** the class default is `0` — a missing key
   cannot silently build a different architecture, and a wrong value fails loudly at
   `load_state_dict` rather than scoring a different model.
3. **The preprocessor must match the checkpoint.** `PFPreProcessor` and `PFPreProcessorMeanPt` have
   **identical `state_dict` keys**, so a mismatched `preproc_type` loads cleanly and silently
   computes the wrong pt feature. The config names `PFPreProcessorMeanPt`, which is what the
   checkpoint was trained with.

Verified in a **fresh clone** of the tip (not a working tree), 2026-09-04 04:26:

```
TransformerEncoder -> PMAEncoder
preproc_type       -> PFPreProcessorMeanPt
num_layers: cfg 0  | class default 0
eval.py-style build + load: OK (PFPreProcessorMeanPt, PMAEncoder)
encoder params: 89,606
checkpoints/ ships exactly one .pth; sorted(glob(...))[-1] selects it
```

The organisers' notebook picks its checkpoint with `sorted(glob("checkpoints/*.pth"))[-1]` and
asserts if that is empty. This branch ships **exactly one** `.pth` via a `.gitignore` exception;
auxiliary checkpoints (`_bestauc`, `_last`) are excluded, because `_last.pth` sorts *after*
`.pth` and would otherwise be graded instead of the best-validation file.

## Measured performance

Sources are JSONs in `~/hackathon-shared/runs/`.

| metric | value | source |
|---|---|---|
| bench `mean_area` (5 families, 20k, R=5) | 0.8260 ± 0.0009 | `d-pma0-aug-meanpt_20260903_223150.json` |
| bench `mean_area` (R=20) | 0.8255 ± 0.0007 | `d-pma0-aug-meanpt-rep20_20260904_001942.json` |
| held-out (ellipse/annulus/diagonal) | 0.8021 ± 0.0006 | `d-pma0-aug-meanpt-heldout_20260903_223745.json` |
| official `eval.py` area (campaign 1) | 0.8790, 0.8786, 0.8780, 0.8795 | `official_d-pma0-aug-meanpt_*.json`, `official_submission-final_*.json` |
| official area, campaign-2 re-verification | **0.8794** | `runs/plots`-independent: `auc_vs_severity.json` from the 2026-09-04 05:16 fresh-clone run |
| stock anchor, for scale | 0.7734 | `anchor-stock-baseline_20260903_194212.json` |

**Read the bench number against the campaign-2 seed floor, not the probe floor.** Three seeds of
the champion configuration measure 0.8250 / 0.8296 / 0.8239 (seed std **0.0030**), so the
separability bar between two three-seed configurations is about **0.0074** — roughly five times the
per-refit probe sigma campaign 1 thresholded on. Several campaign-1 orderings sit below that bar
and are **unproven**, meaning not established by the available evidence rather than refuted.

## Training branch

This branch is deliberately **training-free**: it ships `models.py`, the config and the checkpoint,
which is all `eval.py` needs. The training code is on `wp-d` at commit `9801bdd` (verified present
on the shared remote), which carries the `encoder_class` / `encoder_kwargs` / `use_degradation`
hyperparameters this branch's config deliberately omits — rebuilding the encoder from `eval.py`'s
signature must not depend on them.

```bash
# from wp-d @ 9801bdd
python train.py \
  --data_cfg  configs/data_config_collide1m_small.yaml \
  --train_cfg configs/train_config_d_pma0_aug_meanpt.yaml \
  --data      ~/hack-data/C9_robust_tagging/train/robust_tagging_train_data_small.pt \
  --outdir    checkpoints
```

25 epochs, batch 256, WP-B's dead-region generator as augmentation.

## Campaign-2 re-verification (2026-09-04)

The tip `41d45a9` was certified in campaign 1 and **nothing on the branch has changed**, so that
certification stands. Today's run re-verifies it against the current data files and tooling:

- Fresh clone of `41d45a9`; `eval.py` builds `PFPreProcessorMeanPt` + `PMAEncoder`, both asserted
  with `--expect-preproc` / `--expect-encoder`: **PASS**.
- Clone ships exactly one `.pth`; the scored checkpoint is byte-identical to the shipped one
  (sha256 `2f34628ccd78…`): **PASS**.
- Organisers' `eval.py` over the full 70k eval set, 10 severities: **area 0.8794**, inside the
  campaign-1 range of 0.8780–0.8795 for this checkpoint.

**Re-verification completed on the fixed harness.** The first attempts produced the area but no
`official_*.json`, because of two defects in `accept.sh` (a SIGPIPE race in the free-memory check,
and `--fast` never terminating the run). Both are fixed; the closing run is recorded:

```
official_submission-cert-c2_20260904_054508.json
  branch submission-pma0-meanpt   commit 41d45a9
  official_area 0.8805            stopped_before_tsne true   smoke_test false
accept: PASS -- preproc PFPreProcessorMeanPt, encoder PMAEncoder, one checkpoint,
        scored checkpoint byte-identical to the shipped one (sha256 2f34628ccd78...)
```

Five independent measurements of this checkpoint's official area now exist: campaign 1's
0.8790 / 0.8786 / 0.8780 / 0.8795 and today's 0.8794 and 0.8805. The spread across all six is
0.0025, which is the organisers' unseeded Bernoulli plus an unseeded probe refit, and is the
reason a single official number is never quoted as if it were exact.
