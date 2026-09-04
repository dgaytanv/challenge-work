# How a Deep Sets encoder ships through the grader's `eval.py` (WP-D)

## The constraint

The grader does exactly two things with the encoder:

```python
TransformerEncoder(num_features, embed_size, latent_dim, num_heads, num_layers,
                   linear_dim, num_tokens=None, pairwise)
encoder(preproc(x), delta_r, cls_mask)        # cls_mask [B, N+1], CLS slot first
```

It passes **no keyword arguments** and reads no config key that selects an architecture. So a
different encoder can only ship by *being* `TransformerEncoder`, and every option it needs must be a
constructor **default**.

## The mechanism

`DeepSetsEncoder` was written from the start to that signature — it accepts and ignores
`num_heads`, `num_layers`, `linear_dim`, `num_tokens`, `pairwise` — and to that forward contract.
The switch is therefore two lines at the end of `models.py`:

```python
AttentionTransformerEncoder = TransformerEncoder   # original attention encoder, kept for ablations
TransformerEncoder = DeepSetsEncoder               # the submission switch
```

No change to `eval.py`. No change to the config schema. The attention encoder stays reachable under
its own name, so the ablation rows can still be reproduced from the submission branch.

`mask` handling is shared with the transformer's mask fix: dead rows are re-derived from the input
(`x.abs().sum(-1) == 0`) and OR'd into the incoming mask, because the dataloader builds its mask
before degradation is applied. Column 0 of the `[B, N+1]` mask is the CLS slot and is dropped.

## What must be a default, and why that is a hazard worth naming

`pooling` and `count_feature` are constructor defaults set to the values the shipped checkpoint was
trained with. If a pooling variant ever wins, **the default inside `DeepSetsEncoder` changes, not
the alias line** — a config key would be silently ignored by `eval.py` and the encoder would be
rebuilt with the wrong pooling, loading a state_dict of the wrong shape or, worse, the right shape
with different semantics. The same applies to `preproc_type`, which is a `data:` key read from
`train_config.yaml`: the submission config must name the preprocessor the checkpoint was trained
with, or the preproc `state_dict` loads into the wrong class.

## The shipped branch

The submission is `submission-pma0-meanpt` @ `3ac262f`: `TransformerEncoder = PMAEncoder`
(`num_layers=0`) with `preproc_type: PFPreProcessorMeanPt`, one checkpoint
(`rt_d_pma0_aug_meanpt_encoder_20260903_221125.pth`) shipped via a `.gitignore` exception.

`num_layers` is the one architectural setting `eval.py` *does* pass from the config, so on that
branch the config says `0` **and** the class default is `0`. A missing key therefore cannot
silently build a different architecture, and a wrong value fails loudly at `load_state_dict` —
verified, not assumed. Four branches were built this way during convergence (`submission-deepsets`,
`-deepsets-meanpt`, `-pma0`, `-pma0-meanpt`), one per candidate, because the alias and the
preprocessor must both match the checkpoint being scored.

## Verification

Verified in a **fresh clone**, not in a working tree:

- The organisers' `eval.py`, unmodified, end to end through the alias at full grader settings
  (70k events, 10 severities, grace 1000): **0.8790 and 0.8786**, against the organisers'
  no-degradation reference of 0.8201.
- `accept.sh` asserts the built classes at run time via `--expect-preproc PFPreProcessorMeanPt
  --expect-encoder PMAEncoder`, resolved by importing from the clone so the alias is caught.
- `tests/test_set_encoders.py` passes on the scaffold branch: deletion equivalence, permutation
  invariance, all-dead finiteness, N=200 and N=400.
- `checkpoints/` on the branch tracks no `.pth` (gitignored); the submission spec requires exactly
  one, added via a `.gitignore` exception at submission time.

## A failure mode this branch is deliberately guarded against

The organisers' notebook selects its checkpoint with `sorted(glob("checkpoints/*.pth"))[-1]`.
`.` (0x2E) sorts before `_` (0x5F), so an auxiliary `<stem>_last.pth` sitting beside `<stem>.pth`
sorts *after* it and wins — the notebook would grade the final-epoch checkpoint instead of the
best-validation one, silently and at judging time. Auxiliary checkpoints were moved to
`checkpoints/aux/` for this reason, and smoke-test checkpoints are written to a separate directory
so `checkpoints/` holds only real-run files.
