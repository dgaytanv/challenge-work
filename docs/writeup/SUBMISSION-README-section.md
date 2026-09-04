## Our solution (Fast ML 2026, Challenge C9)

**Model.** `TransformerEncoder` in `src/embedding/models.py` is aliased to a permutation-invariant set encoder
(`PMAEncoder` with no attention body, per the 22:58 ruling, pending its official runs; alias line in models.py): per-candidate MLP, pooling by multihead attention with 4 learned seed queries over surviving candidates (masked), LayerNorm,
linear head to the 6-d latent. The attention encoder is kept as `AttentionTransformerEncoder` for the ablations. The mask is derived from the
input rows so zeroed candidates never enter the pooling.

**Preprocessing.** `preproc_type: PFPreProcessorMeanPt` (pending ruling): pt normalised by the mean surviving pt instead of the
sum, which removes ~95% of the severity-dependent feature drift and ~100% of the 200-vs-400 candidate offset in the feature
distribution that BatchNorm's frozen statistics see. Note: all preprocessor variants share state_dict keys, so the config's
`preproc_type` must match the checkpoint; `accept.sh --expect-preproc` asserts it.

**Training.** `train.py` with `configs/train_config.yaml` (25 epochs, batch 256, patience 5, mixed precision) on
`train/robust_tagging_train_data_small.pt`, with training-time detector degradation from `src/embedding/degradation.py`:
random dead eta-phi regions from five families (rectangles, phi-wedges, eta-strips, coarse cells, isolated towers) with
per-event severity up to 0.85 and a two-epoch ramp, plus exact phi rotation and eta reflection. <TWO-VIEW: FILL if used.>

**Checkpoint.** `checkpoints/<pma0-meanpt checkpoint, from D>` (the only .pth in the directory). Submission branch
`submission-pma0-meanpt` @ `<FILL>` (training-free; training branch `wp-d`, config `train_config_d_pma0_aug_meanpt.yaml`).
Runner-up branch `submission-deepsets-meanpt` @ `791c450`. Pending the 22:58 ruling. Reproduce: `PYTHONPATH=src python train.py --data_cfg configs/data_config_collide1m_small.yaml
--train_cfg configs/train_config.yaml --data <train file> --outdir checkpoints`.

**Numbers.** Our five-family development bench (20k eval events, 5 probe refits): mean AUC-vs-severity area <FILL> +- <FILL>
vs 0.7734 for the stock baseline. Organisers' `eval.py` with `degradation_eval.py` at full settings: area <FILL> (two runs:
<FILL>, <FILL>) vs the shipped 0.8201 reference. Held-out shapes (ellipse, annulus, diagonal band): <FILL>.

**Ablation (mean area on the development bench).** stock 0.7734 -> mask fix on frozen weights 0.7918 -> set encoder, clean
training 0.8036 -> + degradation augmentation 0.8183 -> <FILL further rows>. Full table: `writeup/` and `runs/table.md`.

**Considered and rejected.** Uniform-random candidate dropout as augmentation (too weak), Linformer/MLP-Mixer (fixed token
count breaks 200-vs-400), Lipschitz/Informer architectures (wrong threat model), a dead-fraction input token (probe-incompatible),
raw cosine consistency (rewards a latent offset), tuning to the organisers' 10x10 grid.
