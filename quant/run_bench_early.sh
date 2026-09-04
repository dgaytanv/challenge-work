#!/bin/bash
set -u
cd "$HOME/rt-g"
REF=/home/jovyan/rt-d/checkpoints/rt_d_pma0_aug_meanpt_l1t_encoder_20260903_235356.pth
EVAL=$HOME/hack-data/C9_robust_tagging/eval/robust_tagging_eval_small_l1t.pt
TRAIN=$HOME/hack-data/C9_robust_tagging/train/robust_tagging_train_data_small_l1t.pt
Q=$HOME/hackathon-shared/quant
mkdir -p quant/ckpt
for TAG in gD-bits3-l1t gD-bits4-l1t; do
  [ -f "$Q/$TAG.params.npz" ] || { echo "skip $TAG"; continue; }
  echo "=== $TAG ==="
  python quant/export_quant.py --params $Q/$TAG.params.npz --ref $REF --out quant/ckpt/$TAG.pth \
      --act gelu --n_tokens 400 || { echo "EXPORT FAILED $TAG"; continue; }
  python ~/hackathon-shared/bench/bench_eval.py --repo ~/rt-g --ckpt $HOME/rt-g/quant/ckpt/$TAG.pth \
      --encoder_class QuantizedPMAEncoder --tag $TAG --data $EVAL --eta_max 3.0 \
      --train_data $TRAIN --probe_repeats 5 || echo "BENCH FAILED $TAG"
done
echo "=== early benches done ==="
