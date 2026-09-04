#!/bin/bash
# WP-G campaign 2, G6a: export each trained arm into the torch emulator and bench it on
# the real ruler. Campaign-2 standard: 3 seeds per configuration, L1T eval file,
# --eta_max 3.0, --train_data recorded in the JSON.
set -u
cd "$HOME/c2-g"
REF=/home/jovyan/rt-d/checkpoints/rt_d_pma0_aug_meanpt_l1t_encoder_20260903_235356.pth
EVAL=$HOME/hack-data/C9_robust_tagging/eval/robust_tagging_eval_small_l1t.pt
TRAIN=$HOME/hack-data/C9_robust_tagging/train/robust_tagging_train_data_small_l1t.pt
Q=$HOME/hackathon-shared/quant
mkdir -p quant/ckpt
for SEED in 11 22 33; do
  for ARM in hom het; do
    HOM=0; [ "$ARM" = hom ] && HOM=1
    TAG="g6a-$ARM-bits6-relu-l1t-s$SEED"
    [ -f "$Q/$TAG.params.npz" ] || { echo "skip $TAG (no params)"; continue; }
    echo "=== $TAG $(date -Is)"
    python quant/export_quant.py --params "$Q/$TAG.params.npz" --ref $REF \
        --out quant/ckpt/$TAG.pth --act relu --n_tokens 400 --homogeneous $HOM \
        || { echo "EXPORT FAILED $TAG"; continue; }
    python ~/hackathon-shared/bench/bench_eval.py --repo ~/c2-g \
        --ckpt $HOME/c2-g/quant/ckpt/$TAG.pth --encoder_class QuantizedPMAEncoder \
        --tag $TAG --data $EVAL --eta_max 3.0 --train_data $TRAIN --probe_repeats 5 \
        || echo "BENCH FAILED $TAG"
  done
done
echo "=== G6a benches complete $(date -Is)"
