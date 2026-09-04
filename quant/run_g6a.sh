#!/bin/bash
# WP-G campaign 2, G6a: what does io_stream cost?
#
# hls4ml 1.3.0 refuses io_stream for any HGQ2 model with HETEROGENEOUS activation
# quantization. So the io_stream design is only reachable by forcing the data-lane
# quantizers to one (k, i, f) per tensor. That is a real constraint on the model, not a
# code-generation option, so it has to be re-trained under the constraint and re-measured.
#
# Campaign-2 measurement standard: THREE seeds per configuration. The campaign-1 operating
# point gF-bits6-relu-l1t exists at ONE seed only, and its +-0.0005 is the PROBE std, not a
# seed std, so it cannot enter a seed test. Both arms are therefore run at seeds 11/22/33:
#   g6a-hom-*  homogeneous activations (the io_stream-legal model)
#   g6a-het-*  heterogeneous activations (the campaign-1 configuration, re-run for a
#              like-for-like control; this is a REPRODUCTION of gF-bits6-relu-l1t, and the
#              spread between them is itself a check on the pipeline)
# Everything else is identical: 6-bit cap, ReLU, stage-A seed, L1T train file, eta_max 3.0.
set -u
cd "$HOME/c2-g"
REF=/home/jovyan/rt-d/checkpoints/rt_d_pma0_aug_meanpt_l1t_encoder_20260903_235356.pth
TRAIN=$HOME/hack-data/C9_robust_tagging/train/robust_tagging_train_data_small_l1t.pt
Q=$HOME/hackathon-shared/quant
COMMON="--ckpt $REF --train $TRAIN --eta_max 3.0 --n_tokens 200 --events 10000 --batch 256 \
        --quantized 1 --norm bn --act relu --epochs 10 --lr 3e-4 --max_bits 6 \
        --init_params $Q/gA-float-bn-l1t.params.npz"
for SEED in 11 22 33; do
  for ARM in hom het; do
    HOM=0; [ "$ARM" = hom ] && HOM=1
    TAG="g6a-$ARM-bits6-relu-l1t-s$SEED"
    if [ -f "$Q/$TAG.json" ]; then echo "=== $TAG already done, skipping"; continue; fi
    echo "=== $TAG (homogeneous=$HOM seed=$SEED) $(date -Is)"
    python quant/train_qat.py $COMMON --homogeneous $HOM --seed $SEED --tag "$TAG" \
      || echo "FAILED $TAG"
  done
done
echo "=== G6a training complete $(date -Is)"
