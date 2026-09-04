#!/bin/bash
# WP-G G2 stage B rerun: the three beta points, after fixing the bn_beta/beta collision.
set -u
cd "$HOME/rt-g"
REF=/home/jovyan/rt-d/checkpoints/rt_d_pma0_aug_meanpt_l1t_encoder_20260903_235356.pth
TRAIN=$HOME/hack-data/C9_robust_tagging/train/robust_tagging_train_data_small_l1t.pt
Q=$HOME/hackathon-shared/quant
COMMON="--ckpt $REF --train $TRAIN --eta_max 3.0 --n_tokens 200 --events 10000 --batch 256"
for B in 1e-6 1e-5 1e-4 1e-3; do
  echo "=== beta0=$B ==="
  python quant/train_qat.py $COMMON --tag gB-q-b$B-l1t --quantized 1 --norm bn --act gelu \
      --epochs 8 --lr 3e-4 --beta0 $B --init_params $Q/gA-float-bn-l1t.params.npz || echo "FAILED beta0=$B"
done
echo "=== stage B complete ==="
