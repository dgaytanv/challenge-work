#!/bin/bash
# WP-G G2, CORRECTED beta grid.
#
# The first grid (1e-6 .. 1e-3) was a no-op: every point gave bit-identical EBOPs. Cause:
# EBOPs ~5e8 and the distillation MSE ~0.1, so `add_loss(ebops * beta)` outweighed the task
# loss by ~5000x at beta=1e-6 already, and **Adam is scale-invariant** - multiplying the
# dominant term by 10 leaves the update direction unchanged. The regulariser must be scaled
# so the two terms are COMPARABLE: beta ~ MSE / EBOPs ~ 0.1 / 5e8 ~ 2e-10.
set -u
cd "$HOME/rt-g"
REF=/home/jovyan/rt-d/checkpoints/rt_d_pma0_aug_meanpt_l1t_encoder_20260903_235356.pth
TRAIN=$HOME/hack-data/C9_robust_tagging/train/robust_tagging_train_data_small_l1t.pt
Q=$HOME/hackathon-shared/quant
COMMON="--ckpt $REF --train $TRAIN --eta_max 3.0 --n_tokens 200 --events 10000 --batch 256"
for B in 0 2e-10 2e-9 2e-8; do
  echo "=== beta0=$B ==="
  python quant/train_qat.py $COMMON --tag gC-q-b$B-l1t --quantized 1 --norm bn --act gelu \
      --epochs 8 --lr 3e-4 --beta0 $B --init_params $Q/gA-float-bn-l1t.params.npz || echo "FAILED beta0=$B"
done
echo "=== corrected sweep complete ==="
