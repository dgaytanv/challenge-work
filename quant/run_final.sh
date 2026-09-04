#!/bin/bash
# WP-G: measure the RECOMMENDED operating point directly (6-bit cap AND ReLU together),
# rather than inferring it from the 6-bit GELU row and the 8-bit ReLU row. Tonight's own
# lesson (RULING.md: the winner was an interaction) is that arms are not additive.
set -u
cd "$HOME/rt-g"
REF=/home/jovyan/rt-d/checkpoints/rt_d_pma0_aug_meanpt_l1t_encoder_20260903_235356.pth
TRAIN=$HOME/hack-data/C9_robust_tagging/train/robust_tagging_train_data_small_l1t.pt
Q=$HOME/hackathon-shared/quant
python quant/train_qat.py --ckpt $REF --train $TRAIN --eta_max 3.0 --n_tokens 200 \
  --events 10000 --batch 256 --tag gF-bits6-relu-l1t --quantized 1 --norm bn --act relu \
  --epochs 10 --lr 3e-4 --max_bits 6 --init_params $Q/gA-float-bn-l1t.params.npz || echo "FAILED 6bit-relu"
echo "=== 6-bit ReLU trained ==="
