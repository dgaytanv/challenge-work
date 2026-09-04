#!/bin/bash
# WP-G G2: the accuracy-versus-bits curve.
# beta0 is not a usable axis here (EBOPs identical to 7 s.f. for beta0 = 0 .. 1e-3; see
# hgq_model.build_model docstring), so the sweep caps the learned bit widths directly.
set -u
cd "$HOME/rt-g"
REF=/home/jovyan/rt-d/checkpoints/rt_d_pma0_aug_meanpt_l1t_encoder_20260903_235356.pth
TRAIN=$HOME/hack-data/C9_robust_tagging/train/robust_tagging_train_data_small_l1t.pt
Q=$HOME/hackathon-shared/quant
COMMON="--ckpt $REF --train $TRAIN --eta_max 3.0 --n_tokens 200 --events 10000 --batch 256"
for B in 3 4 6 10; do
  echo "=== max_bits=$B ==="
  python quant/train_qat.py $COMMON --tag gD-bits$B-l1t --quantized 1 --norm bn --act gelu \
      --epochs 8 --lr 3e-4 --max_bits $B --init_params $Q/gA-float-bn-l1t.params.npz || echo "FAILED bits=$B"
done
# ReLU variant, now seeded from stage A like the rest so it is a controlled comparison
echo "=== relu (max_bits 8) ==="
python quant/train_qat.py $COMMON --tag gD-relu-l1t --quantized 1 --norm bn --act relu \
    --epochs 10 --lr 3e-4 --max_bits 8 --init_params $Q/gA-float-bn-l1t.params.npz || echo "FAILED relu"
echo "=== bits sweep complete ==="
