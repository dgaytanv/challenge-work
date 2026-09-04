#!/bin/bash
# WP-G G2: the whole L1T sweep under ONE gpu_small lock acquisition, run sequentially.
# Re-queueing five times on a contended box wastes more wall clock than the runs take.
set -u
cd "$HOME/rt-g"
REF=/home/jovyan/rt-d/checkpoints/rt_d_pma0_aug_meanpt_l1t_encoder_20260903_235356.pth
TRAIN=$HOME/hack-data/C9_robust_tagging/train/robust_tagging_train_data_small_l1t.pt
Q=$HOME/hackathon-shared/quant
COMMON="--ckpt $REF --train $TRAIN --eta_max 3.0 --n_tokens 200 --events 10000 --batch 256"

run () { echo "=== $* ==="; python quant/train_qat.py $COMMON "$@" || echo "FAILED: $*"; }

# Stage A: float BatchNorm student distilled from the float LayerNorm teacher.
# Isolates the cost of the LayerNorm -> BatchNorm swap with no quantization anywhere.
run --tag gA-float-bn-l1t   --quantized 0 --norm bn --act gelu --epochs 12 --lr 1e-3

# Stage B: HGQ2 quantized, seeded from stage A. Isolates the cost of quantization.
# beta0 is the EBOPs regulariser strength: three points trace accuracy vs resources.
for B in 1e-6 1e-5 1e-4; do
  run --tag gB-q-b$B-l1t --quantized 1 --norm bn --act gelu --epochs 8 --lr 3e-4 \
      --beta0 $B --init_params $Q/gA-float-bn-l1t.params.npz
done

# Hardware-friendlier variant: GELU -> ReLU (the LayerNorm -> BatchNorm half is already
# in every row above, because HGQ2 has no quantized LayerNorm at all).
run --tag gB-q-b1e-5-relu-l1t --quantized 1 --norm bn --act relu --epochs 10 --lr 5e-4 --beta0 1e-5
echo "=== sweep complete ==="
