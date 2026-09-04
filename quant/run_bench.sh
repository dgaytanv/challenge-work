#!/bin/bash
# WP-G G3: export each trained variant to the torch emulator's format and bench it on the
# real ruler. L1T eval file, eta_max 3.0, --train_data recorded, five scoring families, R=5.
set -u
cd "$HOME/rt-g"
REF=/home/jovyan/rt-d/checkpoints/rt_d_pma0_aug_meanpt_l1t_encoder_20260903_235356.pth
EVAL=$HOME/hack-data/C9_robust_tagging/eval/robust_tagging_eval_small_l1t.pt
TRAIN=$HOME/hack-data/C9_robust_tagging/train/robust_tagging_train_data_small_l1t.pt
Q=$HOME/hackathon-shared/quant
mkdir -p quant/ckpt

bench_one () {
  TAG=$1; ACT=$2
  echo "=== $TAG (act=$ACT) ==="
  python quant/export_quant.py --params $Q/$TAG.params.npz --ref $REF \
      --out quant/ckpt/$TAG.pth --act $ACT --n_tokens 400 || { echo "EXPORT FAILED $TAG"; return; }
  python ~/hackathon-shared/bench/bench_eval.py --repo ~/rt-g --ckpt $HOME/rt-g/quant/ckpt/$TAG.pth \
      --encoder_class QuantizedPMAEncoder --tag $TAG --data $EVAL --eta_max 3.0 \
      --train_data $TRAIN --probe_repeats 5 || echo "BENCH FAILED $TAG"
}

for B in 1e-6 1e-5 1e-4 1e-3; do
  [ -f "$Q/gB-q-b$B-l1t.params.npz" ] && bench_one gB-q-b$B-l1t gelu
done
[ -f "$Q/gB-q-b1e-5-relu-l1t.params.npz" ] && bench_one gB-q-b1e-5-relu-l1t relu
echo "=== benches complete ==="
