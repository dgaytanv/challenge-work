#!/bin/bash
# WP-I CPU smoke: 1 epoch, --test_mode (10% of the small file), no GPU.
# Purpose is "does train.py run this arm end to end and write a loadable checkpoint",
# not accuracy. num_epochs is forced to 1 in a throwaway copy so the arm configs
# themselves stay exactly one change from the champion.
set -u
cd "$HOME/c2-i"
mkdir -p checkpoints/smokes /tmp/i_smoke
for cfg in "$@"; do
  base=$(basename "$cfg" .yaml)
  tmp=/tmp/i_smoke/${base}_1ep.yaml
  sed 's/^  num_epochs: .*/  num_epochs: 1/' "$cfg" > "$tmp"
  echo "=== SMOKE $base ==="
  CUDA_VISIBLE_DEVICES= RT_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=$HOME/c2-i/src \
    timeout 1800 python train.py \
      --data_cfg configs/data_config_collide1m_small.yaml \
      --train_cfg "$tmp" \
      --data $HOME/hack-data/C9_robust_tagging/train/robust_tagging_train_data_small.pt \
      --outdir checkpoints/smokes --test_mode 2>&1 | tail -6
  echo "--- exit ${PIPESTATUS[0]} ---"
done
echo "=== smokes done ==="
