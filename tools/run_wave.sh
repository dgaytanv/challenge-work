#!/bin/bash
# WP-I: launch a wave of arm runs through the slot scheduler.
#
# All jobs are launched from THIS process and it waits for them, because a QUEUED
# gpu_slot.sh job dies with the process that launched it (WP-H, 03:34). A fire-and-forget
# loop would silently drop every job that did not get an immediate slot.
set -u
cd "$HOME/c2-i"
DATA=$HOME/hack-data/C9_robust_tagging/train/robust_tagging_train_data_small.pt
OUT=$HOME/c2-i/checkpoints/phase1
mkdir -p "$OUT"
pids=()
for cfg in "$@"; do
  tag=$(basename "$cfg" .yaml)
  echo "[wave] launching $tag"
  ~/hackathon-shared/gpu_slot.sh --kind train python train.py \
      --data_cfg configs/data_config_collide1m_small.yaml \
      --train_cfg "$cfg" --data "$DATA" --outdir "$OUT/$tag" \
      > "$OUT/${tag}.log" 2>&1 &
  pids+=($!)
  sleep 2
done
echo "[wave] ${#pids[@]} jobs launched; waiting"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "[wave] done, $fail failure(s)"
for cfg in "$@"; do
  tag=$(basename "$cfg" .yaml)
  echo "--- $tag ---"; tail -3 "$OUT/${tag}.log"
done
