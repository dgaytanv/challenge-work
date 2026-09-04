#!/bin/bash
# WP-I: bench every arm that has a COMPLETED run, at the fairness cap.
# Checkpoint is the one the arm's log names ("Saved best encoder to:"), never a glob:
# duplicate launches from a superseded driver left orphans in two outdirs tonight.
set -u
cd "$HOME/c2-i"
EVAL=$HOME/hack-data/C9_robust_tagging/eval/robust_tagging_eval_small.pt
OUT=$HOME/c2-i/checkpoints/phase1
CAP=${CAP:-4}
outstanding () {
  local q r
  q=$(~/hackathon-shared/gpu_slot.sh --queue 2>/dev/null | grep -c "c2-i" || true)
  r=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read -r p; do
        [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$HOME/c2-i" ] && echo x; done | wc -l)
  echo $((q + r))
}
pids=()
for cfg in "$@"; do
  name=$(basename "$cfg" .yaml); desc=$(echo "$name" | cut -d_ -f2); seed=$(echo "$name" | cut -d_ -f3)
  tag="i-${desc}-${seed}"
  if ls "$HOME/hackathon-shared/runs/${tag}_"*.json >/dev/null 2>&1; then echo "[have] $tag"; continue; fi
  grep -qE "\] released .*rc=0" "$OUT/${name}.log" 2>/dev/null || { echo "[skip] $tag no completed run"; continue; }
  ckpt=$(grep -o "Saved best encoder to: .*\.pth" "$OUT/${name}.log" | tail -1 | sed "s/^Saved best encoder to: //")
  [ -f "$ckpt" ] || { echo "[skip] $tag log names no usable checkpoint"; continue; }
  cls=$(PYTHONPATH=$HOME/c2-i/src python -c "
import sys; sys.path.insert(0,'src')
from embedding.utils.cfg_handler import train_config
print(train_config('$cfg').hp('encoder_class','TransformerEncoder'))")
  while [ "$(outstanding)" -ge "$CAP" ]; do sleep 10; done
  echo "[bench] $tag class=$cls ckpt=$(basename "$ckpt")"
  ( ~/hackathon-shared/gpu_slot.sh --kind bench python ~/hackathon-shared/bench/bench_eval.py \
      --repo ~/c2-i --ckpt "$ckpt" --tag "$tag" --encoder_class "$cls" --train_cfg "$cfg" \
      --probe_repeats 5 --train_data robust_tagging_train_data_small.pt --data "$EVAL" \
      > "$OUT/${name}.bench.log" 2>&1
    echo "[bench-done] $tag $(grep -o 'SUMMARY.*' "$OUT/${name}.bench.log" | tail -1)" ) &
  pids+=($!); sleep 3
done
for p in "${pids[@]}"; do wait "$p" || true; done
echo "=== all benches done ==="
