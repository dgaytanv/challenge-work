#!/bin/bash
# WP-I benches, SEQUENTIAL inside one wrapper (WP-H, 04:33): holds at most one slot so
# ten packages can finish. s11 of every arm first -- those four rows decide the ranking.
# Checkpoint is the path the arm's own log names, never a glob.
set -u
cd "$HOME/c2-i"
EVAL=$HOME/hack-data/C9_robust_tagging/eval/robust_tagging_eval_small.pt
OUT=$HOME/c2-i/checkpoints/phase1
for name in i1p_ff256l1_s11 i3_embed256_s11 i4_seeds8_s11 i6_lat32_s11 \
            i1p_ff256l1_s22 i1p_ff256l1_s33 i4_seeds8_s22 i4_seeds8_s33 \
            i6_lat32_s22 i6_lat32_s33 i3_embed256_s33; do
  cfg=configs/c2/${name}.yaml
  desc=$(echo "$name" | cut -d_ -f2); seed=$(echo "$name" | cut -d_ -f3); tag="i-${desc}-${seed}"
  # A bench EXISTS only if its JSON is in runs/. rc is not a completion signal (WP-H): a
  # wrapper can report rc=0 having never run, from a drain or a kill before slot acquire.
  ls "$HOME/hackathon-shared/runs/${tag}_"*.json >/dev/null 2>&1 && { echo "[have] $tag"; continue; }
  grep -qE "\] released .*rc=0" "$OUT/${name}.log" 2>/dev/null || { echo "[skip] $tag no completed run"; continue; }
  ckpt=$(grep -o "Saved best encoder to: .*\.pth" "$OUT/${name}.log" | tail -1 | sed "s/^Saved best encoder to: //")
  [ -f "$ckpt" ] || { echo "[skip] $tag no usable checkpoint"; continue; }
  cls=$(PYTHONPATH=$HOME/c2-i/src python -c "
import sys; sys.path.insert(0,'src')
from embedding.utils.cfg_handler import train_config
print(train_config('$cfg').hp('encoder_class','TransformerEncoder'))")
  echo "[bench] $tag class=$cls"
  ~/hackathon-shared/gpu_slot.sh --kind bench python ~/hackathon-shared/bench/bench_eval.py \
      --repo ~/c2-i --ckpt "$ckpt" --tag "$tag" --encoder_class "$cls" --train_cfg "$cfg" \
      --probe_repeats 5 --train_data robust_tagging_train_data_small.pt --data "$EVAL" \
      > "$OUT/${name}.bench.log" 2>&1
  if ls "$HOME/hackathon-shared/runs/${tag}_"*.json >/dev/null 2>&1; then
    echo "[bench-done] $tag $(grep -o 'SUMMARY.*' "$OUT/${name}.bench.log" | tail -1)"
  else
    echo "[bench-NORESULT] $tag produced no JSON (drained or killed before acquire)"
  fi
done
echo "=== benches finished ==="
