#!/bin/bash
# WP-I: I1' seeds 22 and 33 only, sequential. A SECOND wrapper alongside the main one, on
# the planner's explicit 04:54 instruction -- these are the only rows that can reach n=3
# before the 05:25 stop, and sequential-behind-the-others cannot get there in time. Two
# slots for ~20 minutes; the bench token cap was removed at 04:33 and slot 4 is reserved
# for benches, so this does not reintroduce the deadlock WP-H fixed.
set -u
cd "$HOME/c2-i"
EVAL=$HOME/hack-data/C9_robust_tagging/eval/robust_tagging_eval_small.pt
OUT=$HOME/c2-i/checkpoints/phase1
for name in i1p_ff256l1_s22 i1p_ff256l1_s33; do
  cfg=configs/c2/${name}.yaml
  seed=$(echo "$name" | cut -d_ -f3); tag="i-ff256l1-${seed}"
  ls "$HOME/hackathon-shared/runs/${tag}_"*.json >/dev/null 2>&1 && { echo "[have] $tag"; continue; }
  ckpt=$(grep -o "Saved best encoder to: .*\.pth" "$OUT/${name}.log" | tail -1 | sed "s/^Saved best encoder to: //")
  [ -f "$ckpt" ] || { echo "[skip] $tag no usable checkpoint"; continue; }
  echo "[bench] $tag ckpt=$(basename "$ckpt")"
  ~/hackathon-shared/gpu_slot.sh --kind bench python ~/hackathon-shared/bench/bench_eval.py \
      --repo ~/c2-i --ckpt "$ckpt" --tag "$tag" --encoder_class PMAEncoderFF256 --train_cfg "$cfg" \
      --probe_repeats 5 --train_data robust_tagging_train_data_small.pt --data "$EVAL" \
      > "$OUT/${name}.bench.log" 2>&1
  if ls "$HOME/hackathon-shared/runs/${tag}_"*.json >/dev/null 2>&1; then
    echo "[bench-done] $tag $(grep -o 'SUMMARY.*' "$OUT/${name}.bench.log" | tail -1)"
  else echo "[bench-NORESULT] $tag no JSON"; fi
done
echo "=== i1p seeds finished ==="
