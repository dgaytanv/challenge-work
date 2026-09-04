#!/bin/bash
# WP-I: planner priority 04:58 -- I4 s22 and s33 before I1' s33. Sequential.
set -u
cd "$HOME/c2-i"
EVAL=$HOME/hack-data/C9_robust_tagging/eval/robust_tagging_eval_small.pt
OUT=$HOME/c2-i/checkpoints/phase1
bench_one () {
  local name=$1 cls=$2 desc=$3 seed=$4 tag
  tag="i-${desc}-${seed}"
  ls "$HOME/hackathon-shared/runs/${tag}_"*.json >/dev/null 2>&1 && { echo "[have] $tag"; return; }
  local cfg=configs/c2/${name}.yaml ckpt
  ckpt=$(grep -o "Saved best encoder to: .*\.pth" "$OUT/${name}.log" | tail -1 | sed "s/^Saved best encoder to: //")
  [ -f "$ckpt" ] || { echo "[skip] $tag no usable checkpoint"; return; }
  echo "[bench] $tag"
  ~/hackathon-shared/gpu_slot.sh --kind bench python ~/hackathon-shared/bench/bench_eval.py \
      --repo ~/c2-i --ckpt "$ckpt" --tag "$tag" --encoder_class "$cls" --train_cfg "$cfg" \
      --probe_repeats 5 --train_data robust_tagging_train_data_small.pt --data "$EVAL" \
      > "$OUT/${name}.bench.log" 2>&1
  if ls "$HOME/hackathon-shared/runs/${tag}_"*.json >/dev/null 2>&1; then
    echo "[bench-done] $tag $(grep -o 'SUMMARY.*' "$OUT/${name}.bench.log" | tail -1)"
  else echo "[bench-NORESULT] $tag no JSON"; fi
}
bench_one i4_seeds8_s22   PMAEncoder8      seeds8  s22
bench_one i4_seeds8_s33   PMAEncoder8      seeds8  s33
bench_one i1p_ff256l1_s33 PMAEncoderFF256  ff256l1 s33
bench_one i6_lat32_s22    PMAEncoder       lat32   s22
bench_one i6_lat32_s33    PMAEncoder       lat32   s33
echo "=== i4-priority wrapper finished ==="
