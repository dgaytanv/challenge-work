#!/bin/bash
# WP-I: remaining Phase-1 benches, sequential, planner priority order (I4 first).
set -u
cd "$HOME/c2-i"
EVAL=$HOME/hack-data/C9_robust_tagging/eval/robust_tagging_eval_small.pt
OUT=$HOME/c2-i/checkpoints/phase1
b () {
  local name=$1 cls=$2 tag=$3 cfg=configs/c2/$1.yaml ckpt
  ls "$HOME/hackathon-shared/runs/${tag}_"*.json >/dev/null 2>&1 && { echo "[have] $tag"; return; }
  ckpt=$(grep -o "Saved best encoder to: .*\.pth" "$OUT/${name}.log" | tail -1 | sed "s/^Saved best encoder to: //")
  [ -f "$ckpt" ] || { echo "[skip] $tag"; return; }
  echo "[bench] $tag"
  ~/hackathon-shared/gpu_slot.sh --kind bench python ~/hackathon-shared/bench/bench_eval.py \
    --repo ~/c2-i --ckpt "$ckpt" --tag "$tag" --encoder_class "$cls" --train_cfg "$cfg" \
    --probe_repeats 5 --train_data robust_tagging_train_data_small.pt --data "$EVAL" \
    > "$OUT/${name}.bench.log" 2>&1
  ls "$HOME/hackathon-shared/runs/${tag}_"*.json >/dev/null 2>&1 \
    && echo "[bench-done] $tag $(grep -o 'SUMMARY.*' "$OUT/${name}.bench.log" | tail -1)" \
    || echo "[bench-NORESULT] $tag"
}
b i4_seeds8_s22   PMAEncoder8     i-seeds8-s22
b i4_seeds8_s33   PMAEncoder8     i-seeds8-s33
b i6_lat32_s22    PMAEncoder      i-lat32-s22
b i6_lat32_s33    PMAEncoder      i-lat32-s33
b i1p_ff256l1_s33 PMAEncoderFF256 i-ff256l1-s33
b i3_embed256_s33 PMAEncoder      i-embed256-s33
echo "=== rest finished ==="
