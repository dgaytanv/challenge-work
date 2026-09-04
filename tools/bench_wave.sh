#!/bin/bash
# WP-I: bench a wave of finished arms. Same launcher-stays-alive rule as run_wave.sh.
#
# --encoder_class is derived from the arm's own config, not hard-coded: benching an arm
# with the wrong class is the campaign-1 failure that produced a fully populated, plausible,
# near-chance row (register: 81 ReLU-trained tensors strict-loaded into a GELU graph).
# Here a mismatch would raise on load_state_dict, but deriving it removes the chance.
set -u
cd "$HOME/c2-i"
EVAL=$HOME/hack-data/C9_robust_tagging/eval/robust_tagging_eval_small.pt
OUT=$HOME/c2-i/checkpoints/phase1
pids=()
for cfg in "$@"; do
  name=$(basename "$cfg" .yaml)                      # e.g. i1p_ff256l1_s11
  desc=$(echo "$name" | cut -d_ -f2)                 # ff256l1
  seed=$(echo "$name" | cut -d_ -f3)                 # s11
  tag="i-${desc}-${seed}"
  cls=$(python - "$cfg" <<'PY'
import sys; sys.path.insert(0,'src')
from embedding.utils.cfg_handler import train_config
print(train_config(sys.argv[1]).hp('encoder_class','TransformerEncoder'))
PY
)
  # Select by uniqueness, not mtime: aux/ holds _bestauc and _last, and _last is written
  # last, so any mtime rule that ever sees aux/ benches the wrong model silently. This glob
  # is non-recursive so aux/ is excluded; the count check makes that checked, not lucky.
  n_ckpt=$(ls "$OUT/$name"/*.pth 2>/dev/null | wc -l)
  if [ "$n_ckpt" -eq 0 ]; then echo "[bench] SKIP $name: no checkpoint in $OUT/$name"; continue; fi
  if [ "$n_ckpt" -ne 1 ]; then echo "[bench] REFUSING $name: $n_ckpt primary checkpoints, cannot guess"; continue; fi
  ckpt=$(ls "$OUT/$name"/*.pth)
  echo "[bench] $tag  class=$cls  ckpt=$(basename "$ckpt")"
  ~/hackathon-shared/gpu_slot.sh --kind bench python ~/hackathon-shared/bench/bench_eval.py \
      --repo ~/c2-i --ckpt "$ckpt" --tag "$tag" --encoder_class "$cls" \
      --train_cfg "$cfg" --probe_repeats 5 \
      --train_data robust_tagging_train_data_small.pt \
      > "$OUT/${name}.bench.log" 2>&1 &
  pids+=($!)
  sleep 2
done
echo "[bench] ${#pids[@]} benches launched; waiting"
for p in "${pids[@]}"; do wait "$p" || true; done
for cfg in "$@"; do
  name=$(basename "$cfg" .yaml)
  [ -f "$OUT/${name}.bench.log" ] && { echo "--- $name ---"; grep -E "SUMMARY|Error|Traceback" "$OUT/${name}.bench.log" | tail -3; }
done
echo "[bench] wave done"
