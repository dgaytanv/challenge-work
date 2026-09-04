#!/bin/bash
# WP-I: run the remaining Phase-1 work while never exceeding the planner's per-package cap
# of 4 OUTSTANDING jobs (queued + running), ruling 04:00.
#
# The cap is measured from the system, not tracked internally, because jobs launched by an
# earlier driver of mine are still running and also count against it. Undercounting would
# put me back over the cap; overcounting only makes me slower.
set -u
cd "$HOME/c2-i"
CAP=${CAP:-4}
DATA=$HOME/hack-data/C9_robust_tagging/train/robust_tagging_train_data_small.pt
EVAL=$HOME/hack-data/C9_robust_tagging/eval/robust_tagging_eval_small.pt
OUT=$HOME/c2-i/checkpoints/phase1

outstanding () {
  local q r
  q=$(~/hackathon-shared/gpu_slot.sh --queue 2>/dev/null | grep -c "c2-i" || true)
  r=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read -r p; do
        [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$HOME/c2-i" ] && echo x; done | wc -l)
  echo $((q + r))
}

wait_for_slot () {
  while [ "$(outstanding)" -ge "$CAP" ]; do sleep 15; done
}

# A run is COMPLETE when gpu_slot.sh has written its "released" line into the arm's own
# log. The presence of a .pth is NOT completion: train.py writes a best-so-far checkpoint
# after every improving epoch, so a run three epochs in already has one. Benching on that
# test would silently report a partially-trained model as the arm's result -- the same
# shape as campaign 1's stale-checkpoint pick.
is_complete () { grep -q "] released" "$OUT/$1.log" 2>/dev/null; }

launch_train () {   # $1 = config
  local cfg=$1 name; name=$(basename "$cfg" .yaml)
  if is_complete "$name"; then echo "[skip-train] $name already finished"; return; fi
  if [ -n "$(ls "$OUT/$name"/*.pth 2>/dev/null)" ]; then
    echo "[wait-train] $name mid-flight (checkpoint exists, not released); waiting"
    local w=0
    while ! is_complete "$name" && [ $w -lt 240 ]; do sleep 15; w=$((w+1)); done
    is_complete "$name" && echo "[wait-train] $name finished" || echo "[wait-train] $name TIMED OUT"
    return
  fi
  wait_for_slot
  echo "[train] $name"
  ~/hackathon-shared/gpu_slot.sh --kind train python train.py \
      --data_cfg configs/data_config_collide1m_small.yaml --train_cfg "$cfg" \
      --data "$DATA" --outdir "$OUT/$name" > "$OUT/${name}.log" 2>&1
  echo "[train-done] $name rc=$?"
}

launch_bench () {   # $1 = config
  local cfg=$1 name desc seed tag cls n ckpt
  name=$(basename "$cfg" .yaml); desc=$(echo "$name" | cut -d_ -f2); seed=$(echo "$name" | cut -d_ -f3)
  tag="i-${desc}-${seed}"
  if ! is_complete "$name"; then echo "[bench] SKIP $tag: training not finished"; return; fi
  n=$(ls "$OUT/$name"/*.pth 2>/dev/null | wc -l)
  [ "$n" -eq 1 ] || { echo "[bench] REFUSING $name: $n primary checkpoints"; return; }
  ckpt=$(ls "$OUT/$name"/*.pth)
  cls=$(python - "$cfg" <<'PY'
import sys; sys.path.insert(0,'src')
from embedding.utils.cfg_handler import train_config
print(train_config(sys.argv[1]).hp('encoder_class','TransformerEncoder'))
PY
)
  wait_for_slot
  echo "[bench] $tag class=$cls"
  ~/hackathon-shared/gpu_slot.sh --kind bench python ~/hackathon-shared/bench/bench_eval.py \
      --repo ~/c2-i --ckpt "$ckpt" --tag "$tag" --encoder_class "$cls" --train_cfg "$cfg" \
      --probe_repeats 5 --train_data robust_tagging_train_data_small.pt \
      --data "$EVAL" > "$OUT/${name}.bench.log" 2>&1
  echo "[bench-done] $tag rc=$? : $(grep -o 'SUMMARY.*' "$OUT/${name}.bench.log" | tail -1)"
}

# train what is missing, then bench everything, each arm-seed pair moving together
for cfg in "$@"; do launch_train "$cfg" & sleep 3; done
wait
echo "=== all training done ==="
for cfg in "$@"; do launch_bench "$cfg" & sleep 3; done
wait
echo "=== all benches done ==="
grep -h -o "SUMMARY.*" "$OUT"/*.bench.log 2>/dev/null | sort
