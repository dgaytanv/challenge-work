#!/bin/bash
# Evaluate gF-bits6-relu-l1t two ways, from a fresh clone of the code that builds it.
#
#   (a) the organisers' eval.py  -- the number that would be graded
#   (b) our bench_eval.py        -- the dev-bench area, comparable to docs/table.md
#
# Usage:  ./evaluate.sh [<repo-url-or-path>]
# Default repo is the shared bare repo on this box.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${1:-$HOME/hackathon-shared/repo.git}"
EVAL_DATA="$HOME/hack-data/C9_robust_tagging/eval/robust_tagging_eval_small_l1t.pt"
WORK="$(mktemp -d)"
trap 'echo "workdir kept at $WORK"' EXIT

echo "== cloning wp-g @ e9939b0"
git clone -q --branch wp-g "$REPO" "$WORK/repo"
cd "$WORK/repo"
git checkout -q e9939b0

# This branch aliases TransformerEncoder = PMAEncoder, which is the WRONG class for this
# checkpoint. Repoint it. Without this eval.py raises a load_state_dict error (loudly, not silently).
printf '\nTransformerEncoder = QuantizedPMAEncoder\n' >> src/embedding/models.py
CKPT="$HERE/gF-bits6-relu-l1t.pth"
sha256sum -c "$HERE/gF-bits6-relu-l1t.pth.sha256" 2>/dev/null || true

echo "== (a) organisers' eval.py"
PYTHONPATH="$PWD/src" python eval.py \
    --train_cfg "$HERE/train_config.yaml" \
    --data_cfg  configs/data_config_eval.yaml \
    --encoder   "$CKPT" \
    --data      "$EVAL_DATA" \
    --outdir    "$WORK/evalPlots"

echo "== (b) our bench_eval.py (L1T file, eta_max 3.0)"
python "$HOME/hackathon-shared/bench/bench_eval.py" \
    --repo "$PWD" \
    --ckpt "$CKPT" \
    --encoder_class QuantizedPMAEncoder \
    --train_cfg "$HERE/train_config.yaml" \
    --tag "reproduce-gF" \
    --data "$EVAL_DATA" \
    --eta_max 3.0 \
    --probe_repeats 5
