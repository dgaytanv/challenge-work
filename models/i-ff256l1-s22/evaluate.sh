#!/bin/bash
# Evaluate i-ff256l1-s22 two ways, from a fresh clone of the code that builds it.
#
#   (a) the organisers' eval.py  -- the number that would be graded
#   (b) our bench_eval.py        -- the dev-bench area, comparable to docs/table.md
#
# Usage:  ./evaluate.sh [<repo-url-or-path>]
# Default repo is the shared bare repo on this box.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${1:-$HOME/hackathon-shared/repo.git}"
EVAL_DATA="$HOME/hack-data/C9_robust_tagging/eval/robust_tagging_eval_small.pt"
WORK="$(mktemp -d)"
trap 'echo "workdir kept at $WORK"' EXIT

echo "== cloning c2-i @ e1c7aa8"
git clone -q --branch c2-i "$REPO" "$WORK/repo"
cd "$WORK/repo"
git checkout -q e1c7aa8

# This branch carries NO alias, so eval.py would build the stock attention encoder and fail
# at load_state_dict. Append the alias that makes eval.py build the right class.
printf '\nTransformerEncoder = PMAEncoderFF256\n' >> src/embedding/models.py
CKPT="$HERE/rt_i1p_ff256l1_s22_encoder_20260904_035107.pth"
sha256sum -c "$HERE/rt_i1p_ff256l1_s22_encoder_20260904_035107.pth.sha256" 2>/dev/null || true

echo "== (a) organisers' eval.py"
PYTHONPATH="$PWD/src" python eval.py \
    --train_cfg "$HERE/train_config.yaml" \
    --data_cfg  configs/data_config_eval.yaml \
    --encoder   "$CKPT" \
    --data      "$EVAL_DATA" \
    --outdir    "$WORK/evalPlots"

echo "== (b) our bench_eval.py (PF file, eta_max 5.0)"
python "$HOME/hackathon-shared/bench/bench_eval.py" \
    --repo "$PWD" \
    --ckpt "$CKPT" \
    --encoder_class PMAEncoderFF256 \
    --train_cfg "$HERE/train_config.yaml" \
    --tag "reproduce-i-ff256l1-s22" \
    --data "$EVAL_DATA" \
    --eta_max 5.0 \
    --probe_repeats 5
