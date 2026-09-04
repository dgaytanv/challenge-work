#!/bin/bash
# Evaluate pma0-stock-preproc two ways, from a fresh clone of the code that builds it.
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

echo "== cloning submission-pma0 @ 99b7d2e"
git clone -q --branch submission-pma0 "$REPO" "$WORK/repo"
cd "$WORK/repo"
git checkout -q 99b7d2e

CKPT="$HERE/rt_d_pma0_aug_encoder_20260903_204550.pth"
sha256sum -c "$HERE/rt_d_pma0_aug_encoder_20260903_204550.pth.sha256" 2>/dev/null || true

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
    --encoder_class PMAEncoder \
    --train_cfg "$HERE/train_config.yaml" \
    --tag "reproduce-pma0-stock" \
    --data "$EVAL_DATA" \
    --eta_max 5.0 \
    --probe_repeats 5
