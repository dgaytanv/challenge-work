#!/bin/bash
# WP-M: watch runs/ for a package's own dev-bench JSON and add the four suites WP-M owns
# (held-out, colleague, L1T, colleague-L1T) for each one, exactly once.
#
#   m_suite_watch.sh <tag-glob> [--once] [--dry-run]
#   m_suite_watch.sh 'h-ref-small-s*'
#
# Why it keys off the JSON rather than off a checkpoint glob: the JSON records the exact `ckpt`
# the owner benched, so the extra suites measure the SAME file the dev row did. Picking the
# checkpoint by newest-mtime instead is how campaign 1 nearly benched a superseded checkpoint
# under a fixed run's tag (register, and WP-H repeated the warning at 03:33).
#
# Skips its own outputs: only records whose families are the five SCORING families are treated as
# a dev row, so `-heldout` / `-colleague` / `-l1t` JSONs can never re-trigger the watcher.
set -u
SHARED=/home/jovyan/hackathon-shared
RUNS="$SHARED/runs"
BENCH="$SHARED/bench"
STATE="$SHARED/runs/.m_suite_done"

GLOB="${1:-h-ref-small-s*}"; shift || true
ONCE=0; DRY=""
for a in "$@"; do
  case "$a" in
    --once) ONCE=1 ;;
    --dry-run) DRY="--dry-run" ;;
  esac
done
touch "$STATE"

is_dev_row () {  # $1 = json path; true iff this is the PF dev row, not one of my own outputs
  python - "$1" <<'PY'
import json, os, sys
SCORING = {"rect", "wedge", "strip", "towers", "cells"}
PF_EVAL = "robust_tagging_eval_small.pt"
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
if not isinstance(d, dict):
    sys.exit(1)
# Families alone is NOT enough: the `-l1t` row this watcher itself writes has the same five
# scoring families, so keying on families would make the watcher re-trigger on its own output
# forever. The `data` field separates them, and it is recorded by bench_eval rather than parsed
# out of a tag. Same discipline as the register's "never group on one field".
if set(d.get("families", {})) != SCORING:
    sys.exit(1)                      # heldout / colleague row: not a dev row, nothing to say
if d.get("data") is None:
    sys.exit(2)                      # campaign-2 bench_eval always records `data`; its absence
                                     # means a stale bench_eval wrote this. Loud, not silent.
sys.exit(0 if os.path.basename(d["data"]) == PF_EVAL else 1)
PY
}

while :; do
  for j in $(ls -t "$RUNS"/${GLOB}_*.json 2>/dev/null); do
    base=$(basename "$j")
    grep -qxF "$base" "$STATE" && continue
    is_dev_row "$j"; dev_rc=$?
    if [ "$dev_rc" = "2" ]; then
      echo "[m-watch] $(date +%H:%M:%S) $base has five scoring families but NO \`data\` field --"
      echo "          written by a bench_eval predating campaign 2. Not treating it as a dev row;"
      echo "          re-bench it with the current bench_eval so the eval file is recorded."
      echo "$base" >> "$STATE"
      continue
    fi
    [ "$dev_rc" = "0" ] || continue
    read -r TAG CKPT REPO SEED TRAIN_DATA < <(python - "$j" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(d.get("tag"), d.get("ckpt"), d.get("repo"),
      d.get("seed") if d.get("seed") is not None else "-",
      d.get("train_data") or "-")
PY
)
    if [ ! -f "$CKPT" ]; then
      echo "[m-watch] $(date +%H:%M:%S) $TAG: checkpoint gone ($CKPT) -- NOT marking done, will retry"
      continue
    fi
    # The train config is not recorded in the bench JSON, so derive it from the tag's seed and
    # verify it exists; refuse rather than guess if it does not.
    CFG=""
    for cand in "$REPO/configs/c2/champion_s${SEED}.yaml" "$REPO/configs/c2/champion.yaml"; do
      [ -f "$cand" ] && { CFG="$cand"; break; }
    done
    if [ -z "$CFG" ]; then
      echo "[m-watch] $(date +%H:%M:%S) $TAG: no train config found under $REPO/configs/c2 -- skipping, NOT marked done"
      continue
    fi
    EXTRA=""
    [ "$SEED" != "-" ] && EXTRA="$EXTRA --seed $SEED"
    [ "$TRAIN_DATA" != "-" ] && EXTRA="$EXTRA --train_data $TRAIN_DATA"
    echo "[m-watch] $(date +%H:%M:%S) suites for $TAG  ckpt=$(basename "$CKPT")  cfg=$(basename "$CFG")"
    "$BENCH/suite_run.sh" "$REPO" "$CKPT" "$TAG" "$CFG" --skip-dev $EXTRA $DRY
    rc=$?
    if [ "$rc" = "0" ] && [ -z "$DRY" ]; then
      echo "$base" >> "$STATE"
      echo "[m-watch] $(date +%H:%M:%S) $TAG suites complete"
    else
      echo "[m-watch] $(date +%H:%M:%S) $TAG rc=$rc -- NOT marked done, will retry"
    fi
  done
  [ "$ONCE" = "1" ] && break
  sleep 30
done
