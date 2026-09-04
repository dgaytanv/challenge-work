#!/bin/bash
# suite_run.sh -- run one checkpoint through every measurement suite, campaign-2 protocol.
#
#   suite_run.sh <repo> <ckpt> <tag> <cfg> [--finalist] [--seed N] [--train_data F]
#                [--encoder_class C] [--skip-l1t] [--skip-dev] [--dry-run]
#
# Suites, and why they are separate (writeup/E-bench-design.md, runs/plots/README.md):
#   dev        five scoring families        <tag>              the ruler we steer on
#   heldout    ellipse/annulus/diagonal     <tag>-heldout      generalisation, never trained on
#   colleague  Group 3's PF suite           <tag>-colleague    independent check, per family
#   l1t        five scoring on the L1T file <tag>-l1t          different acceptance, own block
#   l1t-coll   Group 3's suite on L1T       <tag>-l1t-colleague
# These are FOUR DIFFERENT MEASUREMENTS. Their mean_areas are not comparable to each other and must
# never be averaged or plotted on one axis. The tags differ so a consumer can tell them apart from
# the filename, and each JSON also records its own `data`, `eta_max` and `families_arg`.
#
# --finalist switches the dev bench to --probe_repeats 20 --full and additionally runs three
# official accept.sh --fast runs (the organisers' Bernoulli AND the probe are unseeded, so the
# spread over three runs is the measurement, not a nuisance).
#
# Everything goes through gpu_slot.sh --kind bench, which confines benches to slots 3-4.
set -u

SHARED=/home/jovyan/hackathon-shared
BENCH="$SHARED/bench"
SLOT="$SHARED/gpu_slot.sh"
DATA_ROOT=/home/jovyan/hack-data/C9_robust_tagging
EVAL_PF="$DATA_ROOT/eval/robust_tagging_eval_small.pt"
EVAL_L1T="$DATA_ROOT/eval/robust_tagging_eval_small_l1t.pt"

[ $# -ge 4 ] || { sed -n '2,26p' "$0" >&2; exit 2; }
REPO="$1"; CKPT="$2"; TAG="$3"; CFG="$4"; shift 4

FINALIST=0; SEED=""; TRAIN_DATA=""; ENC="PMAEncoder"; SKIP_L1T=0; SKIP_DEV=0; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --finalist) FINALIST=1; shift ;;
    --seed) SEED="$2"; shift 2 ;;
    --train_data) TRAIN_DATA="$2"; shift 2 ;;
    --encoder_class) ENC="$2"; shift 2 ;;
    --skip-l1t) SKIP_L1T=1; shift ;;
    --skip-dev) SKIP_DEV=1; shift ;;
    --dry-run) DRY=1; shift ;;
    *) echo "suite_run: unknown option $1" >&2; exit 2 ;;
  esac
done

# Fail before spending a slot, not after. A missing checkpoint used to surface as a three-second
# "rc=0" bench whose log held a FileNotFoundError -- register entry: a job that fails fast looks
# exactly like a job that succeeded fast.
for f in "$CKPT" "$CFG" "$SLOT" "$BENCH/bench_eval.py"; do
  [ -e "$f" ] || { echo "suite_run: FAIL -- missing: $f" >&2; exit 1; }
done
[ -d "$REPO" ] || { echo "suite_run: FAIL -- repo is not a directory: $REPO" >&2; exit 1; }

EXTRA=""
[ -n "$SEED" ]       && EXTRA="$EXTRA --seed $SEED"
[ -n "$TRAIN_DATA" ] && EXTRA="$EXTRA --train_data $TRAIN_DATA"

# The finalist protocol changes ONLY the dev bench. The other suites stay at R=5 on 20k events so
# they remain comparable with every screening row already in runs/.
DEV_EXTRA="--probe_repeats 5"
[ "$FINALIST" = "1" ] && DEV_EXTRA="--probe_repeats 20 --full"

run_bench () {  # subtag, families, data, eta_max, extra...
  local subtag="$1" fams="$2" data="$3" eta="$4"; shift 4
  local out_tag="$TAG$subtag"
  echo "[suite] $(date +%H:%M:%S) $out_tag  families=$fams eta_max=$eta $(basename "$data")"
  if [ "$DRY" = "1" ]; then
    echo "        DRY: $SLOT --kind bench python $BENCH/bench_eval.py --repo $REPO --ckpt $CKPT" \
         "--tag $out_tag --encoder_class $ENC --train_cfg $CFG --families $fams --data $data" \
         "--eta_max $eta $* $EXTRA"
    return 0
  fi
  "$SLOT" --kind bench python "$BENCH/bench_eval.py" \
      --repo "$REPO" --ckpt "$CKPT" --tag "$out_tag" --encoder_class "$ENC" \
      --train_cfg "$CFG" --families "$fams" --data "$data" --eta_max "$eta" \
      $* $EXTRA
  local rc=$?
  echo "[suite] $(date +%H:%M:%S) $out_tag rc=$rc"
  [ "$rc" = "0" ] || FAILED="$FAILED $out_tag"
  return $rc
}

FAILED=""
echo "[suite] $(date +%H:%M:%S) START tag=$TAG ckpt=$(basename "$CKPT") finalist=$FINALIST"

# eta_max 5.0 is the PF acceptance and reproduces every campaign-1 PF number; 3.0 is the L1T
# acceptance (runs/plots/README.md gives the per-family calibration that justifies it).
# --skip-dev: the owning package often benches its own dev row at R=5 as part of screening, so
# re-running it here would file a second JSON for the same measurement under the same tag. The
# other four suites are WP-M's to add.
if [ "$SKIP_DEV" = "0" ]; then
  run_bench ""              all       "$EVAL_PF"  5.0 $DEV_EXTRA
else
  echo "[suite] SKIP dev bench (--skip-dev): owner benches it"
fi
run_bench "-heldout"        heldout   "$EVAL_PF"  5.0 --probe_repeats 5
run_bench "-colleague"      colleague "$EVAL_PF"  5.0 --probe_repeats 5
if [ "$SKIP_L1T" = "0" ]; then
  if [ -f "$EVAL_L1T" ]; then
    run_bench "-l1t"           all       "$EVAL_L1T" 3.0 --probe_repeats 5
    run_bench "-l1t-colleague" colleague "$EVAL_L1T" 3.0 --probe_repeats 5
  else
    echo "[suite] SKIP l1t -- $EVAL_L1T not present"
  fi
fi

if [ "$FINALIST" = "1" ]; then
  cat <<'REMINDER'
[suite] ---------------------------------------------------------------------------------
[suite] CERTIFICATION STEP 1, BEFORE ANY OTHER FINALIST STEP (WP-N audit, planner ruling):
[suite]   Nothing in the campaign exercises the GRADER'S constructor path. integration-2 has no
[suite]   TransformerEncoder alias by design, and every bench passes --encoder_class explicitly,
[suite]   so a model can pass every bench here and still fail to build from eval.py's fixed
[suite]   signature -- which is the only path the organisers use.
[suite]   On the submission branch: alias TransformerEncoder to the promoted class
[suite]   (writeup/D-alias-mechanism.md), ship a config whose hyperparameters rebuild it from that
[suite]   signature alone, then run accept.sh --submission from a FRESH CLONE with
[suite]   --expect-preproc / --expect-encoder. Every option the grader cannot pass must be a
[suite]   constructor DEFAULT.
[suite] ---------------------------------------------------------------------------------
REMINDER
  # Officials run on a BRANCH, not a working tree: accept.sh clones it fresh. The branch must
  # alias TransformerEncoder to this encoder and name the matching preproc_type, or the run
  # measures a different model than the one benched above -- PFPreProcessor and
  # PFPreProcessorMeanPt have identical state_dict keys, so a mismatch loads silently.
  if [ -z "${SUITE_BRANCH:-}" ]; then
    echo "[suite] officials SKIPPED: set SUITE_BRANCH=<submission branch> (and optionally"
    echo "        SUITE_EXPECT_PREPROC / SUITE_EXPECT_ENCODER) to run them."
  else
    EXPECT=""
    [ -n "${SUITE_EXPECT_PREPROC:-}" ] && EXPECT="$EXPECT --expect-preproc $SUITE_EXPECT_PREPROC"
    [ -n "${SUITE_EXPECT_ENCODER:-}" ] && EXPECT="$EXPECT --expect-encoder $SUITE_EXPECT_ENCODER"
    for i in 1 2 3; do
      echo "[suite] $(date +%H:%M:%S) official $TAG run $i"
      if [ "$DRY" = "1" ]; then
        echo "        DRY: $SLOT --kind bench $BENCH/accept.sh --fast --tag $TAG $EXPECT $SUITE_BRANCH $CKPT"
        continue
      fi
      # PYTHONUNBUFFERED: accept.sh --fast polls eval.py's log for the area line, and CPython
      # block-buffers stdout to a file. Without it --fast never fires and the run grinds through
      # a full-eval-set t-SNE. Fixed upstream in accept.sh; exported here as belt and braces.
      PYTHONUNBUFFERED=1 "$SLOT" --kind bench "$BENCH/accept.sh" --fast --tag "$TAG" \
          $EXPECT "$SUITE_BRANCH" "$CKPT"
      echo "[suite] $(date +%H:%M:%S) official $TAG run $i rc=$?"
    done
  fi
fi

# Exit non-zero if ANY suite failed. Without this the script ended on a successful `echo` and
# returned 0 even when a suite had crashed, so a caller (my own watcher did exactly this) would
# record a PARTIAL suite as complete and never retry it. The held-out suite crashed on the first
# real run and the script still reported success -- the same "fails by looking like it worked"
# shape as the rest of the register.
if [ -n "$FAILED" ]; then
  echo "[suite] $(date +%H:%M:%S) FAILED tag=$TAG -- suites that did not complete:$FAILED"
  exit 1
fi
echo "[suite] $(date +%H:%M:%S) DONE tag=$TAG (all suites OK)"
