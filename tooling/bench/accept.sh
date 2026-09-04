#!/bin/bash
# End-to-end acceptance check: does the organisers' eval.py run on a candidate, from a CLEAN clone?
#
#   ~/hackathon-shared/bench/accept.sh [--fast] [--tag NAME] [--submission] <repo-or-branch> <ckpt> [outdir]
#
#   <repo-or-branch>  a working clone (e.g. ~/rt-c) whose CURRENT branch is cloned fresh,
#                     or a bare branch name (e.g. wp-c, integration, submission) taken
#                     from the shared remote ~/hackathon-shared/repo.git
#   <ckpt>            checkpoint .pth to evaluate
#   [outdir]          where eval.py writes its plots (default: a temp dir)
#   --fast            stop eval.py once it has printed the area and written auc_vs_severity.json,
#                     skipping the trailing t-SNE plot (a very slow CPU job over the whole eval
#                     set). The area is identical either way -- the t-SNE does not feed it. Use
#                     --fast for the per-candidate official numbers, since this script holds the
#                     main GPU lock; omit it for the final submission check, where "the grader's
#                     script runs to completion" is the thing being verified.
#   --tag NAME        label for the runs/official_<tag>_<ts>.json record (default: branch_commit)
#   --expect-preproc NAME / --expect-encoder NAME
#                     assert which classes eval.py builds in the clone, and FAIL on a mismatch.
#                     PFPreProcessor and PFPreProcessorMeanPt share state_dict keys, so a
#                     checkpoint trained with one loads silently into the other and produces a
#                     plausible but wrong area. The classes are always reported; these assert them.
#   --submission      require the branch to ship EXACTLY ONE checkpoints/*.pth. Use it on the
#                     submission branch: the organisers' notebook does
#                     sorted(glob("checkpoints/*.pth"))[-1] and asserts if that is empty, so a
#                     branch shipping none is unusable to them. Without the flag, zero is fine
#                     (integration legitimately ships none) and only more than one is an error.
#
# Clones the branch into a throwaway directory and runs the notebook's exact eval.py
# invocation (10 severities, sev 0..1, grace 1000, full eval set) against it. Uses
# PYTHONPATH rather than `pip install -e` so the shared global editable install is
# never touched -- several instances share this Python environment.
#
# GPU memory: the preflight below requires ACCEPT_NEED_MIB (default 8000). That default is sized
# for the TRANSFORMER, and the requirement is architecture-dependent, not a property of eval.py:
#   attention encoder   O(batch x heads x N^2)  -- [1024, 8, 401, 401] is ~5 GB, and dominates
#   per-token MLP / set encoder  O(batch x N x embed)  -- two orders of magnitude smaller; under
#                       no_grad no activations are retained, so it peaks far lower still.
# Measured by WP-D for DeepSetsEncoder at eval.py's exact shape: 448 MiB allocated, 514 MiB
# reserved. So for a set encoder, pass ACCEPT_NEED_MIB=3000 (still deliberately generous) rather
# than either forcing the override blindly or assuming 8 GB is needed.
#
# GPU wrappers (one shared A10, two locks):
#   ~/hackathon-shared/gpu_run.sh    full training runs AND this script -- eval.py embeds at
#                                    batch 512+ over 400-candidate events and needs ~8 GB.
#   ~/hackathon-shared/gpu_small.sh  --test_mode smokes and bench_eval.py runs; a separate
#                                    lock so one small job may run beside the main training job.
# Run this script as:  ~/hackathon-shared/gpu_run.sh ~/hackathon-shared/bench/accept.sh <repo> <ckpt>
# It does NOT take a lock itself, so wrapping it never deadlocks.
#
# Prints the "area under AUC-vs-severity curve" line that eval.py emits, and writes it to
# runs/official_<tag>_<ts>.json. Since the organisers shipped src/embedding/degradation_eval.py
# and eval.py now imports THAT, this is the official-for-today number -- but note (a) it is
# explicitly TEMPORARY and stated to change before judging, so nobody tunes to it, and (b) its
# Bernoulli drop uses the global RNG with no seed, so it varies from run to run. It is NOT
# comparable to bench_eval.py's mean_area, which stays the development ruler.
set -euo pipefail

REMOTE="$HOME/hackathon-shared/repo.git"
DATA_ROOT="$HOME/hack-data/C9_robust_tagging"
# ACCEPT_EVAL_PT overrides the eval file for smoke-testing this script itself. Every real
# acceptance run must use the default -- the full eval set is the point of this check.
EVAL_PT="${ACCEPT_EVAL_PT:-$DATA_ROOT/eval/robust_tagging_eval_small.pt}"

SUBMISSION=0
FAST=0
TAG=""
EXPECT_PREPROC=""
EXPECT_ENCODER=""
while [ $# -gt 0 ]; do
    case "$1" in
        --submission) SUBMISSION=1; shift ;;
        --fast) FAST=1; shift ;;
        --tag) TAG="${2:-}"; [ -n "$TAG" ] || { echo "accept: --tag needs a value" >&2; exit 2; }; shift 2 ;;
        --expect-preproc) EXPECT_PREPROC="${2:-}"; shift 2 ;;
        --expect-encoder) EXPECT_ENCODER="${2:-}"; shift 2 ;;
        --) shift; break ;;
        -*) echo "accept: unknown option $1" >&2; exit 2 ;;
        *) break ;;
    esac
done

if [ $# -lt 2 ]; then
    sed -n '2,51p' "$0" >&2   # the whole header block above `set -euo pipefail`
    exit 2
fi

TARGET="$1"
CKPT="$(readlink -f "$2")"
OUTDIR="${3:-}"

[ -f "$CKPT" ] || { echo "accept: checkpoint not found: $CKPT" >&2; exit 1; }
[ -f "$EVAL_PT" ] || { echo "accept: eval data not found: $EVAL_PT" >&2; exit 1; }

if [ -d "$TARGET" ]; then
    SRC="$(readlink -f "$TARGET")"
    BRANCH="$(git -C "$SRC" rev-parse --abbrev-ref HEAD)"
    DIRTY=""
    git -C "$SRC" diff --quiet || DIRTY=" (WARNING: uncommitted changes in $SRC are NOT included)"
    echo "accept: cloning branch '$BRANCH' from $SRC$DIRTY"
else
    SRC="$REMOTE"
    BRANCH="$TARGET"
    echo "accept: cloning branch '$BRANCH' from $REMOTE"
fi

TMP="$(mktemp -d -t accept-XXXXXX)"
CLONE="$TMP/repo"
[ -n "$OUTDIR" ] || OUTDIR="$TMP/eval_plots"
trap 'rm -rf "$TMP"' EXIT

git clone --quiet --branch "$BRANCH" --single-branch "$SRC" "$CLONE"
COMMIT="$(git -C "$CLONE" rev-parse --short HEAD)"
echo "accept: fresh clone at $COMMIT"

for f in eval.py train.py configs/train_config.yaml configs/data_config_eval.yaml src/embedding/models.py; do
    [ -f "$CLONE/$f" ] || { echo "accept: FAIL -- $f missing from the clone" >&2; exit 1; }
done

# --- What eval.py will ACTUALLY build in this clone ------------------------------------
# PFPreProcessor and PFPreProcessorMeanPt have identical state_dict keys, so a MeanPt
# checkpoint loads into the stock preprocessor without error and yields a plausible but
# wrong area. Nothing downstream can detect that, so report what is really being built and
# let the caller assert it. Same for the encoder, which may be an alias.
IDENT="$(cd "$CLONE" && PYTHONPATH="$CLONE/src" python - <<'PYIDENT'
import importlib, yaml
cfg = yaml.safe_load(open("configs/train_config.yaml"))
name = (cfg.get("data") or {}).get("preproc_type", "PFPreProcessor")
pre = getattr(importlib.import_module("embedding.preprocs"), name)
from embedding.models import TransformerEncoder as Enc
print(f"{pre.__name__}|{Enc.__name__}|{name}")
PYIDENT
)" || { echo "accept: FAIL -- could not import the clone's preproc/encoder classes" >&2; exit 1; }
ACTUAL_PREPROC="${IDENT%%|*}"; REST="${IDENT#*|}"
ACTUAL_ENCODER="${REST%%|*}"; CFG_PREPROC="${REST#*|}"
echo "accept: eval.py will build preproc=$ACTUAL_PREPROC (preproc_type: $CFG_PREPROC), encoder=$ACTUAL_ENCODER"

for pair in "preproc:$EXPECT_PREPROC:$ACTUAL_PREPROC" "encoder:$EXPECT_ENCODER:$ACTUAL_ENCODER"; do
    what="${pair%%:*}"; rest="${pair#*:}"; want="${rest%%:*}"; got="${rest#*:}"
    [ -n "$want" ] || continue
    if [ "$want" != "$got" ]; then
        echo "accept: FAIL -- expected $what '$want' but this clone builds '$got'." >&2
        echo "accept:        A checkpoint trained with a different $what can load silently" >&2
        echo "accept:        (identical state_dict keys) and produce a plausible wrong number." >&2
        exit 1
    fi
    echo "accept: $what matches --expect-$what ($got)"
done
if [ "$SUBMISSION" -eq 1 ] && [ -z "$EXPECT_PREPROC$EXPECT_ENCODER" ]; then
    echo "accept: WARNING -- --submission without --expect-preproc/--expect-encoder; the classes"
    echo "accept:          above are reported but nothing is asserted."
fi

# --- Checkpoint-selection hazard -------------------------------------------------------
# The organisers' notebook picks its checkpoint as sorted(glob("checkpoints/*.pth"))[-1].
# '.' (0x2E) sorts before '_' (0x5F), so a sibling "<stem>_last.pth" sorts AFTER "<stem>.pth"
# and wins -- the notebook would silently grade the final-epoch checkpoint instead of the
# best-validation one. Report what that glob would actually select, and whether it is the
# checkpoint we are accepting here.
notebook_pick() {  # $1 = directory holding checkpoints
    { ls -1 "$1"/*.pth 2>/dev/null || true; } | LC_ALL=C sort | tail -1
}

# (a) What ships in the clone. Note the repo .gitignore lists `checkpoints/` and `*.pth`, so a
# fresh clone normally carries none; this fires only if someone force-adds one.
CLONE_PTH_COUNT=$({ ls -1 "$CLONE"/checkpoints/*.pth 2>/dev/null || true; } | wc -l)
if [ "$CLONE_PTH_COUNT" -eq 0 ] && [ "$SUBMISSION" -eq 1 ]; then
    echo "accept: FAIL -- --submission given, but the branch ships no checkpoints/*.pth." >&2
    echo "accept:        The organisers' notebook does sorted(glob('checkpoints/*.pth'))[-1]," >&2
    echo "accept:        gets None, and the next cell asserts. A submission branch must ship" >&2
    echo "accept:        exactly one checkpoint via a .gitignore exception for that file." >&2
    exit 1
elif [ "$CLONE_PTH_COUNT" -eq 0 ]; then
    echo "accept: clone ships no checkpoints/*.pth (expected -- .gitignore excludes them;"
    echo "accept:        pass --submission to require exactly one, for the submission branch)"
elif [ "$CLONE_PTH_COUNT" -eq 1 ]; then
    SHIPPED="$(notebook_pick "$CLONE/checkpoints")"
    echo "accept: clone ships exactly one checkpoint: $(basename "$SHIPPED")"
    # Verify the file we are about to SCORE is byte-identical to the one that SHIPS. Without this
    # the script can certify a submission while having evaluated a different checkpoint entirely.
    SHIPPED_SUM="$(sha256sum "$SHIPPED" | cut -d" " -f1)"
    SCORED_SUM="$(sha256sum "$CKPT" | cut -d" " -f1)"
    if [ "$SHIPPED_SUM" = "$SCORED_SUM" ]; then
        echo "accept: scored checkpoint is byte-identical to the shipped one (sha256 ${SHIPPED_SUM:0:12}...)"
    elif [ "$SUBMISSION" -eq 1 ]; then
        echo "accept: FAIL -- the checkpoint being scored is NOT the one the branch ships." >&2
        echo "accept:        ships:  $(basename "$SHIPPED")  sha256 ${SHIPPED_SUM:0:12}..." >&2
        echo "accept:        scored: $(basename "$CKPT")  sha256 ${SCORED_SUM:0:12}..." >&2
        echo "accept:        The organisers would run the shipped file, so this PASS would not" >&2
        echo "accept:        describe what they receive. Point <ckpt> at the shipped checkpoint." >&2
        exit 1
    else
        echo "accept: WARNING -- scored checkpoint differs from the one the branch ships"
        echo "accept:          ships ${SHIPPED_SUM:0:12}..., scoring ${SCORED_SUM:0:12}...; pass --submission to make this fatal"
    fi
else
    echo "accept: FAIL -- the branch ships $CLONE_PTH_COUNT checkpoints/*.pth files." >&2
    ls -1 "$CLONE"/checkpoints/*.pth 2>/dev/null | sed 's/^/accept:        /' >&2
    echo "accept:        The organisers' notebook takes sorted(glob(...))[-1] and would pick" >&2
    echo "accept:        $(basename "$(notebook_pick "$CLONE/checkpoints")")" >&2
    echo "accept:        A submission branch must ship exactly one .pth, or none." >&2
    exit 1
fi

# (a2) Checkpoints in SUBDIRECTORIES (checkpoints/smokes/, checkpoints/aux/) are harmless: the
# notebook globs "checkpoints/*.pth", which is not recursive, so they can never be auto-selected.
# They still ship with the branch, so list them and their total size -- the submitter should see
# what is being handed over even when it cannot be picked by mistake.
NESTED=$(find "$CLONE/checkpoints" -mindepth 2 -name '*.pth' 2>/dev/null | sort || true)
if [ -n "$NESTED" ]; then
    NESTED_N=$(printf '%s\n' "$NESTED" | wc -l)
    NESTED_SZ=$(du -ch $NESTED 2>/dev/null | tail -1 | cut -f1)
    echo "accept: note -- branch also ships $NESTED_N .pth in checkpoints/ subdirectories ($NESTED_SZ,"
    echo "accept:        outside the notebook's non-recursive checkpoints/*.pth glob, so not selectable):"
    printf '%s\n' "$NESTED" | sed "s|$CLONE/|accept:          |"
fi

# (b) The hazard that actually bites: the notebook runs in a WORKING directory, not a clone.
if [ -d "${SRC:-}/checkpoints" ]; then
    WORK_PICK="$(notebook_pick "$SRC/checkpoints")"
    WORK_COUNT=$({ ls -1 "$SRC"/checkpoints/*.pth 2>/dev/null || true; } | wc -l)
    if [ -n "$WORK_PICK" ] && [ "$(readlink -f "$WORK_PICK")" != "$CKPT" ]; then
        echo "accept: WARNING -- in the working tree $SRC, the notebook would auto-select"
        echo "accept:          $(basename "$WORK_PICK")"
        echo "accept:          but this run is accepting $(basename "$CKPT")"
        echo "accept:          ($WORK_COUNT .pth files present). Before submitting, leave exactly the"
        echo "accept:          intended checkpoint in checkpoints/ and move the rest aside."
    elif [ -n "$WORK_PICK" ]; then
        echo "accept: working-tree checkpoints/ would auto-select the same file we are accepting"
    fi
fi

mkdir -p "$OUTDIR"

# eval.py embeds at a hard-coded batch of 1024. With 400 candidates that is a
# [1024, 8, 401, 401] attention matrix, about 5 GB, and it OOMs if anything else is
# using the A10. Fail here with a readable message instead of 200 lines of traceback.
NEED_MIB=${ACCEPT_NEED_MIB:-8000}
if command -v nvidia-smi >/dev/null 2>&1; then
    # awk 'NR==1', not `head -1`. head exits after the first line and closes the pipe; with five
    # GPUs nvidia-smi is still writing, dies of SIGPIPE, and under `set -o pipefail` the command
    # substitution returns 141 and this script exits BEFORE eval.py runs -- reported as a bare
    # signal code with no message. It is a RACE, not deterministic: if all five lines fit the pipe
    # buffer before head exits, nothing happens, which is why it passed on one GPU and passes
    # intermittently on five. awk drains stdin to EOF, so the writer never sees a closed pipe.
    FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk 'NR==1')"
    if [ -n "$FREE_MIB" ] && [ "$FREE_MIB" -lt "$NEED_MIB" ]; then
        echo "accept: FAIL -- only ${FREE_MIB} MiB free on the GPU, eval.py needs about ${NEED_MIB} MiB." >&2
        echo "accept: run this through the full-run lock:  ~/hackathon-shared/gpu_run.sh $0 $*" >&2
        exit 1
    fi
fi

echo "accept: eval data $EVAL_PT"
if [ -n "${ACCEPT_EVAL_PT:-}" ]; then
    echo "accept: WARNING -- ACCEPT_EVAL_PT is set, so this is a SMOKE TEST of the script, not an"
    echo "accept:          acceptance run. The area it prints is not a result. Unset it for the real check."
fi
echo "accept: running the organisers' eval.py (this takes a while)..."

LOG="$TMP/eval.log"
: > "$LOG"
# Deliberately no `tail -f` streamer here: a background tailer sharing this script's stdout
# proved fragile when accept.sh is itself run under a wrapper or a pipeline. Follow progress
# with the path printed below; the whole log is echoed when the run finishes.
echo "accept: eval.py log -> $LOG  (tail -f it to follow progress)"

set +e
(
    cd "$CLONE"
    # PYTHONPATH, not pip install -e: the global editable install is shared by every instance.
    # `exec` so that $! below is the python process itself, not a wrapping subshell. That
    # assumption proved WRONG in practice on 2026-09-04: the poll loop's `kill -0 "$EVAL_PID"`
    # failed on its first iteration, so the loop exited immediately, --fast never fired and the
    # script parked in `wait` for the whole t-SNE. Evidence: wchan do_wait, only child python,
    # and six one-second samples with no `sleep` child. So record the pid from INSIDE the
    # subshell, where $$ is this shell and becomes the exec'd process, and trust that file.
    # $BASHPID, NOT $$: `$$` is the PARENT shell's pid and does not change inside a subshell,
    # so `echo $$` here recorded accept.sh's own pid and the liveness test stayed wrong.
    # $BASHPID is this subshell, which `exec` below turns into the python process.
    echo "$BASHPID" > "$TMP/eval.pid"
    exec env PYTHONUNBUFFERED=1 RT_NUM_THREADS="${RT_NUM_THREADS:-3}" \
        OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}" \
        PYTHONPATH="$CLONE/src" python eval.py \
        --train_cfg configs/train_config.yaml \
        --data_cfg configs/data_config_eval.yaml \
        --encoder "$CKPT" \
        --data "$EVAL_PT" \
        --outdir "$OUTDIR" \
        --sev_min 0.0 \
        --sev_max 1.0 \
        --num_severities 10 \
        --grace_period 1000
) >>"$LOG" 2>&1 &
EVAL_PID=$!

# eval.py prints the area (and writes auc_vs_severity.json) before the t-SNE plot, which is
# a very slow CPU job over the whole eval set. --fast stops once the area is on disk.
FAST_STOPPED=0
# Prefer the pid the subshell recorded; fall back to $! if the file is not there yet.
for _ in 1 2 3 4 5; do [ -s "$TMP/eval.pid" ] && break; sleep 1; done
if [ -s "$TMP/eval.pid" ]; then
    _p="$(cat "$TMP/eval.pid" 2>/dev/null || true)"
    if [ -n "$_p" ] && grep -qa "eval.py" "/proc/$_p/cmdline" 2>/dev/null; then EVAL_PID="$_p"; fi
fi
# ARTEFACT_DEADLINE: independent of any pid test. Once the JSON exists, --fast stops the run
# within ~2 s even if the liveness test above is wrong again -- a guard whose loop can exit for
# an unrelated reason is not a guard, which is exactly how the first version failed.
ARTEFACT_SEEN=0
while kill -0 "$EVAL_PID" 2>/dev/null || { [ "$FAST" -eq 1 ] && [ "$ARTEFACT_SEEN" -eq 0 ] \
        && [ -s "$OUTDIR/auc_vs_severity.json" ]; }; do
    # Poll for the artefact, NOT for a log line: eval.py's stdout is redirected and therefore
    # block-buffered, so the printed area does not appear until the process exits. The JSON is
    # written the moment the area is computed, which is the event we actually want.
    if [ "$FAST" -eq 1 ] && [ -s "$OUTDIR/auc_vs_severity.json" ]; then
        ARTEFACT_SEEN=1
        sleep 1                       # let the json finish flushing
        # SIGTERM alone is NOT enough. eval.py is inside sklearn's t-SNE, a C extension that does
        # not run Python signal handlers until it returns, so the process ignores SIGTERM for as
        # long as the t-SNE takes -- and the script then blocks in `wait` on a process it believes
        # it has stopped. Measured 2026-09-04: SIGTERM left it running, SIGKILL ended it at once.
        # Escalate, and confirm death rather than assuming it.
        kill "$EVAL_PID" 2>/dev/null
        for _ in 1 2 3 4 5; do kill -0 "$EVAL_PID" 2>/dev/null || break; sleep 1; done
        kill -9 "$EVAL_PID" 2>/dev/null || true
        pkill -9 -f "python eval.py .*--outdir $OUTDIR" 2>/dev/null || true
        FAST_STOPPED=1
        break
    fi
    sleep 2
done
wait "$EVAL_PID"; RC=$?
[ "$FAST_STOPPED" -eq 1 ] && RC=0    # we killed it deliberately, after the number was written
set -e
cat "$LOG"

echo
if [ "$RC" -ne 0 ]; then
    echo "accept: FAIL -- eval.py exited $RC (branch $BRANCH @ $COMMIT)"
    exit "$RC"
fi

AREA="$(grep -oP 'area under AUC-vs-severity curve: \K[0-9.]+' "$LOG" | tail -1 || true)"
if [ -z "$AREA" ] && [ -s "$OUTDIR/auc_vs_severity.json" ]; then
    # --fast stopped eval.py before it flushed stdout. Recompute the area from the artefact the
    # same way eval.py does (trapezoid over severities, normalised by the severity range).
    AREA="$(python - "$OUTDIR/auc_vs_severity.json" <<'PYAREA'
import json, sys
d = json.load(open(sys.argv[1]))
s, a = d["severities"], d["aucs"]
den = s[-1] - s[0]
print(f"{(sum((s[i+1]-s[i])*(a[i+1]+a[i])/2 for i in range(len(s)-1))/den if den > 0 else a[0]):.4f}")
PYAREA
)"
    echo "accept: area recovered from auc_vs_severity.json (eval.py was stopped before it flushed stdout)"
fi
if [ -z "$AREA" ]; then
    echo "accept: FAIL -- eval.py finished but printed no area line (branch $BRANCH @ $COMMIT)"
    exit 1
fi

# Copy the curve out before the temp dir is removed, unless the caller chose their own outdir.
if [ "$OUTDIR" = "$TMP/eval_plots" ]; then
    KEEP="$HOME/hackathon-shared/runs/accept_${BRANCH//\//_}_${COMMIT}"
    mkdir -p "$KEEP"
    cp "$OUTDIR"/*.png "$OUTDIR"/*.json "$KEEP"/ 2>/dev/null || true
    echo "accept: plots kept in $KEEP"
fi

# Machine-readable record so ablation_table.py can join the official area onto the bench row.
# Kept in a separate schema ("kind": "official") from bench_eval.py's JSONs -- this number comes
# from the organisers' degradation_eval.py, NOT from our five-family bench, and the two are not
# comparable to each other.
[ -n "$TAG" ] || TAG="${BRANCH//\//_}_${COMMIT}"
OFFICIAL_JSON="$HOME/hackathon-shared/runs/official_${TAG}_$(date +%Y%m%d_%H%M%S).json"
mkdir -p "$HOME/hackathon-shared/runs"
SMOKE=false; [ -n "${ACCEPT_EVAL_PT:-}" ] && SMOKE=true
python - "$OFFICIAL_JSON" "$TAG" "$BRANCH" "$COMMIT" "$CKPT" "$AREA" "$EVAL_PT" \
         "$FAST_STOPPED" "$SMOKE" "$OUTDIR/auc_vs_severity.json" \
         "$ACTUAL_PREPROC" "$ACTUAL_ENCODER" <<'PYEOF' || true
import json, sys, os, time
out, tag, branch, commit, ckpt, area, data, fast, smoke, curve = sys.argv[1:11]
preproc_cls, encoder_cls = sys.argv[11], sys.argv[12]
rec = {
    "kind": "official", "tag": tag, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "branch": branch, "commit": commit,
    "ckpt": ckpt, "official_area": float(area), "data": data,
    "stopped_before_tsne": fast == "1", "smoke_test": smoke == "true",
    "preproc_class": preproc_cls, "encoder_class": encoder_cls,
    "degradation": "embedding.degradation_eval (organisers', TEMPORARY - stated to change)",
    "note": "Bernoulli drop uses the global RNG, so this number varies run to run; "
            "not comparable to bench_eval.py mean_area.",
}
try:
    with open(curve) as f:
        rec.update(json.load(f))       # severities + aucs
except OSError:
    pass
with open(out, "w") as f:
    json.dump(rec, f, indent=2)
print(f"accept: wrote {out}")
PYEOF

STOPPED_NOTE=""
[ "$FAST_STOPPED" -eq 1 ] && STOPPED_NOTE="  (--fast: stopped before the t-SNE plot)"
echo "accept: PASS  branch=$BRANCH commit=$COMMIT ckpt=$(basename "$CKPT")  eval.py area=$AREA$STOPPED_NOTE"
