#!/bin/bash
# Slot scheduler for the 5 A10s (campaign 2).
#
#   gpu_slot.sh [--slots N] [--kind train|bench] [--abandon-if-launcher-dies] cmd...
#   gpu_slot.sh --queue              list jobs currently waiting for a slot
#   gpu_slot.sh --drain <pid|all>    stop waiting job(s) by pid
#
# Takes the first free slot with a non-blocking flock, sets CUDA_VISIBLE_DEVICES, runs cmd.
#
# LAUNCHER POLICY (changed 04:00 by planner ruling after WP-L lost four runs):
#   Default: a QUEUED job KEEPS WAITING when the process that launched it exits. Detached
#   launches (`setsid nohup ... &` from a shell that returns) are the campaign norm, and the old
#   default silently reaped those jobs whenever all five slots were busy -- invisible unless you
#   read gpu.log.
#   --abandon-if-launcher-dies restores the campaign-1 behaviour for queue scripts that want a
#   killed queue to take its pending jobs with it. Use --drain to clean up by hand instead.
# An ACQUIRED job is never killed by this wrapper, under either policy.
set -u
# GPU_SLOT_LOCKDIR/LOG overridable so the scheduler can be tested in isolation
LOCKDIR=${GPU_SLOT_LOCKDIR:-/home/jovyan/hackathon-shared/locks}
QDIR=$LOCKDIR/queue
RUNDIR=$LOCKDIR/running
LOG=${GPU_SLOT_LOG:-/home/jovyan/hackathon-shared/gpu.log}
mkdir -p "$LOCKDIR" "$QDIR" "$RUNDIR"

if [ "${1:-}" = "--queue" ]; then
    shopt -s nullglob; found=0
    printf "%-8s %-9s %-22s %s\n" PID QUEUED CWD COMMAND
    for f in "$QDIR"/*.job; do
        pid=$(basename "$f" .job)
        if ! kill -0 "$pid" 2>/dev/null; then rm -f "$f"; continue; fi
        found=1; . "$f"
        printf "%-8s %-9s %-22s %s\n" "$pid" "$JQ_TIME" "$(basename "$JQ_CWD")" "${JQ_CMD:0:70}"
    done
    [ $found -eq 0 ] && echo "(nothing queued)"
    echo
    printf "running slots per package:\n"
    shopt -s nullglob
    summary=$(for r in "$RUNDIR"/*.run; do
        rp=$(basename "$r" .run)
        kill -0 "$rp" 2>/dev/null || { rm -f "$r"; continue; }
        ( . "$r"; echo "$JR_PKG" )
    done | sort | uniq -c)
    if [ -n "$summary" ]; then echo "$summary" | sed 's/^/  /'; else echo "  (none)"; fi
    exit 0
fi
if [ "${1:-}" = "--drain" ]; then
    target="${2:-}"; [ -n "$target" ] || { echo "usage: gpu_slot.sh --drain <pid|all>" >&2; exit 2; }
    shopt -s nullglob; n=0
    for f in "$QDIR"/*.job; do
        pid=$(basename "$f" .job)
        if ! kill -0 "$pid" 2>/dev/null; then rm -f "$f"; continue; fi
        if [ "$target" = "all" ] || [ "$target" = "$pid" ]; then
            . "$f"
            echo "[slot -] drained queued pid $pid ($JQ_CWD): ${JQ_CMD:0:60}" | tee -a "$LOG"
            kill -TERM "$pid" 2>/dev/null; rm -f "$f"; n=$((n+1))
        fi
    done
    echo "drained $n queued job(s)"; exit 0
fi

NSLOTS=1; KIND=train; ABANDON=0; PRIORITY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --slots) NSLOTS="$2"; shift 2 ;;
    --kind)  KIND="$2";  shift 2 ;;
    --abandon-if-launcher-dies) ABANDON=1; shift ;;
    --priority) PRIORITY=1; shift ;;
    --) shift; break ;;
    *) break ;;
  esac
done
[ $# -gt 0 ] || { echo "usage: gpu_slot.sh [--slots N] [--kind train|bench] [--priority] [--abandon-if-launcher-dies] cmd..." >&2; exit 2; }
case "$KIND" in
  train) ALLOWED=(0 1 2 3 4) ;;
  bench) ALLOWED=(0 1 2 3 4) ;;  # 04:26 ruling: benches may use ANY free slot
  *) echo "gpu_slot.sh: --kind must be train or bench" >&2; exit 2 ;;
esac
MAX_BENCH=3   # enforced by benchtok{0..2}.lock, not by counting (counting races)
if [ "$NSLOTS" -gt "${#ALLOWED[@]}" ]; then
  echo "gpu_slot.sh: --slots $NSLOTS exceeds the $KIND allowance (${#ALLOWED[@]})" >&2; exit 2
fi

export RT_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
[ -d "$PWD/src/embedding" ] && export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

QFILE="$QDIR/$$.job"
# Removes the QUEUE entry only. The bench token must be held for the whole life of the job --
# cleanup_q is called immediately after a slot is acquired, so releasing the token here made the
# 3-bench cap apply only while queued, which is exactly not the point. Fds close on exit anyway.
cleanup_q() { rm -f "$QFILE"; return 0; }
# A drained job must NOT look like a completed one. Without this, --drain's SIGTERM ends up
# reported as rc=0 by the documented `gpu_slot.sh ...; echo rc=$?` pattern, so a bench that never
# ran reads as a bench that finished -- a missing row that looks present. Exit 75 while QUEUED.
# (After a slot is acquired these signals are ignored instead; a running job is never killed here.)
trap 'cleanup_q; exit 75' TERM INT HUP
trap cleanup_q EXIT

BENCH_FD=""
take_bench_token() {   # atomic: hold one of MAX_BENCH tokens for as long as this job runs
  local t f
  for t in $(seq 0 $((MAX_BENCH-1))); do
    exec {f}>"$LOCKDIR/benchtok$t.lock" || continue
    if flock -n "$f"; then BENCH_FD="$f"; return 0; fi
    exec {f}>&-
  done
  return 1
}
drop_bench_token() { [ -n "$BENCH_FD" ] && { exec {BENCH_FD}>&-; BENCH_FD=""; }; }

FDS=(); IDX=()
try_acquire() {
  FDS=(); IDX=()
  local i f
  for i in "${ALLOWED[@]}"; do
    exec {f}>"$LOCKDIR/gpu$i.lock" || continue
    if flock -n "$f"; then FDS+=("$f"); IDX+=("$i"); else exec {f}>&-; fi
    [ "${#IDX[@]}" -eq "$NSLOTS" ] && return 0
  done
  for f in "${FDS[@]}"; do exec {f}>&-; done   # release partial grabs, never deadlock
  FDS=(); IDX=()
  return 1
}

# Fair share groups by PACKAGE. cwd basename is the natural key, but a job launched from the
# shared bench directory would take pkg=bench, collapsing every package's benches into ONE
# fair-share bucket and defeating the whole mechanism. If cwd is not a package dir, fall back to
# the --repo argument in the command, which bench_eval always carries.
PKG=$(basename "$PWD")
if [[ "$PKG" != c2-* ]]; then
  for ((_i=1; _i<=$#; _i++)); do
    if [ "${!_i}" = "--repo" ]; then _j=$((_i+1)); _r="${!_j:-}"; [ -n "$_r" ] && PKG=$(basename "$_r"); break; fi
  done
fi

# --priority is honoured ahead of fair share and is restricted to the packages the planner named.
PRIO_PKGS=" c2-h c2-m "
GRANTS="$LOCKDIR/priority_grants"     # written only by the planner: one cwd or pid per line
prio_granted() {
  [ -r "$GRANTS" ] || return 1
  local line
  while IFS= read -r line; do
    line="${line%%#*}"; line="$(echo "$line" | tr -d '[:space:]')"
    [ -z "$line" ] && continue
    [ "$line" = "$PWD" ] && return 0
    [ "$line" = "$(basename "$PWD")" ] && return 0
    [ "$line" = "$$" ] && return 0
  done < "$GRANTS"
  return 1
}
if [ "$PRIORITY" -eq 1 ] && [[ "$PRIO_PKGS" != *" $PKG "* ]]; then
  if prio_granted; then
    echo "[slot -] --priority granted to pkg=$PKG cwd=$PWD by planner grant file" | tee -a "$LOG"
  else
    echo "[slot -] --priority requested by pkg=$PKG which is not authorised (allow-list: c2-h c2-m; no grant in $GRANTS); ignoring" | tee -a "$LOG"
    PRIORITY=0
  fi
fi

# --- fair share -------------------------------------------------------------
# When a slot frees, the waiting job whose PACKAGE holds the fewest running slots goes first,
# ties broken by queue time. Each waiter computes this independently from the shared registries;
# flock -n still arbitrates, so a race costs ordering, never correctness.
running_for() {   # $1 = package -> count of slots that package currently holds
  local n=0 r rp
  shopt -s nullglob
  for r in "$RUNDIR"/*.run; do
    rp=$(basename "$r" .run)
    if ! kill -0 "$rp" 2>/dev/null; then rm -f "$r" 2>/dev/null; continue; fi
    # A holder can release (and delete its .run) between the glob and this read; tolerate it.
    rpkg=$( ( . "$r" 2>/dev/null; echo "${JR_PKG:-}" ) )
    [ "$rpkg" = "$1" ] && n=$((n+1))
  done
  echo "$n"
}
running_benches() {
  local n=0 r rp
  shopt -s nullglob
  for r in "$RUNDIR"/*.run; do
    rp=$(basename "$r" .run)
    if ! kill -0 "$rp" 2>/dev/null; then rm -f "$r" 2>/dev/null; continue; fi
    rkind=$( ( . "$r" 2>/dev/null; echo "${JR_KIND:-train}" ) )
    [ "$rkind" = "bench" ] && n=$((n+1))
  done
  echo "$n"
}
BENCH_RESERVED_SLOT=4
bench_queued() {
  local f fp qk
  shopt -s nullglob
  for f in "$QDIR"/*.job; do
    fp=$(basename "$f" .job)
    [ "$fp" = "$$" ] && continue
    kill -0 "$fp" 2>/dev/null || { rm -f "$f" 2>/dev/null; continue; }
    qk=$( ( . "$f" 2>/dev/null; echo "${JQ_KIND:-train}" ) )
    [ "$qk" = "bench" ] && return 0
  done
  return 1
}
train_queued() {
  local f fp
  shopt -s nullglob
  for f in "$QDIR"/*.job; do
    fp=$(basename "$f" .job)
    [ "$fp" = "$$" ] && continue
    kill -0 "$fp" 2>/dev/null || { rm -f "$f" 2>/dev/null; continue; }
    qkind=$( ( . "$f" 2>/dev/null; echo "${JQ_KIND:-train}" ) )
    [ "$qkind" = "train" ] && return 0
  done
  return 1
}
my_turn() {       # 0 if no queued job outranks me
  local f fp best_pkg_n best_t mine_n mine_t other_n other_t
  mine_n=$(running_for "$PKG"); mine_t="$MY_QTIME"
  shopt -s nullglob
  for f in "$QDIR"/*.job; do
    fp=$(basename "$f" .job)
    [ "$fp" = "$$" ] && continue
    if ! kill -0 "$fp" 2>/dev/null; then rm -f "$f" 2>/dev/null; continue; fi
    # A competitor can acquire (and delete its job file) between the glob above and this read.
    # Sourcing a vanished file leaves JQ_* unset, which under `set -u` printed an unbound-variable
    # error on every poll. Read the file ONCE, tolerate its disappearance, and default every field.
    read -r other_t other_n_pkg other_p < <(
        ( . "$f" 2>/dev/null; echo "${JQ_EPOCH:-} ${JQ_PKG:-} ${JQ_PRIO:-0}" ) )
    [ -z "$other_t" ] && continue
    other_n=$(running_for "${other_n_pkg:-_none_}")
    # priority beats fair share outright, in both directions
    if [ "${other_p:-0}" -gt "$PRIORITY" ]; then return 1; fi
    if [ "${other_p:-0}" -lt "$PRIORITY" ]; then continue; fi
    if [ "$other_n" -lt "$mine_n" ]; then return 1; fi
    if [ "$other_n" -eq "$mine_n" ] && [ "$other_t" -lt "$mine_t" ]; then return 1; fi
  done
  return 0
}
# ----------------------------------------------------------------------------

parent=$PPID
# A detached launch (setsid) is already reparented to init by the time we read $PPID, and
# `kill -0 1` fails for a non-root user, so watching pid 1 would abandon EVERY detached job
# immediately. If there is no real launcher to watch, say so and do not abandon.
if [ "$ABANDON" -eq 1 ] && [ "$parent" -le 1 ]; then
  echo "[slot -] --abandon-if-launcher-dies: no launcher to watch (parent=$parent, detached); ignoring the flag" | tee -a "$LOG"
  ABANDON=0
fi
queued_logged=0
MY_QTIME=$(date +%s)
while true; do
  if [ "$KIND" = "train" ] && bench_queued; then
    # hold slot 4 open for a waiting bench; training may use it only when no bench waits
    ALLOWED=(); for _s in 0 1 2 3 4; do [ "$_s" = "$BENCH_RESERVED_SLOT" ] || ALLOWED+=("$_s"); done
  elif [ "$KIND" = "train" ]; then
    ALLOWED=(0 1 2 3 4)
  fi
  if [ "$queued_logged" -eq 0 ] || my_turn; then try_acquire && break; fi
  if [ "$queued_logged" -eq 0 ]; then
    MY_QTIME=$(date +%s)
    { echo "JQ_TIME='$(date +%H:%M:%S)'"; echo "JQ_CWD='$PWD'"; echo "JQ_PKG='$PKG'";
      echo "JQ_EPOCH=$MY_QTIME"; echo "JQ_KIND='$KIND'"; echo "JQ_PRIO=$PRIORITY";
      echo "JQ_CMD='$*'"; echo "JQ_PARENT=$parent"; } > "$QFILE"
    echo "[slot -] queued ($(date +%H:%M:%S)) kind=$KIND slots=$NSLOTS parent=$parent cwd=$PWD: $*" | tee -a "$LOG"
    queued_logged=1
  fi
  if [ "$ABANDON" -eq 1 ] && ! kill -0 "$parent" 2>/dev/null; then
    echo "[slot -] parent $parent died while QUEUED (--abandon-if-launcher-dies); abandoning: $*" | tee -a "$LOG"
    exit 130
  fi
  sleep 5
done
cleanup_q; trap - EXIT

DEVS=$(IFS=,; echo "${IDX[*]}")
export CUDA_VISIBLE_DEVICES="$DEVS"
RFILE="$RUNDIR/$$.run"
{ echo "JR_PKG='$PKG'"; echo "JR_SLOT='$DEVS'"; echo "JR_CWD='$PWD'"; echo "JR_KIND='$KIND'"; } > "$RFILE"
trap 'rm -f "$RFILE"' EXIT
PTAG=""; [ "$PRIORITY" -eq 1 ] && PTAG=" PRIORITY(pkg=$PKG)"
echo "[slot $DEVS] acquired ($(date +%H:%M:%S)) kind=$KIND$PTAG pid=$$ cwd=$PWD pkg=$PKG: $*" | tee -a "$LOG"
trap '' HUP INT TERM     # ACQUIRED: this wrapper never kills the job from here on
"$@"
rc=$?
rm -f "$RFILE"
echo "[slot $DEVS] released ($(date +%H:%M:%S)) rc=$rc: $*" | tee -a "$LOG"
exit $rc
