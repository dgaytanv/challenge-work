#!/bin/bash
# WP-G G7: drive one hls4ml project through Vitis HLS. One job = one invocation.
# Usage: run_synth.sh <project_dir> <csim> <synth> <cosim> <vsynth> [logname]
# NEVER more than two of these at once (8 cores shared with five training jobs).
set -u
PRJ=$1; CSIM=$2; SYNTH=$3; COSIM=$4; VSYNTH=$5; LOG=${6:-build}
export PATH=/tools/Xilinx/Vitis/2024.2/bin:/tools/Xilinx/Vitis_HLS/2024.2/bin:$PATH
cat > "$PRJ/build_opt.tcl" <<TCL
array set opt {
    reset      1
    csim       $CSIM
    synth      $SYNTH
    cosim      $COSIM
    validation 0
    export     0
    vsynth     $VSYNTH
    fifo_opt   0
}
TCL
cd "$PRJ" || exit 2
echo "[synth] $(date -Is) start $PRJ csim=$CSIM synth=$SYNTH cosim=$COSIM vsynth=$VSYNTH"
vitis-run --tcl build_prj.tcl --mode hls
rc=$?
echo "[synth] $(date -Is) done rc=$rc"
exit $rc
