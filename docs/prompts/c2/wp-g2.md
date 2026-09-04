# WP-G (continued): HGQ2 on the champion trained and evaluated on L1T, then FPGA synthesis (session: to be assigned)
Read `prompts/c2/00-common-v2.md`, `prompts/00-common.md`, `prompts/wp-g.md` (campaign 1 brief + Amendment G-L1T), `writeup/G-quantization.md`
and `reports/g-0055.md` in full. You continue WP-G; branch `wp-g` (@def4c15) is yours, clone to `~/c2-g`. Nothing in it is discarded.

## State you inherit (all measured, all in runs/*.json and writeup/G-quantization.md)
- Float L1T reference: `d-pma0-aug-meanpt-l1t` 0.8117 ± 0.0003 (champion recipe trained on the L1T train file, generator eta_max 3).
- Keras port exact (1e-5), two exact restructurings (89,606 → 60,198 params), four constant input channels folded at export.
- LayerNorm → trainable QBatchNormalization: swap cost 0.0039 (the whole accuracy cost of the line).
- Operating point: 6-bit cap + ReLU (`gF-bits6-relu-l1t`) 0.8089 ± 0.0005, EBOPs 3.80e8 at N=200; torch emulator and hls4ml C-sim bit-exact.
- Blockers: hls4ml 1.3.0 has no masked softmax (additive −64 bias used), no LayerNorm, no io_stream with heterogeneous activation
  quantization; the fully unrolled io_parallel design at N=400 did not compile in 45 min. Synthesis was impossible: no tools.

## What is new
`/tools/Xilinx/Vivado/2024.2` and `/tools/Xilinx/Vitis/2024.2` are installed with a licence server (`XILINXD_LICENSE_FILE` set). Verify
`vitis_hls -version` and `vivado -version` first and report the outputs; if the licence fails, report the exact error and stop that thread.

## G6: a synthesisable design at the full token count (the blocker to remove)
Two routes, run both to the point of a resource/latency estimate, then choose on numbers:
- G6a `io_stream` with HOMOGENEOUS activation quantizers (per-tensor, not per-channel), weights still heterogeneous. Re-train stage B from
  stage A under that constraint at the 6-bit cap with ReLU, bench on L1T (`--eta_max 3.0 --train_data ...l1t.pt`, 3 seeds per the
  campaign-2 standard), report the accuracy cost of the homogeneity constraint against `gF-bits6-relu-l1t`.
- G6b `io_parallel` with the token loop folded: a per-token sub-graph synthesised once and iterated (hls4ml reuse factor, or a
  hand-written wrapper around the hls4ml per-token block with the PMA pooling as a streaming reduction). Numbers unchanged by construction;
  prove it with the emulator at N=400.
Target part: `xcvu13p-flga2577-2-e` (hls4ml default, L1-trigger class) at 200 MHz clock unless the user says otherwise; state it in every table.

## G7: synthesis
Vitis HLS C-synthesis of the chosen design at N=400 (and N=16 as the sanity anchor): latency (cycles and µs), initiation interval, LUT/FF/DSP/BRAM
utilisation and percentage of the part, both from the HLS report. Then Vivado out-of-context logic synthesis (`vivado_synth`) for post-synthesis
utilisation and timing; report whether 200 MHz closes. C/RTL co-simulation on ≥ 16 events, bit-compared against the Keras model. Every number
in a table with the design (G6a/G6b), N, precision, part and clock beside it. If synthesis at N=400 exceeds 3 h, report the partial log and ask.

## G8: track the champion
If campaign 2 promotes a new champion (planner ruling), retrain it on L1T with WP-H's loader (`degradation_eta_max 3.0`, 3 seeds), redo G1
acceptance, stage A/B at the operating cap, and re-synthesise. Until then the certified champion's L1T model is the design.

## Rules
Campaign-2 measurement standard applies (3 seeds, seed test, JSON fields). GPU only through `gpu_slot.sh`. Synthesis runs on CPU and are
long: state expected wall time before each, run under `nohup` with a log in `~/c2-g/synth/`, and never run more than two at once (8 cores shared).
Reports `reports/g-<HHMM>.md` every 60 min; first report = tool versions + licence check + plan for G6a/G6b.
