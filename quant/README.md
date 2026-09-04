# WP-G: HGQ2 quantization of the certified winner

Owner: session `jovyan-31 [32ac07]`. Branch `wp-g` off `submission-pma0-meanpt` @41d45a9. **Ships nothing.**

## What stays in float on the host, and why
The preprocessor is *almost* elementwise. `preproc_meanpt_np` in `keras_port.py` splits it as:

* **Elementwise, foldable, quantizable in principle**: `tanh(dxy)`, the pdgId one-hot, the charge lookup,
  and the frozen batch norm (in eval mode a per-channel affine map, so applying it everywhere and then
  zeroing invalid rows is *identical* to torch's `flat[valid] = bn(flat[valid])` — see the test).
* **One per-event reduction**: `log(pt_i / mean surviving pt)` needs `sum(pt)` and `count` over the event
  before any per-candidate work can finish. That is a two-pass dependency over 400 candidates.

For G1-G3 the whole preprocessor runs in float on the host and the quantized model starts at the 14
features. This is stated rather than hidden: the quantized region is the **encoder**, 89,606 parameters.

## Files
* `keras_port.py` — G1 float Keras 3 re-implementation + torch weight loader (tested).
* `hgq_model.py`  — G2 HGQ2 quantized model, in the restructured hardware-shaped form.

## Campaign 2 additions

* `g6_build.py` — G6/G7. Converts the HGQ2 model to an hls4ml project at the full token
  count. `--pf 1` folds the token loop (one per-token datapath iterated N times) and
  `--einsum_rf` folds the attention·value einsum; `--homogeneous 1` builds the
  per-tensor-activation variant that io_stream would need. Writes the resulting hls4ml
  graph into the JSON so the fold is evidenced, not asserted.
* `make_tb_data.py` — writes `tb_data/tb_input_features.dat` / `tb_output_predictions.dat`
  from REAL L1T eval events plus the Keras latents, so C-simulation and C/RTL
  co-simulation are bit-compared against Keras rather than eyeballed.
* `parse_synth.py` — every number in the writeup's Synthesis table, sourced from the
  tool's own `*_csynth.xml` / utilisation report, with percentages taken from the tool's
  own "Available" figures for the part.
* `run_g6a.sh` / `run_g6a_bench.sh` — the 3-seed homogeneous-vs-heterogeneous arms.
* `../synth/run_synth.sh` — drives one hls4ml project through Vitis HLS. Never more than
  two at once on this 8-core box.
* `../synth/udevshim/udevshim.c` and `../synth/udev_stub.c` — **source only, NOT BUILT.**
  Vivado 2024.2 and 2025.2 both crash (SIGABRT / SIGSEGV) inside libudev's
  `udev_enumerate_scan_devices`, called from the FlexLM feature checkout, *before* they
  reach the licence server, because this container has no `/run/udev`. `LD_PRELOAD` is
  bypassed (libXil_lmgr11 `dlopen`s libudev and resolves on that handle); the only route
  left is a no-op stand-in `libudev.so.1` earlier on `LD_LIBRARY_PATH` for the Vivado
  process. Building a stand-in for a system library is a human decision, so these are
  left uncompiled. Nothing in this package uses them.
