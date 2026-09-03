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
