# G — HGQ2 quantization of the certified encoder, and the road to hls4ml

WP-G, session `jovyan-31 [32ac07]`, branch `wp-g` (ships nothing; the competition entry is untouched).
Every number below is sourced to a JSON under `~/hackathon-shared/quant/` or to a `runs/` bench file.

**What was quantized.** Per the L1T amendment (planner, 23:43), the target is D's L1T-trained float
reference `rt_d_pma0_aug_meanpt_l1t_encoder_20260903_235356.pth` (bench `d-pma0-aug-meanpt-l1t`
mean_area 0.8117 ± 0.0003, clean AUC 0.9067), not the PF competition winner. The PF-trained winner
evaluated on L1T (`winner-l1t-eta3`, 0.8063 ± 0.0006) appears only as a reference row.

---

## 0. Environment (G0)

Nothing was installed at 23:11. Installed: **hgq 0.2.0** (HGQ2), keras 3.15.1 on the **torch** backend,
hls4ml 1.3.0, da4ml 0.5.2, quantizers 1.2.2. No TensorFlow is needed anywhere.

> **Naming trap, recorded because it silently gives the wrong library.** PyPI `hgq` is HGQ**1**
> (max version 0.2.6, Keras-2/TensorFlow only). HGQ2 is the distribution **`hgq2`**, which installs
> the *module* `hgq`. Installing the module name as the prompt phrases it gets you HGQ1.

**No Vitis/Vivado HLS exists on this box** — not on PATH, no `/opt/Xilinx`, `/tools/Xilinx`,
`/opt/intelFPGA`. So **there are no synthesis numbers in this report**: no LUT/FF/DSP/BRAM, no latency,
no II from a real synthesis run. hls4ml does bundle `ap_types/` and `build_lib.sh`, so C-simulation with
g++ is possible, and that is what section 4 reports instead.

---

## 1. The port (G1) — PASS

`quant/keras_port.py` re-implements `PFPreProcessorMeanPt` + `PMAEncoder(num_layers=0)` in Keras 3.
`tests/test_port.py`, 2000 real eval events, clean and `Degradation(severity=0.5)`:

| reference | view | preproc max&#124;d&#124; | **latent max&#124;d&#124;** | tol | frozen-probe ΔAUC |
|---|---|---|---|---|---|
| L1T (D's) | clean | 7.63e-06 | **7.63e-06** | 1e-4 | +0.0e0 |
| L1T (D's) | degraded s=0.5 | 7.63e-06 | **1.00e-05** | 1e-4 | +0.0e0 |
| PF (winner) | clean | 7.63e-06 | **1.14e-05** | 1e-4 | +0.0e0 |
| PF (winner) | degraded s=0.5 | 7.63e-06 | **1.24e-05** | 1e-4 | -1.9e-06 |

Relative latent error 1e-6 = float32 round-off. Parameter count identical (89,606). The probe is fit
**once** on the torch latents and applied frozen to both latent sets, so ΔAUC carries no refit noise at
all — it is a sharper test than re-fitting and comparing means.
Sources: `quant/g1_port_test_L1T.json`, `quant/g1_port_test.json`.

Numerics that had to be matched explicitly, each of which would have silently shifted the latents:
keras `LayerNormalization` defaults to **epsilon 1e-3** where torch `nn.LayerNorm` uses 1e-5; torch
`nn.Linear` stores its kernel transposed relative to keras `Dense`; torch `nn.GELU` is the exact erf form.

### What stays in float on the host
The quantized region is **the encoder only** (89,606 parameters). The preprocessor is split as:
* **elementwise and foldable** — `tanh(dxy)`, the pdgId one-hot, the charge lookup, and the frozen
  BatchNorm. In eval mode that BatchNorm is a per-channel affine, and applying it to *every* row and then
  zeroing invalid rows is **bit-identical** to torch's `flat[valid] = bn(flat[valid])` followed by
  `where(valid, x, 0)` — verified at max&#124;d&#124; 7.6e-06. (The bias makes invalid rows nonzero; the final
  masking kills them. This is exactly where such a fold usually goes wrong.)
* **one per-event reduction that does NOT fold** — `log(pt_i / mean surviving pt)` needs `sum(pt)` and a
  live count before any per-candidate work can finish. It is a two-pass dependency over 400 candidates and
  it stays in float on the host. Stated, not hidden.

---

## 2. Two exact restructurings, before any quantization (33% of the parameters, free)

`tests/test_restructure.py`, verified to **9.06e-06** (L1T) and **1.67e-05** (PF) on 2000 events, clean and degraded:

* **R1 — the seed queries are constants.** With `num_layers=0` nothing per-event feeds the 4 learned
  seeds, so `q = q_proj(seeds)` is a compile-time constant and the score collapses to a single Dense:
  `scores[b,h,s,n] = A[(h,s),:]·h[b,n] + c[(h,s)]`, with
  `A[(h,s),:] = Σ_d q_flat[s,h·D+d]·W_k[h·D+d,:]/√D`. **`q_proj` + `k_proj` (33,024 params) become one
  Dense 128→32 (4,128 params).**
* **R2 — `h = phi(x)·keep` is redundant.** The masked softmax already gives dead tokens weight 0 and
  `out = Σ_n attn[n]·v[n]`, so `v[dead]` never reaches the output. A 128-wide per-token multiplier disappears.

**89,606 → 60,198 parameters (−29,408, −33%) at zero accuracy cost.** Source: `quant/g2_restructure_test_L1T.json`.

---

## 3. Input channels: four of fourteen are constants

Measured over **8,000,000 live candidates per file** (`quant/fold_channels.py`, assertion-guarded):

| | PF eval | L1T eval |
|---|---|---|
| `is_pf` | **constant +1** | **constant +1** |
| `pdg130`, `pdg1`, `pdg2` | **constant 0** | **constant 0** |

`|pdgId|` only ever takes {0, 11, 13, 22, 211} in either file — 130 (h0), 1 (h_HF) and 2 (egamma_HF) never
occur, so three of the seven one-hot channels are dead. A constant channel folds into the next layer's bias
**exactly**: `y = Σ_c W[c,:]x[c] + b` becomes `Σ_{c live} W[c,:]x[c] + (b + Σ_{c const} W[c,:]v_c)`.

**First Dense: 14 → 10 inputs on both files. Multiplies 1792 → 1280, 28.6% fewer.** Verified max&#124;d&#124; of the
fold = **3.55e-15** (float64 exact algebra). Sources: `quant/g_fold_channels_L1T.json`, `..._PF.json`.

> **Caveat (D, 00:12), adopted.** The fold is exact only *while* those channels stay constant. A future data
> drop containing a 130 or an h_HF would be silently mis-scored by a folded model. Hence the assertion, and
> hence the data file each fold was verified against is recorded in the JSON.

A side effect worth stating: 61% of *real* candidates carry `pdgId == 0`, so the one-hot block is all-zero
for the majority of live candidates. That is why the network cannot infer the validity bit from its own
features and why the mask must be an explicit input (§5).

---

## 4. Dynamic range and the cost of the dxysig tail

Raw `dxysig` reaches 6.5e4, which looks alarming. It is not, because the preprocessor's BatchNorm divides
by a std that the same tail inflates: the **encoder input** sees &#124;max&#124; 65.3 (L1T) / 68.8 (PF).
Per-channel integer bits of the 14 encoder inputs (BN statistics from each train split, γ=1 β=0, 20k events):

| channel | max&#124;x&#124; | p99.9 | ratio | i_bits | i_bits at p99.9 |
|---|---|---|---|---|---|
| pt | 12.3 | 6.96 | 1.8 | 5 | 4 |
| dxy | 22.8 | 22.15 | **1.00** | 6 | 6 |
| **dxysig** | 65.3 | 13.74 | **4.8** | **8** | **5** |
| eta / phi | 1.8 / 1.7 | — | 1.0 | 2 | 2 |

**dxysig costs 3 integer bits for 0.1% of candidates** — at a fixed word length its fractional resolution is
8× coarser than 99.9% of the data needs. **dxy is not an outlier problem** (ratio 1.00): six integer bits are
genuinely required because `tanh` saturates and the BN std is tiny. If anything is ever squashed in the
preprocessor it should be dxysig alone. Sources: `quant/g2_input_range_L1T.json`, `..._PF.json`.

---

## 5. The mask, and the finding a reviewer of the FPGA design needs

HGQ2's `QSoftmax` supports a mask natively, but **hls4ml 1.3.0 cannot convert it**:
`converters/keras_v3/hgq2/softmax.py` raises `NotImplementedError('Masked softmax not supported yet')` for a
2-input softmax. So the mask is applied as an **additive bias on the scores** before a plain 1-input softmax:
`scores' = scores + mask_add`, `mask_add ∈ {0, −MASK_BIG}`, supplied as a model input.

### The margin is set by the dead token's score, not by the live range
Under R2 a dead token does **not** carry zero — it carries `phi(0)`, so its score is a fixed 32-vector
`A·phi(0) + c`, measured at **[−2.79, +14.03]** on the certified model. Live scores span **[−20.53, +59.07]**
(20k events; the range is severity-independent to 1e-2). **In the worst of the 32 channels the best live score
is only +0.35 above the dead score.** The entire safety margin is therefore the bias itself, not the geometry.

At the original `MASK_BIG = 32` the worst gap was **−32.35**: one dead token carries exp(−32.35) = 8.9e-15 of
the best live token's weight, and 400 of them 3.6e-12 — safe, but only 32 e-folds, and **nothing in QAT
constrains the scores from growing**. That concern proved exactly right: during stage A training the live
score maximum grew from **+59 to +332**, a 5.6× drift in 12 epochs.

`MASK_BIG` is therefore **64**, costing one integer bit of score range (7 → 8), and
`hgq_model.assert_mask_margin()` recomputes the gap after every training run and **raises** below 20 e-folds
allowing for `n_tokens` dead tokens. Measured margins: stage A **60.6 e-folds**, ReLU variant **59.4 e-folds**.

---

## 6. LayerNorm: HGQ2 has none, and the swap is not cosmetic

`grep -rn "LayerNorm" hgq/` returns **zero hits**. HGQ2 has `QBatchNormalization`, `QBatchNormDense` and
`QEinsumDenseBatchnorm` but **no quantized LayerNorm at all**. The encoder uses LayerNorm in three places
(twice in the per-candidate MLP, once on the 512-d pooled vector), so the LayerNorm→BatchNorm swap is **the
only fully-quantizable path**, not an optional extra row.

### Calibrating it correctly matters by a factor of six
LayerNorm normalises **per token across channels**, so what it divides by is the spread of the 128 (or 512)
channels *within one token*, not the spread of one channel across tokens. Seeding a BatchNorm with
per-channel statistics uses the wrong denominator whenever the channel means are spread out:

| norm | per-channel var | per-token var | ratio | spread of channel means |
|---|---|---|---|---|
| phi.1 | 0.4217 | 0.4754 | 1.13 | 0.2334 |
| phi.4 | 0.1802 | 0.3760 | 2.09 | 0.4438 |
| **norm_pooled** | 0.0349 | 0.7843 | **22.50** | 0.8657 |

Cold-initialisation error on the latent: **5165 with per-channel statistics, 879 with per-token statistics**,
against a latent scale of 10.8. The remaining 879 is genuine per-token variation that no population affine can
absorb — that is precisely the quantity the swap is measured for, and it is why `QBatchNormalization` is
**trained** (batch statistics during training, running statistics frozen and foldable at export) rather than
folded cold.

*Caveat carried, not assumed away:* keras BatchNormalization cannot exclude dead rows from its batch
statistics, so the running statistics are taken over live and dead tokens together.

---

## 7. Emulation on the real ruler (G3) — EXACT

`QuantizedPMAEncoder` in `src/embedding/models.py` emulates HGQ2's fixed-point arithmetic in torch, taking its
semantics from `quantizers/fixed_point/_fixed_point_ops.py` (saturate-then-round for SAT/SAT_SYM,
round-then-wrap for WRAP; RND = `floor(x+0.5)`, RND_CONV = round-half-even). Weight quantizers are applied
**once at export** (they are data-independent, so this is exact); only the data-lane quantizers run at inference.

> **torch emulator vs HGQ2 Keras, 1000 events × 400 candidates:
> max&#124;d&#124; = 0.000e+00, and 100.00% of latent elements are identical to the last bit.**
> A layer-by-layer trace is 0.000e+00 at all 15 taps.

Not "within tolerance" — exactly zero. So `bench_eval.py --encoder_class QuantizedPMAEncoder` measures the
actual quantized arithmetic with no emulation gap to caveat. Source: `quant/g3_emulation_test.json`.

---

## 7b. Results on the real ruler

All rows: L1T eval file, `--eta_max 3.0`, `--train_data robust_tagging_train_data_small_l1t.pt`
recorded, five scoring families, R=5 probe refits. Each Δ is against **its own** reference, so cost
is attributable to a stage rather than blended.

| row | what it is | mean_area | clean AUC | Δ vs its reference | separable? |
|---|---|---|---|---|---|
| `d-pma0-aug-meanpt-l1t` | D's float L1T reference (LayerNorm, GELU, fp32) | **0.8117 ± 0.0003** | 0.9067 | — | — |
| `g-stageA-l1t` | + R1/R2 + **LayerNorm→BatchNorm**, still float | **0.8078 ± 0.0007** | 0.8976 | **−0.0039** vs float | **yes** (3.9×) |
| `gB-q-b1e-6-l1t` | + **HGQ2 quantization** (~8 bit) | **0.8079 ± 0.0011** | 0.9019 | **+0.0001** vs stage A | no (thr 0.0017) |
| `winner-l1t-eta3` | PF-trained winner, evaluated on L1T (reference only) | 0.8063 ± 0.0006 | — | — | — |

**The entire measured cost is the LayerNorm→BatchNorm swap (−0.0039 ± 0.0008, separable).
Quantization to ~8 bits is free within the probe floor (+0.0001 ± 0.0013).**

Precision of the quantized row, so that "free" is falsifiable (planner ruling, 00:41):
mean weight bits **8.84**, mean activation bits **9.06**, EBOPs **5.056e8**, score-path integer bits **7**,
mask margin **60.2 e-folds**. (§7d shows the same statement holds down to a 6-bit cap, and fails at 4.)

**A free reproducibility check.** `gB-q-b1e-6-l1t` and `gD-bits10-l1t` are independent training runs whose
bit-width cap never binds, so they are the same model configuration trained twice and benched twice. They give
**0.8079 ± 0.0011** and **0.8087 ± 0.0003** — a spread of 0.0008 across the whole train-and-bench pipeline.
That is the scale below which no difference in this report should be believed, and it is consistent with the
per-refit probe floor rather than larger than it.

Note the hardware-shaped, LayerNorm-free, 60,198-parameter quantized model at 0.8079 still sits **above**
the PF-trained competition winner evaluated on the same file (0.8063).

### The dead-row caveat, bounded
Stage A's BatchNorm statistics are taken over live **and** dead tokens together. Keras 3's
`BatchNormalization.call` does take a `mask` and route it to `_moments` (verified: the moving mean differs by
2.4e-3 with and without), so this is testable, and it was tested — `gE-stageA-maskedbn-l1t`, identical
protocol with the dead tokens excluded from the batch statistics:

| | distill rms | mean_area |
|---|---|---|
| stage A, statistics over all tokens | 0.3889 | **0.8078 ± 0.0007** |
| stage A, statistics over live tokens only | **0.3682** | **0.8044 ± 0.0007** |

Restricting the statistics to live slots does **not** recover the swap cost — it makes the bench *worse*. So
the 0.0039 is the swap itself, not dead-row contamination. Reported as a bound, not as a decomposition.

That row also carries a warning about this package's whole objective: masked BN **matches the float teacher
better** (rms 0.3682 vs 0.3889) while scoring **worse on the ruler** (0.8044 vs 0.8078). Distillation loss and
bench mean_area are therefore not monotonically related, which is a concrete instance of the limitation
recorded in §9.5 — arm 1 optimises teacher agreement, and teacher agreement is not the metric.

---

## 7c. Methods failure worth recording: the resource regulariser measured nothing

The prompt asks for a sweep of HGQ2's `beta` resource regulariser. **That sweep is degenerate for this
model and reports nothing**, which took four fully-populated, plausible-looking, identical rows to notice.

Measured EBOPs at the end of training, seeded identically:

| beta0 | 0 | 1e-6 | 1e-5 | 1e-4 | 1e-3 |
|---|---|---|---|---|---|
| EBOPs | 5.056406e8 | 5.056406e8 | 5.056406e8 | 5.056406e8 | 5.056406e8 |

Identical to **seven significant figures** across a range including zero; integer bits identical to the last
bit; mean weight bits moving 0.014 over a 1000× range of beta. The term is genuinely in the loss (it scales
exactly 10× with beta: 514 → 5137) and genuinely produces gradients (39 of 70 trainable variables receive
nonzero gradient from it). Two causes compound:

1. **HGQ2's default quantizer configs already carry `MonoL1(1e-8)` regularisers on the bit-width variables**
   (`kbi_weight_default`, `kif_datalane_default`), which push widths down with no help from `beta`.
2. **Adam is scale-invariant.** EBOPs ≈ 5e8 while the distillation MSE ≈ 0.1, so at beta = 1e-6 the resource
   term already outweighs the task loss by ~5000×. Once every term acting on a width has the same sign,
   Adam's step is set by the learning rate, not by the loss magnitude, and multiplying the dominant term by
   1000 changes nothing.

**Transferable rule:** before sweeping any regulariser strength under Adam, check that the two loss terms are
within about an order of magnitude of each other. Otherwise the sweep silently measures the learning rate.
HGQ2's default `beta0 = 1e-5` is not a universal setting — it is calibrated for models whose EBOPs are far
smaller than a 200-token × 128-wide encoder's.

The accuracy-versus-bits curve in §7d therefore sweeps a **direct cap on the learned bit widths** instead
(weights: total bits; data lanes: fractional bits, with the integer part left free so the score path cannot
saturate — see §5).

---

## 7d. Accuracy versus bits

Axis: a hard cap on the learned bit widths (weights: total bits; data lanes: fractional bits, integer part
left free). Every point is seeded from stage A and trained identically (8 epochs, lr 3e-4, 10k events/epoch,
clean + generator-degraded views), so the only difference between rows is the cap.

| bit cap | mean w bits | mean act bits | score int bits | EBOPs (N=200) | distill rms | mean_area | clean AUC | Δ vs stage A |
|---|---|---|---|---|---|---|---|---|
| 3 | 3.88 | 2.92 | 7 | 6.064e+07 | 3.003 | 0.5000 ± 0.0000 (**dead**) | 0.5000 | — |
| 4 | 4.88 | 6.84 | 7 | 1.840e+08 | 1.686 | **0.7797 ± 0.0018** | 0.8547 | -0.0282 |
| 6 | 6.84 | 9.05 | 7 | 3.806e+08 | 0.366 | **0.8080 ± 0.0001** | 0.8978 | +0.0002 |
| 8 (uncapped) | 8.84 | 9.06 | 7 | 5.056e+08 | 0.265 | **0.8079 ± 0.0011** | 0.9019 | +0.0001 |
| 10 (cap not binding) | 8.82 | 9.06 | 7 | 5.056e+08 | 0.265 | **0.8087 ± 0.0003** | 0.9025 | +0.0009 |
| 8, **ReLU** | 8.78 | 8.59 | 7 | 5.059e+08 | 0.231 | **0.8078 ± 0.0009** | 0.9014 | -0.0000 |
| **6, ReLU** (operating point) | 6.79 | 8.58 | 7 | **3.804e+08** | 0.319 | **0.8089 ± 0.0005** | 0.8991 | +0.0010 |

**GELU → ReLU is free.** The ReLU row is 0.8078 ± 0.0009 against GELU's 0.8079 ± 0.0011 and stage A's
0.8078 ± 0.0007 — indistinguishable, and its distillation rms is actually the *best* of any row (0.231 vs
0.265). This matters more than the accuracy tie: **ReLU needs no lookup table at all**, whereas GELU costs a
`QUnaryFunctionLUT` per activation site, and it is that LUT which forced the homogeneous-table constraint and
tripped both hls4ml bugs (§8). The hardware-friendlier variant the prompt asked for is therefore free on
accuracy and strictly better on implementability. The earlier ReLU row that benched at 0.5226 was a harness
bug of mine, not a property of ReLU — see §8b.1.

**There is a cliff between 3 and 4 bits, and it is a total collapse rather than a degradation.**
At a 3-bit cap the encoder emits **a single constant latent**: over 500 eval events the per-dimension
standard deviation is exactly 0.0000 and there is **1 distinct latent vector**. The probe therefore has
nothing to separate and returns AUC 0.5000 ± 0.0000 — the zero variance is the signature, not a coincidence.
At 4 bits the same measurement gives 500 distinct latents with healthy per-dimension spread
(std 0.53–1.75), so the model is alive. The 3-bit row's EBOPs (6.06e7) should not be read as an operating
point; it is the cost of a dead network.

### The figure
`runs/plots/quant_curve_latest.png`, regenerated from the run JSONs by `quant/plot_curve.py`, with
`quant_curve_latest.txt` beside it as a plain-text table so **no number on the results page depends on
reading a chart**. Two stacked panels sharing the x axis rather than one dual-axis chart; EBOPs annotated
once per bit cap (they are a function of the cap, not the activation); float and stage A reference lines
labelled inside the axes; the collapsed 3-bit row drawn **on the bottom spine** with a downward caret and its
true value in the label, because the y range is focused on the region where the live rows differ and a
collapsed row would otherwise be silently clipped out of view.

*Verification caveat, stated because it is unusual:* image reads were failing on hook timeouts for every
session tonight, so **this figure was never visually inspected by anyone**. Instead it is checked
programmatically on each render — every text artist is tested for leaving the canvas, overflowing its axes,
and colliding with another. That check caught three real defects in the first version (one label off the
canvas, one overflowing, and a collision) and now reports clean. A programmatic check is weaker than a look;
it is what was available.

### Chosen operating point

**A 6-bit cap with ReLU — measured directly, not inferred.** `gF-bits6-relu-l1t`:
**mean_area 0.8089 ± 0.0005, clean AUC 0.8991, EBOPs 3.804e8, mask margin 60.5 e-folds.**

It was worth measuring rather than composing from the 6-bit GELU row and the 8-bit ReLU row, because this
project's own ruling records that ablation arms are **not** additive (the competition winner was an
interaction between the PMA readout and MeanPt, neither of which separated alone). Here the combination
happens to be benign, but that is now a measurement.

Against the unquantized stage A reference (0.8078 ± 0.0007) the difference is +0.00104 against a threshold of
0.00115 — **tied**, and nominally above. Against the 6-bit GELU row it is +0.00086, which sits **inside the
0.0008 pipeline reproducibility** measured in §7b, so **no claim is made that ReLU beats GELU**: every row at
a cap of 6 or more is mutually tied on accuracy.

6-bit ReLU is therefore chosen on **cost and implementability, not on its point estimate**: 3.80e8 EBOPs
against 5.06e8 for the 8- and 10-bit rows (25% cheaper), and ReLU needs **no lookup table at all**, which
removes the single component that caused most of §8's hls4ml trouble. Going to 4 bits costs 0.028 mean_area,
far outside any floor; 3 bits collapses. The knee is sharp and 6 bits is the point to build.

---

## 8. hls4ml (G4)

**C-simulation is bit-exact.** N=16, io_parallel, Vitis backend, compiled with g++ only:
**csim vs Keras max&#124;d&#124; = 0.000e+00, 100% of elements exact** (`quant/g4_hls4ml_N16_io_parallel.json`).
So the whole chain — float → HGQ2 Keras → torch emulator → hls4ml C++ — is exact at every link.

### N = 400 does not compile on this box
Conversion at the full 400-candidate budget **succeeds** (the project and firmware are written), but the g++
compilation of the fully-unrolled io_parallel design was **time-boxed and killed at 2,710 s (45.2 min)**
without finishing, at ~1.1 GB RSS. That is a result rather than an absence: **io_parallel is forced on us**
(§8.3, io_stream cannot carry heterogeneous activation quantization), and io_parallel at N=400 unrolls every
per-token layer 400 times — 15.7 M multiplies in one flattened design. The N=16 csim is bit-exact and the
arithmetic is N-independent by construction (every quantizer on a token-axis tensor is homogeneous over
`(batch, token)`), so the numerical claim carries; what does not carry is any statement that a 400-token
design is buildable. On a real target this points at io_stream with homogeneous activations, or at folding
the token loop, neither of which is done here.

### What hls4ml could not handle, and what was done about each

| # | Blocker | Resolution |
|---|---|---|
| 1 | **Masked softmax**: `hgq2/softmax.py` raises `NotImplementedError('Masked softmax not supported yet')` for a 2-input QSoftmax. | Mask applied as an additive −64 score bias before a plain softmax (§5). Exact, costs one integer bit. |
| 2 | **No quantized LayerNorm in HGQ2 at all.** | Replaced by trained `QBatchNormalization` (§6). Cost measured, not assumed. |
| 3 | **io_stream impossible**: `NotImplementedError('Heterogenous quantization for activations is only supported with IOType=io_parallel')`. | **Unresolved, and fundamental.** Per-channel activation bitwidths *are* what HGQ2 is for, so "io_stream" and "HGQ2 high-granularity quantization" are mutually exclusive in hls4ml 1.3.0. Reported as io_parallel. io_stream is reachable only by forcing the activation quantizers homogeneous, i.e. by discarding the granularity that motivates HGQ2. |
| 4 | **hgq 0.2.0 ships no `QActivation`** although hls4ml has a handler for one (version skew), so a LUT is the only quantized activation available. | Used `QUnaryFunctionLUT`. This is why the GELU→ReLU variant matters: ReLU needs no table. |
| 5 | **hls4ml requires a homogeneous activation table** (`assert not layer._allow_heterogeneous_table`). | `allow_heterogeneous_table=False`. Physically right anyway — one LUT is shared by all 128 channels in hardware. |
| 6 | **hls4ml bug**: its LUT handler calls `.numpy()` on variables that still require grad (torch backend only). | Avoided without patching, by `model.trainable = False` before conversion. |
| 7 | **hls4ml bug**: its LUT handler assumes the activation's output quantizer was built on a **rank-2** tensor (`layer.oq(table[None, ...])[0]`). Ours is per-token `[B,N,C]`, hence rank 3, and the expand fails. | Handler reimplemented rank-generically in `quant/hls4ml_patch.py`. Documented as an hls4ml bug; the model is unchanged. |

hls4ml also warns `result_t for phi_act0 propagated multiple times. Bit-exactness may be compromised.`
(consecutive quantizers: our BatchNorm feeds directly into the activation LUT). The measured csim difference
is nevertheless exactly zero, so the warning did not bite here.

### NOT DONE — synthesis
**No LUT/FF/DSP/BRAM, latency or II numbers exist in this report**, because no Vitis/Vivado HLS is installed
on this box (§0). This is a missing deliverable, not a deferred one, and nothing below substitutes for it.

### Resource proxy, in place of synthesis
What can be reported honestly is the quantity HGQ2 optimises and the HGQ literature uses as the LUT proxy:
**EBOPs**, the bit-weighted MAC count `Σ bits(input) × bits(weight)`. For the ~8-bit quantized model at the
full **N = 400** candidate budget (`quant/g4_resource_gB-q-b1e-6-l1t_N400.json`):

| layer | EBOPs | multiplies | mean weight bits | mean activation bits |
|---|---|---|---|---|
| phi_0 | 3.60e7 | 716,800 | 9.00 | 6.64 |
| phi_1 | **4.20e8** | 6,553,600 | 9.00 | 9.00 |
| score (fused q·k, R1) | 1.05e8 | 1,638,400 | 9.00 | 9.00 |
| v | **4.20e8** | 6,553,600 | 9.00 | 9.00 |
| combine (attn·v) | 1.12e7 | 204,800 | — | — |
| softmax | 1.17e6 | — | — | — |
| out_proj | 4.60e6 | 65,536 | 9.00 | 9.72 |
| bottleneck | 2.46e5 | 3,072 | 9.00 | 11.03 |
| **TOTAL** | **1.010e9** | **15,735,808** | | |

Two things a hardware reviewer should take from this. **`phi_1` and `v` are 83% of the cost** (4.2e8 each) —
both are 128×128 per-token Denses, and they are where any further width reduction should be aimed.
And R1 matters exactly here: without it the score path would be a second 128×128 per-token Dense
(another ~4.2e8) instead of the 1.05e8 shown. EBOPs scale linearly in N for every per-token layer, as expected
(5.056e8 at N=200 → 1.010e9 at N=400, exactly 2×); `out_proj` and `bottleneck` are N-independent.

---

## 8b. Four silent failures from this package

Each produced a fully populated, plausible-looking, wrong answer rather than an error. Recorded because the
defence generalises, not because they were interesting individually.

1. **The activation as a constructor kwarg.** `QuantizedPMAEncoder(act='gelu')` — but `bench_eval.py` and
   `eval.py` build the encoder from the signature alone and pass no kwargs. A ReLU-trained checkpoint
   strict-loaded into a GELU-default class: 81 tensors matched, nothing raised, and the model benched at
   **mean_area 0.5226, clean AUC 0.6250**, near chance in every family — while its distillation rms was a
   healthy 0.269. It was plotted as a genuine "ReLU fails" result before being caught.
   *Defence:* anything the eval harness will not pass must travel **inside the checkpoint**. The activation
   is now a state_dict buffer. This is the same rule the `--expect-preproc` guard exists for.
2. **Saving the weights but not the quantization.** `save_params` stored kernels and BatchNorm parameters
   only. In HGQ2 the per-tensor bit widths **are** the quantization, and they are learned during QAT and set
   by `trace_minmax`. Reloading therefore reverted every quantizer to its `i0 = 2` default, and the export
   wrapped scores of magnitude 42 into [−4, 4). Caught only by the emulation test (max|d| 19.4 on a latent of
   scale 14). *Defence:* save every variable by path, not a curated list.
3. **A regulariser sweep that measured nothing** (§7c). Four identical rows.
   *Defence:* check the two loss terms are within an order of magnitude before sweeping either.
4. **`homogeneous_axis=(0,1)` applied to the 2-D tail.** Correct for `(batch, token)` tensors, but after the
   flatten axis 1 is the *channel* axis, so it collapsed all 512 channels onto one shared bit width —
   discarding exactly the granularity HGQ2 exists for, with no error and only a slightly worse number.

A fifth, near-miss rather than a failure: the additive mask margin (§5) was set by the **dead** token's score,
not the live range, and in the worst channel had only 0.35 of headroom before the bias was applied. It was
correct at MASK_BIG = 32 but for a reason that no longer held once QAT grew the live scores 5.6×.

---

## 8c. Campaign 2 — a synthesisable design at N=400, and synthesis (G6, G7)

*Added by WP-G in campaign 2 (4 Sep, from 03:34). Campaign 1's §8 ends with two untried routes to a
buildable 400-token design and no synthesis at all, because the box had no vendor tools. It now has them.
Nothing above is retracted except where this section says so explicitly.*

### 8c.0 Tools: Vitis HLS synthesises, Vivado crashes before the licence server

| step | result |
|---|---|
| `vitis_hls -version` | v2024.2, SW Build 5238294, exit 0 |
| `vivado -version` | v2024.2, SW Build 5239630, exit 0 |
| **Vitis HLS `csynth_design` on `xcvu13p-flga2577-2-e`** | **works, no licence error** (trivial kernel: II 4, depth 11, Estimated Fmax 381.24 MHz, 79 s of synthesis inside 727 s of wall clock) |
| **Vivado `synth_design` on the same part** | **crashes** — `realloc(): invalid pointer` / `Abnormal program termination (6)` in 2024.2; `Abnormal program termination (11)` in 2025.2 |

`-version` checks out no feature, so it is not the licence test; the test above is. Both Vivado crashes have
the same stack — `libc realloc` ← `libudev.so.1 udev_enumerate_scan_devices` ← `libXil_lmgr11.so` ←
`XilFNP::XilFlex::XF_lc_checkout` ← `HLRegMgr::checkoutFeature` — so **Vivado dies inside the FlexLM device
scan, before it ever reaches `2100@xilinxd.xilinx-dev`**. This container has no `/run/udev`. It is not a
licence refusal and the licence server may well be healthy; we cannot tell from here. Stacks:
`~/c2-g/synth/vivlic/hs_err_pid23142.log`, `~/c2-g/synth/vivlic25/hs_err_pid113719.log`.

`LD_PRELOAD` does not help — `libXil_lmgr11` `dlopen()`s libudev and resolves on that handle, verified by the
real symbol still appearing in the stack. The only remaining route is a no-op stand-in `libudev.so.1` earlier
on `LD_LIBRARY_PATH` for the Vivado process; the source is in the branch at `synth/udevshim/udevshim.c`,
**deliberately not compiled**, because standing in for a system library is a human decision.
**Consequence: there is no Vivado post-synthesis utilisation and no post-synthesis timing closure answer in
this report.** Where a clock number appears below it is Vitis HLS's own estimate, labelled as such.

### 8c.1 io_stream is impossible — campaign 1's §8.3 diagnosis is RETRACTED

§8.3 says io_stream is blocked by *heterogeneous* activation quantization and is "reachable only by forcing
the activation quantizers homogeneous". **That is wrong.** With every data-lane quantizer forced per-tensor
(`build_model(homogeneous=True)`, `heterogeneous_axis=()`), hls4ml still raises:

```
NotImplementedError: Heterogenous quantization for activations is only supported with IOType=io_parallel
```

The message misleads. `backends/fpga/passes/hgq_proxy_model.py::ProcessFixedPointQuantizerLayer.transform`
opens with `if model.config.config['IOType'] != 'io_parallel': raise NotImplementedError(...)` — **the guard
tests IOType only and never inspects the quantizer's granularity** — and hls4ml inserts a
`FixedPointQuantizer` layer for *every* HGQ2 quantizer. So **every HGQ2 model is refused for io_stream**,
homogeneous or not. Two further blockers, each fatal alone: `backends/vivado/passes/einsum.py:55` asserts
`io_type == 'io_parallel'` and our attention pooling is a `QEinsum`; and the io_stream softmax path requires
`axis = -1`, while ours reduces over axis 1, the token axis.
Source: `quant/g6a_iostream_verdict.json`.

### 8c.2 What the homogeneity constraint costs anyway (G6a) — nothing on the ruler, +4.6 % EBOPs

Measured even though io_stream is dead, because it prices the route for any future hls4ml that lifts the
guard, and because it gives the operating point its first **seed** std. Six QAT runs, 3 seeds ×
{per-channel, per-tensor activations}, everything else identical (6-bit cap, ReLU, stage-A seed, L1T train
file, generator `eta_max` 3.0). Benched on the L1T eval file with `--eta_max 3.0`,
`--train_data robust_tagging_train_data_small_l1t.pt`, `--probe_repeats 5`, `QuantizedPMAEncoder`.

| arm | per-seed (s11 / s22 / s33) | **mean ± seed std (n=3)** | probe std | EBOPs (N=200) |
|---|---|---|---|---|
| per-channel activations (`g6a-het-*`, the campaign-1 configuration) | 0.8071 / 0.8090 / 0.8096 | **0.8086 ± 0.0013** | 0.0004–0.0007 | **3.806e8** |
| per-tensor activations (`g6a-hom-*`, what io_stream would need) | 0.8077 / 0.8089 / 0.8095 | **0.8087 ± 0.0009** | 0.0003–0.0004 | **3.981e8 (+4.6 %)** |

**Δ = +0.00009 against a seed-test threshold of 0.00274 → not separable.** So on this model HGQ2's
per-channel activation granularity buys **resources, not accuracy**. The per-channel arm's three-seed mean
**0.8086 ± 0.0013 reproduces campaign 1's single-seed `gF-bits6-relu-l1t` (0.8089 ± 0.0005)**.

> *A hypothesis I published and then falsified.* Mid-run I reported the two arms as producing bit-identical
> outputs, on the evidence that their distillation rms, max|d| and mask margin agreed to four decimals and
> that their kernels are bit-identical. The mechanism was right as far as it went — the homogeneous quantizer
> takes `i = max over channels`, so it only ever widens the range — but the conclusion was wrong. The
> distillation metrics are computed on the *calibration* sample, where no channel exceeds its per-channel
> range. On eval data the arms diverge, because the data-lane quantizers use `WRAP` overflow: a channel whose
> per-channel integer range is too narrow **wraps** where the per-tensor range does not. The bench, not the
> identity, is the claim.

### 8c.3 The fold (G6b): the token loop, and the wall that was actually there

**The fold needed no model change and no hand-written wrapper.** From the hls4ml source:
`hgq.layers.QDense.parallelization_factor` (default −1 → `prod(shape[1:-1])`, i.e. 400 copies — campaign 1's
design) → `converters/keras_v3/hgq2/_base.py::QDenseHandler` copies it into the layer config when the input
rank > 1 → `model/optimizer/passes/multi_dense.py` rewrites the Dense as a Conv1D carrying it →
`vivado_backend.init_conv1d` sets `n_partitions = out_width // pf`. **pf = 1 at N = 400 is one per-token
datapath iterated 400 times.** Evidenced in the emitted `firmware/parameters.h`, not asserted:

| layer | campaign 1 | with `--pf 1 --einsum_rf 400` |
|---|---|---|
| `phi_0`, `phi_1`, `score`, `v` | unrolled 400× | `n_partitions = 400`, `n_pixels = 1` |
| `combine` (attn·v `QEinsum`) | 204,800 multipliers | `reuse_factor = 400`, **`multiplier_limit = 512`** |
| `softmax` | pf = 32 | pf = 1 |

The einsum takes no `parallelization_factor`, but its template emits
`#pragma HLS PIPELINE II = reuse_factor` and `ALLOCATION mul limit = total/reuse_factor`, so `ReuseFactor` is
its fold knob. Conversion at N=400 takes **6 s** (campaign 1: converted, then failed to compile in 45 min).
**C-simulation vs Keras at N=16: max|d| = 0.000e+00, 100.00 % of elements exact** — the fold changes nothing
numerically, by construction.

#### The real N=400 wall was hls4ml's quantizer code generation, not the arithmetic
Vitis HLS on the folded N=400 design, after 37 min of source analysis:
`WARNING: [HLS 200-1995] There were 56,931,767 instructions in the design after the 'Compile/Link' phase`
(the tool's own warning threshold is 50 M). Its design-size report says where:

| function | instructions |
|---|---|
| `phi_0` / `phi_1` / `score` / `v` (folded Denses) | 29,888 / 257,888 / 257,888 / 257,888 |
| **`nrm0_iq` / `phi_act0_iq` / `nrm1_iq` / `phi_act1_iq` / `score_iq` (quantizers)** | **7.76 M / 7.78 M / 7.78 M / 7.78 M / 6.15 M** |

`backends/fpga/passes/hgq_proxy_model.py::generate_mask_fn` **broadcasts the (k, b, i) masks to the full
tensor shape and emits one line of C++ per element** — 51,200 lines for a `[400,128]` tensor —
unconditionally, whether or not the mask varies. **That, not the multipliers, is what campaign 1's
"io_parallel does not compile at N=400" actually was.**

#### Patch P3, and it is exact
Our data-lane quantizers are homogeneous over `(batch, token)` by construction — they must be, one datapath
processes every token — so every token repeats the same mask. P3 (`quant/hls4ml_patch.py`) emits the mask
once inside a loop over tokens. It folds over the **largest leading block the mask is genuinely constant
across** (the einsum's `[400,8,4]` and `[400,8,16]` inputs vary over the head axis but not over tokens, which
a naive last-axis split would have missed), it **verifies** that constancy rather than assuming it, and it
falls back to hls4ml's own generator verbatim where it does not hold.

| | generated `nnet_code_gen.h` | instructions after Compile/Link | source analysis |
|---|---|---|---|
| hls4ml 1.3.0 as shipped, N=400 | 397,699 lines | 56,931,767 | 2,245 s |
| **+ P3, N=400** | **8,744 lines (45×)** | **1,234,610 (46×)** | **235 s (9.5×)** |
| + P3, N=16 | — | 364,842 | 88 s |

**Verified bit-exact: N=16 csim vs Keras max|d| = 0.000e+00, 100 % of elements** (`quant/g6_g6b_N16_pf1_foldq.json`).
Same arithmetic, smaller source.

### 8c.4 The design now compiles and is exact at N=400

| | campaign 1 | campaign 2 (folded + P3) |
|---|---|---|
| hls4ml conversion at N=400 | succeeded | 6 s |
| g++ compile of the N=400 io_parallel design | **killed at 2,710 s, unfinished** | **183 s, finished** |
| C-simulation vs Keras at N=400 (16 events) | never obtained | **max&#124;d&#124; = 0.000e+00, 100.00 % of elements** |
| C-simulation vs Keras at N=16 | max&#124;d&#124; = 0.000e+00 | max&#124;d&#124; = 0.000e+00 (unchanged by the fold) |

Design on every row: G6b, `xcvu13p-flga2577-2-e`, 5 ns (200 MHz), 6-bit cap + ReLU (`gF-bits6-relu-l1t`),
io_parallel, `parallelization_factor` 1, einsum `ReuseFactor` 400.
Sources: `quant/g6_g6b_N400_pf1_foldq_csim.json`, `quant/g6_g6b_N16_pf1_foldq.json`.

**Campaign 1's open item 2 — "buildability at 400 tokens is unproven" — is now half closed.** The design
builds and the arithmetic is exact at the token count the eval set actually uses. What is still unproven is
the hardware half, below.

### 8c.5 Synthesis (G7): NO resource, latency or II numbers. What happened instead.

**There are no LUT / FF / DSP / BRAM figures, no latency and no initiation interval in this report.** Three
Vitis HLS C-synthesis runs on `xcvu13p-flga2577-2-e` at 5 ns were started and none completed:

| run | started | stopped | elapsed | phase reached |
|---|---|---|---|---|
| **N=400, folded (P3)** — the deliverable | 05:09 | **11:10** | **6 h 00 m** | Compile/Link done (1,234,610 instructions), **stuck in Unroll/Inline for 5 h 55 m** |
| N=16 anchor, folded (P3) | 05:24 | 08:10 | 2 h 46 m | same |
| N=1 calibration | 07:16 | 08:10 | 54 m | same |
| N=400, hls4ml as shipped | 04:02 | 05:09 | 1 h 07 m | same (stopped to free the machine for the folded build) |
| N=16, hls4ml as shipped | 04:53 | 05:24 | 31 m | same (replaced by the folded anchor) |

The N=400 run was given **twice** its 3 h cap by planner ruling and still did not leave the Unroll/Inline
phase. **It did not diverge or error**: `clang` held 99 % CPU for the whole six hours and RSS grew smoothly
from 1.27 to 2.66 GB. It was stopped, not killed by a failure. Partial artefacts:
`synth/FINAL_partial_N400_pf1_csynth_1110.log`, `synth/FINAL_partial_N400_pf1_design_size_1110.rpt`, and
`synth/partial_*` / `synth/N*_unpatched_*` for the other four. Outcome recorded in
`quant/g7_csynth_outcome.json` with the exact command to reproduce it.

**Where the time goes, attributed rather than guessed.** The N=1 run is the control: with one token there is
nothing for P3 to fold, so it falls back to hls4ml's own generator and emits **byte-for-byte the same 2,689
lines with and without P3** (checked against a `WPG_NO_P3=1` build). That design sat in the same phase for
52 minutes. So the long phase is **neither P3 nor the token count**. With the token loop folded the design is
dominated by its **N-independent** parts — 383,350 instructions at N=1 against 364,842 at N=16 — chiefly the
two 512-wide tail quantizers `norm_pooled_iq` and `bottleneck_iq` at ~77.8 k instructions each, because
hls4ml spends ~152 LLVM instructions per `ap_fixed` conversion and the tail alone has 1,024 of them. It is a
property of **this architecture through hls4ml's io_parallel path in Vitis HLS 2024.2 on this machine**, and
it follows that neither waiting longer nor shrinking N is likely to help.

**What P3 did fix is the phase before it, and there the effect is unambiguous** (§8c.3): 46× fewer
instructions after Compile/Link, 9.5× faster source analysis, and a g++ compile that finishes instead of
being killed. P3 was never a claim about Unroll/Inline.

> **The missing numbers, stated plainly.** There is **no LUT, FF, DSP, BRAM or URAM figure, no latency in
> cycles or microseconds, and no initiation interval** anywhere in this report, from any tool. Nor is there a
> C/RTL co-simulation, because it needs the C-synthesis it would be compared against — the 16 real L1T events
> and their Keras latents are written and waiting in both N=400 projects' `tb_data/`. Nor is there any Vivado
> result (§8c.0). The cycle counts in §8c.6b are what the emitted structure implies, not what a tool
> measured, and they are labelled as such wherever they appear.

### 8c.6 Resource proxy at the operating point, N=400 — a proxy, not a synthesis result

Since no synthesis completed, the only cost figure available is the one campaign 1 used: EBOPs, the
bit-weighted MAC count, which is the HGQ literature's LUT proxy. For **`gF-bits6-relu-l1t`, the chosen 6-bit
ReLU operating point, at N=400** (`quant/g4_resource_gF-bits6-relu-l1t_N400.json`):

| layer | EBOPs | multiplies |
|---|---|---|
| `phi_1` | 3.149e8 | 6,553,600 |
| `v` | 3.149e8 | 6,553,600 |
| `score` (fused q·k, R1) | 7.872e7 | 1,638,400 |
| `phi_0` | 2.703e7 | 716,800 |
| `combine` (attn·v) | 1.145e7 | 204,800 |
| everything else | 1.28e7 | 68,608 |
| **TOTAL** | **7.605e8** | **15,735,808** |

`phi_1` and `v` are **83 %** of the cost, as they were for the 8-bit model. Against campaign 1's ~8-bit row
(1.010e9 EBOPs at N=400) the 6-bit ReLU operating point is **25 % cheaper on the proxy at no measured
accuracy cost**. It is a proxy and it is not a substitute for the synthesis that did not finish.

### 8c.6b "Does it fit?" — the proxy answer, with the number that actually matters

Campaign 1's **15,735,808** is the multiply COUNT PER EVENT of the *unfolded* design, where every multiply is
its own piece of hardware. That is not the hardware count of the folded design, and quoting it as one would
overstate the cost by two orders of magnitude. The multipliers **instantiated** are read from the emitted
`firmware/parameters.h` (`quant/fit_estimate.py`, `quant/g7_fit_g6b_N400_pf1.json`):

| block | multipliers instantiated | note |
|---|---|---|
| `phi_0` 14→128 | 1,792 | per-token datapath: built **once**, |
| `phi_1` 128→128 | 16,384 | iterated 400 times |
| `score` 128→32 (fused q·k, R1) | 4,096 | |
| `v` 128→128 | 16,384 | |
| — per-token subtotal — | **38,656** | |
| `combine` (attn·v einsum, `ReuseFactor` 400) | 512 | once per event |
| `out_proj` (4 seeds, `n_partitions = 1`, all four in parallel) | **65,536** | once per event |
| `bottleneck` 512→6 | 3,072 | once per event |
| **TOTAL INSTANTIATED** | **107,776** | |

**The fold is 146× on hardware: 15,735,808 → 107,776.** That is the difference between a design that cannot
exist and one that is the same order of magnitude as the part.

Against `xcvu13p-flga2577-2-e` (**12,288 DSP48E2, 1,728,000 LUTs**):
* 107,776 multipliers is **8.8× the DSP count**, so most of them must be LUT logic in any case. That is the
  normal outcome for 7-bit × 8-bit operands — a DSP48E2 is 27×18 and is wasted on them — not a problem.
* At **25–60 LUTs per small fixed-point multiplier** (a deliberate range; a point estimate here would be
  false precision) that is **2.7 M – 6.5 M LUTs against a 1.73 M budget**. **As configured, `pf = 1` probably
  does not fit.**
* **`out_proj` alone is 61 % of the multipliers**, and for no good reason: its input is `[4, 128]`, so
  `out_width = 4` and hls4ml gives it `n_partitions = 1` — all four seed positions unrolled. Folding it the
  same way as the token loop (`ParallelizationFactor = 1`, a one-line config change, not applied in the
  synthesised project) takes the total to **58,624 multipliers → 1.5 M – 3.5 M LUTs**, straddling the budget.
* Beyond that, `ReuseFactor` on `phi_1` and `v` is the remaining dial: `rf = 4` takes each from 16,384 to
  4,096 multipliers, at 4× the token-pass cycles.

### The fitting configuration — built, bit-exact, and NOT synthesised

Both levers applied (`out_proj` `ParallelizationFactor` 1, `ReuseFactor` 4 on `phi_1` and `v`), built and
C-simulated. Same weights, same arithmetic, same part and clock; only the unrolling differs.

| | `g6b_N400_pf1` | **`g6b_N400_fit`** (recommended) |
|---|---|---|
| `phi_0` / `phi_1` / `score` / `v` multipliers | 1,792 / 16,384 / 4,096 / 16,384 | 1,792 / **4,096** / 4,096 / **4,096** |
| `out_proj` | 65,536 (`n_partitions` 1) | **16,384** (`n_partitions` 4) |
| `combine` einsum / `bottleneck` | 512 / 3,072 | 512 / 3,072 |
| **multipliers instantiated** | **107,776** | **34,048** |
| fold vs the unfolded design | 146× | **462×** |
| × the part's 12,288 DSP48s | 8.8× | 2.8× |
| LUT proxy (25–60 LUT/mult) vs 1,728,000 | 2.69 M – 6.47 M — **over** | **0.85 M – 2.04 M — straddles** |
| token pass at 200 MHz | 400 cycles = **2.00 µs** | 1,600 cycles = **8.00 µs** |
| **C-simulation vs Keras at N=400** | **max&#124;d&#124; = 0.000e+00, 100 %** | **max&#124;d&#124; = 0.000e+00, 100 %** |
| C-synthesis | attempted, did not complete (§8c.5) | **not attempted** |

Folding changes no number, and for the fitting configuration that is now **measured, not assumed**
(`quant/g6_g6b_N400_fit.json`). Configurations and their emitted `parameters.h` are committed under
`quant/hls_config/`; `quant/fit_estimate.py` reads the multiplier counts back out of them.

**Latency and throughput.** The token pass bounds the initiation interval: **400 cycles = 2.00 µs** at
`pf = 1`, **1,600 cycles = 8.00 µs** for the fitting configuration. Total latency adds the depth of the
dataflow chain (the 400-wide softmax reduction and the tail), which is not knowable without the synthesis.

**Everything in this subsection except the multiplier counts is a proxy.** The multiplier counts are exact
and read from the emitted firmware; the LUT figures are not a synthesis result and must not be quoted as one.

### 8c.6c Time and size, every run

All on the same 8-core box; wall clock, not CPU. Load average was 15-20 until ~05:30 (five other
packages training) and 2.8-4.8 afterwards, so the early rows are inflated relative to the late ones.

#### Wall clock

| job | design | started | ended | elapsed | outcome |
|---|---|---|---|---|---|
| Vitis HLS licence probe | 8-tap `ap_fixed` dot product | 03:39:41 | 03:51:44 | **12 m 03 s** | **completed** — 79 s of `csynth` inside it, the rest device-data load |
| Vivado 2024.2 licence probe | 1-line multiplier | 03:52:25 | 04:00:17 | 7 m 52 s | **crashed** (SIGABRT, udev) |
| Vivado 2025.2 licence probe | same | 04:05:23 | 04:18:25 | 13 m 02 s | **crashed** (SIGSEGV, udev) |
| Vivado 2024.2 + `LD_PRELOAD` shim | same | 04:20:50 | 04:21:25 | 35 s | **crashed** — preload bypassed (`dlopen`) |
| `csynth` N=400, hls4ml as shipped | G6b, pf 1 | 04:02:15 | 05:09:10 | 1 h 06 m 55 s | stopped to free the machine |
| `csynth` N=16, hls4ml as shipped | G6b, pf 1 | 04:52:59 | 05:24:46 | 31 m 47 s | stopped; anchor swapped to the folded build |
| **`csynth` N=400, + P3** | **the deliverable** | 05:09:18 | 11:10:36 | **6 h 01 m 18 s** | **stopped at the cap, still in Unroll/Inline** |
| `csynth` N=16, + P3 | anchor | 05:24:49 | ~08:10 | ~2 h 45 m | stopped |
| `csynth` N=1, (P3 a no-op here) | calibration control | 07:16:06 | 08:10:42 | 54 m 36 s | stopped |

Not one C-synthesis of this architecture completed, at any token count, patched or not.

#### Time inside the toolchain, per design

| design | hls4ml convert | Vitis source analysis | g++ `csim` compile | csim result |
|---|---|---|---|---|
| N=400, as shipped | 6 s | **2,245 s** | **killed at 2,710 s, unfinished** (campaign 1) | never obtained |
| **N=400, + P3** | 2 s | **235 s (9.6×)** | **183 s (finished)** | **max&#124;d&#124; 0.000e+00, 100 %** |
| N=400, fitting config | 1 s | not attempted | 183 s | **max&#124;d&#124; 0.000e+00, 100 %** |
| N=16, as shipped | 3 s | 205 s | 239 s | max&#124;d&#124; 0.000e+00, 100 % |
| N=16, + P3 | 1 s | 88 s | 48 s | max&#124;d&#124; 0.000e+00, 100 % |
| N=1 | 1 s | 110 s | — | — |

#### Generated source, and instructions after Compile/Link

| design | `nnet_code_gen.h` lines | `firmware/` on disk | instructions after Compile/Link |
|---|---|---|---|
| N=400, as shipped | **397,699** | 30 M | **56,931,767** (over Vitis's own 50 M warning threshold) |
| **N=400, + P3** | **8,744 (45×)** | 7.2 M | **1,234,610 (46×)** |
| N=400, fitting config | 8,753 | 11 M | not synthesised |
| N=16, as shipped | 17,539 | 6.1 M | 2,510,511 |
| N=16, + P3 | 2,600 (6.7×) | 3.5 M | 364,842 |
| N=1 (P3 a no-op: byte-identical) | 2,689 | 2.7 M | 383,350 |

Note the last two rows: **N=1 is *larger* than N=16** once the token loop is folded, which is the whole
attribution in one line — the design is dominated by its N-independent parts.

#### Where the instructions are, at N=400: what P3 touched and what it did not

| function | as shipped | + P3 | ratio |
|---|---|---|---|
| `phi_0_iq` | 738,801 | 1,908 | **387×** |
| `nrm0_iq` | 7,764,001 | 19,933 | **389×** |
| `phi_act0_iq` | 7,778,401 | 19,969 | **390×** |
| `nrm1_iq` | 7,781,601 | 19,977 | **389×** |
| `phi_act1_iq` | 7,781,601 | 19,977 | **389×** |
| `score_iq` | 6,145,601 | 15,887 | **387×** |
| `quantizer` (mask input) | 1,940,001 | 4,989 | **389×** |
| `v_iq` | 6,145,601 | 15,887 | **387×** |
| `quantizer_2` (attn → einsum) | 1,932,001 | 4,969 | **389×** |
| `quantizer_3` (v → einsum) | 7,775,201 | 19,961 | **389×** |
| `out_proj_iq` | 77,769 | 19,965 | 3.9× (rank 2: folds over the 4 seeds only) |
| **`norm_pooled_iq`** | **77,821** | **77,821** | **1× — untouched** |
| **`bottleneck_iq`** | **77,729** | **77,729** | **1× — untouched** |
| `phi_0` / `phi_1` / `score` / `v` convolutions | 29,888 / 257,888 / 257,888 / 257,888 | unchanged | 1× (already folded by `pf`) |
| `softmax` / `einsum` / `out_proj` conv / `bottleneck` dense | 107,969 / 698 / 866 / 595 | unchanged | 1× |

Every token-axis quantizer folds ~390×. The two **rank-1, 512-wide tail quantizers do not fold at all** —
there is no repeated axis in them — and after P3 they are **the two largest functions in the design**, at
12.6 % of its instructions between them. That is the attribution, and it is why the next change is the tail
quantization rather than anything to do with tokens (§9 item 1).

#### Hardware size: multipliers instantiated

| configuration | `phi_0` | `phi_1` | `score` | `v` | einsum | `out_proj` | `bottleneck` | **total** | vs unfolded |
|---|---|---|---|---|---|---|---|---|---|
| unfolded (campaign 1) | 716,800 | 6,553,600 | 1,638,400 | 6,553,600 | 204,800 | 65,536 | 3,072 | **15,735,808** | 1× |
| `pf = 1` | 1,792 | 16,384 | 4,096 | 16,384 | 512 | 65,536 | 3,072 | **107,776** | **146×** |
| **fitting config** | 1,792 | **4,096** | 4,096 | **4,096** | 512 | **16,384** | 3,072 | **34,048** | **462×** |

Part budget for scale: `xcvu13p-flga2577-2-e` has **12,288 DSP48E2** and **1,728,000 LUTs**.

### 8c.7 Operational notes for anyone repeating this on a shared box
* **Killing `vitis_hls` does not kill its `clang` children.** Both runs I stopped left orphaned `clang`
  processes at 95 % CPU for 40–60 minutes, invisible in the Vitis logs. Kill the process group, or check
  `ps` afterwards.
* `set_part xcvu13p-flga2577-2-e` alone costs ~10 minutes of device-data load per invocation on this box; a
  trivial 8-tap dot product took 79 s of synthesis inside 727 s of wall clock.
* hls4ml 1.3.0 emits `config_array_partition -maximum_size 4096`, which **Vitis 2024.2 rejects**
  (`ERROR: [HLS 200-101] config_array_partition: Unknown option '-maximum_size'`; it wants
  `syn.array_partition.complete_threshold`). The build continues, so the intended partition threshold is
  simply never applied. Version skew worth knowing about.

## 9. What does NOT work yet

1. **No completed synthesis.** Vitis HLS runs and needs no licence, but no C-synthesis of this
   architecture has finished (§8c.5); Vivado crashes before the licence server (§8c.0). The single most promising change is **the N-independent tail quantizers**: `norm_pooled_iq` and
   `bottleneck_iq` are ~77.8 k LLVM instructions each and dominate the phase Vitis is stuck in.
   Quantizing *before* the flatten, or giving the 512-wide tail a per-tensor quantizer instead of
   a per-channel one, would cut that by ~512×. That is the next thing to try, and it is a model
   change with an accuracy cost that would have to be measured, not assumed.
2. ~~**io_stream is unreachable** without giving up per-channel activation quantization (§8.3), and the
   io_parallel design it forces **does not compile at N=400** on this box (time-boxed at 2,710 s). The
   bit-exactness result stands at N=16; buildability at 400 tokens is unproven.~~
   **SUPERSEDED (campaign 2), both halves.** *First half wrong:* giving up per-channel quantization does
   **not** unlock io_stream — hls4ml's guard tests IOType only, so every HGQ2 model is refused, and the
   `QEinsum` in the attention pooling is refused independently (§8c.1). io_stream is unreachable, but for
   different reasons than stated. *Second half now false:* with the token loop folded and P3 applied the
   N=400 design **compiles in 183 s** and C-simulates **bit-exact** (§8c.4). Buildability at 400 tokens is
   proven; only the hardware cost is still unmeasured (§8c.5).
3. **The preprocessor is not quantized.** The per-event `mean(pt)` reduction stays in float on the host (§1).
4. **The 14→10 channel fold is verified and quantified but not yet applied** to the benched model; the benched
   model keeps all 14 inputs so the float/quantized comparison stays controlled. Applying it is exact.
5. **arm 2 (the original contrastive + consistency objective) is not yet run.** Everything here is arm 1,
   distillation to the float teacher, which by construction can only *match* the float row, never beat it —
   it inherits the teacher's drift rather than reducing it. "Matches float" must not be read as "quantization
   is free" without arm 2.
6. ~~**Two untried routes to a buildable 400-token design.**~~ **BOTH TRIED (campaign 2).**
   (a) io_stream with homogeneous activation quantizers: **impossible**, and the accuracy cost was measured
   anyway — a 3-seed tie at +4.6 % EBOPs (§8c.1, §8c.2). (b) Fold the token loop: **done, and it is the
   design** — one `parallelization_factor`, no model change, 146× fewer instantiated multipliers, C-simulation
   bit-exact at N=400 (§8c.3, §8c.4, §8c.6b).

7. **BatchNorm batch statistics include dead rows** (§6). Now *bounded* rather than open: excluding them
   makes the bench worse (0.8044 vs 0.8078), so the swap cost is the swap, not the contamination. The
   by-product — better teacher agreement with worse bench — is the sharper finding and it undercuts arm 1's
   objective, see item 5.

8. **No Vivado at all.** Out-of-context logic synthesis, post-synthesis utilisation and the "does 200 MHz
   close" answer are all missing, because Vivado 2024.2 and 2025.2 both crash inside the FlexLM udev device
   scan before reaching the licence server on a container with no `/run/udev` (§8c.0). The workaround —
   a no-op stand-in `libudev.so.1` on `LD_LIBRARY_PATH` for the Vivado process — is written but deliberately
   **not built**; it is a human decision, not one to make in passing.
9. **C/RTL co-simulation has not run.** It needs the C-synthesis it co-simulates against. The 16 real L1T
   events and their Keras latents are written and waiting in both N=400 projects' `tb_data/`.
10. **The fitting configuration is unsynthesised.** `g6b_N400_fit` (34,048 multipliers, §8c.6b) is the
   recommended operating point on the proxy and is bit-exact in C-simulation, but no synthesis has been
   attempted on it, so its LUT/FF/DSP/BRAM, latency and II are unknown like everything else here.
