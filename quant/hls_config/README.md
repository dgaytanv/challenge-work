# Committed hls4ml configurations for the two G6b operating points

Both are the same weights (`gF-bits6-relu-l1t`, 6-bit cap + ReLU), the same arithmetic and the same
`xcvu13p-flga2577-2-e` at 5 ns (200 MHz). They differ only in how much of the design is unrolled, and both
are **bit-exact against Keras in C-simulation at N=400** (max|d| = 0.000e+00, 100 % of elements) — folding
changes no number, which is the whole point of it.

| | `g6b_N400_pf1` | `g6b_N400_fit` (recommended) |
|---|---|---|
| token loop | `ParallelizationFactor` 1 | same |
| `phi_1`, `v` | `ReuseFactor` 1 | **`ReuseFactor` 4** |
| `out_proj` | default (`n_partitions` 1: all 4 seeds unrolled) | **`ParallelizationFactor` 1 (`n_partitions` 4)** |
| einsum `combine` | `ReuseFactor` 400 | same |
| **multipliers instantiated** | **107,776** | **34,048** |
| fold vs the unfolded design | 146× | **462×** |
| token pass | 400 cycles = 2.00 µs | 1,600 cycles = 8.00 µs |
| LUT proxy (25–60 LUT/mult) vs 1,728,000 | 2.69 M – 6.47 M — **over** | 0.85 M – 2.04 M — **straddles** |
| C-synthesis | attempted, did not complete | **not attempted** |

Reproduce with `quant/g6_build.py`:

```
--pf 1 --einsum_rf 400                                  # g6b_N400_pf1
--pf 1 --einsum_rf 400 --out_proj_pf 1 --dense_rf 4     # g6b_N400_fit
```

`parameters.h` is committed for each so the multiplier counts can be checked without rebuilding;
`quant/fit_estimate.py` reads them and is where the table above comes from.
