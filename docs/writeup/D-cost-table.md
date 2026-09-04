# Cost of the encoders (WP-D)

## Parameters and step time

Batch 128, N=200 candidates, fwd+bwd+Adam, A10. Parameter counts are exact. Step times are the
**median of 25 timed steps** measured back to back in one process; medians rather than means
because the A10 was shared with other instances' training runs throughout and the mean is dominated
by contention spikes (the Deep Sets max of 172 ms against a min of 3.3 ms is one such spike, not the
model). Ratios measured in the same process are meaningful; absolute values are an upper bound.

| encoder | params | median ms/step | min | max |
|---|---|---|---|---|
| `TransformerEncoder` (4 blocks) | 2.376 M | 208.8 | 130.1 | 273.8 |
| **`DeepSetsEncoder`** | **0.053 M** | **9.8** | 3.3 | 172.2 |
| `PMAEncoder(num_layers=0)` | 0.090 M | 11.9 | 5.3 | 24.0 |
| `PMAEncoder(num_layers=2)` | 1.276 M | 135.2 | 64.1 | 176.6 |

Deep Sets is **45x smaller** and roughly **20x faster per step** than the transformer under identical
shared-GPU conditions. On an idle GPU measured earlier in the session the gap was wider (8.7 ms
against 312.8 ms, 36x); the honest statement is "more than an order of magnitude", not a precise
factor, because the machine was never quiet enough to pin it down.

The scaling reason is structural: attention is O(batch x heads x N^2) and a per-token MLP is
O(batch x N x embed). At N=200 that is already a factor of ~10 in the dominant term, and the eval
set has N=400, where it is ~20.

`PMAEncoder(num_layers=0)` costs almost the same as Deep Sets, confirming that the expense is the
self-attention stack and not the pooling readout.

## Inference memory

Measured at `eval.py`'s exact embedding shape (batch 1024, 400 candidates, `no_grad`), Deep Sets
peaks at **448 MiB allocated / 514 MiB reserved**. The transformer at the same shape needs about
8 GB, dominated by the `[1024, 8, 401, 401]` attention tensor. This is why `accept.sh`'s default
8000 MiB preflight is wrong for a set encoder and `ACCEPT_NEED_MIB=3000` is the appropriate
(still conservative) override.

## The shipped encoder's cost

The submission is `PMAEncoder(num_layers=0)`: 0.090 M parameters, 11.9 ms median per training step
— **26x fewer parameters and roughly 18x faster per step than the stock transformer**, while
scoring 0.8260 against its 0.7734. With `num_layers=0` there is no self-attention stack at all;
the only attention is the pooling readout over 4 learned seed queries, which is O(N) in candidates.
That it costs almost exactly what Deep Sets costs (11.9 ms against 9.8 ms) confirms the expense in
the original model was the self-attention stack rather than the readout.

## What this bought in practice

Deep Sets full training runs (25 epochs, batch 256, 80k events) took ~15 minutes each and were
approved to run **lock-free**, because at ~600 MiB and largely dataloader-bound they neither need
nor can saturate the shared A10. That is what allowed WP-D to carry the extra arms — the clean/aug
pair, the pooling variants, the two-view and JSD arms, and the 400-candidate arm — while the
transformer instances queued on the main GPU lock.

The corollary for the ruling: if Deep Sets and a transformer variant land within the probe noise
floor of each other, Deep Sets is far cheaper to retrain, ship and reproduce, and its
deletion-equivalence property is exact rather than approximate.
