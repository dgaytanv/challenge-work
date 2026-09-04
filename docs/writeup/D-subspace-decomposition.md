# Where the degradation drift actually goes (WP-D)

Hypothesis proposed by WP-C; this is the measurement. Tool: `tests/latent_subspace_decomposition.py`
on branch `wp-d`. Numbers below are severity 0.5, 20000 eval events, WP-B's generator.

## The question

`d-deepsets-clean` reaches mean_area 0.8017 while its latent moves ~0.86 of the population spread
under degradation. If that motion were harmful the AUC would collapse. So where does it go?

## Method

Train the grader's own probe (`EvalMLP`, the recipe from `eval.py`) on clean latents. For each event
linearise the probe at `z_c`: `g = d(class margin)/dz`. To first order that single direction is the
only thing the probe's decision responds to. Split `d = z_d - z_c` into the component along `g_hat`
and the component orthogonal to it.

Magnitudes alone can mislead, so the same split is also done operationally: feed the probe
`z_c + d_par` and `z_c + d_perp` separately and see which one actually costs AUC. That check is
**not** first-order, so agreement between the two readings is real evidence rather than a restatement.

## Result

| | d-deepsets-clean | d-deepsets-aug |
|---|---|---|
| population spread `\|\|z_c - mu\|\|` | 1.2953 | 1.9489 |
| total drift / spread | 0.856 | 0.726 |
| **probe-relevant `d_par` / spread** | **0.211** | **0.261** |
| null-space `d_perp` / spread | 0.810 | 0.653 |
| harmful fraction, squared magnitude | 0.112 | 0.201 |
| AUC clean | 0.8868 | 0.8915 |
| AUC on `z_c + d_perp` | 0.8860 | 0.8869 |
| AUC on `z_c + d_par` | 0.8556 | 0.8677 |
| AUC on full degraded `z_d` | 0.8560 | 0.8628 |

On the clean-trained model the null-space component is **3.8x larger** than the probe-relevant one
and costs **0.0008** AUC. The small probe-relevant component costs **0.0312** and reproduces
essentially the entire degradation cost. The drift that matters is a small, nearly orthogonal
fraction of the drift that exists.

**Consequence for the consistency loss.** An MSE on raw displacement weights by squared magnitude,
so about **89%** of its gradient budget goes on motion that costs ~2.5% of the AUC loss. This is
what motivated the JSD logit-consistency arm: the four-class classifier's directions are a
label-legal proxy for the probe directions, available at training time when the probe is not.

**Augmentation does not fix it.** WP-B's generator improves mean_area by +0.0147, but `d_par/spread`
goes *up* (0.211 -> 0.261) while `d_perp/spread` falls (0.810 -> 0.653) and the population spread
grows 50%. So augmentation removed the harmless part and separated the classes relative to the
drift; it did not make the latent more invariant in the direction that matters.

## Rigid shift vs per-event motion

Tested because it was proposed that most of the displacement is a rigid translation of the whole
cloud, which an MSE term penalises and a cosine term cannot. On `d-deepsets-aug`:

| component | size | AUC when fed alone |
|---|---|---|
| common (rigid) `mean(d)` | 0.503 x spread, 41.3% by squared magnitude | 0.8823 (costs 0.0099) |
| per-event residual `d - mean(d)` | 0.525 x spread | 0.8569 (costs 0.0353) |
| full degraded | 0.726 x spread | 0.8614 |

**These numbers are at severity 0.5 only, and the rigid fraction is strongly severity-dependent.**
WP-A measured the shared-shift fraction per (family, severity) on `d-deepsets-clean` and it rises
from 0.18 at s=0.2 to 0.99 at s=1.0. So "the drift is 41% rigid" is a statement about the middle of
the severity range, not about the model: at high severity the drift is almost entirely a rigid
shift, and at low severity almost none of it is. Any single-number rigid fraction is a cell-mean
over the severity grid, and two models' means are only comparable when their AUC-vs-severity curves
have similar shape — which is exactly what a successful two-view arm changes. A severity sweep of
this decomposition is queued to replace the single-point number.

With that qualification: at s=0.5 the drift is 41% rigid, so "most of the displacement is a rigid
shift" is not true there; the rigid half is the **less** harmful half by 3.6x; and the per-event
residual alone is worse than the full degraded latent, so the rigid part mildly compensates.
Whether the rigid component is still the less harmful one at s=1.0, where it is 99% of the drift,
is not yet measured and should not be assumed. Separately, WP-C's `consistency_terms` centres both
views by the *same* detached clean batch mean, so a common translation survives centring and the
centred cosine does respond to it.

This was measured on `DeepSetsEncoder`, whose latent geometry differs sharply from the transformer's
(offset-to-spread 3.9x versus 30x), so the premise may still hold there. It should be re-measured on
the transformer before being used as a reason for anything.

## Caveats

- The probe direction comes from a probe fit on this model's clean latents, not the grader's own
  refit. `g` is a faithful proxy for the operator, not the operator.
- The split is first-order; the operational AUC check is not, and they agree.
- Across two probe inits the harmful fraction read 0.282 then 0.268. The conclusion is stable, the
  third digit is not.

## Related: the latent offset pathology

`tests/latent_offset_control.py` reports the shuffled-pair control that started this thread. On
`d-deepsets-clean`, raw-latent cosine reads matched 0.9762 against shuffled 0.9353, because the
latent carries a common offset 3.9x the per-event spread. Centred, the same checkpoint reads
matched 0.5974 against shuffled 0.0105. WP-C's transformer was 30x, where matched 0.999933 against
shuffled 0.999557 made a raw cosine consistency term carry almost no gradient at all. The control
has to be run per encoder: the pathology is general to the setup, its severity is not.
