# How to plot example events from WP-B's generator (tested recipes)

Both verified on real data. Import: `from embedding.degradation import Degradation`,
run with `PYTHONPATH=$HOME/rt-<your-wp>/src`. `x` is RAW `[B, N, 7]`.

## Recipe A — one family at an EXACT severity (best for a severity-ladder figure)
The public train-mode path samples `s ~ U(0, s_max)`, so it cannot give you a fixed severity.
For a clean "same event at s = 0.2/0.4/0.6/0.8" figure, call the family helpers directly:

```python
d = Degradation(severity=None)
eta, phi = x[..., 1], x[..., 2]
s = torch.full((x.shape[0],), 0.6)          # exact severity, per event
rect   = d._drop_rect(eta, phi, s, None)
wedge  = d._drop_bands(phi, s, 2*math.pi, 0.1, 0.4, 0.6, 1.0, 16, True,  None)
strip  = d._drop_bands(eta, s, 10.0,      1.0, 2.5, 0.5, 1.0,  8, False, None)
cells  = d._drop_cells_multi(eta, phi, s, (0.25, 0.5, 1.0), None)
towers = d._drop_cells(eta, phi, s, 0.1, None, p_lo=1.0, p_hi=1.0, compensate=False)
# each is a [B, N] bool mask: True = dropped. Apply with:
x_deg = torch.where((mask & (x[..., 0] > 0)).unsqueeze(-1), torch.zeros_like(x), x)
```
Measured dropped fractions (8 events): at s=0.3 rect 0.221 / wedge 0.263 / strip 0.234 / cells 0.309 /
towers 0.300; at s=0.6 rect 0.444 / wedge 0.416 / strip 0.381 / cells 0.603 / towers 0.613. Rect and the
band families undershoot slightly at high `s` because regions overlap; that is expected.

## Recipe B — public API, one family, every confound switched off
```python
d = Degradation(
    severity=None, families=("rect",),   # or wedge / strip / cells / towers
    p_clean=0.0,        # never skip the event
    s_max=0.6,          # severity ~ U(0, 0.6)
    curriculum=False,   # no warm-up ramp; otherwise s_max ramps over 600 calls
    rotate_phi=False, reflect_eta=False,   # keep eta/phi comparable to the clean panel
    p_charged_only=0.0, p_neutral_only=0.0, p_pt_scale=0.0,
)
x_deg = d(x)
```

## Two traps for a plotting script
1. **`rotate_phi`/`reflect_eta` default to ON.** Leave them on and the degraded panel is a *rotated and
   reflected* copy of the clean one, so the dead region will not line up visually with the missing
   candidates. Switch both off for any before/after figure.
2. **`p_pt_scale` defaults to 0.10.** That mode leaves rows ALIVE with reduced pt rather than zeroing them,
   so ~10% of events will show a region that is dimmed rather than empty. Correct behaviour, confusing in a
   figure. Set `p_pt_scale=0.0`.

With both off, surviving rows are bit-identical to the clean event (verified), so a scatter of
`pt > 0` candidates before/after differs only by the removed region.

## Suggested figure
One row per family (rect, wedge, strip, cells, towers), columns = clean + s in {0.2, 0.4, 0.6, 0.8},
scatter in (eta, phi) with marker size ~ log pt. eta spans [-5, 5], phi spans (-pi, pi].
