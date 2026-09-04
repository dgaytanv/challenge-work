"""Held-out corruption families: the honesty check.

These shapes appear in NEITHER the default benchmark (rect / wedge / strip / towers / cells)
NOR in WP-B's training generator, which draws from the same five shape classes. If a model's
area holds up here as well as it does on the default families, the gain is real robustness
rather than a fit to our own dead maps.

Families:
  ellipse   rotated ellipses in eta-phi -- curved boundaries, no axis-aligned edges
  annulus   rings around a random centre -- dead region with a live hole inside it
  diagonal  tilted bands, phi correlated with eta -- the one structure no axis-aligned family makes

Same contract as BenchDegradation: seeded, nested in severity (the dead set at s=0.4 is a
subset of the dead set at s=0.6), severity 0 returns x unchanged, forward(x) -> x.

Standalone self-check:
  python ~/hackathon-shared/bench/heldout_degradation.py
"""
import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_degradation import (  # noqa: E402  (path juggling must come first)
    CELL,
    ETA_MAX,
    ETA_MIN,
    N_ETA,
    N_PHI,
    PHI_CELL,
    BenchDegradation,
)

HELDOUT_FAMILIES = ("ellipse", "annulus", "diagonal")
MAX_SHAPES = 500


def _cell_centres(device=None):
    eta_c = ETA_MIN + (torch.arange(N_ETA, device=device, dtype=torch.float32) + 0.5) * CELL
    phi_c = -math.pi + (torch.arange(N_PHI, device=device, dtype=torch.float32) + 0.5) * PHI_CELL
    return eta_c.unsqueeze(1), phi_c.unsqueeze(0)  # [N_ETA, 1], [1, N_PHI]


def _wrap(d):
    return (d + math.pi) % (2 * math.pi) - math.pi


def build_heldout_map(family: str, severity: float, seed: int = 4321) -> torch.Tensor:
    """Return a [N_ETA, N_PHI] tensor of per-cell drop probabilities.

    Shapes are drawn from one seeded generator in a fixed order and accumulated with
    torch.maximum, stopping once the covered area reaches `severity`. Because the draw
    sequence does not depend on severity, a lower severity is a prefix of a higher one,
    which is what makes the maps nested.
    """
    if family not in HELDOUT_FAMILIES:
        raise ValueError(f"unknown held-out family {family}, choose from {HELDOUT_FAMILIES}")
    grid = torch.zeros(N_ETA, N_PHI)
    if severity <= 0:
        return grid

    g = torch.Generator().manual_seed(seed + HELDOUT_FAMILIES.index(family) * 1000)
    eta_c, phi_c = _cell_centres()

    def u(a, b):
        return a + (b - a) * torch.rand(1, generator=g).item()

    for _ in range(MAX_SHAPES):
        if (grid > 0).float().mean().item() >= severity:
            break
        e0, p0 = u(ETA_MIN, ETA_MAX), u(-math.pi, math.pi)
        d_eta = eta_c - e0                      # [N_ETA, 1]
        d_phi = _wrap(phi_c - p0)               # [1, N_PHI]

        if family == "ellipse":
            a, b, theta = u(0.6, 2.5), u(0.3, 1.2), u(0.0, math.pi)
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            xr = d_eta * cos_t + d_phi * sin_t
            yr = -d_eta * sin_t + d_phi * cos_t
            hit = (xr / a) ** 2 + (yr / b) ** 2 < 1.0
            p = 1.0
        elif family == "annulus":
            r_out = u(0.8, 2.5)
            r_in = max(0.0, r_out - u(0.2, 0.8))
            r = torch.sqrt(d_eta ** 2 + d_phi ** 2)
            hit = (r >= r_in) & (r < r_out)
            p = 0.95
        else:  # diagonal
            slope, width = u(-1.2, 1.2), u(0.2, 0.7)
            # Band of half-width width/2 about the line phi = slope * eta + p0.
            hit = _wrap(phi_c - (slope * eta_c + p0)).abs() < width / 2
            p = 1.0

        hit = hit.expand(N_ETA, N_PHI)
        grid = torch.where(hit, torch.clamp(grid, min=p), grid)

    return grid


class HeldoutDegradation(BenchDegradation):
    """BenchDegradation with a held-out dead map. Same forward contract, different shapes."""

    def __init__(self, family: str, severity: float, seed: int = 4321):
        nn.Module.__init__(self)  # skip BenchDegradation.__init__, which only knows its own families
        self.family, self.severity, self.seed = family, float(severity), seed
        self.register_buffer("grid", build_heldout_map(family, severity, seed))
        self._gen = None
        # BenchDegradation.forward indexes the grid with (eta - self.eta_min) / CELL. Those two
        # attributes were added to BenchDegradation.__init__ by the eta_max work (23:44) AFTER this
        # class was written (20:02), and because this __init__ deliberately skips the parent's, the
        # inherited forward began raising AttributeError: 'HeldoutDegradation' object has no
        # attribute 'eta_min'. The whole held-out suite -- promotion-rule item (b) -- was
        # unrunnable from 23:44 until this fix; found 03:49 on the first campaign-2 reference run.
        #
        # The values are the MODULE defaults, not a caller's --eta_max, and that is deliberate:
        # build_heldout_map() takes no eta_max and paints its shapes on the fixed ETA_MIN..ETA_MAX
        # grid, so the forward must index that same grid. It matches what bench_eval already
        # records as `eta_max_applies: false` for this suite -- eta_max rescales the five scoring
        # families only. Setting these from a caller's eta_max would silently mis-index the map.
        self.eta_max, self.eta_min = float(ETA_MAX), float(ETA_MIN)


def _self_check():
    severities = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ok = True
    for fam in HELDOUT_FAMILIES:
        prev = None
        areas = []
        for s in severities:
            grid = build_heldout_map(fam, s)
            areas.append((grid > 0).float().mean().item())
            if prev is not None:
                nested = bool(((grid > 0) | ~(prev > 0)).all())
                if not nested:
                    ok = False
                    print(f"  !! {fam}: map at s={s} is not a superset of the previous severity")
            prev = grid
        print(f"  {fam:8s} dead area by severity: " + "  ".join(f"s={s:.1f}:{a:.2f}" for s, a in zip(severities, areas)))

    x = torch.rand(64, 200, 7)
    x[..., 1] = x[..., 1] * 10 - 5          # eta in [-5, 5]
    x[..., 2] = x[..., 2] * 2 * math.pi - math.pi
    x[..., 0] = x[..., 0] * 10 + 1          # pt > 0, every row valid

    unchanged = torch.equal(HeldoutDegradation("ellipse", 0.0)(x), x)
    print(f"  severity 0 returns input unchanged: {unchanged}")
    ok = ok and unchanged

    for fam in HELDOUT_FAMILIES:
        fracs = []
        for s in severities[1:]:
            out = HeldoutDegradation(fam, s)(x)
            fracs.append(((out[..., 0] == 0) & (x[..., 0] > 0)).float().mean().item())
            kept = ~((out[..., 0] == 0) & (x[..., 0] > 0))
            if not torch.equal(out[kept], x[kept]):
                ok = False
                print(f"  !! {fam}: surviving rows were modified")
        print(f"  {fam:8s} dropped candidate fraction: " + "  ".join(f"{f:.2f}" for f in fracs))

    print("held-out self-check:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_self_check())
