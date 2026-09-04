"""Fixed benchmark corruptions for cross-instance comparison.

NOT a training augmentation. Every instance evaluates against these identical,
seeded dead maps so numbers are comparable. Do not tune models to these maps.

Severity s in [0, 1] is (approximately) the fraction of the eta-phi plane
(|eta| < 5, full phi) covered by dead regions. Maps are nested in severity:
the dead set at s=0.4 is a subset of the dead set at s=0.6 for a given family.

Families (shapes taken from real detector failures, see the literature review):
  rect    HEM-like rectangles, d_eta in [0.8, 2.0], d_phi in [0.4, 1.0], p_drop = 1.0
  wedge   phi-wedges spanning all eta, d_phi in [0.1, 0.3], p_drop = 0.9
  strip   thin eta strips, d_eta in [1.0, 2.0], d_phi in [0.15, 0.3], p_drop = 1.0
  towers  isolated 0.1 x 0.1 dead towers, Bernoulli(s) per tower, p_drop = 1.0
  cells   coarse 0.5 x 0.5 blocks, round(s * n_blocks) dead, p_drop ~ U(0.5, 1.0)
"""
import math
import torch
import torch.nn as nn

ETA_MIN, ETA_MAX = -5.0, 5.0
CELL = 0.1
N_ETA = int(round((ETA_MAX - ETA_MIN) / CELL))      # 100
N_PHI = int(round(2 * math.pi / CELL))               # 62
PHI_CELL = 2 * math.pi / N_PHI
FAMILIES = ("rect", "wedge", "strip", "towers", "cells")


def _paint(grid, eta0, deta, phi0, dphi, p, eta_min=ETA_MIN):
    n_eta = grid.shape[0]
    i0 = int((eta0 - eta_min) / CELL)
    i1 = min(n_eta, i0 + max(1, int(round(deta / CELL))))
    j0 = int((phi0 + math.pi) / PHI_CELL)
    nj = max(1, int(round(dphi / PHI_CELL)))
    js = torch.tensor([(j0 + k) % N_PHI for k in range(nj)])
    block = grid[i0:i1, js]
    grid[i0:i1, js] = torch.maximum(block, torch.full_like(block, p))


def build_map(family: str, severity: float, seed: int = 1234,
              eta_max: float = ETA_MAX) -> torch.Tensor:
    """Return a [n_eta, N_PHI] tensor of per-cell drop probabilities.

    `eta_max` sets the acceptance the families are painted over: the plane is
    [-eta_max, eta_max] x (-pi, pi]. Default 5.0 reproduces every PF-file number ever taken with
    this module. Use 3.0 for the L1T/PUPPI acceptance, where a [-5, 5] plane would make severity
    mean "fraction of a plane two thirds of which contains no candidates" -- measured on
    winner-l1t, rect at eta_max 5 removes 0.28/0.52/0.77/0.93/1.00 of candidates at s=0.2..1.0
    against a design target of s.

    Only the PLANE is rescaled. The shape priors (d_eta ranges and so on) are left in absolute eta
    units, so a rectangle is the same physical size on either acceptance; what changes is how much
    of the plane it covers, which is what severity is defined against.
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family}, choose from {FAMILIES}")
    eta_min = -eta_max
    n_eta = int(round((eta_max - eta_min) / CELL))
    g = torch.Generator().manual_seed(seed + FAMILIES.index(family) * 1000)
    grid = torch.zeros(n_eta, N_PHI)
    if severity <= 0:
        return grid
    n_cells = n_eta * N_PHI

    def u(a, b):
        return a + (b - a) * torch.rand(1, generator=g).item()

    if family in ("rect", "wedge", "strip"):
        for _ in range(500):
            if (grid > 0).float().mean().item() >= severity:
                break
            if family == "rect":
                deta, dphi, p = u(0.8, 2.0), u(0.4, 1.0), 1.0
            elif family == "wedge":
                deta, dphi, p = eta_max - eta_min, u(0.1, 0.3), 0.9
            else:
                deta, dphi, p = u(1.0, 2.0), u(0.15, 0.3), 1.0
            eta0 = u(eta_min, eta_max - deta) if family != "wedge" else eta_min
            phi0 = u(-math.pi, math.pi)
            _paint(grid, eta0, deta, phi0, dphi, p, eta_min)
    elif family == "towers":
        field = torch.rand(n_eta, N_PHI, generator=g)
        grid[field < severity] = 1.0
    elif family == "cells":
        be, bp = 5, 5
        n_be, n_bp = n_eta // be, math.ceil(N_PHI / bp)
        n_blocks = n_be * n_bp
        order = torch.randperm(n_blocks, generator=g)
        probs = 0.5 + 0.5 * torch.rand(n_blocks, generator=g)
        k = int(round(severity * n_blocks))
        for b in order[:k].tolist():
            ie, ip = divmod(b, n_bp)
            grid[ie * be:(ie + 1) * be, ip * bp:min(N_PHI, (ip + 1) * bp)] = probs[b]
    return grid


class BenchDegradation(nn.Module):
    """Drop-in replacement for embedding.degradation.Degradation with the same forward contract."""

    def __init__(self, family: str, severity: float, seed: int = 1234,
                 eta_max: float = ETA_MAX):
        super().__init__()
        self.family, self.severity, self.seed = family, float(severity), seed
        self.eta_max, self.eta_min = float(eta_max), -float(eta_max)
        self.register_buffer("grid", build_map(family, severity, seed, eta_max))
        self._gen = None

    def dead_area_fraction(self) -> float:
        return (self.grid > 0).float().mean().item()

    def expected_drop_fraction(self) -> float:
        """Expected fraction of candidates dropped if candidates were uniform in the plane."""
        return self.grid.mean().item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.severity <= 0:
            return x
        if self._gen is None or self._gen.device != x.device:
            self._gen = torch.Generator(device=x.device)
            self._gen.manual_seed(self.seed + 7)
        pt, eta, phi = x[..., 0], x[..., 1], x[..., 2]
        valid = pt > 0
        i = ((eta - self.eta_min) / CELL).floor().long().clamp(0, self.grid.shape[0] - 1)
        j = ((phi + math.pi) / PHI_CELL).floor().long().clamp(0, N_PHI - 1)
        p = self.grid[i, j]
        drop = torch.bernoulli(p, generator=self._gen).bool() & valid
        if drop.any():
            x = x.clone()
            x[drop] = 0.0
        return x
