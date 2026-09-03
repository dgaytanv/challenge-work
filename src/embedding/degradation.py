"""Detector degradation: dead eta-phi regions plus exact collision symmetries.

Two modes, keyed on ``severity``:

* ``severity is None``  -> TRAIN mode. Broad random family of dead regions, plus a
  random global phi rotation and eta reflection. Used as training augmentation.
* ``severity`` is a float -> EVAL mode. Deterministic corruption at a fixed severity,
  used by ``eval.py``'s severity sweep. ``severity == 0`` returns the input unchanged.

Conventions
-----------
x is RAW, shape [B, N, F] with F >= 7 and columns
``[pt, eta, phi, dxy, dxysig, is_pf, pdgId, ...]``.
A candidate is "valid" iff ``pt > 0``; rows that are already all-zero (padding, or
candidates killed by an earlier degradation) are never touched. Dropping a candidate
zeroes its whole row, which is what ``TransformerEncoder`` re-derives its attention
mask from.

Severity semantics
------------------
``severity`` / the internally sampled ``s`` is the *target fraction of the eta-phi
plane that is dead*, and every family is parameterised so the expected fraction of
candidates dropped is approximately ``s``. This makes the families comparable to each
other and matches the grader's description of severity as "% of the eta-phi plane".
See ``_drop_bands`` / ``_drop_cells`` for how the shape priors are rescaled to hit ``s``.
"""

import math
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

# Detector acceptance seen in the data: eta in [-5, 5], phi in (-pi, pi].
ETA_MAX = 5.0
ETA_SPAN = 2 * ETA_MAX
PHI_SPAN = 2 * math.pi
PLANE_AREA = ETA_SPAN * PHI_SPAN

FAMILIES = ("rect", "wedge", "strip", "cells", "towers")

# Charged candidates (used by the "drop only charged / only neutral" milder modes).
CHARGED_PDGIDS = (11, 13, 211)


def wrap_phi(phi: torch.Tensor) -> torch.Tensor:
    """Wrap an angle (or angle difference) into [-pi, pi)."""
    return (phi + math.pi) % PHI_SPAN - math.pi


class Degradation(nn.Module):
    """Simulate detector degradation as dead regions of the eta-phi plane.

    Parameters
    ----------
    severity
        ``None`` -> train mode (random). A float -> eval mode at that fixed severity.
    rotate_phi, reflect_eta
        Exact symmetries of pp collisions, applied in train mode only. Free augmentation.
    p_clean
        Probability that a training event gets no dead regions at all (symmetries still apply).
    s_max
        Upper end of the sampled dead-area fraction in train mode.
    warmup_calls, curriculum
        ``s_max`` ramps linearly from ``CURRICULUM_S_START`` to ``s_max`` over the first
        ``warmup_calls`` train-mode forward passes. Set ``curriculum=False`` to disable.
    p_charged_only, p_neutral_only, p_pt_scale
        Probabilities of the milder failure modes: kill only charged / only neutral
        candidates inside the region, or merely scale their pt down instead of zeroing.
    families
        Which dead-region families to sample from in train mode.
    seed
        Seed for the eval-mode RNG, making the eval sweep reproducible.
    eval_cell_size, eval_drop_prob
        Eval-mode corruption: coarse cells of this size, a fraction ``severity`` of them
        dead, each dead cell killing candidates with this probability.
    """

    CURRICULUM_S_START = 0.2

    def __init__(
        self,
        severity: Union[float, None] = None,
        *,
        rotate_phi: bool = True,
        reflect_eta: bool = True,
        p_clean: float = 0.15,
        s_max: float = 0.85,
        warmup_calls: int = 600,
        curriculum: bool = True,
        p_charged_only: float = 0.05,
        p_neutral_only: float = 0.05,
        p_pt_scale: float = 0.10,
        pt_scale_range: Sequence[float] = (0.3, 0.9),
        families: Sequence[str] = FAMILIES,
        seed: int = 0,
        eval_cell_size: float = 0.5,
        eval_drop_prob: float = 0.8,
        eta_max: float = ETA_MAX,
    ):
        super().__init__()
        self.severity = severity
        self.rotate_phi = rotate_phi
        self.reflect_eta = reflect_eta
        self.p_clean = p_clean
        self.s_max = s_max
        self.warmup_calls = warmup_calls
        self.curriculum = curriculum
        self.p_charged_only = p_charged_only
        self.p_neutral_only = p_neutral_only
        self.p_pt_scale = p_pt_scale
        self.pt_scale_range = tuple(pt_scale_range)
        unknown = [f for f in families if f not in FAMILIES]
        if unknown:
            raise ValueError(f"unknown degradation families: {unknown}")
        self.families = tuple(families)
        self.seed = seed
        self.eval_cell_size = eval_cell_size
        self.eval_drop_prob = eval_drop_prob
        self.eta_max = eta_max
        self.eta_span = 2 * eta_max
        self.plane_area = self.eta_span * PHI_SPAN

        # Plain python int on purpose: a CUDA buffer would force a device sync
        # (.item()) on every forward. The module is never saved in a checkpoint.
        self.calls = 0
        self._gens = {}
        self._cell_lut = {}
        self._charged_ids = None

    # ------------------------------------------------------------------ RNG

    def _gen(self, device: torch.device) -> Optional[torch.Generator]:
        """Eval mode draws from a private, seeded generator so a sweep is reproducible.

        The generator is *not* reset per call: it advances across batches, so the whole
        sweep is a deterministic function of ``seed`` and the (fixed) batch order.
        Train mode returns ``None``, i.e. the global RNG.
        """
        if self.severity is None:
            return None
        key = (device.type, device.index)
        if key not in self._gens:
            g = torch.Generator(device=device)
            g.manual_seed(self.seed)
            self._gens[key] = g
        return self._gens[key]

    def _u(self, shape, ref: torch.Tensor, lo: float = 0.0, hi: float = 1.0, gen=None) -> torch.Tensor:
        r = torch.rand(shape, device=ref.device, dtype=ref.dtype, generator=gen)
        return r if (lo == 0.0 and hi == 1.0) else lo + (hi - lo) * r

    # ------------------------------------------------------- region families

    def _drop_bands(
        self, coord, s, span, w_lo, w_hi, p_lo, p_hi, m_max, wrap, gen
    ) -> torch.Tensor:
        """Parallel bands in one coordinate, spanning the other coordinate fully.

        ``wrap=True`` -> phi wedges; ``wrap=False`` -> eta strips. The width prior
        (w_lo, w_hi) fixes how many bands are used; the width is then rescaled so that
        ``M * width * p / span == s``, i.e. the expected dropped fraction is ``s``.
        """
        B, N = coord.shape
        w0 = self._u((B,), coord, w_lo, w_hi, gen)
        p = self._u((B,), coord, p_lo, p_hi, gen)
        m = torch.clamp(torch.round(s * span / w0), min=1, max=m_max)
        w = torch.clamp(s * span / (m * p), min=0.0, max=span)

        centers = self._u((B, m_max), coord, -span / 2, span / 2, gen)
        active = torch.arange(m_max, device=coord.device).view(1, -1) < m.view(-1, 1)

        d = coord.unsqueeze(-1) - centers.unsqueeze(1)          # [B, N, m_max]
        if wrap:
            d = wrap_phi(d)
        inside = (d.abs() <= (w / 2).view(-1, 1, 1)) & active.unsqueeze(1)
        hit = inside.any(dim=-1)
        return hit & (self._u((B, N), coord, gen=gen) < p.view(-1, 1))

    def _drop_rect(self, eta, phi, s, gen, k_max: int = 4) -> torch.Tensor:
        """K axis-aligned rectangles. Shape priors set the aspect ratio, s sets the size."""
        B, N = eta.shape
        k = torch.randint(1, k_max + 1, (B,), device=eta.device, generator=gen)
        active = torch.arange(k_max, device=eta.device).view(1, -1) < k.view(-1, 1)  # [B, k_max]

        d_eta0 = self._u((B, k_max), eta, 0.5, 2.5, gen)
        d_phi0 = self._u((B, k_max), eta, 0.3, 1.2, gen)
        p = self._u((B, k_max), eta, 0.5, 1.0, gen)

        # Rescale the sampled boxes so their total (probability-weighted) area is s * plane.
        base = ((d_eta0 * d_phi0 * p) * active).sum(dim=1) / self.plane_area          # [B]
        g = torch.sqrt(torch.clamp(s / base.clamp_min(1e-6), min=0.0, max=64.0))
        d_eta = torch.clamp(d_eta0 * g.view(-1, 1), max=self.eta_span)
        d_phi = torch.clamp(d_phi0 * g.view(-1, 1), max=PHI_SPAN)

        c_eta = self._u((B, k_max), eta, -self.eta_max, self.eta_max, gen)
        c_phi = self._u((B, k_max), eta, -math.pi, math.pi, gen)

        in_eta = (eta.unsqueeze(-1) - c_eta.unsqueeze(1)).abs() <= (d_eta / 2).unsqueeze(1)
        in_phi = wrap_phi(phi.unsqueeze(-1) - c_phi.unsqueeze(1)).abs() <= (d_phi / 2).unsqueeze(1)
        inside = in_eta & in_phi & active.unsqueeze(1)
        killed = inside & (self._u((B, N, k_max), eta, gen=gen) < p.unsqueeze(1))
        return killed.any(dim=-1)

    def _cell_index(self, eta, phi, cell: float):
        n_eta = max(1, int(math.ceil(self.eta_span / cell)))
        n_phi = max(1, int(math.ceil(PHI_SPAN / cell)))
        i = torch.clamp(((eta + self.eta_max) / cell).floor().long(), 0, n_eta - 1)
        j = torch.clamp(((phi + math.pi) / cell).floor().long(), 0, n_phi - 1)
        return i * n_phi + j, n_eta * n_phi

    def _cell_lut_for(self, sizes, device):
        """Cached per-size grid dimensions and flat-field offsets for the fused grid."""
        key = (tuple(sizes), device.type, device.index)
        if key not in self._cell_lut:
            n_eta = [max(1, int(math.ceil(self.eta_span / c))) for c in sizes]
            n_phi = [max(1, int(math.ceil(PHI_SPAN / c))) for c in sizes]
            counts = [a * b for a, b in zip(n_eta, n_phi)]
            offsets, run = [], 0
            for c in counts:
                offsets.append(run)
                run += c
            t = lambda v, dt: torch.tensor(v, device=device, dtype=dt)
            self._cell_lut[key] = (t(list(sizes), torch.float32), t(n_eta, torch.long),
                                   t(n_phi, torch.long), t(offsets, torch.long), run)
        return self._cell_lut[key]

    def _drop_cells_multi(self, eta, phi, s, sizes, gen, p_lo=0.3, p_hi=1.0):
        """Coarse cells whose size is chosen per event, over a single fused grid.

        The three grids are laid out end to end in one flat field so the whole family
        costs one random field and one gather instead of one per cell size.
        """
        B, N = eta.shape
        cell_t, n_eta_t, n_phi_t, off_t, total = self._cell_lut_for(sizes, eta.device)
        choice = torch.randint(0, len(sizes), (B,), device=eta.device, generator=gen)
        cell = cell_t[choice].to(eta.dtype).view(-1, 1)
        n_eta = n_eta_t[choice].view(-1, 1)
        n_phi = n_phi_t[choice].view(-1, 1)
        off = off_t[choice].view(-1, 1)

        i = torch.clamp(((eta + self.eta_max) / cell).floor().long(), min=0).minimum(n_eta - 1)
        j = torch.clamp(((phi + math.pi) / cell).floor().long(), min=0).minimum(n_phi - 1)
        idx = off + i * n_phi + j

        p_mean = 0.5 * (p_lo + p_hi)
        frac = torch.clamp(s / p_mean, max=1.0).view(-1, 1)
        field = self._u((B, total), eta, gen=gen) < frac
        probs = self._u((B, total), eta, p_lo, p_hi, gen)
        return field.gather(1, idx) & (self._u((B, N), eta, gen=gen) < probs.gather(1, idx))

    def _drop_cells(self, eta, phi, s, cell, gen, p_lo=0.3, p_hi=1.0, compensate=True):
        """A random field over a coarse eta-phi grid: a fraction of cells go dead."""
        B, N = eta.shape
        idx, n_cells = self._cell_index(eta, phi, cell)
        p_mean = 0.5 * (p_lo + p_hi)
        # Fraction of cells to kill so that the expected candidate loss is s.
        frac = torch.clamp(s / p_mean, max=1.0) if compensate else s
        field = self._u((B, n_cells), eta, gen=gen) < frac.view(-1, 1)
        probs = self._u((B, n_cells), eta, p_lo, p_hi, gen) if p_hi > p_lo else None
        hit = field.gather(1, idx)
        if probs is None:
            return hit & (self._u((B, N), eta, gen=gen) < p_lo)
        return hit & (self._u((B, N), eta, gen=gen) < probs.gather(1, idx))

    # ------------------------------------------------------------- symmetries

    def _apply_symmetries(self, x, valid, gen):
        if not (self.rotate_phi or self.reflect_eta):
            return x
        B = x.shape[0]
        out = x.clone()
        if self.rotate_phi:
            dphi = self._u((B, 1), x, -math.pi, math.pi, gen)
            out[..., 2] = torch.where(valid, wrap_phi(x[..., 2] + dphi), x[..., 2])
        if self.reflect_eta:
            sign = torch.where(self._u((B, 1), x, gen=gen) < 0.5, -1.0, 1.0).to(x.dtype)
            out[..., 1] = torch.where(valid, x[..., 1] * sign, x[..., 1])
        return out

    # ---------------------------------------------------------------- forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[-1] < 7:
            raise ValueError(f"expected x of shape [B, N, >=7], got {tuple(x.shape)}")
        if self.severity is not None:
            return self._forward_eval(x, float(self.severity))
        return self._forward_train(x)

    def _forward_eval(self, x, severity: float) -> torch.Tensor:
        if severity <= 0:
            return x
        gen = self._gen(x.device)
        valid = x[..., 0] > 0
        s = torch.full((x.shape[0],), min(severity, 1.0), device=x.device, dtype=x.dtype)
        drop = self._drop_cells(
            x[..., 1], x[..., 2], s, self.eval_cell_size, gen,
            p_lo=self.eval_drop_prob, p_hi=self.eval_drop_prob, compensate=False,
        )
        return torch.where((drop & valid).unsqueeze(-1), torch.zeros_like(x), x)

    def _current_s_max(self) -> float:
        if not self.curriculum or self.warmup_calls <= 0:
            return self.s_max
        t = min(1.0, self.calls / self.warmup_calls)
        return self.CURRICULUM_S_START + (self.s_max - self.CURRICULUM_S_START) * t

    def _forward_train(self, x) -> torch.Tensor:
        B, N, _ = x.shape
        gen = None
        valid = x[..., 0] > 0

        x = self._apply_symmetries(x, valid, gen)
        eta, phi = x[..., 1], x[..., 2]

        s_max = self._current_s_max()
        self.calls += 1

        s = self._u((B,), x, 0.0, s_max, gen)
        clean = self._u((B,), x, gen=gen) < self.p_clean
        s = torch.where(clean, torch.zeros_like(s), s)

        masks = []
        for fam in self.families:
            if fam == "rect":
                masks.append(self._drop_rect(eta, phi, s, gen))
            elif fam == "wedge":
                masks.append(self._drop_bands(phi, s, PHI_SPAN, 0.1, 0.4, 0.6, 1.0, 16, True, gen))
            elif fam == "strip":
                masks.append(self._drop_bands(eta, s, self.eta_span, 1.0, 2.5, 0.5, 1.0, 8, False, gen))
            elif fam == "cells":
                masks.append(self._drop_cells_multi(eta, phi, s, (0.25, 0.5, 1.0), gen))
            elif fam == "towers":
                masks.append(self._drop_cells(eta, phi, s, 0.1, gen, p_lo=1.0, p_hi=1.0, compensate=False))
        masks = torch.stack(masks, dim=0)                                   # [F, B, N]
        fam_idx = torch.randint(0, len(self.families), (B,), device=x.device, generator=gen)
        drop = masks[fam_idx, torch.arange(B, device=x.device)]             # [B, N]
        drop = drop & valid & ~clean.view(-1, 1)

        # Milder failure modes: only charged / only neutral / pt merely scaled down.
        mode_u = self._u((B, 1), x, gen=gen)
        charged_only = mode_u < self.p_charged_only
        neutral_only = (mode_u >= self.p_charged_only) & (mode_u < self.p_charged_only + self.p_neutral_only)
        scale_only = (mode_u >= self.p_charged_only + self.p_neutral_only) & (
            mode_u < self.p_charged_only + self.p_neutral_only + self.p_pt_scale
        )

        if self._charged_ids is None or self._charged_ids.device != x.device:
            self._charged_ids = torch.tensor(CHARGED_PDGIDS, device=x.device, dtype=x.dtype)
        is_charged = (x[..., 6].abs().unsqueeze(-1) == self._charged_ids).any(dim=-1)
        drop = drop & ~(charged_only & ~is_charged) & ~(neutral_only & is_charged)

        zero_mask = drop & ~scale_only
        scale_mask = drop & scale_only

        out = torch.where(zero_mask.unsqueeze(-1), torch.zeros_like(x), x)
        if self.p_pt_scale > 0:
            factor = self._u((B, 1), x, *self.pt_scale_range, gen=gen)
            out[..., 0] = torch.where(scale_mask, out[..., 0] * factor, out[..., 0])
        return out
