"""Group 3's degradation families, ported for our bench.

Source: ~/hackathon-shared/colleague-group3, branch `dorian`, commit 37168c1,
`src/embedding/degradation.py`. The mask-generating logic below is copied VERBATIM from that
file -- `_rand`, `_delta_phi`, `_spatial_mask` and the family definitions are theirs, unmodified,
so a number produced here is a number produced by their code. Only the wrapper is ours.

Two suites, matching their `compare_robustness.py`:
  colleague        (their HELD-OUT suite)  ellipse, cell_dropout, edge_truncation
  colleague-train  (their TRAINING suite)  rectangle, eta_band, phi_wedge, multi_patch, candidate_loss

TWO THINGS TO KNOW BEFORE READING ANY NUMBER FROM THIS FILE:

1. **Eta convention mismatch.** Their families assume `eta in [-3, 3]` (an L1T/PUPPI acceptance);
   our PF-candidate data spans `[-5, 5]`. We do NOT alter their code for this, on instruction. The
   consequence is that every eta-bounded family (`eta_band`, `cell_dropout`, `edge_truncation`,
   `rectangle`, `ellipse`, `multi_patch`) can only ever touch the `|eta| <= 3` core, so even
   severity 1.0 leaves the forward region untouched. `bench_eval.py` prints the dropped fraction at
   each severity, which is where that shows up: if dropped fraction saturates well below 1.0, the
   family is bounded by the convention, not by the model. Compare against our own families, which
   reach 0.76-1.00 at s=1.0.

2. **Their severity grid is 0.2-0.8; ours is 0.2-1.0.** We keep ours so the columns line up with the
   rest of the table, which means the top two severity points are outside the range their families
   were designed and tuned for.

Seeding follows `compare_robustness.py`: `seed + 100*family_idx + severity_idx`, with `family_idx`
the index WITHIN ITS OWN SUITE (their loop is `enumerate(args.families)`, whose default is the
held-out suite in the order above) and `severity_idx` the index into the severity list excluding
the leading zero (`args.severities[1:]`).

**We use a local `torch.Generator`, not `torch.manual_seed`, and that difference is deliberate.**
Their script seeds the global RNG. Doing that inside `bench_eval.py` would reseed the generator our
probe refits draw from, silently correlating the repeats and destroying the noise-floor measurement
that every separability verdict in the table depends on. A `Generator` seeded identically yields
the same sequence, so the corruption is unchanged; only the global side effect is removed.
"""
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn

COLLEAGUE_HELDOUT_FAMILIES = ("ellipse", "cell_dropout", "edge_truncation")
COLLEAGUE_TRAIN_FAMILIES = ("rectangle", "eta_band", "phi_wedge", "multi_patch", "candidate_loss")
DEFAULT_SEED = 0


# ----------------------------------------------------------------------------------------------
# VERBATIM from colleague-group3 @37168c1 src/embedding/degradation.py -- do not edit.
# ----------------------------------------------------------------------------------------------
@dataclass
class DegradationMetadata:
    family: list
    severity: torch.Tensor
    dropped_fraction: torch.Tensor
    dropped_pt_fraction: torch.Tensor


class ColleagueCore(nn.Module):
    """Their `Degradation`, renamed only to avoid colliding with ours on import."""

    ALL_FAMILIES = (
        "rectangle", "eta_band", "phi_wedge", "multi_patch",
        "candidate_loss", "ellipse", "cell_dropout", "edge_truncation",
    )
    DEFAULT_TRAIN_FAMILIES = (
        "rectangle", "eta_band", "phi_wedge", "multi_patch", "candidate_loss",
    )

    def __init__(
        self,
        severity: Optional[float] = 0.0,
        enabled: bool = True,
        max_train_severity: float = 0.8,
        clean_probability: float = 0.25,
        families: Optional[Sequence[str]] = None,
        family_weights: Optional[Sequence[float]] = None,
        family: Optional[str] = None,
        high_severity_probability: float = 0.3,
    ):
        super().__init__()
        self.severity = severity
        self.enabled = enabled
        self.max_train_severity = max_train_severity
        self.clean_probability = clean_probability
        self.families = tuple(families or self.DEFAULT_TRAIN_FAMILIES)
        self.family = family
        self.high_severity_probability = high_severity_probability
        if severity is not None and not 0.0 <= severity <= 1.0:
            raise ValueError("severity must be in [0, 1] or None")
        if not 0.0 <= max_train_severity <= 1.0:
            raise ValueError("max_train_severity must be in [0, 1]")
        if not 0.0 <= clean_probability <= 1.0:
            raise ValueError("clean_probability must be in [0, 1]")
        if not 0.0 <= high_severity_probability <= 1.0:
            raise ValueError("high_severity_probability must be in [0, 1]")
        unknown = set(self.families) - set(self.ALL_FAMILIES)
        if family is not None and family not in self.ALL_FAMILIES:
            unknown.add(family)
        if unknown:
            raise ValueError(f"unknown degradation families: {sorted(unknown)}")
        weights = torch.tensor(
            family_weights if family_weights is not None else [1.0] * len(self.families),
            dtype=torch.float32,
        )
        if len(weights) != len(self.families) or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("family_weights must be non-negative and match families")
        self.register_buffer("family_weights", weights / weights.sum())

    @staticmethod
    def _rand(shape, x, generator=None):
        return torch.rand(shape, device=x.device, dtype=x.dtype, generator=generator)

    @staticmethod
    def _delta_phi(phi, center):
        return torch.remainder(phi - center + torch.pi, 2.0 * torch.pi) - torch.pi

    def _sample_severities(self, batch_size, x, max_severity, generator):
        uniform = self._rand((batch_size,), x, generator) * max_severity
        high = self._rand((batch_size,), x, generator).pow(1.0 / 3.0) * max_severity
        choose_high = self._rand((batch_size,), x, generator) < self.high_severity_probability
        severities = torch.where(choose_high, high, uniform)
        clean = self._rand((batch_size,), x, generator) < self.clean_probability
        return severities.masked_fill(clean, 0.0)

    def _spatial_mask(self, x, severity, family, generator):
        b, n = x.shape[:2]
        eta, phi = x[..., 1], x[..., 2]
        valid = x[..., 0] > 0
        s = severity.clamp(0, 1)
        if family == "candidate_loss":
            return valid & (self._rand((b, n), x, generator) < s[:, None])
        if family == "eta_band":
            span = 6.0 * s
            center = -3.0 + span / 2 + self._rand((b,), x, generator) * (6.0 - span)
            return valid & ((eta - center[:, None]).abs() <= span[:, None] / 2)
        if family == "phi_wedge":
            span = 2 * torch.pi * s
            center = -torch.pi + self._rand((b,), x, generator) * (2 * torch.pi)
            return valid & (self._delta_phi(phi, center[:, None]).abs() <= span[:, None] / 2)
        if family == "edge_truncation":
            side = self._rand((b,), x, generator) < 0.5
            low, high = -3.0 + 6.0 * s, 3.0 - 6.0 * s
            return valid & torch.where(side[:, None], eta <= low[:, None], eta >= high[:, None])
        if family == "cell_dropout":
            eta_cell = ((eta + 3.0) / 0.6).floor().long().clamp(0, 9)
            phi_cell = ((phi + torch.pi) / (torch.pi / 6)).floor().long().clamp(0, 11)
            dead = self._rand((b, 120), x, generator) < s[:, None]
            return valid & dead.gather(1, eta_cell * 12 + phi_cell)
        if family == "multi_patch":
            out = torch.zeros((b, n), device=x.device, dtype=torch.bool)
            patch_s = 1.0 - (1.0 - s).pow(1.0 / 3.0)
            for _ in range(3):
                out |= self._spatial_mask(x, patch_s, "rectangle", generator)
            return out

        aspect = torch.exp((self._rand((b,), x, generator) * 2.0 - 1.0) * 1.2)
        eta_frac = torch.sqrt(s * aspect).clamp(max=1.0)
        phi_frac = (s / eta_frac.clamp_min(1e-6)).clamp(max=1.0)
        eta_span, phi_span = 6.0 * eta_frac, 2.0 * torch.pi * phi_frac
        eta_center = -3.0 + eta_span / 2 + self._rand((b,), x, generator) * (6.0 - eta_span)
        phi_center = -torch.pi + self._rand((b,), x, generator) * (2.0 * torch.pi)
        de = (eta - eta_center[:, None]).abs() / (eta_span[:, None] / 2).clamp_min(1e-6)
        dp = self._delta_phi(phi, phi_center[:, None]).abs() / (phi_span[:, None] / 2).clamp_min(1e-6)
        region = (de.square() + dp.square() <= 1.0) if family == "ellipse" else ((de <= 1) & (dp <= 1))
        return valid & region

    def apply(self, x: torch.Tensor, *, generator=None, max_severity: Optional[float] = None):
        b = x.shape[0]
        if not self.enabled or (self.severity is None and not self.training):
            zero = torch.zeros(b, device=x.device, dtype=x.dtype)
            return x, DegradationMetadata(["clean"] * b, zero, zero, zero)
        valid = x[..., 0] > 0
        if self.severity is None:
            cap = self.max_train_severity if max_severity is None else max_severity
            severities = self._sample_severities(b, x, cap, generator)
        else:
            severities = torch.full((b,), float(self.severity), device=x.device, dtype=x.dtype)
        if self.family is not None:
            available, family_ids = (self.family,), torch.zeros(b, device=x.device, dtype=torch.long)
        else:
            available = self.families
            family_ids = torch.multinomial(self.family_weights, b, replacement=True, generator=generator)
        names = [available[i] for i in family_ids.tolist()]
        drop = torch.zeros_like(valid)
        for i, name in enumerate(available):
            rows = family_ids == i
            if rows.any():
                drop[rows] = self._spatial_mask(x[rows], severities[rows], name, generator)
        drop &= severities[:, None] > 0
        degraded = x.masked_fill(drop.unsqueeze(-1), 0.0)
        dropped_fraction = drop.sum(1) / valid.sum(1).clamp_min(1)
        dropped_pt_fraction = (x[..., 0] * drop).sum(1) / x[..., 0].sum(1).clamp_min(1e-8)
        return degraded, DegradationMetadata(names, severities, dropped_fraction, dropped_pt_fraction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply(x)[0]
# ----------------------------------------------------------------------------------------------
# End verbatim section.
# ----------------------------------------------------------------------------------------------


def suite_of(family: str):
    """(suite tuple, index within it) -- the index their seeding scheme uses."""
    for suite in (COLLEAGUE_HELDOUT_FAMILIES, COLLEAGUE_TRAIN_FAMILIES):
        if family in suite:
            return suite, suite.index(family)
    raise ValueError(f"unknown colleague family {family!r}")


class ColleagueDegradation(nn.Module):
    """Their families behind `BenchDegradation`'s interface: (family, severity) -> forward(x) -> x."""

    def __init__(self, family: str, severity: float, seed: int = DEFAULT_SEED,
                 severity_index: Optional[int] = None):
        super().__init__()
        _, fam_idx = suite_of(family)
        self.family, self.severity, self.seed = family, float(severity), seed
        # severity_idx as in their loop over `args.severities[1:]`; bench_eval passes ours.
        self.severity_index = severity_index if severity_index is not None else 0
        self._core = ColleagueCore(severity=float(severity), family=family)
        self._core.eval()
        self._family_index = fam_idx
        self._gen = None

    def _generator(self, device):
        if self._gen is None or self._gen.device != device:
            g = torch.Generator(device=device)
            g.manual_seed(self.seed + 100 * self._family_index + self.severity_index)
            self._gen = g
        return self._gen

    def dead_area_fraction(self) -> float:
        """Nominal only. Their severity is a target plane fraction under an eta in [-3,3]
        convention, so on our [-5,5] data this OVERSTATES the reachable area. The dropped
        fraction that bench_eval prints is the number to trust."""
        return self.severity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.severity <= 0:
            return x
        return self._core.apply(x, generator=self._generator(x.device))[0]
