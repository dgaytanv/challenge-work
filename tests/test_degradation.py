"""Tests for embedding.degradation.Degradation (WP-B).

Run with:
    PYTHONPATH=$HOME/rt-b/src python tests/test_degradation.py
"""
import math
import os
import sys
import time

import torch

from embedding.degradation import Degradation, wrap_phi

DATA = os.path.expanduser("~/hack-data/C9_robust_tagging/train/robust_tagging_train_data_small.pt")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def real_events(n=2048, pad_last=0):
    """Real raw events [n, 200, 7]; optionally zero the last `pad_last` slots as padding."""
    x = torch.load(DATA, map_location="cpu")[:n, :, :7].clone()
    if pad_last:
        x[:, -pad_last:, :] = 0.0
    return x.to(DEV)


def dropped_fraction(x_in, x_out):
    """Per-event fraction of originally-valid candidates that were zeroed."""
    valid = x_in[..., 0] > 0
    killed = (x_out.abs().sum(-1) == 0) & valid
    return killed.sum(1).float() / valid.sum(1).clamp_min(1).float()


def test_eval_severity_zero_identity():
    x = real_events(256)
    out = Degradation(severity=0.0).to(DEV).eval()(x)
    assert torch.equal(out, x), "severity 0 must return the input unchanged"
    out0 = Degradation(severity=0).to(DEV).eval()(x)
    assert torch.equal(out0, x)
    print("[ok] eval severity 0 is the identity")


def test_eval_monotonic_and_deterministic():
    x = real_events(1024)
    fracs = []
    for s in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        d = Degradation(severity=s).to(DEV).eval()
        fracs.append(dropped_fraction(x, d(x)).mean().item())
    print("[ok] eval dropped fraction vs severity:",
          " ".join(f"{s:.1f}->{f:.3f}" for s, f in zip((0.0, .2, .4, .6, .8, 1.0), fracs)))
    assert all(b >= a - 1e-6 for a, b in zip(fracs, fracs[1:])), f"not monotonic: {fracs}"
    assert fracs[-1] > 0.5, f"severity 1.0 should kill most of the event, got {fracs[-1]:.3f}"

    a = Degradation(severity=0.5, seed=7).to(DEV).eval()(x)
    b = Degradation(severity=0.5, seed=7).to(DEV).eval()(x)
    assert torch.equal(a, b), "same seed must give the same eval corruption"
    c = Degradation(severity=0.5, seed=8).to(DEV).eval()(x)
    assert not torch.equal(a, c), "different seeds should differ"
    print("[ok] eval mode is deterministic given the seed")


def test_padding_rows_stay_zero():
    x = real_events(512, pad_last=50)
    pad = slice(-50, None)
    for d in (Degradation(severity=None).to(DEV).train(),
              Degradation(severity=0.6).to(DEV).eval()):
        out = d(x)
        assert torch.equal(out[:, pad, :], torch.zeros_like(out[:, pad, :])), \
            "padding rows must remain exactly zero"
    print("[ok] padding rows stay zero in train and eval mode")


def test_survivors_bit_identical_without_symmetries():
    x = real_events(1024)
    d = Degradation(severity=None, rotate_phi=False, reflect_eta=False, p_pt_scale=0.0,
                    curriculum=False).to(DEV).train()
    out = d(x)
    alive = out.abs().sum(-1) != 0
    assert torch.equal(out[alive], x[alive]), "surviving rows must be bit-identical"
    assert alive.sum() < x[..., 0].gt(0).sum(), "expected at least some candidates dropped"
    print(f"[ok] survivors bit-identical with symmetries off ({alive.float().mean():.3f} alive)")


def test_symmetries_are_exact():
    x = real_events(1024)
    d = Degradation(severity=None, p_clean=1.0, p_pt_scale=0.0, curriculum=False).to(DEV).train()
    out = d(x)  # p_clean=1 -> no dead regions, symmetries only
    assert torch.equal(out.abs().sum(-1) != 0, x.abs().sum(-1) != 0), "no row should die"
    for col, name in ((0, "pt"), (3, "dxy"), (4, "dxysig"), (5, "is_pf"), (6, "pdgId")):
        assert torch.equal(out[..., col], x[..., col]), f"{name} must be untouched"
    assert torch.allclose(out[..., 1].abs(), x[..., 1].abs(), atol=1e-5), "eta only reflected"
    # phi is rotated by one constant angle per event
    dphi = wrap_phi(out[..., 2] - x[..., 2])
    spread = (wrap_phi(dphi - dphi[:, :1])).abs().max().item()
    assert spread < 1e-4, f"phi rotation must be global per event, spread {spread}"
    assert out[..., 2].abs().max() <= math.pi + 1e-5, "phi must stay wrapped"
    print(f"[ok] symmetries exact: pt/dxy/dxysig/is_pf/pdgId untouched, "
          f"phi rotation spread {spread:.2e}")


def test_train_distribution_spans():
    x = real_events(2048)
    for curriculum in (False, True):
        d = Degradation(severity=None, curriculum=curriculum).to(DEV).train()
        fr = torch.cat([dropped_fraction(x[i:i + 256], d(x[i:i + 256]))
                        for i in range(0, 2048, 256)]).cpu()
        edges = torch.linspace(0, 1, 11)
        hist = torch.histogram(fr, bins=edges).hist
        tag = "curriculum ON " if curriculum else "curriculum OFF"
        print(f"  {tag} dropped-fraction histogram over {len(fr)} events "
              f"(min {fr.min():.3f} max {fr.max():.3f} mean {fr.mean():.3f}):")
        for lo, hi, c in zip(edges[:-1], edges[1:], hist):
            bar = "#" * int(40 * c / max(hist.max().item(), 1))
            print(f"    [{lo:.1f},{hi:.1f}) {int(c):5d} {bar}")
        if not curriculum:
            assert fr.min() < 0.01, f"needs some clean events, min {fr.min():.3f}"
            assert fr.max() > 0.8, f"needs to reach 0.8 dead, max {fr.max():.3f}"
            occupied = (hist > 0).sum().item()
            assert occupied >= 9, f"distribution should cover the range, {occupied}/10 bins"
    print("[ok] train dropped fraction spans [0, 0.8+]")


def test_family_coverage():
    """Each family alone should track its target severity."""
    x = real_events(1024)
    for fam in ("rect", "wedge", "strip", "cells", "towers"):
        d = Degradation(severity=None, families=(fam,), p_clean=0.0, curriculum=False,
                        p_charged_only=0.0, p_neutral_only=0.0, p_pt_scale=0.0).to(DEV).train()
        fr = dropped_fraction(x, d(x))
        print(f"  {fam:7s} mean {fr.mean():.3f} max {fr.max():.3f}")
        assert fr.max() > 0.6, f"{fam} never reaches a high dead fraction (max {fr.max():.3f})"
    print("[ok] every family reaches high severity")


def test_timing():
    if DEV != "cuda":
        print("[skip] timing: no GPU")
        return
    x = real_events(2048)[:128]
    d = Degradation(severity=None).to(DEV).train()
    for _ in range(10):
        d(x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        d(x)
    torch.cuda.synchronize()
    ms = (time.time() - t0) * 1000 / 100
    print(f"[ok] train forward [128, 200, 7] on GPU: {ms:.2f} ms/call")
    assert ms < 5.0, f"too slow: {ms:.2f} ms"


def test_eval_400_tokens():
    """Nothing may depend on a fixed token count (eval has 400)."""
    ev = os.path.expanduser("~/hack-data/C9_robust_tagging/eval/robust_tagging_eval_small.pt")
    x = torch.load(ev, map_location="cpu")[:512, :, :7].to(DEV)
    assert x.shape[1] == 400, x.shape
    for d in (Degradation(severity=None).to(DEV).train(), Degradation(severity=0.6).to(DEV).eval()):
        out = d(x)
        assert out.shape == x.shape
    print("[ok] works on the 400-candidate eval tensor")


def test_symmetries_only_mode():
    """dead_regions=False: the module must be symmetries-only, never dropping a candidate."""
    x = real_events(1024)
    d = Degradation(severity=None, dead_regions=False).to(DEV).train()
    out = d(x)
    assert torch.equal(out.abs().sum(-1) != 0, x.abs().sum(-1) != 0), "no candidate may be dropped"
    for col in (0, 3, 4, 5, 6):
        assert torch.equal(out[..., col], x[..., col])
    # Two views built by composing: symmetry module first, degrader second, so both views
    # share one rotation/reflection (the pattern WP-C's two-view loop uses).
    sym = Degradation(severity=None, dead_regions=False).to(DEV).train()
    deg = Degradation(severity=None, rotate_phi=False, reflect_eta=False,
                      p_pt_scale=0.0).to(DEV).train()
    x_sym = sym(x)
    x_c, x_d = x_sym, deg(x_sym)
    alive = x_d.abs().sum(-1) != 0
    assert torch.equal(x_d[alive], x_c[alive]), "views must agree exactly on surviving candidates"
    assert not torch.equal(x_d, x_c), "the degraded view should actually differ"
    print("[ok] dead_regions=False is symmetries-only; compose-then-degrade keeps views aligned")


def test_pt_scale_mode_keeps_rows_alive():
    """The pt-scaling milder mode leaves rows ALIVE with a reduced pt, by design.

    Consequence for a two-view loop: with p_pt_scale > 0 the degraded view can differ
    from the clean view on candidates that were not dropped. That is the intended
    "partially working region", not a bug -- but it means a test asserting that
    survivors are bit-identical must set p_pt_scale=0.
    """
    x = real_events(1024)
    deg = Degradation(severity=None, rotate_phi=False, reflect_eta=False, p_clean=0.0,
                      p_charged_only=0.0, p_neutral_only=0.0, p_pt_scale=1.0,
                      curriculum=False).to(DEV).train()
    out = deg(x)
    assert torch.equal(out.abs().sum(-1) != 0, x.abs().sum(-1) != 0), \
        "pt-scale mode must not kill any candidate"
    changed = (out[..., 0] != x[..., 0])
    assert changed.any(), "pt-scale mode should reduce some pt values"
    assert (out[..., 0][changed] < x[..., 0][changed]).all(), "pt must only be scaled down"
    for col in (1, 2, 3, 4, 5, 6):
        assert torch.equal(out[..., col], x[..., col]), "only pt may change in pt-scale mode"
    print(f"[ok] pt-scale mode keeps rows alive with reduced pt "
          f"({changed.float().mean():.3f} of candidates scaled)")


def test_curriculum_never_exceeds_s_max():
    """The ramp must not hand back more severity than s_max, even for tiny s_max."""
    for s_max in (0.0, 0.1, 0.85):
        d = Degradation(severity=None, s_max=s_max, curriculum=True)
        assert d._current_s_max() <= s_max + 1e-9, \
            f"curriculum start {d._current_s_max()} exceeds s_max {s_max}"
        d.calls = 10 ** 6
        assert abs(d._current_s_max() - s_max) < 1e-9, "ramp must end exactly at s_max"
    # s_max=0 with curriculum on must really drop nothing, without relying on p_clean.
    x = real_events(512)
    d = Degradation(severity=None, s_max=0.0, p_clean=0.0, curriculum=True).to(DEV).train()
    assert torch.equal(d(x).abs().sum(-1) != 0, x.abs().sum(-1) != 0), "s_max=0 must drop nothing"
    print("[ok] curriculum never exceeds s_max (incl. s_max=0)")


if __name__ == "__main__":
    print(f"device: {DEV}")
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print("\nALL TESTS PASSED")
