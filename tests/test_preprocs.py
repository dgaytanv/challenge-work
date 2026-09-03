"""WP-B2: tests for the pt-normalisation preproc variants.

Run with:
    ~/hackathon-shared/gpu_small.sh python tests/test_preprocs.py
"""
import os
import torch

from embedding.degradation import Degradation
from embedding.preprocs import (PFPreProcessor, PFPreProcessorAbsPt, PFPreProcessorMaxPt,
                                _PFPreProcessorPtVariant)
from embedding.utils.data_utils import EPS

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN = os.path.expanduser("~/hack-data/C9_robust_tagging/train/robust_tagging_train_data_small.pt")


class _StockRatio(_PFPreProcessorPtVariant):
    """The stock pt rule expressed through the variant base, to prove the base is faithful."""
    def pt_feature(self, pt_raw, valid):
        s = torch.where(valid, pt_raw, torch.zeros_like(pt_raw)).sum(dim=-1, keepdim=True)
        return torch.log(pt_raw / (s + EPS))


def events(n=1024):
    return torch.load(TRAIN, map_location="cpu")[:n, :, :7].to(DEV)


def test_variant_base_matches_stock():
    """The refactored forward must reproduce PFPreProcessor exactly, on clean AND degraded input."""
    x = events()
    a, b = PFPreProcessor({}).to(DEV).eval(), _StockRatio({}).to(DEV).eval()
    b.load_state_dict(a.state_dict())
    for name, xin in (("clean", x), ("degraded", Degradation(severity=0.6).to(DEV).eval()(x))):
        with torch.no_grad():
            ya, yb = a(xin), b(xin)
        d = (ya - yb).abs().max().item()
        assert d < 1e-5, f"{name}: variant base differs from PFPreProcessor by {d}"
        print(f"[ok] variant base reproduces PFPreProcessor on {name} input (max diff {d:.2e})")


def test_zero_rows_stay_zero():
    """Zeroed candidates must stay all-zero: the encoder derives its mask from that."""
    x = events()
    xd = Degradation(severity=0.6).to(DEV).eval()(x)
    dead = xd[..., 0] == 0
    for cls in (PFPreProcessorAbsPt, PFPreProcessorMaxPt):
        p = cls({}).to(DEV).eval()
        with torch.no_grad():
            y = p(xd)
        assert torch.equal(y[dead], torch.zeros_like(y[dead])), f"{cls.__name__} leaks on dead rows"
    print("[ok] dead candidates stay exactly zero for both variants")


def test_pt_feature_shift_under_degradation():
    """The point of B2: how much does the pt feature of a SURVIVOR move when others die?

    Measured before any batch norm, so it is purely the normalisation rule.
    """
    x = events(2048)
    print(f"\n  mean |shift| of the pt feature on surviving candidates (nats):")
    print(f"  {'severity':>9} {'stock (sum_pt)':>15} {'AbsPt':>10} {'MaxPt':>10}")
    rows = []
    for s in (0.2, 0.4, 0.6, 0.8):
        xd = Degradation(severity=s).to(DEV).eval()(x)
        alive = (xd[..., 0] > 0) & (x[..., 0] > 0)
        out = []
        for cls in (_StockRatio, PFPreProcessorAbsPt, PFPreProcessorMaxPt):
            p = cls({}).to(DEV).eval()
            with torch.no_grad():
                f_c = p.pt_feature(x[..., 0], x[..., 0] > 0)
                f_d = p.pt_feature(xd[..., 0], xd[..., 0] > 0)
            out.append((f_d - f_c)[alive].abs().mean().item())
        rows.append((s, *out))
        print(f"  {s:9.1f} {out[0]:15.4f} {out[1]:10.4f} {out[2]:10.4f}")
    for s, stock, abspt, maxpt in rows:
        assert abspt < 1e-6, f"AbsPt must be exactly invariant, got {abspt}"
        assert maxpt < stock, f"MaxPt ({maxpt}) should move less than stock ({stock}) at sev {s}"
    print("[ok] AbsPt is exactly invariant; MaxPt moves less than stock at every severity")


if __name__ == "__main__":
    print(f"device: {DEV}")
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
    print("\nALL TESTS PASSED")
