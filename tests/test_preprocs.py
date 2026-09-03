"""WP-B2: tests for the pt-normalisation preproc variants.

Run with:
    ~/hackathon-shared/gpu_small.sh python tests/test_preprocs.py
"""
import os
import torch

from embedding.degradation import Degradation
from embedding.preprocs import (PFPreProcessor, PFPreProcessorAbsPt, PFPreProcessorMaxPt,
                                PFPreProcessorMeanPt, _PFPreProcessorPtVariant)
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
    for cls in (PFPreProcessorAbsPt, PFPreProcessorMaxPt, PFPreProcessorMeanPt):
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
    print(f"  {'severity':>9} {'stock (sum_pt)':>15} {'AbsPt':>10} {'MaxPt':>10} {'MeanPt':>10}")
    rows = []
    for s in (0.2, 0.4, 0.6, 0.8):
        xd = Degradation(severity=s).to(DEV).eval()(x)
        alive = (xd[..., 0] > 0) & (x[..., 0] > 0)
        out = []
        for cls in (_StockRatio, PFPreProcessorAbsPt, PFPreProcessorMaxPt, PFPreProcessorMeanPt):
            p = cls({}).to(DEV).eval()
            with torch.no_grad():
                f_c = p.pt_feature(x[..., 0], x[..., 0] > 0)
                f_d = p.pt_feature(xd[..., 0], xd[..., 0] > 0)
            out.append((f_d - f_c)[alive].abs().mean().item())
        rows.append((s, *out))
        print(f"  {s:9.1f} {out[0]:15.4f} {out[1]:10.4f} {out[2]:10.4f} {out[3]:10.4f}")
    for s, stock, abspt, maxpt, meanpt in rows:
        assert abspt < 1e-6, f"AbsPt must be exactly invariant, got {abspt}"
        assert maxpt < stock, f"MaxPt ({maxpt}) should move less than stock ({stock}) at sev {s}"
        assert meanpt < stock, f"MeanPt ({meanpt}) should move less than stock ({stock}) at sev {s}"
    print("[ok] AbsPt exactly invariant; MaxPt and MeanPt both move less than stock everywhere")


def test_candidate_count_offset():
    """Train events have 200 candidates, eval events 400. A denominator that grows with
    the candidate count offsets the feature between train and eval before any degradation.

    Same candidates, scored in a 200-slot event vs a 400-slot event.
    """
    x400 = torch.load(os.path.expanduser(
        "~/hack-data/C9_robust_tagging/eval/robust_tagging_eval_small.pt"),
        map_location="cpu")[:512, :, :7].to(DEV)
    x200 = x400[:, :200, :].clone()
    shared = x200[..., 0] > 0
    print(f"\n  mean |offset| on the SAME candidates, 200-slot vs 400-slot event (nats):")
    out = {}
    for cls in (_StockRatio, PFPreProcessorAbsPt, PFPreProcessorMaxPt, PFPreProcessorMeanPt):
        p = cls({}).to(DEV).eval()
        with torch.no_grad():
            f4 = p.pt_feature(x400[..., 0], x400[..., 0] > 0)[:, :200]
            f2 = p.pt_feature(x200[..., 0], x200[..., 0] > 0)
        out[cls.__name__] = (f4 - f2)[shared].abs().mean().item()
        print(f"    {cls.__name__:24s} {out[cls.__name__]:.4f}")
    # NOTE: this truncation is pt-BIASED -- the data is sorted by pt, so the first 200
    # slots are the LEADING 200. The top 200 carry ~66% of the event's pt but only 50% of
    # the count, so mean_pt over-corrects (it assumes count and pt scale together) and only
    # partially removes the offset. AbsPt and MaxPt remove it exactly, by construction.
    assert out["_StockRatio"] > 0.3, "expected the stock rule to carry a large count offset"
    assert out["PFPreProcessorMeanPt"] < out["_StockRatio"], "MeanPt should reduce the offset"
    assert out["PFPreProcessorAbsPt"] < 1e-6 and out["PFPreProcessorMaxPt"] < 1e-6
    print(f"[ok] count offset: stock {out['_StockRatio']:.4f} -> MeanPt "
          f"{out['PFPreProcessorMeanPt']:.4f} ({100*(1-out['PFPreProcessorMeanPt']/out['_StockRatio']):.0f}% "
          f"reduced, NOT removed); AbsPt/MaxPt exactly 0")


def test_train_vs_eval_feature_distribution():
    """The offset that actually matters: the real 200-candidate train file versus the real
    400-candidate eval file. This is what the encoder sees at train time vs grading time."""
    tr = torch.load(TRAIN, map_location="cpu")[:2048, :, :7].to(DEV)
    ev = torch.load(os.path.expanduser(
        "~/hack-data/C9_robust_tagging/eval/robust_tagging_eval_small.pt"),
        map_location="cpu")[:2048, :, :7].to(DEV)
    print(f"\n  mean pt feature, real train (200 cands) vs real eval (400 cands):")
    print(f"    {'rule':24s} {'train':>9} {'eval':>9} {'shift':>9}")
    for cls in (_StockRatio, PFPreProcessorAbsPt, PFPreProcessorMaxPt, PFPreProcessorMeanPt):
        p = cls({}).to(DEV).eval()
        with torch.no_grad():
            a = p.pt_feature(tr[..., 0], tr[..., 0] > 0)[tr[..., 0] > 0].mean().item()
            b = p.pt_feature(ev[..., 0], ev[..., 0] > 0)[ev[..., 0] > 0].mean().item()
        print(f"    {cls.__name__:24s} {a:9.4f} {b:9.4f} {b-a:9.4f}")
    print("[ok] train/eval feature offset reported per rule")


if __name__ == "__main__":
    print(f"device: {DEV}")
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
    print("\nALL TESTS PASSED")
