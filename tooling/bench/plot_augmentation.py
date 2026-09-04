#!/usr/bin/env python
"""What WP-B's training augmentation does to an event (planner request, 22:18).

  python ~/hackathon-shared/bench/plot_augmentation.py

Writes runs/plots/augmentation_examples.png and runs/plots/augmentation_dropped_hist.png.
CPU only; reads the eval file and `integration`'s src/embedding/degradation.py, modifies nothing.

Two figures, and they use the generator differently on purpose:

* examples  -- the confounds B documents in writeup/B-generator-plotting-recipe.md are switched
  off (`rotate_phi`/`reflect_eta`, which would make the degraded panel a rotated copy so the dead
  region does not line up; `p_pt_scale`, which dims rows instead of removing them; `p_clean`,
  which would leave a panel untouched). Severity is drawn from train mode's own distribution,
  s ~ U(0, s_max), and then the family helper is called directly (B's recipe A) so the panel
  title can state the EXACT severity -- the public train path samples s internally and never
  reports it. What is drawn is therefore train mode with the confounds off, not a different
  corruption: the helpers are the ones `_forward_train` itself calls.
* histogram -- the public train path with every default left ON except the curriculum, because
  the question there is what the distribution training actually sees looks like. So p_clean's
  15% spike at zero and the milder failure modes are present, and they are meant to be.
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/rt-f/src"))
import plot_common as pc

import numpy as np
import torch

SCRIPT = os.path.abspath(__file__)
EVAL_PT = os.path.expanduser("~/hack-data/C9_robust_tagging/eval/robust_tagging_eval_small.pt")
FAMS = ("rect", "wedge", "strip", "cells", "towers")
SEED = 20260903
S_MAX = 0.85          # Degradation's own default upper end of the sampled dead-area fraction
LADDER = (0.2, 0.4, 0.6, 0.8)
SURVIVOR = "#0072B2"
# Local to this figure; deliberately NOT from the tag colour registry, which is for run tags.
FAM_COLORS = {"rect": "#0072B2", "wedge": "#E69F00", "strip": "#009E73",
              "cells": "#CC79A7", "towers": "#D55E00"}
DROPPED = "#9A9A9A"


def load_events(n, path=EVAL_PT):
    """First n events of the eval file as RAW [n, N, 7] (column 7 is the label, dropped)."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    t = obj[0] if isinstance(obj, (list, tuple)) else obj
    if isinstance(t, dict):
        t = t.get("features", t.get("data", next(iter(t.values()))))
    t = t.float()
    return t[:n, :, :7].clone()


def family_mask(deg, x, fam, s):
    """B's recipe A: the same helpers `_forward_train` calls, at an exact per-event severity."""
    eta, phi = x[..., 1], x[..., 2]
    if fam == "rect":
        return deg._drop_rect(eta, phi, s, None)
    if fam == "wedge":
        return deg._drop_bands(phi, s, 2 * math.pi, 0.1, 0.4, 0.6, 1.0, 16, True, None)
    if fam == "strip":
        return deg._drop_bands(eta, s, 10.0, 1.0, 2.5, 0.5, 1.0, 8, False, None)
    if fam == "cells":
        return deg._drop_cells_multi(eta, phi, s, (0.25, 0.5, 1.0), None)
    if fam == "towers":
        return deg._drop_cells(eta, phi, s, 0.1, None, p_lo=1.0, p_hi=1.0, compensate=False)
    raise ValueError(fam)


def panel(ax, x_ev, drop, title):
    pt, eta, phi = x_ev[..., 0], x_ev[..., 1], x_ev[..., 2]
    valid = pt > 0
    size = 3.0 + 22.0 * (torch.log1p(pt) / max(float(torch.log1p(pt[valid]).max()), 1e-6)) ** 2
    keep = valid & ~drop
    gone = valid & drop
    if gone.any():
        ax.scatter(eta[gone], phi[gone], s=size[gone], c=DROPPED, marker="x",
                   linewidths=0.7, alpha=0.85, zorder=2)
    ax.scatter(eta[keep], phi[keep], s=size[keep], c=SURVIVOR, marker="o",
               linewidths=0, alpha=0.85, zorder=3)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-math.pi, math.pi)
    ax.set_title(title, fontsize=7.5)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.2)


def examples(n_events=3):
    import matplotlib.pyplot as plt
    from embedding.degradation import Degradation

    x = load_events(max(n_events, 1))
    deg = Degradation(severity=None)          # helpers only; nothing sampled from it
    torch.manual_seed(SEED)
    s_sampled = torch.rand(x.shape[0]) * S_MAX          # train mode's own s ~ U(0, s_max)

    ncol = 1 + len(FAMS)
    nrow = n_events + 1                                  # + the severity ladder
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.55 * ncol, 2.45 * nrow), squeeze=False)

    for i in range(n_events):
        xe = x[i:i + 1]
        nvalid = int((xe[0, :, 0] > 0).sum())
        panel(axes[i][0], xe[0], torch.zeros(xe.shape[1], dtype=torch.bool),
              f"event {i}: clean, {nvalid} candidates")
        for j, fam in enumerate(FAMS):
            torch.manual_seed(SEED + 1000 * i + j)       # reproducible region draw per panel
            s = s_sampled[i:i + 1]
            m = family_mask(deg, xe, fam, s)[0] & (xe[0, :, 0] > 0)
            frac = float(m.sum()) / max(nvalid, 1)
            panel(axes[i][j + 1], xe[0], m, f"{fam}  s={float(s):.2f}  dropped {frac:.2f}")

    # ---- last row: one event, one family, the severity ladder (B's recipe A, exact severities)
    xe = x[0:1]
    nvalid = int((xe[0, :, 0] > 0).sum())
    panel(axes[nrow - 1][0], xe[0], torch.zeros(xe.shape[1], dtype=torch.bool),
          f"event 0: clean, {nvalid} candidates")
    for j, s_val in enumerate(LADDER):
        # ONE seed for the whole ladder, not one per rung: the region centres and count are drawn
        # from it and only their extent scales with s, so a shared seed grows the same regions and
        # the dropped fraction rises monotonically. Reseeding per rung redraws the geometry and
        # the ladder comes out non-monotonic, which reads as a bug in the generator rather than
        # as the resampling it actually is.
        torch.manual_seed(SEED + 77)
        s = torch.full((1,), float(s_val))
        m = family_mask(deg, xe, "rect", s)[0] & (xe[0, :, 0] > 0)
        frac = float(m.sum()) / max(nvalid, 1)
        panel(axes[nrow - 1][j + 1], xe[0], m, f"rect  s={s_val:.1f}  dropped {frac:.2f}")
    axes[nrow - 1][len(LADDER) + 1].axis("off")
    axes[nrow - 1][len(LADDER) + 1].text(
        0.0, 0.5, "severity ladder: one event,\none set of regions, four fixed\nseverities. The regions grow;\nthey are not redrawn.\n\nOne event, so the dropped\nfraction is a sample, not the\nfamily's expectation. rect\nundershoots s where regions\noverlap (measured by B).",
        fontsize=7.5, color="#555555", va="center")

    for r in range(nrow):
        axes[r][0].set_ylabel("phi", fontsize=8)
    for c in range(ncol):
        if axes[nrow - 1][c].axison:          # the ladder row's spare panel is switched off
            axes[nrow - 1][c].set_xlabel("eta", fontsize=8)

    fig.suptitle("WP-B training augmentation: surviving candidates (blue) and dropped ones (grey x), "
                 "marker area ~ log pt", fontsize=11)
    fig.tight_layout(rect=(0, 0.055, 1, 0.965))
    return pc.save(fig, "augmentation_examples", SCRIPT,
                   "train mode with rotate_phi/reflect_eta/p_pt_scale/p_clean off so the dead region "
                   f"lines up with the clean panel (B's recipe) · seed {SEED}")


def dropped_hist(n_events=4000, eval_severity=0.5):
    import matplotlib.pyplot as plt
    from embedding.degradation import Degradation

    x = load_events(n_events)
    valid = (x[..., 0] > 0)
    nvalid = valid.sum(dim=1).clamp(min=1).float()

    torch.manual_seed(SEED)
    train = Degradation(severity=None, curriculum=False)      # every other default left ON
    xt = train(x)
    ft = ((valid & (xt[..., 0] == 0)).sum(dim=1).float() / nvalid).numpy()

    ev = Degradation(severity=eval_severity)                  # its own seeded RNG
    xv = ev(x)
    fv = ((valid & (xv[..., 0] == 0)).sum(dim=1).float() / nvalid).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
    bins = np.linspace(0, 1, 51)
    for ax, f, title, colour in (
        (axes[0], ft, f"train mode (severity=None, curriculum off, s_max={S_MAX})", SURVIVOR),
        (axes[1], fv, f"eval mode at severity={eval_severity}", "#E69F00"),
    ):
        ax.hist(f, bins=bins, color=colour, alpha=0.85)
        ax.axvline(float(f.mean()), color="#333333", linestyle="--", linewidth=1.2,
                   label=f"mean {f.mean():.3f}")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("fraction of the event's candidates dropped")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
        ax.text(0.02, 0.96, f"n={len(f)}\nmedian {np.median(f):.3f}\nexactly 0: {(f == 0).mean():.1%}",
                transform=ax.transAxes, fontsize=8, va="top", color="#444444")
    axes[0].set_ylabel("events")
    fig.suptitle("Per-event dropped fraction: what training sees vs what the sweep applies", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return pc.save(fig, "augmentation_dropped_hist", SCRIPT,
                   "train mode keeps every default except the curriculum, so p_clean's 15% and the "
                   f"milder failure modes are included on purpose · seed {SEED}")


def generate():
    pc.use_agg()
    import matplotlib.pyplot as plt
    written = []
    for fn in (examples, dropped_hist):
        try:
            written += fn() or []
        except Exception as exc:
            pc.log(f"{fn.__name__} failed: {exc!r}")
    plt.close("all")
    pc.log("wrote " + ", ".join(os.path.basename(w) for w in written))
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=3)
    ap.add_argument("--hist_events", type=int, default=4000)
    ap.parse_args()
    generate()


def family_tails(n_events=8000):
    """B's request: the dropped-fraction distribution PER FAMILY, and the high-severity tail mass.

    The aggregate histogram draws the family uniformly per event, so its tail above 0.8 (0.8%) is an
    average over five families with different geometry. Rect undershoots its target severity more
    than the others because overlapping rectangles double-count dead area, so its own tail should be
    thinner -- which strengthens the extrapolation reading of the rect s=1.0 inversions rather than
    weakening it. This measures the conditional distribution given the family, with every other
    train-mode default left on so the numbers are comparable to the aggregate.
    """
    import matplotlib.pyplot as plt
    from embedding.degradation import Degradation

    x = load_events(n_events)
    valid = (x[..., 0] > 0)
    nvalid = valid.sum(dim=1).clamp(min=1).float()

    stats, data = {}, {}
    for fam in FAMS:
        torch.manual_seed(SEED)
        d = Degradation(severity=None, families=(fam,), curriculum=False)
        f = ((valid & (d(x)[..., 0] == 0)).sum(dim=1).float() / nvalid).numpy()
        data[fam] = f
        stats[fam] = {"mean": float(f.mean()), "median": float(np.median(f)),
                      "gt0.6": float((f > 0.6).mean()), "gt0.8": float((f > 0.8).mean()),
                      "eq0": float((f == 0).mean())}
    torch.manual_seed(SEED)
    agg = ((valid & (Degradation(severity=None, curriculum=False)(x)[..., 0] == 0)
            ).sum(dim=1).float() / nvalid).numpy()
    stats["all (family drawn per event)"] = {
        "mean": float(agg.mean()), "median": float(np.median(agg)),
        "gt0.6": float((agg > 0.6).mean()), "gt0.8": float((agg > 0.8).mean()),
        "eq0": float((agg == 0).mean())}

    pc.log(f"per-family dropped fraction over {n_events} events (train mode, curriculum off):")
    pc.log(f"  {'family':32s} {'mean':>6s} {'median':>7s} {'>0.6':>7s} {'>0.8':>7s} {'=0':>7s}")
    for k, v in stats.items():
        pc.log(f"  {k:32s} {v['mean']:6.3f} {v['median']:7.3f} {v['gt0.6']:6.2%} {v['gt0.8']:6.2%} {v['eq0']:6.2%}")

    fig, axes = plt.subplots(1, len(FAMS), figsize=(3.1 * len(FAMS), 3.6), sharex=True, sharey=True)
    bins = np.linspace(0, 1, 41)
    for ax, fam in zip(axes, FAMS):
        f, st = data[fam], stats[fam]
        ax.hist(f, bins=bins, color=FAM_COLORS[fam], alpha=0.85)
        ax.axvline(0.8, color="#333333", linestyle=":", linewidth=1.0)
        ax.set_title(f"{fam}\nmean {st['mean']:.3f}   >0.8: {st['gt0.8']:.2%}", fontsize=9)
        ax.set_xlabel("dropped fraction", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("events")
    fig.suptitle("Dropped fraction per corruption family (train mode, curriculum off); "
                 "dotted line = 0.8, the edge of the trained regime", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return pc.save(fig, "augmentation_family_tails", SCRIPT,
                   f"conditional on the family, every other train default on · n={n_events} · seed {SEED}")
