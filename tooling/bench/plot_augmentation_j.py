#!/usr/bin/env python
"""WP-J campaign 2: what the NEW generator families look like, and what J5's placement fix does.

  python ~/hackathon-shared/bench/plot_augmentation_j.py [--repo ~/c2-j]

Writes runs/plots/j1_colleague_families.{png,pdf} and runs/plots/j5_placement.{png,pdf}.
CPU only; reads the eval file and --repo's src/embedding/degradation.py, modifies nothing.

Deliberately a SEPARATE script from plot_augmentation.py rather than an extension of it. That one
is WP-F's, it is in active use for the five scoring families, and it hardcodes ~/rt-f/src; adding a
--repo switch to it mid-campaign risks a conflict for a figure only WP-J needs. The panel drawing
below follows its conventions on purpose (blue survivors, grey x for dropped, marker area ~ log pt,
same axes) so the two sets of figures can be read side by side. Credit for the convention: WP-F.

The confounds are switched off exactly as WP-B's recipe A prescribes -- rotate_phi/reflect_eta
would make the degraded panel a rotated copy so the dead region would not line up with the clean
one, p_pt_scale dims rows instead of removing them, and p_clean would leave a panel untouched. The
family helpers called here are the ones `_forward_train` itself calls, so this is train mode with
the confounds off, not a different corruption. Severity is stated exactly on every panel because
the helper is called directly; the public train path samples s internally and never reports it.
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_common as pc

import numpy as np
import torch

SCRIPT = os.path.abspath(__file__)


def content_key(*parts):
    """Key the inspection claim to the SCRIPT plus the figure's parameters.

    plot_common retires an inspection as soon as the key changes, so this must move whenever the
    rendered content could have moved and not otherwise. The script's own bytes cover the drawing
    and the family code paths; the parameters cover the rest. It deliberately does NOT hash the
    PNG: save() writes the footer before the file exists, so a PNG hash could only ever describe
    the previous render.
    """
    import hashlib
    h = hashlib.md5(open(SCRIPT, "rb").read()).hexdigest()[:10]
    return h + ":" + ":".join(str(q) for q in parts)
EVAL_PT = os.path.expanduser("~/hack-data/C9_robust_tagging/eval/robust_tagging_eval_small.pt")
SEED = 20260904
SURVIVOR = "#0072B2"
DROPPED = "#9A9A9A"
C_FAMS = ("c_rect", "c_eta_band", "c_phi_wedge", "c_multi_patch", "c_cand_loss")
OUR_FAMS = ("rect", "wedge", "strip")
LADDER = (0.2, 0.4, 0.6, 0.8, 1.0)


def load_events(n, path=EVAL_PT):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    t = obj[0] if isinstance(obj, (list, tuple)) else obj
    if isinstance(t, dict):
        t = t.get("features", t.get("data", next(iter(t.values()))))
    return t.float()[:n, :, :7].clone()


def family_mask(deg, x, fam, s):
    """The same helpers `_forward_train` dispatches to, at an exact per-event severity."""
    eta, phi = x[..., 1], x[..., 2]
    valid = x[..., 0] > 0
    if fam == "rect":
        return deg._drop_rect(eta, phi, s, None, valid=valid)
    if fam == "wedge":
        return deg._drop_bands(phi, s, 2 * math.pi, 0.1, 0.4, 0.6, 1.0, 16, True, None)
    if fam == "strip":
        if deg.on_target_placement:
            return deg._drop_bands(deg._eta_cdf(eta, valid), s, 1.0,
                                   1.0 / deg.eta_span, 2.5 / deg.eta_span, 0.5, 1.0, 8, False, None)
        return deg._drop_bands(eta, s, deg.eta_span, 1.0, 2.5, 0.5, 1.0, 8, False, None)
    if fam == "cells":
        return deg._drop_cells_multi(eta, phi, s, (0.25, 0.5, 1.0), None)
    if fam == "towers":
        return deg._drop_cells(eta, phi, s, 0.1, None, p_lo=1.0, p_hi=1.0, compensate=False)
    if fam == "c_cand_loss":
        return deg._c_cand_loss(eta, s, None)
    if fam == "c_eta_band":
        return deg._c_eta_band(eta, s, None)
    if fam == "c_phi_wedge":
        return deg._c_phi_wedge(phi, s, None)
    if fam == "c_rect":
        return deg._c_rect(eta, phi, s, None)
    if fam == "c_multi_patch":
        return deg._c_multi_patch(eta, phi, s, None)
    raise ValueError(fam)


def panel(ax, x_ev, drop, title):
    pt, eta, phi = x_ev[..., 0], x_ev[..., 1], x_ev[..., 2]
    valid = pt > 0
    size = 3.0 + 22.0 * (torch.log1p(pt) / max(float(torch.log1p(pt[valid]).max()), 1e-6)) ** 2
    keep, gone = valid & ~drop, valid & drop
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


def colleague_families(n_events=3):
    import matplotlib.pyplot as plt
    from embedding.degradation import Degradation

    x = load_events(n_events)
    deg = Degradation(severity=None)
    torch.manual_seed(SEED)
    s_sampled = torch.rand(x.shape[0]) * 0.85

    ncol, nrow = 1 + len(C_FAMS), n_events + 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.55 * ncol, 2.45 * nrow), squeeze=False)
    for i in range(n_events):
        xe = x[i:i + 1]
        nvalid = int((xe[0, :, 0] > 0).sum())
        panel(axes[i][0], xe[0], torch.zeros(xe.shape[1], dtype=torch.bool),
              f"event {i}: clean, {nvalid} candidates")
        for j, fam in enumerate(C_FAMS):
            torch.manual_seed(SEED + 1000 * i + j)
            s = s_sampled[i:i + 1]
            m = family_mask(deg, xe, fam, s)[0] & (xe[0, :, 0] > 0)
            panel(axes[i][j + 1], xe[0], m,
                  f"{fam}  s={float(s):.2f}  dropped {float(m.sum()) / max(nvalid, 1):.2f}")

    # Bottom row: one family across the severity ladder, one seed for the whole row so the region
    # grows rather than being redrawn (WP-F's point; reseeding per rung reads as a generator bug).
    xe = x[0:1]
    nvalid = int((xe[0, :, 0] > 0).sum())
    panel(axes[nrow - 1][0], xe[0], torch.zeros(xe.shape[1], dtype=torch.bool),
          f"event 0: clean, {nvalid} candidates")
    for j, s_val in enumerate(LADDER):
        torch.manual_seed(SEED + 77)
        s = torch.full((1,), float(s_val))
        m = family_mask(deg, xe, "c_rect", s)[0] & (xe[0, :, 0] > 0)
        panel(axes[nrow - 1][j + 1], xe[0], m,
              f"c_rect  s={s_val:.1f}  dropped {float(m.sum()) / max(nvalid, 1):.2f}")

    for r in range(nrow):
        axes[r][0].set_ylabel("phi", fontsize=8)
    for c in range(ncol):
        axes[nrow - 1][c].set_xlabel("eta", fontsize=8)
    fig.suptitle("WP-J J1: Group 3's five TRAINING families, ported verbatim (their eta [-3,3] "
                 "replaced by ours). Blue survives, grey x dropped, marker area ~ log pt",
                 fontsize=11)
    # Two explicit lines rather than wrap=True: wrapping is to the FIGURE width, so the text ran
    # to both margins and clipped. Fixed width, centred, with the rect leaving room below it.
    fig.text(0.5, 0.030,
             "Every panel is ONE event, so its dropped fraction is a single sample, not the "
             "family's expectation.\nThe bottom ladder tracks s closely for this event; over 4000 "
             "events c_rect reaches 0.841 at s=1.0, not 1.00.\nThe per-family calibration is the "
             "claim; the table is in reports/j-0345.md.",
             ha="center", va="bottom", fontsize=8.5, color="#555555", linespacing=1.5)
    fig.tight_layout(rect=(0, 0.085, 1, 0.965))
    return pc.save(fig, "j1_colleague_families", SCRIPT,
                   "train mode with rotate_phi/reflect_eta/p_pt_scale/p_clean off so the dead "
                   f"region lines up with the clean panel (WP-B recipe A) · seed {SEED}",
                   content_key=content_key("j1", SEED, n_events))


def placement(n_cal=4000):
    """J5: the same three families before and after the placement fix, plus their calibration."""
    import matplotlib.pyplot as plt
    from embedding.degradation import Degradation

    x = load_events(max(n_cal, 1))
    xe = x[0:1]
    nvalid = int((xe[0, :, 0] > 0).sum())
    valid_all = x[..., 0] > 0
    eta_all, phi_all = x[..., 1], x[..., 2]

    fig = plt.figure(figsize=(16.0, 9.6))
    gs = fig.add_gridspec(3, 5, width_ratios=[1, 1, 1, 1, 1.25], hspace=0.34, wspace=0.26)

    cal = {}
    for r, fam in enumerate(OUR_FAMS):
        for c, (otp, label) in enumerate(((False, "champion"), (True, "J5"))):
            deg = Degradation(severity=None, on_target_placement=otp)
            for k, s_val in enumerate((0.6, 1.0)):
                torch.manual_seed(SEED + 7 * r)
                s = torch.full((1,), float(s_val))
                m = family_mask(deg, xe, fam, s)[0] & (xe[0, :, 0] > 0)
                ax = fig.add_subplot(gs[r, c * 2 + k])
                panel(ax, xe[0], m,
                      f"{fam} · {label} · s={s_val:.1f} · dropped {float(m.sum()) / max(nvalid, 1):.2f}")
                if c * 2 + k == 0:
                    ax.set_ylabel("phi", fontsize=8)
                if r == 2:
                    ax.set_xlabel("eta", fontsize=8)
            # calibration over the whole slice, which is the claim the panels only illustrate
            row = []
            for s_val in LADDER:
                torch.manual_seed(SEED + 31)
                sv = torch.full((x.shape[0],), float(s_val))
                mm = family_mask(deg, x, fam, sv) & valid_all
                row.append(float(mm.sum()) / float(valid_all.sum()))
            cal[(fam, label)] = row

        ax = fig.add_subplot(gs[r, 4])
        ax.plot(LADDER, LADDER, color="#999999", linestyle=":", linewidth=1.2, label="target = s")
        ax.plot(LADDER, cal[(fam, "champion")], color="#D55E00", marker="s", markersize=4,
                linewidth=1.6, label="champion")
        ax.plot(LADDER, cal[(fam, "J5")], color="#0072B2", marker="o", markersize=4,
                linewidth=1.6, label="J5")
        e0 = np.mean([abs(a - b) for a, b in zip(cal[(fam, "champion")], LADDER)])
        e1 = np.mean([abs(a - b) for a, b in zip(cal[(fam, "J5")], LADDER)])
        ax.set_title(f"{fam}: mean |err| {e0:.3f} -> {e1:.3f}", fontsize=8.5)
        ax.set_xlim(0.1, 1.05)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper left")
        ax.set_ylabel("fraction dropped", fontsize=8)
        if r == 2:
            ax.set_xlabel("requested severity s", fontsize=8)

    fig.suptitle("WP-J J5: our three eta-shaped families before and after the placement fix. "
                 f"Panels are one event; the right column is all {n_cal} events, which is the claim",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.035, 1, 0.955))
    return pc.save(fig, "j5_placement", SCRIPT,
                   "panels share one seed per family row so champion and J5 draw the same shape "
                   f"priors and only the placement and scaling differ · seed {SEED}",
                   content_key=content_key("j5", SEED, n_cal))


def generate():
    pc.use_agg()
    import matplotlib.pyplot as plt
    written = []
    for fn in (colleague_families, placement):
        try:
            written += fn() or []
        except Exception as exc:
            pc.log(f"{fn.__name__} failed: {exc!r}")
    plt.close("all")
    pc.log("wrote " + ", ".join(os.path.basename(w) for w in written))
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.expanduser("~/c2-j"),
                    help="repo whose src/embedding/degradation.py is plotted")
    ap.add_argument("--events", type=int, default=3)
    args = ap.parse_args()
    sys.path.insert(0, os.path.join(os.path.expanduser(args.repo), "src"))
    generate()
