"""WP-N: architecture diagram of the certified champion. Boxes and arrows, matplotlib only.

Every parameter count is READ FROM THE MODEL, not typed: the script builds PMAEncoder and
PFPreProcessorMeanPt and counts, then asserts the encoder total is 89,606. A diagram with
hand-typed counts is register class 21 (a number from memory beside sourced ones) waiting to happen.
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_common as pc

SCRIPT = os.path.abspath(__file__)
INK, MUTED, ACC, DASH = "#222222", "#666666", "#0072B2", "#B22222"


def counts():
    repo = os.path.expanduser("~/c2-n")
    sys.path.insert(0, os.path.join(repo, "src"))
    from embedding.models import PMAEncoder
    from embedding.preprocs import PFPreProcessorMeanPt
    pp = PFPreProcessorMeanPt({})
    enc = PMAEncoder(num_features=pp.num_features, embed_size=128, latent_dim=6, num_heads=8,
                     num_layers=0, linear_dim=None, num_tokens=None, pairwise=False)
    c = {n: sum(p.numel() for p in m.parameters()) for n, m in enc.named_children()}
    c["preproc"] = sum(p.numel() for p in pp.parameters())
    c["total"] = sum(p.numel() for p in enc.parameters())
    assert c["total"] == 89606, f"encoder total is {c['total']}, expected 89606"
    return c


def box(ax, x, y, w, h, title, lines, params=None, color=INK, lw=1.4, ls="-"):
    """Title, then the parameter count directly under it, then the body. The first version put the
    count at the box FLOOR and the body flowed down into it -- 'params' printed on top of a body
    line in four of five boxes. Anchoring both to the TOP means the body can only run out of box,
    which the layout gate can see, rather than colliding with something, which it reported but did
    not block on."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010,rounding_size=0.014",
                                fc="white", ec=color, lw=lw, ls=ls, zorder=3))
    ax.text(x + w / 2, y + h - 0.040, title, ha="center", va="top", fontsize=9.2,
            fontweight="bold", color=color, zorder=4)
    top = y + h - 0.088
    if params is not None:
        ax.text(x + w / 2, top, f"{params:,} params", ha="center", va="top",
                fontsize=8.0, color=ACC, fontweight="bold", zorder=4)
        top -= 0.040
    ax.text(x + w / 2, top, "\n".join(lines), ha="center", va="top", fontsize=7.1,
            color=INK, zorder=4, linespacing=1.55)


def arrow(ax, x0, x1, y, label):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=13,
                                 color=MUTED, lw=1.2, zorder=2))
    ax.text((x0 + x1) / 2, y + 0.018, label, ha="center", va="bottom", fontsize=7.0, color=MUTED)


def main():
    c = counts()
    fig, ax = plt.subplots(figsize=(18.0, 8.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    yb, hb = 0.505, 0.375
    GAP = 0.056
    ws = [0.126, 0.157, 0.143, 0.168, 0.140]
    xs, cur = [], 0.010
    for w in ws:
        xs.append(cur); cur += w + GAP

    box(ax, xs[0], yb, ws[0], hb, "input event",
        ["a SET of up to N candidates", "N = 200 train / 400 eval", "", "7 raw features:",
         "pt, eta, phi, dxy,", "dxy_sig, is_pf, pdgId", "", "dead candidates are",
         "all-zero rows"])
    arrow(ax, xs[0] + ws[0], xs[1], yb + hb / 2, "[N, 7]")

    box(ax, xs[1], yb, ws[1], hb, "PFPreProcessorMeanPt",
        ["log(pt / mean surviving pt)", "eta, phi, tanh(dxy), dxy_sig,", "is_pf, pdgId one-hot (7)",
         "= 14 features", "", "BatchNorm over the 5", "continuous features", "",
         "attention mask derived from the", "zeroed rows (the mask fix)"], params=c["preproc"])
    arrow(ax, xs[1] + ws[1], xs[2], yb + hb / 2, "[N, 14]")

    box(ax, xs[2], yb, ws[2], hb, "per-candidate MLP  phi",
        ["Linear 14 -> 128", "LayerNorm, GELU", "Linear 128 -> 128", "LayerNorm, GELU",
         "", "shared across all", "candidates"], params=c["phi"])
    arrow(ax, xs[2] + ws[2], xs[3], yb + hb / 2, "[N, 128]")

    box(ax, xs[3], yb, ws[3], hb, "PMA pooling",
        ["4 learned seed queries x 8 heads", "attend over the SURVIVING",
         "candidates (masked softmax)", "then out_proj", "",
         "output [4, 128] -> flatten 512", "", "num_layers = 0:", "NO transformer body"], params=c["pma"], color=ACC, lw=1.9)
    arrow(ax, xs[3] + ws[3], xs[4], yb + hb / 2, "[512]")

    box(ax, xs[4], yb, ws[4], hb, "readout",
        ["LayerNorm(512)", f"{c['norm_pooled']:,}", "", "Linear 512 -> 6",
         f"{c['bottleneck']:,}", "", "= THE LATENT", "the grader scores"],
        params=c["norm_pooled"] + c["bottleneck"], color=ACC, lw=1.9)

    ax.text(xs[4] + ws[4] / 2, yb - 0.035, "latent [6]", ha="center", va="top", fontsize=8.6,
            fontweight="bold", color=ACC)

    yt, ht = 0.075, 0.315
    box(ax, 0.155, yt, 0.36, ht, "TRAINING ONLY  (not shipped in the scored path)",
        ["dead-region augmentation on the INPUT:", "five families, phi rotation, eta reflection",
         "", "Projector 6 -> 12  ->  classifier (4 classes)", "",
         "loss = cross-entropy  +  0.05 x supervised contrastive"], color=DASH, ls="--", lw=1.3)
    ax.add_patch(FancyArrowPatch((xs[2] + ws[2] / 2, yb), (0.36, yt + ht), arrowstyle="-|>", mutation_scale=11,
                                 color=DASH, lw=1.1, ls="--", zorder=2))

    box(ax, 0.575, yt, 0.345, ht, "GRADER  (frozen probe)",
        ["EvalMLP probe FIT on CLEAN latents", "then APPLIED to DEGRADED latents",
         "of the same events", "", "score = area under AUC vs severity", "",
         "so the target is: the degraded latent must land", "where the clean latent landed"],
        color=DASH, ls="--", lw=1.3)
    ax.add_patch(FancyArrowPatch((xs[4] + ws[4] / 2, yb), (0.835, yt + ht), arrowstyle="-|>", mutation_scale=11,
                                 color=DASH, lw=1.1, ls="--", zorder=2))

    ax.text(0.5, 0.975, "Certified champion: PMAEncoder(num_layers=0) + PFPreProcessorMeanPt",
            ha="center", va="top", fontsize=13.5, fontweight="bold", color=INK)
    ax.text(0.5, 0.935, f"{c['total']:,} encoder parameters  ·  a permutation-invariant set encoder: "
                        f"no transformer body, no positional structure",
            ha="center", va="top", fontsize=9.2, color=MUTED)

    import hashlib
    key = hashlib.sha256((repr(sorted(c.items())) + open(SCRIPT, "rb").read().hex()[:64])
                         .encode()).hexdigest()[:12]
    pc.save(fig, "architecture", SCRIPT,
            f"PMAEncoder(num_layers=0) + PFPreProcessorMeanPt, {c['total']:,} encoder parameters; "
            f"submission-pma0-meanpt @41d45a9", content_key=key)
    open(os.path.join(pc.PLOTS_DIR, "architecture_key.txt"), "w").write(key)
    print({k: v for k, v in sorted(c.items())})


if __name__ == "__main__":
    main()
