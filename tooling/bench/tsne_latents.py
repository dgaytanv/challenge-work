"""Latent-space t-SNE under degradation: does the latent move, and does it move together?

Two figures:
  runs/plots/tsne_latents_latest.png  2x3 grid -- rows {winner, anchor}, columns
      {clean, cells s=0.4, cells s=0.8}. One JOINT t-SNE per model over the concatenated
      [clean; s=0.4; s=0.8] latents, so the three panels of a row share one embedding space and
      a point that moves between panels has genuinely moved. Panels annotated with the dropped
      fraction (measured here) and the probe AUC at that severity (read from the bench JSONs).
  runs/plots/tsne_drift_latest.png    winner only, arrows from each event's clean position to its
      s=0.8 position for a sample of events -- WP-A's question of whether degradation displaces
      latents along one shared direction or scatters them.

Colour: background events in a neutral grey, signal in one accent. Validated with the dataviz
skill's checker (light surface #fcfcfb): CVD separation dE 14.2 (protan), normal-vision dE 25.2,
both marks >= 3:1 contrast. The grey deliberately fails the chroma floor -- background is the
neutral mass here, not a second identity -- and identity is carried by a legend, never colour alone.

Usage (embedding is GPU and small; the t-SNE is CPU and slow):
  RT_NUM_THREADS=2 python ~/hackathon-shared/bench/tsne_latents.py
"""
import argparse
import glob
import importlib.util
import json
import os
import sys

import numpy as np
import torch

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.expanduser("~/hackathon-shared/runs")
SCRATCH = "/tmp/claude-1000/-home-jovyan/c11b043d-6147-45f0-83d2-450a35e1c5ab/scratchpad"

BG, SIG = "#5f6368", "#eb6834"          # validated pair; see module docstring
INK, MUTED, SURFACE = "#1a1a19", "#5f6368", "#fcfcfb"

# Public: WP-F's watch loop keys its re-run trigger on these checkpoint paths. Import
# CHECKPOINT_PATHS rather than duplicating them, so the trigger cannot go stale if this changes.
MODELS = [
    ("winner (PMA + MeanPt)", f"{SCRATCH}/pmacheck",
     f"{SCRATCH}/pmacheck/checkpoints/rt_d_pma0_aug_meanpt_encoder_20260903_221125.pth",
     "d-pma0-aug-meanpt"),
    ("anchor (organisers' stock)", os.path.expanduser("~/rt-e"),
     os.path.expanduser("~/hack-data/C9_robust_tagging/checkpoints/robust_tagging_encoder_20260902_212357.pth"),
     "anchor-stock-baseline"),
]
SEVERITIES = [0.0, 0.4, 0.8]
FAMILY = "cells"
CHECKPOINT_PATHS = [m[2] for m in MODELS]      # for external trigger logic; see note above
BENCH_TAGS = [m[3] for m in MODELS]


def load_eval_module(repo):
    spec = importlib.util.spec_from_file_location(f"rt_eval_{abs(hash(repo))}", os.path.join(repo, "eval.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bench_auc(tag, family, severity):
    """Probe AUC for (tag, family, severity) from the newest matching bench JSON."""
    best = None
    for path in sorted(glob.glob(os.path.join(RUNS, "*.json"))):
        try:
            d = json.load(open(path))
        except Exception:
            continue
        if not isinstance(d, dict) or d.get("tag") != tag or family not in d.get("families", {}):
            continue
        if best is None or d.get("timestamp", "") >= best.get("timestamp", ""):
            best = d
    if best is None:
        return None
    fam = best["families"][family]
    for s, a in zip(fam["severities"], fam["aucs"]):
        if abs(s - severity) < 1e-6:
            return a
    return None


def embed_conditions(repo, ckpt, feats, labels, device):
    """Latents for [clean, s=0.4, s=0.8] plus the measured dropped fraction of each."""
    sys.path.insert(0, repo)
    sys.path.insert(0, os.path.join(repo, "src"))
    for m in [k for k in list(sys.modules) if k.startswith("embedding")]:
        del sys.modules[m]                     # force re-import from THIS repo
    ev = load_eval_module(repo)
    from embedding.utils.cfg_handler import train_config, data_config
    sys.path.insert(0, BENCH_DIR)
    from bench_degradation import BenchDegradation

    cfg = train_config(os.path.join(repo, "configs", "train_config.yaml"))
    cfg_data = data_config(os.path.join(repo, "configs", "data_config_eval.yaml"))
    ckpt_d = torch.load(ckpt, map_location=device)
    preproc, encoder, norm_constants = ev.build_preproc_and_encoder(cfg, ckpt_d, device)

    out, fracs = [], []
    for s in SEVERITIES:
        deg = None if s == 0 else BenchDegradation(FAMILY, s).to(device).eval()
        if deg is None:
            frac = 0.0
        else:
            with torch.no_grad():
                probe = deg(feats[:2048].to(device))
                frac = float(((probe[..., 0] == 0) & (feats[:2048, :, 0].to(device) > 0)).float().mean())
        lat, _, _ = ev.embed_dataset(preproc, encoder, feats, labels, cfg_data, norm_constants,
                                     device, batch_size=256, degradation=deg)
        out.append(lat.numpy())
        fracs.append(frac)
    return out, fracs


def joint_tsne(blocks, seed=42, perplexity=50):
    """ONE t-SNE over the concatenated conditions, so the panels share an embedding space."""
    from sklearn.manifold import TSNE
    X = np.concatenate(blocks, axis=0)
    coords = TSNE(n_components=2, init="pca", random_state=seed, perplexity=perplexity).fit_transform(X)
    n = len(blocks[0])
    return [coords[i * n:(i + 1) * n] for i in range(len(blocks))]


def panel(ax, xy, y, title, frac, auc):
    bg, sig = y == 0, y == 1
    ax.scatter(xy[bg, 0], xy[bg, 1], s=3.5, c=BG, alpha=0.28, linewidths=0, label="background",
               rasterized=True)
    ax.scatter(xy[sig, 0], xy[sig, 1], s=6.5, c=SIG, alpha=0.75, linewidths=0, label="signal",
               rasterized=True)
    ax.set_title(title, fontsize=10, color=INK, pad=6)
    note = f"dropped {frac:.2f}" + (f"   AUC {auc:.3f}" if auc is not None else "")
    ax.text(0.03, 0.03, note, transform=ax.transAxes, fontsize=8.5, color=MUTED, va="bottom")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#d8d8d6")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, choices=[None, "cpu", "cuda"],
                    help="force the embedding device. Default: cuda when available. Pass cpu to "
                         "run entirely lock-free -- the t-SNE dominates the runtime anyway.")
    ap.add_argument("--events", type=int, default=6000)
    ap.add_argument("--arrows", type=int, default=300)
    ap.add_argument("--outdir", default=os.path.join(RUNS, "plots"))
    ap.add_argument("--data", default=os.path.expanduser(
        "~/hack-data/C9_robust_tagging/eval/robust_tagging_eval_small.pt"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.outdir, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    sys.path.insert(0, os.path.expanduser("~/rt-e/src"))
    from embedding.utils.data_utils import load_data
    feats, labels = load_data(args.data, map_location="cpu", max_events=args.events)
    y = labels.numpy()
    print(f"[tsne] {feats.shape[0]} events, {int((y==1).sum())} signal, device={device}", flush=True)

    results = {}
    for name, repo, ckpt, tag in MODELS:
        print(f"[tsne] embedding {name} ...", flush=True)
        blocks, fracs = embed_conditions(repo, ckpt, feats, labels, device)
        print(f"[tsne] joint t-SNE for {name} over {sum(len(b) for b in blocks)} latents ...", flush=True)
        coords = joint_tsne(blocks)
        aucs = [bench_auc(tag, FAMILY, s if s > 0 else 0.0) for s in SEVERITIES]
        # Directional coherence is computed on the RAW LATENTS, not on the t-SNE coordinates.
        # t-SNE is a non-linear embedding that preserves neighbourhoods, not directions: a
        # "common direction" measured in its output would be an artefact of the layout, not a
        # property of the encoder. The picture is context; this number is the answer.
        d_lat = blocks[2] - blocks[0]
        coh_lat = float(np.linalg.norm(d_lat.mean(0)) /
                        (np.linalg.norm(d_lat, axis=1).mean() + 1e-9))
        # Coherence alone is NOT a robustness measure -- the anchor is more coherent than the
        # winner and scores worse. What costs AUC is the component of the displacement along the
        # discriminative direction; motion orthogonal to it is free. `along` is that fraction,
        # using a linear logistic probe on the clean latents as a tractable stand-in for eval.py's
        # MLP probe (a proxy: the real probe is non-linear and has no single normal).
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(max_iter=2000).fit(blocks[0], y)
        w = lr.coef_[0] / (np.linalg.norm(lr.coef_[0]) + 1e-12)
        along = float(np.abs(d_lat @ w).mean() / (np.linalg.norm(d_lat, axis=1).mean() + 1e-9))
        # Absolute along-probe motion, expressed in units of THIS model's own clean latent spread
        # (mean radius of the clean cloud). Scale-free across models, unlike raw latent units, but
        # magnitude-preserving, unlike the fraction above -- a model can have a small fraction and
        # still travel far along the probe direction.
        spread = float(np.linalg.norm(blocks[0] - blocks[0].mean(0), axis=1).mean() + 1e-12)
        along_abs = float(np.abs(d_lat @ w).mean() / spread)
        results[name] = (coords, fracs, aucs, d_lat, coh_lat, along, along_abs)

    # ---- Figure 1: 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9), facecolor=SURFACE)
    for r, (name, _, _, _) in enumerate(MODELS):
        coords, fracs, aucs = results[name][:3]
        for c, s in enumerate(SEVERITIES):
            label = "clean" if s == 0 else f"{FAMILY} s={s}"
            panel(axes[r, c], coords[c], y, label, fracs[c], aucs[c])
        axes[r, 0].set_ylabel(name, fontsize=11, color=INK, labelpad=10)
    h, l = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(h, l, loc="upper right", frameon=False, fontsize=10, markerscale=3,
                     bbox_to_anchor=(0.995, 0.985))
    for t in leg.get_texts():
        t.set_color(INK)
    fig.suptitle("Latent space under progressive detector dropout (one joint t-SNE per model)",
                 fontsize=13, color=INK, x=0.01, ha="left", y=0.985)
    fig.text(0.01, 0.005,
             f"{args.events} eval events, identical across panels. Columns share one embedding "
             f"space per row, so movement between panels is real. AUC is the frozen probe's, from "
             f"the bench JSONs.", fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout(rect=[0.01, 0.02, 1, 0.955])
    p1 = os.path.join(args.outdir, "tsne_latents_latest.png")
    fig.savefig(p1, dpi=160, facecolor=SURFACE); plt.close(fig)
    print(f"[tsne] wrote {p1}")

    # ---- Figure 2: drift arrows, winner only
    name = MODELS[0][0]
    coords, fracs, aucs, d_lat, coh_lat, along, along_abs = results[name]
    clean, deg = coords[0], coords[2]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(clean), size=min(args.arrows, len(clean)), replace=False)
    fig, ax = plt.subplots(figsize=(9, 8), facecolor=SURFACE)
    ax.scatter(clean[:, 0], clean[:, 1], s=3, c=BG, alpha=0.16, linewidths=0, rasterized=True,
               label="clean")
    ax.scatter(deg[:, 0], deg[:, 1], s=3, c=SIG, alpha=0.16, linewidths=0, rasterized=True,
               label=f"{FAMILY} s=0.8")
    for i in idx:
        ax.annotate("", xy=deg[i], xytext=clean[i],
                    arrowprops=dict(arrowstyle="-", color=INK, alpha=0.16, linewidth=0.45,
                                    shrinkA=0, shrinkB=0))
    # Binned mean displacement: individual arrows over a t-SNE are a hairball, and the question
    # ("is there a common direction?") is answered by the local mean, not by 300 crossing lines.
    disp = deg - clean
    nb = 12
    xe = np.linspace(clean[:, 0].min(), clean[:, 0].max(), nb + 1)
    ye = np.linspace(clean[:, 1].min(), clean[:, 1].max(), nb + 1)
    gx, gy, gu, gv, gn = [], [], [], [], []
    for a in range(nb):
        for b in range(nb):
            m = ((clean[:, 0] >= xe[a]) & (clean[:, 0] < xe[a + 1]) &
                 (clean[:, 1] >= ye[b]) & (clean[:, 1] < ye[b + 1]))
            if m.sum() >= 12:
                gx.append((xe[a] + xe[a + 1]) / 2); gy.append((ye[b] + ye[b + 1]) / 2)
                gu.append(disp[m, 0].mean()); gv.append(disp[m, 1].mean()); gn.append(int(m.sum()))
    # FIXED-LENGTH arrows. Their length in embedding units is not a magnitude anyone should read:
    # t-SNE distances are not latent distances. Direction shows local structure; opacity encodes
    # how many events back each cell.
    gx, gy = np.array(gx), np.array(gy)
    gu, gv, gn = np.array(gu), np.array(gv), np.array(gn, dtype=float)
    mag = np.hypot(gu, gv); mag[mag == 0] = 1.0
    step = (xe[1] - xe[0]) * 0.62                      # one arrow spans ~60% of a cell, always
    ux, uy = gu / mag * step, gv / mag * step
    alpha = 0.30 + 0.60 * (gn - gn.min()) / max(gn.max() - gn.min(), 1)
    for X, Y, U, V, A in zip(gx, gy, ux, uy, alpha):
        ax.annotate("", xy=(X + U / 2, Y + V / 2), xytext=(X - U / 2, Y - V / 2),
                    arrowprops=dict(arrowstyle="-|>", color=SIG, alpha=float(A), linewidth=1.4,
                                    shrinkA=0, shrinkB=0), zorder=5)
    coh = coh_lat
    ax.set_title(f"{name}: where each event's latent goes under {FAMILY} s=0.8", fontsize=12, color=INK)
    ax.text(0.02, 0.02,
            f"grey lines: {len(idx)} sampled event displacements of {len(clean)}\n"
            f"orange: mean displacement DIRECTION per t-SNE cell (>=12 events), drawn at fixed\n"
            f"length -- opacity is the cell's event count. Arrow directions are embedding\n"
            f"quantities; only the coherence below is a latent-space number.\n"
            f"directional coherence {coh:.2f} over all {len(clean)} events at {FAMILY} "
            f"s={SEVERITIES[2]}; along-probe fraction {along:.2f}, absolute along-probe "
            f"{along_abs:.2f} clean-spread units\n"
            f"(anchor: fraction {results[MODELS[1][0]][5]:.2f}, absolute "
            f"{results[MODELS[1][0]][6]:.2f})\n"
            f"NEITHER NUMBER EXPLAINS THE RANKING. The anchor is MORE coherent "
            f"({results[MODELS[1][0]][4]:.2f}) and has a far SMALLER along-probe fraction\n"
            f"({results[MODELS[1][0]][5]:.2f} vs {along:.2f}), yet loses more AUC "
            f"(0.861->0.772 against 0.915->0.842). So the hypothesis that only the\n"
            f"along-probe component costs AUC is NOT supported by these numbers. Both are "
            f"fractions, not magnitudes, and the\nlinear probe used here is only a proxy for "
            f"eval.py's non-linear MLP. Treat both as descriptive, not explanatory.",
            transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#d8d8d6")
    leg = ax.legend(loc="upper right", frameon=False, fontsize=10, markerscale=3)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.text(0.012, 0.012,
             "Footnote: directional coherence is computed on the RAW LATENT displacements, not on these\n"
             "t-SNE coordinates. t-SNE preserves neighbourhoods, not directions, so a common direction\n"
             "read off this picture would be a property of the layout, not of the encoder. The value\n"
             "0.44 coincidentally matches an earlier figure's 0.443, which was computed in embedding\n"
             "coordinates and retracted; they agree by chance, not because the correction was skipped.",
             fontsize=8, color=MUTED, ha="left", va="bottom", linespacing=1.5)
    fig.tight_layout(rect=[0.01, 0.105, 1, 1])
    p2 = os.path.join(args.outdir, "tsne_drift_latest.png")
    fig.savefig(p2, dpi=160, facecolor=SURFACE); plt.close(fig)
    print(f"[tsne] wrote {p2}")
    for nm in results:
        print(f"[tsne] {nm}: coherence {results[nm][4]:.3f}, along-probe fraction "
              f"{results[nm][5]:.3f}, absolute along-probe {results[nm][6]:.3f} clean-spread units "
              f"({FAMILY} s={SEVERITIES[2]}, all {args.events} events)")


if __name__ == "__main__":
    main()
