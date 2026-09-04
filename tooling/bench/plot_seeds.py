"""WP-N: seed-spread figure for the ruling. One panel: every Phase-1 configuration's seeds as
points, the reference band shaded, dev bench only.

Two things this figure exists to prevent, both from the register:
  * class 7 / class 1 -- two benches of the SAME checkpoint are two probe measurements, not two
    seeds. They are merged on ckpt_sha256 (falling back to the checkpoint basename), and the
    spread between them is reported separately as the probe floor. Plotting them as two seed
    points would inflate the seed std with probe noise, which is the exact confusion class 22 is
    about, running the other way.
  * class 24 -- an n=1 configuration has an UNMEASURED seed spread, not a zero one. Those are
    drawn as a bare point and labelled "1 seed", never with an error bar.
"""
import glob, json, os, re, sys, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_common as pc

SCRIPT = os.path.abspath(__file__)
REF = "h-ref-small"


def load():
    rows = []
    for f in sorted(glob.glob(os.path.join(pc.RUNS_DIR, "*.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if "families" not in d or "mean_area" not in d or d.get("seed") is None:
            continue
        if not pc.is_scoring(d):
            continue                      # dev bench only, per the order
        if d["tag"].startswith("n-audit"):
            continue                      # my own re-measurement is not a configuration
        d["_file"] = os.path.basename(f)
        rows.append(d)
    return rows


def main():
    rows = load()
    cfg = collections.defaultdict(dict)   # base -> seed -> list of runs
    for r in rows:
        base = re.sub(r"-s\d+$", "", r["tag"])
        cfg[base].setdefault(r["seed"], []).append(r)

    order = [REF] + sorted(b for b in cfg if b != REF)
    order = [b for b in order if b in cfg]

    # Height must cover the notes too: each merged-duplicate row adds a footnote line, and with
    # 19 configurations the note block grew past the bottom of the canvas and the layout gate
    # (correctly) refused to publish. Compute the note count BEFORE sizing.
    n_notes = 1 + sum(1 for b in cfg for sd in [cfg[b]]
                      for seed, runs in cfg[b].items()
                      if len(runs) > 1)
    fig, ax = plt.subplots(figsize=(11.0, 0.42 * len(order) + 3.0 + 0.17 * n_notes))
    ref_mean = ref_sd = None
    dupes = []
    stats = {}
    for base in order:
        per_seed = []
        for seed, runs in sorted(cfg[base].items()):
            ident = {(r.get("ckpt_sha256") or os.path.basename(r.get("ckpt") or "")) for r in runs}
            if len(runs) > 1 and len(ident) == 1:
                vals = [r["mean_area"] for r in runs]
                dupes.append((base, seed, max(vals) - min(vals), len(runs)))
                per_seed.append((seed, float(np.mean(vals)), runs))
            else:
                for r in runs:
                    per_seed.append((seed, r["mean_area"], [r]))
        ms = [m for _, m, _ in per_seed]
        stats[base] = (per_seed, float(np.mean(ms)),
                       float(np.std(ms, ddof=1)) if len(ms) > 1 else None)
    if REF in stats:
        _, ref_mean, ref_sd = stats[REF]

    if ref_mean is not None and ref_sd is not None:
        ax.axvspan(ref_mean - ref_sd, ref_mean + ref_sd, color="#D62728", alpha=0.13, zorder=0,
                   label=f"reference {ref_mean:.4f} ± {ref_sd:.4f} (seed std, n=3)")
        ax.axvline(ref_mean, color="#D62728", ls="--", lw=1.4, zorder=1)

    for i, base in enumerate(order):
        per_seed, mean, sd = stats[base]
        y = len(order) - 1 - i
        col = pc.LEADER_COLOR if base == REF else "#333333"
        for seed, m, _ in per_seed:
            ax.plot([m], [y], "o", ms=6, color=col, zorder=4)
            dy = (8, -13, 15)[[sd for sd, _, _ in per_seed].index(seed) % 3]
            ax.annotate(f"s{seed}", (m, y), textcoords="offset points", xytext=(0, dy),
                        ha="center", fontsize=6.2, color="#666666")
        if sd is not None:
            ax.errorbar([mean], [y], xerr=[sd], fmt="D", ms=7, color=col, capsize=4,
                        elinewidth=1.4, zorder=5)
            lab = f"{mean:.4f} ± {sd:.4f}  (n={len(per_seed)})"
        else:
            lab = f"{mean:.4f}   1 seed (spread unmeasured)"
        ax.text(1.005, y, lab, transform=ax.get_yaxis_transform(), va="center",
                fontsize=7.4, color="#222222")

    # The ruling is about whether anything clears a band 0.0030 wide, and two collapsed arms
    # (k5_whiten 0.70, k2_normlatent 0.80) stretched the axis until that band was a sliver. Focus
    # the axis on the decision region and park off-range rows at the edge with their value printed
    # beside them -- nothing is hidden, and the comparison the ruling turns on becomes readable.
    allm = [m for b in order for _, m, _ in stats[b][0]]
    lo = min([m for m in allm if m > (ref_mean or 0) - 0.02] or allm)
    hi = max(allm)
    pad = max(0.002, 0.12 * (hi - lo))
    xlo, xhi = lo - pad, hi + pad
    off = [(b, stats[b][1]) for b in order if stats[b][1] < xlo]
    for b, m in off:
        y = len(order) - 1 - order.index(b)
        ax.annotate(f"◀ {m:.4f}", (xlo, y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=7.2, color="#B22222", fontweight="bold")
    ax.set_xlim(xlo, xhi)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([order[len(order) - 1 - k] for k in range(len(order))], fontsize=8)
    ax.set_xlabel("dev-bench mean_area (five scoring families, 20k events)")
    ax.set_title("Phase 1: every configuration's seeds, against the reference seed band")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="lower left", fontsize=7.4)
    ax.set_ylim(-0.8, len(order) - 0.2)

    note = []
    if ref_sd is not None:
        note.append(f"separability bar between two 3-seed configs = 3*sqrt(2*{ref_sd:.4f}^2/3) "
                    f"= {3*np.sqrt(2*ref_sd**2/3):.4f}")
    for base, seed, spread, n in dupes:
        note.append(f"{base} s{seed}: {n} benches of ONE checkpoint (same sha256) merged to one "
                    f"seed point; their spread {spread:.6f} is PROBE noise, not seed noise")
    band = 0.055 + 0.021 * len(note)
    fig.subplots_adjust(left=0.20, right=0.66, top=0.94, bottom=band + 0.04)
    for k, t in enumerate(note):
        fig.text(0.02, band - 0.019 * k - 0.012, t, fontsize=6.4, color="#444444")

    ck = pc.__dict__.get("hashlib") or __import__("hashlib")
    key = ck.sha256((repr(sorted((b, tuple(sorted(s for s, _, _ in stats[b][0]))) for b in order))
                     + open(SCRIPT, "rb").read().hex()[:64]).encode()).hexdigest()[:12]
    pc.save(fig, "seeds", SCRIPT, "dev bench only", content_key=key)
    open(os.path.join(pc.PLOTS_DIR, "seeds_key.txt"), "w").write(key)
    print("configs:", {b: sorted(s for s, _, _ in stats[b][0]) for b in order})


if __name__ == "__main__":
    main()
