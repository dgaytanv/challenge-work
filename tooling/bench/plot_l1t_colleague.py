"""WP-N: colleague suite on the L1T eval file, with Group 3's own checkpoints beside ours.

A separate script rather than a switch on plot_curves, for WP-J's reason: plot_curves is in active
use for the ruling figures tonight and this figure is needed once.

The point of the figure is a comparison that is NOT like-for-like, so the caveats are drawn on it
rather than left to a caption: Group 3's models are trained on the L1T file and ours are not, and
their published numbers use a severity grid that stops at 0.8 while ours goes to 1.0.
"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_common as pc

SCRIPT = os.path.abspath(__file__)
WANT = [("d-pma0-aug-meanpt-l1t-colleague", "our recipe, TRAINED ON L1T (like-for-like)", "ours_l1t"),
        ("c3-best-colleague-l1t",      "Group 3 best (paired consistency)", True),
        ("c3-control-colleague-l1t",   "Group 3 control",                   True),
        ("c-winner-l1t-eta3",          "our campaign-1 winner (PF-trained)", False),
        ("h-ref-small-s22-l1t-colleague", "our reference s22 (PF-trained)",  False),
        ("h-ref-small-s33-l1t-colleague", "our reference s33 (PF-trained)",  False)]
G3 = {"#7B3294", "#C2A5CF"}


def newest(tag):
    fs = sorted(glob.glob(os.path.join(pc.RUNS_DIR, f"{tag}_*.json")))
    if not fs:
        return None, None
    # More than one bench of one model is duplicate MEASUREMENT, not a second row: take the newest
    # and say so, rather than plotting both (register class 7).
    d = json.load(open(fs[-1]))
    return d, (os.path.basename(fs[-1]), len(fs))


def main():
    rows = []
    dupes = []
    for tag, label, is_g3 in WANT:
        d, meta = newest(tag)
        if d is None:
            pc.log(f"l1t_colleague: {tag} absent, skipped")
            continue
        if meta[1] > 1:
            dupes.append(f"{tag}: {meta[1]} benches of one model, newest used ({meta[0]})")
        rows.append((tag, label, is_g3, d, meta[0]))
    if not rows:
        pc.log("l1t_colleague: no rows"); return

    fams = ["c_ellipse", "c_cell_dropout", "c_edge_truncation"]
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 6.2), squeeze=False)
    axes = axes.ravel()
    g3_cols = ["#7B3294", "#C2A5CF"]
    ours = ["#0072B2", "#009E73", "#E69F00"]
    gi = oi = 0
    handles, labels = [], []
    for tag, label, is_g3, d, fn in rows:
        if is_g3 == "ours_l1t":
            # The like-for-like row: our recipe on THEIR training file. Triangle because train_data
            # is a non-standard file (the marker convention on every other figure), drawn on top.
            col, ls, lw, mk, z = "#0072B2", "-", 2.8, "^", 6
        elif is_g3:
            col, ls, lw, mk, z = g3_cols[gi % 2], "-", 2.4, "o", 4; gi += 1
        else:
            col, ls, lw, mk, z = ours[oi % 3], "--", 1.6, "o", 3; oi += 1
        for i, fam in enumerate(fams):
            e = (d.get("families") or {}).get(fam)
            if not e:
                continue
            std = e.get("aucs_std")
            ax = axes[i]
            if std and len(std) == len(e["aucs"]):
                ax.errorbar(e["severities"], e["aucs"], yerr=std, color=col, ls=ls, lw=lw,
                            marker=mk, ms=4.5 if mk == "^" else 3.5, capsize=2, elinewidth=0.8,
                            zorder=z)
            else:
                ax.plot(e["severities"], e["aucs"], color=col, ls=ls, lw=lw, marker=mk,
                        ms=4.5 if mk == "^" else 3.5, zorder=z)
        h, = axes[0].plot([], [], color=col, ls=ls, lw=lw, marker=mk, ms=4.5 if mk == "^" else 3.5)
        handles.append(h)
        lab = f"{label}   {d['mean_area']:.4f} ± {d.get('mean_area_std', 0):.4f}  (sev 0-1.0)"
        if is_g3:
            # Their published numbers stop at 0.8. Print OUR value on THEIR grid beside ours, so the
            # two can be compared without anyone recomputing it or comparing across grids by mistake.
            a = []
            for fam in fams:
                e = (d.get("families") or {}).get(fam)
                if not e:
                    continue
                sv = np.array(e["severities"]); au = np.array(e["aucs"]); m = sv <= 0.8
                a.append(np.trapezoid(au[m], sv[m]) / (sv[m][-1] - sv[m][0]))
            if a:
                lab += f"   |   {np.mean(a):.4f} on their 0-0.8 grid"
        labels.append(lab)
    for i, fam in enumerate(fams):
        axes[i].set_title(fam, fontsize=10)
        axes[i].axhline(0.5, color="#999999", lw=0.8, ls=":", zorder=0)
        axes[i].set_xlabel("severity")
        axes[i].grid(True, alpha=0.3)
    axes[0].set_ylabel("AUC (MLP probe on latents)")
    fig.suptitle("Colleague-group-3 suite on the L1T eval file — their checkpoints beside ours", y=0.985)

    note = ("Triangle = trained on the L1T file, the same file Group 3 trained on: that row and the "
            "Group 3 rows ARE like-for-like.\n"
            "The dashed rows are PF-trained and are shown for continuity only — for THOSE, this is not "
            "a like-for-like comparison of methods.\n"
            "All rows here use ONE probe implementation (ours), so the rows are comparable to each "
            "other even though they are not comparable to Group 3's published numbers:\n"
            "their summary sweeps severity to 0.8, ours to 1.0. Our c3-best restricted to their grid "
            "is 0.8107 against their published 0.822427.")
    for k, t in enumerate(dupes):
        note += f"\n{t}"
    fig.subplots_adjust(left=0.06, right=0.985, top=0.88, bottom=0.52)
    fig.legend(handles, labels, loc="lower center", ncol=1, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.28))
    fig.text(0.045, 0.015, note, fontsize=7.0, color="#333333", va="bottom")

    import hashlib
    key = hashlib.sha256((repr(sorted((t, round(d["mean_area"], 6)) for t, _, _, d, _ in rows))
                          + open(SCRIPT, "rb").read().hex()[:64]).encode()).hexdigest()[:12]
    pc.save(fig, "l1t_colleague", SCRIPT, "colleague suite, L1T eval file", content_key=key)
    open(os.path.join(pc.PLOTS_DIR, "l1t_colleague_key.txt"), "w").write(key)
    for t, _, _, d, fn in rows:
        print("%-32s %.4f  %s" % (t, d["mean_area"], fn))


if __name__ == "__main__":
    main()
