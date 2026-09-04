#!/usr/bin/env python
"""F3: the story of the night in one panel -- bench mean_area against the wall-clock time of the run.

  python ~/hackathon-shared/bench/plot_progress.py

x is the timestamp in the JSON filename (when the bench finished), y is mean_area with the probe
error bar where the run carries repeats. The anchor is a horizontal line; points are coloured by
architecture, which is the one place the per-tag colour map of F1/F2 does not apply.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_common as pc

SCRIPT = os.path.abspath(__file__)

# Measured by E (21:35, stock transformer, 20k, 10 refits on frozen embeddings): sigma 0.0008.
PROBE_3SIGMA = 0.0025

ARCH_COLOR = {"transformer": "#0072B2", "set encoder": "#E69F00"}


def architecture(run):
    cls = str(run.get("encoder_class") or "")
    if cls == "TransformerEncoder":
        return "transformer"
    if cls:
        return "set encoder"          # DeepSetsEncoder, PMAEncoder, ...
    return "unknown"


def draw(rows, anchor):
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6.5))
    if anchor is not None:
        ax.axhline(anchor, color=pc.ANCHOR_COLOR, linestyle="--", linewidth=1.6, zorder=1,
                   label=f"anchor {pc.ANCHOR_TAG}  {anchor:.4f}")
        ax.axhspan(anchor - PROBE_3SIGMA, anchor + PROBE_3SIGMA, color=pc.ANCHOR_COLOR,
                   alpha=0.10, linewidth=0, zorder=0,
                   label=f"±3σ probe floor, transformer ({PROBE_3SIGMA:.4f}); "
                         f"the set encoders' is wider")

    seen_arch = set()
    anns = []
    for i, r in enumerate(rows):
        arch = architecture(r)
        color = ARCH_COLOR.get(arch, "#666666")
        sd = pc.repeats_std(r, "mean_area_std")
        marker = "s" if r.get("full") else "o"
        ax.errorbar([r["_wallclock"]], [r["mean_area"]], yerr=([sd] if sd else None),
                    fmt=marker, color=color, markersize=9, capsize=3, elinewidth=1.1,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=4,
                    label=arch if arch not in seen_arch else None)
        seen_arch.add(arch)
        note = pc.display_key(r)
        if sd is None:
            note += " (n=1)"
        # Alternate side and direction so labels never land on a marker or on each other.
        # Four offsets rather than two: with twelve rows the two-way alternation put adjacent
        # labels on top of each other, which check_layout reported as a collision.
        dx, ha = ((10, "left") if i % 2 == 0 else (-10, "right"))
        dy = (11, -15, 20, -24)[i % 4]
        # A leader line, so a displaced label still says which marker it belongs to. Without one,
        # the two rep20 labels sat far from their points and which belonged to which was ambiguous
        # -- not a collision, so check_layout could not see it; found by looking. WP-N.
        anns.append(ax.annotate(
            note, (r["_wallclock"], r["mean_area"]),
            textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=6.8, color="#333333",
            arrowprops=dict(arrowstyle="-", lw=0.5, color="#999999",
                            shrinkA=0, shrinkB=4, relpos=(1.0 if ha == "right" else 0.0, 0.5))))

    # Measured repulsion. The fixed four-way offset alternation is blind: it spreads labels by
    # INDEX, not by where they actually landed, so a cluster of rows at similar times and scores
    # still overlapped. Push overlapping labels apart on the rendered boxes and re-measure, which
    # is the same principle as check_visibility -- decide on the raster, not on the intent.
    MAX_SHIFT = 26.0                     # points; a label must stay near its own marker
    base_dy = [a.get_position()[1] for a in anns]
    try:
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        for _ in range(120):
            boxes = [a.get_window_extent(renderer=rend) for a in anns]
            moved = False
            for i in range(len(anns)):
                for j in range(i + 1, len(anns)):
                    a, b = boxes[i], boxes[j]
                    ox = min(a.x1, b.x1) - max(a.x0, b.x0)
                    oy = min(a.y1, b.y1) - max(a.y0, b.y0)
                    if ox <= 2 or oy <= 2:
                        continue
                    # Bounded, damped, and clamped. The first version pushed each label by half
                    # the overlap every sweep with no limit; in a cluster of mutually overlapping
                    # labels that compounds, and it flung them to y = -36579 on a 650 px canvas.
                    # The publication gate caught it, which is the gate earning its keep on its
                    # author. Small steps, and never further than MAX_SHIFT from where the
                    # deterministic offset put it, so a label stays near its own marker.
                    up, dn = (i, j) if a.y0 >= b.y0 else (j, i)
                    step = min(oy / 2 + 0.5, 3.0)
                    for who, sign in ((up, +1), (dn, -1)):
                        x, y = anns[who].get_position()
                        newy = y + sign * step
                        if abs(newy - base_dy[who]) <= MAX_SHIFT:
                            anns[who].set_position((x, newy))
                            moved = True
            if not moved:
                break
            fig.canvas.draw()
    except Exception as exc:
        pc.log(f"progress: label repulsion could not run ({exc!r}); labels left at fixed offsets")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("wall-clock time of the bench run (JSON timestamp)")
    ax.set_ylabel("mean_area over the five bench families")
    ax.set_title("Progress through the night: bench mean_area vs wall clock")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", frameon=True)
    ax.margins(x=0.16, y=0.16)
    fig.autofmt_xdate()
    fig.tight_layout()
    extra = ("squares = --full eval set, circles = 20k · error bar = std over probe refits, "
             "(n=1) rows have none · colour = architecture here, not the per-tag map")
    return pc.save(fig, "progress", SCRIPT, extra)


def generate():
    pc.use_agg()
    import matplotlib.pyplot as plt
    bench, _ = pc.load_runs()
    rows = [r for r in pc.latest_per_key(bench) if pc.is_scoring(r) and r.get("_wallclock")]
    dropped = len([r for r in pc.latest_per_key(bench) if not r.get("_wallclock")])
    if dropped:
        pc.log(f"{dropped} run(s) have no parsable timestamp in the filename and are not on the time axis")
    if not rows:
        pc.log("no timestamped bench runs, nothing to draw")
        return []
    rows.sort(key=lambda r: r["_wallclock"])
    anchor = next((r["mean_area"] for r in rows if r["tag"] == pc.ANCHOR_TAG), None)
    written = draw(rows, anchor)
    plt.close("all")
    pc.log("wrote " + ", ".join(os.path.basename(w) for w in written))
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=int, default=300)
    args = ap.parse_args()
    if args.watch:
        generate()
        pc.watch(generate, interval=args.interval)
    else:
        generate()


if __name__ == "__main__":
    main()
