#!/usr/bin/env python
"""F2: the organisers' own sweep (eval.py + their degradation_eval.py) for every tag that has an
official JSON, against their reference "(No degradation)" curve, in their styling.

  python ~/hackathon-shared/bench/plot_official.py

Their Bernoulli drop is unseeded, so repeated runs of one checkpoint differ. Where a tag has more
than one official JSON it is drawn as a mean line with a min-max band over the runs.

This is a DIFFERENT corruption from bench_eval.py's five families. Nothing on these axes is
comparable to mean_area, and nothing here is ever tuned to -- the organisers marked the grid
TEMPORARY and stated it will change before judging.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_common as pc

SCRIPT = os.path.abspath(__file__)


def group(official):
    """{tag: [records...]} in wall-clock order, keeping only records that carry a curve."""
    by_tag = {}
    for r in sorted(official, key=lambda r: (r.get("_wallclock") or 0, r["_path"])):
        by_tag.setdefault(r["tag"], []).append(r)
    return by_tag


def colour_aliases(by_tag):
    """{tag: tag-whose-colour-to-use} for tags that are the same model under another name.

    One model reaches runs/ under more than one tag: the final acceptance record is written as
    `submission-final`, and a-pma's two official runs were filed as `a-pma-probe` and `a-pma`.
    Those are the same checkpoint and should not claim two colours.

    The key is (checkpoint basename, branch), matching what E pools the final table's official
    column on, so the figure and the table group identically. Commit is deliberately NOT in the
    key: the same checkpoint on the same branch has records at commits that differ only by
    README edits (3ac262f/41d45a9, b50612d/99b7d2e, baa5d4d/791c450, verified by E), and keying
    on commit splits pairs that are one measurement. The checkpoint basename stays in the key --
    branch alone, or checkpoint alone, is not enough. Checkpoint alone is what merged the anchor
    with the mask-fix rows that share the stock weights under different code.

    Every record of a tag is considered, not just its first: keying off one record makes the
    result depend on which happened to be written first, which is how this passed by luck before.
    """
    first, alias = {}, {}
    for tag in sorted(by_tag, key=lambda t: by_tag[t][0].get("_path", "")):
        keys = {(os.path.basename(r["ckpt"]), r.get("branch"))
                for r in by_tag[tag] if r.get("ckpt") and r.get("branch")}
        if len(keys) != 1:
            continue                     # a tag spanning checkpoints or branches keeps its own colour
        key = keys.pop()
        if key in first and first[key] != tag:
            alias[tag] = first[key]
        else:
            first.setdefault(key, tag)
    return alias


def draw(by_tag, baseline, leader, donors=None):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8.5, 6))
    if baseline:
        ax.plot(baseline["severities"], baseline["aucs"], marker="o", markersize=4,
                linestyle="--", color="red", linewidth=1.8,
                label=baseline.get("label", "(No degradation)"), zorder=3)

    area_only = []
    alias = colour_aliases(by_tag)
    donors = donors or {}
    for tag, recs in sorted(by_tag.items(), key=lambda kv: -np.mean([r["official_area"] for r in kv[1]])):
        # Same ownership rule as the bench figures: a tag takes the colour of the scoring row
        # that owns its checkpoint, so one model is one colour everywhere.
        ck = os.path.basename(recs[0]["ckpt"]) if recs[0].get("ckpt") else None
        color = pc.color_for(donors.get(ck) or alias.get(tag, tag), leader=leader)
        curves = [(r["severities"], r["aucs"]) for r in recs
                  if r.get("severities") and r.get("aucs")
                  and len(r["severities"]) == len(r["aucs"])]
        areas = [r["official_area"] for r in recs]
        n = len(recs)
        lbl = f"{tag}   area {np.mean(areas):.4f}"
        if n > 1:
            lbl += f"  (n={n}, spread {max(areas) - min(areas):.4f})"
        else:
            lbl += "  (n=1)"
        if tag in alias:
            lbl += f"  [same ckpt+branch as {alias[tag]}]"
        if not curves:
            area_only.append(f"{tag}: area {np.mean(areas):.4f} (n={n}, no per-severity curve on disk)")
            continue
        # Runs of one tag can differ in sweep length; align on the severities they all share.
        common = sorted(set.intersection(*[set(s) for s, _ in curves]))
        if not common:
            area_only.append(f"{tag}: area {np.mean(areas):.4f} (n={n}, no common severity grid)")
            continue
        stacked = np.array([[a[s.index(x)] for x in common] for s, a in curves])
        mean = stacked.mean(axis=0)
        ax.plot(common, mean, marker="o", markersize=4, color=color, linewidth=1.8, label=lbl, zorder=4)
        if len(curves) > 1:
            ax.fill_between(common, stacked.min(axis=0), stacked.max(axis=0),
                            color=color, alpha=0.18, linewidth=0, zorder=2)

    # eval.py's x axis verbatim. The y axis is the one deliberate departure from their figure:
    # eval.py labels it "linear probe", but its train_linear_probe builds an EvalMLP
    # (128-64-32 ReLU, models.py:313). Planner's ruling -- say what the probe actually is.
    ax.set_xlabel(pc.X_LABEL)
    ax.set_ylabel(pc.Y_LABEL)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_title("Organisers' sweep (eval.py + degradation_eval.py, TEMPORARY grid)")
    if not by_tag:
        ax.text(0.5, 0.5, "no official runs recorded yet\n(accept.sh writes runs/official_<tag>_<ts>.json)",
                transform=ax.transAxes, ha="center", va="center", fontsize=11, color="#777777")
    if area_only:
        ax.text(0.01, 0.02, "\n".join(["area recorded without a curve:"] + area_only),
                transform=ax.transAxes, fontsize=7, color="#555555", va="bottom")
    ax.legend(fontsize=8, loc="lower left", frameon=True)
    fig.tight_layout()
    extra = "organisers' corruption; NOT comparable to bench mean_area · band = min-max over unseeded repeats"
    return pc.save(fig, "official", SCRIPT, extra)


def generate():
    pc.use_agg()
    import matplotlib.pyplot as plt
    bench, official = pc.load_runs()
    leader = (pc.leader_tag(bench) or "").replace(" (full)", "").replace(" (heldout)", "") or None
    written = draw(group(official), pc.load_baseline(), leader, pc.ckpt_colour_donors(bench))
    plt.close("all")
    pc.log(f"wrote {', '.join(os.path.basename(w) for w in written)} "
           f"({len(official)} official JSON(s), {len(group(official))} tag(s))")
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
