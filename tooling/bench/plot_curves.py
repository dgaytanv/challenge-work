#!/usr/bin/env python
"""F1: AUC vs severity per bench family, our attempts against the anchor.

  python ~/hackathon-shared/bench/plot_curves.py                 # default tag set, one figure
  python ~/hackathon-shared/bench/plot_curves.py --tags a,b,c    # explicit tags
  python ~/hackathon-shared/bench/plot_curves.py --watch         # regenerate every figure on new JSONs

Default tag set: the anchor, planner-maskfix-stockckpt, and every other bench tag in runs/ ranked
by mean_area, capped at 8 curves per panel. Held-out-only runs are excluded by default (their
families are a different, non-scoring set); --include-heldout puts them back.

This is our five-family development bench. Never put an official area on these axes.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_common as pc

SCRIPT = os.path.abspath(__file__)
# The final figure's rows, in the planner's order (22:28). Anything else is added only if it
# benches above LATE_ARM_MIN, so a late arm that beats the probable winner still appears.
FIXED = [
    "d-pma0-aug-meanpt",            # the planner's ruled winner; holds the accent (see RULED_WINNER)
    "a-pma",                        # benched 0.8275 at 22:47, above the winner but not separably
    "d-deepsets-aug-meanpt",
    "d-pma0-aug",
    "d-deepsets-aug-rep10",
    "e-stock-maskfix",
    pc.MASKFIX_TAG,                 # planner-maskfix-stockckpt
    pc.ANCHOR_TAG,
    "c-twoview-cos1",
    "d-deepsets-clean",             # last, so this is the row the cap drops if it binds
]
LATE_ARM_MIN = 0.8186


def select(bench, tags=None, include_heldout=False, cap=pc.MAX_CURVES, heldout_only=False,
           collapse=False, suite="scoring"):
    # NOTE: every tag is its own row. Folding re-benches of one checkpoint together was tried and
    # reverted: the grouping key can only be the checkpoint path, and anchor-stock-baseline,
    # a-maskfix-stockckpt and planner-maskfix-stockckpt all share the stock checkpoint while being
    # three different measurements, so it silently dropped the anchor. Commit does not rescue it
    # either -- every run in runs/ has a distinct commit, including the leader's own re-bench.
    for r in bench:
        if pc.is_retracted(r):
            pc.log(f"excluding {r['tag']}: {pc.RETRACTED[r['tag']]}")
    bench = [r for r in bench if not pc.is_retracted(r)]
    if suite == "altdata":
        moved = [r["tag"] for r in bench if pc.is_g_sweep_row(r)]
        if moved:
            pc.log(f"altdata: {', '.join(moved)} are WP-G sweep rows, not the operating point; "
                   f"they belong on WP-G's accuracy-vs-bit-cap figure")
        bench = [r for r in bench if not pc.is_g_sweep_row(r)]
    if suite == "colleague":
        # User order, 05:5x: colleague_latest shows OUR models only. Group 3's own checkpoints are
        # preserved on colleague_group3_latest, not deleted -- the comparison is a result, it is
        # just not what this figure is for any more. Match on the c3- tag prefix, which is the
        # convention agreed with WP-M for their rows.
        theirs = [r["tag"] for r in bench if r["tag"].startswith("c3-")]
        if theirs:
            pc.log(f"colleague: excluding Group 3 rows {', '.join(sorted(set(theirs)))} by user "
                   f"order; they are preserved on colleague_group3_latest")
        bench = [r for r in bench if not r["tag"].startswith("c3-")]
    rows = pc.rank(bench)
    if heldout_only:
        suite = "heldout"
    if suite != "scoring":
        # A non-scoring suite figure holds runs from THAT suite and nothing else, so the mean
        # panel is structurally incapable of averaging across suites. colleague-group3's
        # "ellipse" is a different implementation from our held-out "ellipse" despite the shared
        # name, so this separation is about more than tidiness.
        return [r for r in rows if pc.suite(r) == suite][:cap]
    rows = [r for r in rows if pc.is_scoring(r)]
    by_tag = {}
    for r in rows:
        by_tag.setdefault(r["tag"], r)          # rank() already put the newest/best first
    if tags:
        want = [t.strip() for t in tags.split(",") if t.strip()]
        keyed = {pc.display_key(r): r for r in rows}
        chosen = []
        for t in want:                          # the caller's order is the legend order
            r = keyed.get(t) or by_tag.get(t)
            if r is None:
                pc.log(f"no bench JSON for tag {t!r}, skipping it")
            elif r not in chosen:
                chosen.append(r)
        return chosen[:cap]
    # The pinned rows in the planner's order, then any late arm that beat the probable winner.
    pinned = [by_tag[t] for t in FIXED if t in by_tag]
    for t in FIXED:
        if t not in by_tag:
            pc.log(f"pinned tag {t!r} has no bench JSON yet")
    # A late arm that beats LATE_ARM_MIN is, by definition, competitive with the pinned set, and
    # is the row a reader most needs to see. The pin list has grown to exactly the cap, so
    # appending late arms after the pins meant they could never fit and the TOP-SCORING row was
    # silently absent. Reserve their slots first and trim the pins from the END, which is the
    # order the planner set them in for exactly this purpose.
    late = [r for r in rows if r not in pinned and r.get("mean_area", 0.0) > LATE_ARM_MIN]
    for r in late:
        pc.log(f"{pc.display_key(r)} benched {r['mean_area']:.4f} > {LATE_ARM_MIN}, adding it")
    late = late[:cap]
    room = max(0, cap - len(late))
    if len(pinned) > room:
        pc.log(f"cap {cap} binds: dropping pinned "
               f"{', '.join(pc.display_key(r) for r in pinned[room:])} to make room for "
               f"{len(late)} late arm(s)")
    keep = pinned[:room] + late
    keep.sort(key=lambda r: r.get("mean_area", 0.0), reverse=True)
    return keep


def ckpt_alias(rows):
    """{tag: tag-whose-colour-to-use} keyed on the checkpoint, for a SINGLE-suite figure.

    E's colleague tags (c-winner, c-deepsets-meanpt) do not carry a "-colleague" suffix, so
    base_tag cannot map them onto their scoring row, and they would each claim a fresh colour.
    Safe here and only here: this figure holds one suite, so the checkpoint identifies the model
    unambiguously. It is NOT safe on the scoring figure, where the anchor and both mask-fix rows
    share the stock checkpoint under different code -- that grouping dropped the anchor earlier.
    """
    first, alias = {}, {}
    for r in rows:
        ck = r.get("ckpt")
        if not ck:
            continue
        key = os.path.basename(ck)
        tag = r["tag"]
        if key in first and first[key] != tag:
            alias[tag] = first[key]
        else:
            first.setdefault(key, tag)
    return alias


def draw(rows, leader, outstem="curves", title=None, mean_panel=True, per_family_legend=False,
         extra_note=None, colour_by_ckpt=False, donors=None, mean_by_family_suite=False,
         footer_note=None, all_runs=None):
    import matplotlib.pyplot as plt
    import numpy as np

    fams = pc.families_of(rows)
    donors = donors or {}
    # Only qualify a label with eta_max when the figure actually mixes more than one -- a constant
    # suffix on every row is noise. E adds the field with the L1T re-runs; rows without it are
    # left unqualified rather than assumed.
    # None counts as its own category: mixing "eta_max 3" rows with rows that never recorded it
    # is exactly when the qualifier is needed, and requiring two RECORDED values hid it.
    etas = {(pc.eta_max(r) if pc.eta_max_applies(r) else None) for r in rows}
    show_eta = len(etas) > 1 and any(e is not None for e in etas)
    # Three training states, not two: a row that RECORDS the standard file has said something,
    # a row that records nothing has not, and the legend must not turn the second into the first.
    train_states = {pc.train_state(r) for r in rows}
    show_train = len(train_states) > 1
    # A mean panel may only average within one family set. The altdata figure holds the scoring
    # five and colleague-group3's three on the same eval file, so it gets one mean panel per
    # family suite rather than one that silently averages a five-family row with a three-family
    # one -- the same reason the colleague figure has no mean panel at all.
    suites = []
    if mean_panel:
        if mean_by_family_suite:
            seen = []
            for r in rows:
                fs = pc.family_suite(r)
                if fs not in seen:
                    seen.append(fs)
            suites = seen
        else:
            suites = [None]
    panels = fams + [f"mean over {s} families" if s else "mean over families" for s in suites]
    alias = ckpt_alias(rows) if colour_by_ckpt else {}
    ncol = 3
    nrow = (len(panels) + ncol - 1) // ncol
    # The legend, the optional note and the footer live in a band BELOW the panels. Size that band
    # in inches and convert to a figure fraction, rather than guessing a fraction: with the mean
    # panel dropped the colleague figure is a single row, and a fraction tuned for a two-row
    # figure put the legend on top of the axes.
    # Columns must follow the longest label, not be fixed at 3. Qualifiers like
    # "[trained on the standard file, eta_max 5]" only appear once a figure MIXES train states or
    # eta conventions, so the legend can grow wider between two renders with no code change and no
    # new rows -- which is how a clipped legend shipped twice tonight. Estimate the widest label
    # and drop to 2 or 1 columns before it overflows; the layout gate then has nothing to catch.
    # ncol must be decided BEFORE the band is sized, or the band is sized for three columns and the
    # legend is later drawn in one -- which is how a 10-row legend ended up on top of the bottom
    # panels, with 14 collisions logged and nothing blocking. Predict the label width here from the
    # same pieces the draw loop uses (display_key + the qualifiers, whose visibility is already
    # decided above) rather than from display_key alone, which was the estimate that was too short.
    legend_ncol = 1 if per_family_legend else 3
    if not per_family_legend:
        qual_pad = (len("  [trained on the standard file]") if show_train else 0) + \
                   (len(", eta_max not recorded") if show_eta else 0)
        widest = max((len(pc.display_key(r)) + qual_pad + 26 for r in rows), default=40)
        while legend_ncol > 1 and widest * 7.1 * legend_ncol > (panel_w_probe := 4.4) * ncol * 100 * 0.96:
            legend_ncol -= 1
    legend_rows = max(1, -(-len(rows) // legend_ncol))
    band_in = 0.78 + 0.20 * legend_rows + (0.52 if extra_note else 0.0)
    panel_h, panel_w = 3.5, 4.4
    fig_h = panel_h * nrow + band_in
    fig, axes = plt.subplots(nrow, ncol, figsize=(panel_w * ncol, fig_h), squeeze=False)
    axes = axes.ravel()

    # Planner ruling: one model is one colour. Identity is the recorded checkpoint (with register
    # class 1's second discriminator, see pc.model_identity), falling back to the tag with -repNN
    # stripped. Within an identity the highest-R measurement is drawn solid and any lower-R twin as
    # a thin dashed line in the same colour, with ONE legend entry naming both. Before this, a
    # model's two measurements took two colours and the lower-R one was drawn underneath its own
    # twin and was invisible -- 294 px against a 9902 px median.
    ident = pc.model_identity(rows)
    primary = {}
    for r in rows:
        k = ident.get(r["tag"], r["tag"])
        if k not in primary or pc.probe_repeats(r) > pc.probe_repeats(primary[k]):
            primary[k] = r
    reps_by_ident = {}
    for r in rows:
        reps_by_ident.setdefault(ident.get(r["tag"], r["tag"]), set()).add(pc.probe_repeats(r))

    # Resolve every row's colour up front, guaranteeing uniqueness within this figure.
    colour_keys = [(donors.get(os.path.basename(r["ckpt"])) if r.get("ckpt") else None)
                   or alias.get(r["tag"]) or ident.get(r["tag"], pc.base_tag(r)) for r in rows]
    seen_key, ordered = set(), []
    for k in colour_keys:
        if k not in seen_key:
            seen_key.add(k)
            ordered.append(k)
    colours = pc.distinct_colours(ordered, leader=leader)

    handles, labels = [], []
    data_min = 1.0
    # WP-N, register class 13. Rows arrive in rank order (best first) and matplotlib draws later
    # calls ON TOP, so without zorder the BEST rows are buried by every row below them. Measured on
    # the 01:28 render: the certified submission showed 287 px of its own colour against 21164 for
    # the worst-scoring row -- the least visible line on a figure whose footer tells the reader to
    # find it by colour. Iterating in reverse would fix the occlusion and silently reverse the
    # legend, which is built from these handles, so the order stays and zorder carries the fix:
    # rank 0 gets the highest zorder. The accent takes a further boost so the ruled row is never
    # occluded by a row it ties with.
    n_rows = len(rows)
    for rank_i, r in enumerate(rows):
        key = pc.display_key(r)
        quals = []
        sus = pc.suspicious_collapse(r)
        if sus:
            pc.log(f"SUSPICIOUS {r['tag']}: {sus}")
            quals.append("SUSPICIOUS - see log")
        for note in (pc.tag_note(r), pc.encoder_note(r)):
            if note:
                quals.append(note)
        n_runs = pc.run_count(all_runs or rows, key)
        if n_runs > 1:
            quals.append(f"{n_runs} runs pooled in table")
        if show_train:
            st = pc.train_state(r)
            quals.append({"nonstandard": f"trained on {pc.train_data(r)}",
                          "standard": "trained on the standard file",
                          "unrecorded": "train file not recorded"}[st])
        if show_eta:
            if pc.eta_max_applies(r):
                quals.append(f"eta_max {pc.eta_max(r):g}")
            else:
                quals.append("eta_max n/a" if r.get("eta_max") is not None
                             else "eta_max not recorded")
        if quals:
            key += "  [" + ", ".join(quals) + "]"
        # Colour keys on the bare TAG, so a tag's 20k, --full and held-out rows share one colour
        # across every figure; the variant is carried by the marker and the legend text instead.
        # Colour follows the MODEL: a row measured on another eval file or another suite takes
        # the colour of the scoring row that owns its checkpoint, when exactly one does.
        donor = donors.get(os.path.basename(r["ckpt"])) if r.get("ckpt") else None
        color = colours[donor or alias.get(r["tag"]) or ident.get(r["tag"], pc.base_tag(r))]
        is_anchor = r["tag"] == pc.ANCHOR_TAG
        # Two rows of one model can share a colour on a mixed figure; the dash separates them.
        dashed = is_anchor or (mean_by_family_suite and pc.family_suite(r) != "scoring")
        # Marker carries where the WEIGHTS came from: triangle = trained on a non-standard file
        # (the L1T line), square = benched on the --full eval set, circle otherwise.
        marker = "s" if r.get("full") else ("^" if pc.alt_train(r) else "o")
        z = 3.0 + (n_rows - rank_i)                      # best rank -> highest zorder
        if leader is not None and pc.base_tag(r) == leader:
            z += n_rows                                  # the accent is never occluded by a tie
        ik = ident.get(r["tag"], r["tag"])
        is_twin = primary.get(ik) is not r
        style = dict(color=color, marker=marker,
                     markersize=(2.0 if is_twin else (4.5 if marker == "^" else 3.5)),
                     linewidth=(0.9 if is_twin else 1.8),
                     linestyle=":" if is_twin else ("--" if dashed else "-"),
                     zorder=(z - 0.5) if is_twin else z)
        # ---- per-family panels
        per_sev = {}
        for i, fam in enumerate(fams):
            ent = (r.get("families") or {}).get(fam)
            if not ent or "severities" not in ent or "aucs" not in ent:
                continue
            sev, auc = ent["severities"], ent["aucs"]
            if len(sev) != len(auc) or not sev:
                pc.log(f"{pc.display_key(r)}/{fam}: severities and aucs disagree in length, skipping the panel")
                continue
            std = ent.get("aucs_std") if pc.repeats_std(r, "mean_area_std") is not None else None
            ax = axes[i]
            if std and len(std) == len(auc):
                ax.errorbar(sev, auc, yerr=std, capsize=2, elinewidth=0.8, **style)
            else:
                ax.plot(sev, auc, **style)
            data_min = min(data_min, min(auc))
            for s, a in zip(sev, auc):
                per_sev.setdefault(s, []).append(a)
        # ---- mean-over-families panel: only severities every family in this run reached
        nfam = sum(1 for f in fams if f in (r.get("families") or {}))
        xs = sorted(s for s, v in per_sev.items() if len(v) == nfam and nfam)
        if xs and mean_panel:
            mkey = pc.family_suite(r) if mean_by_family_suite else None
            if mkey in suites:
                ys = [float(np.mean(per_sev[s])) for s in xs]
                axes[len(fams) + suites.index(mkey)].plot(xs, ys, **style)
        # ONE legend entry per model. The twin is drawn but not listed again; its existence is
        # carried in the primary's label as the set of repeat counts, so no measurement is hidden
        # and no reader hunts the panel for a colour that is under another line.
        if not is_twin:
            h, = axes[0].plot([], [], **style)
            handles.append(h)
            want_per_family = (per_family_legend is True or
                               (per_family_legend == "auto" and pc.family_suite(r) != "scoring"))
            if want_per_family:
                per = "   ".join(f"{f} {r['families'][f]['area']:.4f}"
                                 for f in fams if f in (r.get("families") or {}))
                lab = f"{key}   {per}"
            else:
                lab = f"{key}   {pc.area_label(r)}"
            reps = sorted(reps_by_ident.get(ik, set()))
            if len(reps) > 1:
                lab += "  (R " + "/".join(str(n) for n in reps) + ")"
            labels.append(lab)

    ylo = min(0.45, data_min - 0.02)
    for i, name in enumerate(panels):
        ax = axes[i]
        ax.axhline(0.5, color="#999999", linewidth=0.8, linestyle=":", zorder=0)
        ax.text(0.02, 0.505, "chance", fontsize=6, color="#777777", transform=ax.get_yaxis_transform())
        ax.set_title(name, fontsize=10)
        # 0.45-1.0 as briefed. A below-chance point is a real result (the frozen probe reading
        # the signal backwards, not merely losing it), so the floor drops to fit one rather than
        # clipping it off the bottom of the panel -- and every panel keeps the SAME floor, so the
        # figure stays comparable across families and across the night.
        ax.set_ylim(ylo, 1.0)
        ax.set_xlim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        # eval.py's x axis verbatim; the y axis corrects eval.py's "linear probe", since its
        # train_linear_probe actually builds an EvalMLP (128-64-32 ReLU). Planner's ruling.
        ax.set_xlabel(pc.X_LABEL, fontsize=8)
        ax.set_ylabel(pc.Y_LABEL, fontsize=8)
        ax.tick_params(labelsize=8)
    for ax in axes[len(panels):]:
        ax.axis("off")

    suptitle = title or "AUC vs severity, five-family development bench (bench_eval.py)"
    fig.suptitle(suptitle, fontsize=13 if len(suptitle) < 95 else 10.5, wrap=True)
    legend_y = (0.34 + (0.46 if extra_note else 0.0)) / fig_h
    # Decide columns from the ACTUAL label strings, not from an estimate made before they existed.
    # The first attempt guessed the width from display_key plus a constant and still clipped: the
    # rendered label also carries the area, the +- and the (R n/m) suffix. Measuring the strings we
    # are about to draw is both simpler and correct.
    if not per_family_legend and labels:
        widest_chars = max(len(x) for x in labels)
        fig_px = fig.get_size_inches()[0] * fig.dpi
        while legend_ncol > 1 and widest_chars * 7.1 * legend_ncol > fig_px * 0.96:
            legend_ncol -= 1
            pc.log(f"{outstem}: legend narrowed to {legend_ncol} column(s) after measuring the "
                   f"real labels; the band was sized for more, so the figure keeps slack rather "
                   f"than overlapping the panels")
    fig.legend(handles, labels, loc="lower center", ncol=min(legend_ncol, max(1, len(labels))),
               bbox_to_anchor=(0.5, legend_y), fontsize=8, frameon=False,
               title={"heldout": "tag   held-out mean area ± std over probe refits",
                      "colleague": "tag   area per family (no suite mean: see the note)",
                      "altdata": "tag   area over its own family set "
                                 "(scoring rows: mean ± std; colleague rows: per family)",
                      }.get(outstem, "tag   mean_area ± std over probe refits"),
               title_fontsize=8)
    fig.tight_layout(rect=(0, band_in / fig_h, 1, 1 - 0.38 / fig_h))
    # The footer now wraps to two lines on wide figures, so the note must clear more than one
    # line's height -- check_layout caught them overlapping the moment the wrap landed.
    note_y = 0.34 / fig_h
    if extra_note:
        # Centred long text overflows BOTH edges once it exceeds the canvas; check_layout found
        # the altdata caveat running from x=-269 to x=1589 on a 1320px figure, i.e. its first and
        # last words were simply not there. Wrap it to the figure width.
        import textwrap
        per_line = max(70, int(panel_w * ncol * fig.dpi / 5.6))
        fig.text(0.5, note_y, "\n".join(textwrap.wrap(extra_note, per_line)),
                 fontsize=7.5, color="#8B2500", ha="center", va="bottom", linespacing=1.35)
    extra = {"heldout": "held-out families, never averaged with the scoring five",
             "colleague": "colleague-group3's suite; their 'ellipse' is their own implementation, "
                          "not ours; never averaged with the scoring five",
             }.get(outstem, "bench families; NOT comparable to the organisers' official area")
    extra += " · squares = --full eval set"
    if show_train:
        extra += " · triangles = trained on a non-standard train file"
    if footer_note:
        extra += " · " + footer_note
    if ylo < 0.45:
        extra += f" · y floor lowered to {ylo:.2f} to keep a below-chance point on the panel"
    # Only claim the accent when the accented row is actually on this figure.
    if leader and any((donors.get(os.path.basename(r["ckpt"])) if r.get("ckpt") else None)
                      == leader or r["tag"] == leader for r in rows):
        extra += f" · blue = certified submission ({leader}), which is a ruling and not a ranking"
    if any(r["tag"] == pc.ANCHOR_TAG for r in rows):
        extra += " · red dashed = anchor"
    # Fingerprint the DATA, so an inspection is retired by a change of content and not by a redraw.
    import hashlib
    # Include the script's own hash: a change to the drawing code changes what a reader sees even
    # when every number is identical. My identity/legend rewrite altered the colours and the legend
    # of this figure while leaving the rows untouched, and a data-only key carried a 03:42
    # inspection forward onto a figure nobody had looked at.
    with open(SCRIPT, "rb") as _fh:
        _script_sha = hashlib.sha256(_fh.read()).hexdigest()[:8]
    ck = hashlib.sha256((repr(sorted(
        (pc.display_key(r), round(float(r.get("mean_area") or 0.0), 6)) for r in rows
    )) + _script_sha).encode()).hexdigest()[:12]
    out = pc.save(fig, outstem, SCRIPT, extra, content_key=ck)
    # WP-N class 13: verify on the RASTER that every row is actually visible. Occlusion is a
    # property of what was drawn, not of what was requested, so it has to be measured on pixels.
    # Only rows that were actually DRAWN on this figure. Adding the leader and the anchor
    # unconditionally reported "0 px -- effectively invisible" for rows that are simply not on the
    # heldout/colleague/altdata suites, which is a false positive, and a check that cries wolf is
    # register class 4 all over again: the next real hit gets skimmed past with the noise.
    drawn = {k: v for k, v in colours.items()}
    if leader and leader in drawn:
        drawn[leader] = pc.LEADER_COLOR
    if pc.ANCHOR_TAG in drawn:
        drawn[pc.ANCHOR_TAG] = pc.ANCHOR_COLOR
    try:
        pc.check_visibility(os.path.join(pc.PLOTS_DIR, outstem + "_latest.png"), drawn, outstem)
    except Exception as exc:
        pc.log(f"{outstem}: visibility check errored ({exc!r}) -- treat as UNCHECKED")
    return out


SUITE_TITLE = {
    "heldout": "AUC vs severity, HELD-OUT corruption families "
               "(generalisation check, not part of mean_area)",
    "colleague": "AUC vs severity, colleague-group3's corruption suite "
                 "(independent check, not part of mean_area)",
    "altdata": "AUC vs severity on robust_tagging_eval_small_l1t.pt (L1T acceptance, eta in "
               "[-3, 3]) — not comparable to any area measured on the standard eval file",
}
SUITE_STEM = {"scoring": "curves", "heldout": "heldout", "colleague": "colleague",
              "altdata": "altdata"}

# From E, 23:36: their edge_truncation keeps eta <= -3+6s OR eta >= 3-6s, so the two half-planes
# overlap once s >= 0.5 and the family degenerates into "drop nearly everything", chosen by a coin
# flip per event. The s >= 0.6 points of that panel are not edge truncation and must not be read
# as one. This is also why this figure quotes per-family areas and draws no mean-over-families
# panel: a mean across a family that stops being itself halfway through is not a summary of
# anything.
ALTDATA_NOTE = (
    "g-stageA-l1t is NOT a quantization result: it is R1+R2 (exact) plus LayerNorm->BatchNorm "
    "(not exact, forced -- HGQ2 has no quantized LayerNorm) plus 12 epochs of distillation. Its "
    "-0.0039 +- 0.0007 against d-pma0-aug-meanpt-l1t is the measured price of that swap, and it "
    "still sits above winner-l1t-eta3. Quantization on top of it costs +0.000088 (tied)."
)

COLLEAGUE_NOTE = (
    "c_edge_truncation keeps eta <= -3+6s OR eta >= 3-6s (E): the half-planes overlap from "
    "s = 0.5, so its s >= 0.6 points are 'drop nearly everything by a per-event coin flip', "
    "not edge truncation. Read that panel below s = 0.6 only."
)


def generate(tags=None, include_heldout=False, cap=pc.MAX_CURVES, heldout_only=False,
             collapse=True, suite="scoring"):
    pc.use_agg()
    import matplotlib.pyplot as plt
    bench, _ = pc.load_runs()
    if not bench:
        pc.log("no bench JSONs in runs/, nothing to draw")
        return []
    if heldout_only:
        suite = "heldout"
    rows = select(bench, tags, include_heldout, cap, heldout_only, collapse, suite)
    if not rows:
        pc.log(f"no {suite} bench JSONs yet, {SUITE_STEM.get(suite, suite)} figure not drawn"
               if suite != "scoring" else "no rows selected, nothing to draw")
        return []
    written = draw(rows, pc.leader_tag(bench), all_runs=bench,
                   outstem=SUITE_STEM.get(suite, suite),
                   title=SUITE_TITLE.get(suite),
                   mean_panel=(suite != "colleague"),
                   per_family_legend=(True if suite == "colleague"
                                      else ("auto" if suite == "altdata" else False)),
                   extra_note=({"colleague": COLLEAGUE_NOTE, "altdata": ALTDATA_NOTE}.get(suite)),
                   colour_by_ckpt=(suite != "scoring"),
                   donors=(pc.ckpt_colour_donors(bench) if suite != "scoring" else None),
                   mean_by_family_suite=(suite == "altdata"),
                   footer_note=("eval file robust_tagging_eval_small_l1t.pt; dashed = "
                                "colleague-suite families" if suite == "altdata" else None))
    plt.close("all")
    pc.log("wrote " + ", ".join(os.path.basename(w) for w in written))
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default=None, help="comma-separated tags; default is the ranked set")
    ap.add_argument("--include-heldout", action="store_true")
    ap.add_argument("--heldout", action="store_true",
                    help="draw heldout_<ts>.png from held-out-family runs only (F6)")
    ap.add_argument("--suite", default=None,
                    choices=["scoring", "heldout", "colleague", "altdata"],
                    help="which corruption suite to draw; overrides --heldout")
    ap.add_argument("--cap", type=int, default=pc.MAX_CURVES)
    ap.add_argument("--watch", action="store_true", help="poll runs/ and regenerate on every new JSON")
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--solo", action="store_true",
                    help="with --watch, regenerate only this figure (default: all three + README)")
    args = ap.parse_args()

    if not args.watch:
        generate(args.tags, args.include_heldout, args.cap, args.heldout,
                 suite=args.suite or ("heldout" if args.heldout else "scoring"))
        return

    def regen():
        written = generate(args.tags, args.include_heldout, args.cap)
        for suite in ("heldout", "colleague", "altdata"):
            written += generate(None, False, args.cap, suite=suite) or []
        written += pc.refresh_tsne() or []
        if not args.solo:
            import plot_official, plot_progress, plot_readme
            for mod in (plot_official, plot_progress):
                try:
                    written += mod.generate() or []
                except Exception as exc:
                    pc.log(f"{mod.__name__} failed, keeping its previous figure: {exc!r}")
            try:
                plot_readme.write_readme()
            except Exception as exc:
                pc.log(f"README regeneration failed: {exc!r}")
        return written

    regen()
    pc.watch(regen, interval=args.interval)


if __name__ == "__main__":
    main()
