"""Shared plumbing for WP-F's figures (plot_curves.py, plot_official.py, plot_progress.py).

Read-only over ~/hackathon-shared/runs. Nothing here writes a JSON, touches the GPU, or takes a
lock; the only things written are PNG/PDF/README under runs/plots/ plus the colour registry.

Two measurements live in runs/ and are NEVER put on one axis:
  * bench runs   (bench_eval.py)  -- our five development families, field `mean_area`
  * official runs (accept.sh)     -- the organisers' degradation_eval.py, field `official_area`
"""
import datetime as _dt
import glob
import hashlib
import json
import os
import re
import sys
import time

SHARED = os.path.expanduser("~/hackathon-shared")
RUNS_DIR = os.path.join(SHARED, "runs")
PLOTS_DIR = os.path.join(RUNS_DIR, "plots")
COLOR_REGISTRY = os.path.join(PLOTS_DIR, "colors.json")
ANCHOR_TAG = "anchor-stock-baseline"
MASKFIX_TAG = "planner-maskfix-stockckpt"
BASELINE_JSON = os.path.expanduser("~/rt-f/baseline_no_degradation_auc.json")

# Same order as ablation_table.py so the two artefacts read the same way.
FAMILY_ORDER = ["rect", "wedge", "strip", "towers", "cells",
                "ellipse", "annulus", "diagonal",
                "cell_dropout", "edge_truncation",
                "c_ellipse", "c_cell_dropout", "c_edge_truncation"]

# Three corruption suites. A run belongs to exactly one and they are NEVER put on one axis or
# averaged together: different shapes, different code, different difficulty.
SCORING = {"rect", "wedge", "strip", "towers", "cells"}          # bench_eval.py, the ruler
HELDOUT = {"ellipse", "annulus", "diagonal"}                     # our generalisation check
# Both spellings: E is adding a "c_" prefix so the ellipse name collision cannot recur at the
# data level, and JSONs written before that change carry the bare names.
COLLEAGUE = {"ellipse", "cell_dropout", "edge_truncation",
             "c_ellipse", "c_cell_dropout", "c_edge_truncation"}

# NOTE: "ellipse" appears in BOTH HELDOUT and COLLEAGUE, and the two are different
# implementations of the name -- colleague-group3 ships its own degradation module. So the family
# set alone cannot classify a run whose only family is ellipse. The tag suffix decides first.

# eval.py's own axis says "linear probe", but train_linear_probe builds an EvalMLP
# (Linear 128 - ReLU - 64 - ReLU - 32 - ReLU - out, models.py:313). Planner ruled the
# figures say MLP; x keeps eval.py's wording exactly.
X_LABEL = "Dropout severity (% of eta-phi plane)"
Y_LABEL = "AUC (MLP probe on latents)"

# The WP-F brief capped panels at 8; the planner raised it to 10 at 22:15 and to 12 at 00:22, so
# the two-view and clean ablation rows stay visible now that both R=20 re-benches qualify as late
# arms. See register class 10: a pin is a floor, a cap is a ceiling, and the two must be checked
# against each other whenever either moves.
MAX_CURVES = 12

# ---------------------------------------------------------------- colours
# Okabe-Ito, colour-blind safe, with the washed-out yellow dropped and two dark neutrals added.
ANCHOR_COLOR = "#D62728"   # reserved: the anchor / the organisers' "(No degradation)" reference
LEADER_COLOR = "#0072B2"   # reserved: whoever currently leads runs/table.md on mean_area
# Okabe-Ito (yellow dropped, it is illegible on white) then Tol's qualitative set, both
# colour-blind-safe. Longer than MAX_CURVES on purpose: with exactly as many colours as rows,
# one extra tag exhausts the palette and the allocator has to hand out a duplicate.
PALETTE = [
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#666666",  # grey
    "#8C6D1F",  # dark gold
    "#332288",  # indigo
    "#117733",  # dark green
    "#882255",  # wine
    "#44AA99",  # teal
    "#AA4499",  # magenta
    "#999933",  # olive
    "#661100",  # dark brown
    "#6699CC",  # steel blue
    "#DDAA33",  # amber
    "#004488",  # navy
    "#994455",  # brick
    "#6699AA",  # slate
    "#997700",  # bronze
]


# Hues within this many degrees of the reserved colours are skipped.
#
# WP-N, register class 14. This was a hard-coded (0.0, 210.0) while LEADER_COLOR is #0072B2, whose
# hue is 202 -- the guard protected a hue the accent does not use. And it was consulted only by
# _generated_colour(), the palette-exhaustion path, never by the curated PALETTE, which contains
# four entries inside the band: #56B4E9 (202, EXACTLY the accent's hue), #6699CC (210), #004488
# (210), #6699AA (195). So curves_latest shipped three blue-family lines under a footer asserting
# that blue identifies one specific row. Derive the band from the colours themselves so the two
# cannot drift apart again, and apply it to both allocation paths.
_RESERVED_BAND_DEG = 18.0


def _hue_of(hexcolor):
    import colorsys
    h = hexcolor.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return colorsys.rgb_to_hsv(r, g, b)[0] * 360.0


def _reserved_hues():
    """Derived from the reserved colours, never hard-coded beside them."""
    return (_hue_of(ANCHOR_COLOR), _hue_of(LEADER_COLOR))


def _near_reserved(hexcolor):
    """True if a colour sits in a reserved hue band and is saturated enough for the hue to read."""
    import colorsys
    h = hexcolor.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    hh, sat, _ = colorsys.rgb_to_hsv(r, g, b)
    if sat <= 0.2:                       # greys have no hue to confuse
        return False
    hh *= 360.0
    return any(min(abs(hh - x), 360 - abs(hh - x)) <= _RESERVED_BAND_DEG for x in _reserved_hues())


_RESERVED_HUES = _reserved_hues()



def _hue_sat_val(hexcolor):
    import colorsys
    h = hexcolor.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return colorsys.rgb_to_hsv(r, g, b)


def _perceptual_gap(a, b):
    """Rough separation between two colours: hue degrees, collapsed when either is desaturated.

    Exact-string uniqueness is not enough. The registry currently holds #E69F00 and #DDAA33, which
    are 0.5 degrees apart in hue -- different strings, indistinguishable as two lines on a panel.
    """
    ha, sa, va = _hue_sat_val(a)
    hb, sb, vb = _hue_sat_val(b)
    if sa <= 0.2 or sb <= 0.2:              # one is grey: separate them by lightness instead
        return abs(va - vb) * 360.0
    d = abs(ha * 360.0 - hb * 360.0)
    return min(d, 360.0 - d)


def _farthest_colour(used, candidates):
    """The candidate that maximises the minimum perceptual gap to everything already in use.

    Taking the FIRST free candidate is what clustered a repair pass onto hues 227/243/247 beside an
    existing 250 -- four blue-purples where the point of the repair was distinguishability.
    """
    best, best_gap = None, -1.0
    for c in candidates:
        if c in used or _near_reserved(c):
            continue
        gap = min((_perceptual_gap(c, u) for u in used), default=360.0)
        if gap > best_gap:
            best, best_gap = c, gap
    return best

def _generated_colour(i, used):
    """Deterministic extra colour i: golden-angle hue, fixed S/V, reserved hues avoided."""
    import colorsys
    h = (i * 137.508) % 360.0
    for _ in range(24):
        if all(min(abs(h - r), 360 - abs(h - r)) > 18.0 for r in _RESERVED_HUES):
            break
        h = (h + 37.0) % 360.0
    r, g, b = colorsys.hsv_to_rgb(h / 360.0, 0.62, 0.72)
    return "#%02X%02X%02X" % (int(r * 255), int(g * 255), int(b * 255))


def _load_registry():
    try:
        with open(COLOR_REGISTRY) as f:
            reg = json.load(f)
        if isinstance(reg, dict):
            return reg
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _save_registry(reg):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    tmp = COLOR_REGISTRY + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(reg, f, indent=2, sort_keys=True)
        os.replace(tmp, COLOR_REGISTRY)
    except OSError as exc:
        log(f"could not persist the colour registry: {exc}")


def color_for(key, leader=None):
    """One fixed colour per tag for the whole night, persisted in runs/plots/colors.json.

    Two colours are reserved and are NOT handed out by the registry:
      red    -- the anchor (and the organisers' reference curve, which is the same checkpoint)
      blue   -- whichever tag currently leads on mean_area; if the lead changes hands, the old
                leader falls back to its registry colour. This is the ONLY colour that moves,
                and every figure's footer says so.
    """
    if key == ANCHOR_TAG:
        return ANCHOR_COLOR
    if leader is not None and key == leader:
        return LEADER_COLOR
    import fcntl
    os.makedirs(PLOTS_DIR, exist_ok=True)
    lock_path = COLOR_REGISTRY + ".lock"
    try:
        lf = open(lock_path, "a+")
    except OSError:
        lf = None
    try:
        if lf is not None:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        # Re-read INSIDE the lock: the watcher process may have allocated since we last looked.
        reg = _load_registry()
        # Class 4's lesson, applied to class 14: fixing the allocator does not fix the entries the
        # broken allocator already wrote to disk. Repair persisted assignments that sit in a
        # reserved hue band, in the lock, and say so in the log -- a silent reassignment would be
        # its own silent failure, since a tag's colour is supposed to be stable for the night.
        offenders = sorted(t for t, c in reg.items() if _near_reserved(c))
        if offenders:
            for t in offenders:
                used = set(reg.values()) - {reg[t]}
                pool = list(PALETTE) + [_generated_colour(i, used) for i in range(len(reg) + 24)]
                alt = _farthest_colour(used, pool)
                if alt is None:
                    log(f"colour repair: no non-reserved colour available for {t!r}; left as {reg[t]}")
                    continue
                log(f"colour repair: {t!r} {reg[t]} sits in a reserved hue band "
                    f"(anchor/accent) -> reassigned {alt}")
                reg[t] = alt
            _save_registry(reg)
        if key not in reg:
            used = set(reg.values())
            # Class 14: filter the CURATED palette too, not just the generated fallback. A curated
            # colour inside a reserved band is exactly as confusable as a generated one.
            free = [c for c in PALETTE if c not in used and not _near_reserved(c)]
            if not free:
                blocked = [c for c in PALETTE if c not in used and _near_reserved(c)]
                if blocked:
                    log(f"palette: {len(blocked)} remaining curated colour(s) sit in a reserved "
                        f"hue band and were skipped for {key!r}; generating instead")
            if not free:
                # The curated palette has run out. Twice tonight the fallback was "reuse
                # PALETTE[0]", which put four rows on one figure in the same colour and, the
                # second time, five. Generate instead: golden-angle hue stepping at fixed
                # saturation and lightness gives an unlimited deterministic sequence, and hues
                # near the reserved red and blue are skipped so the anchor and the leader accent
                # stay unique. A generated colour is worse than a curated one; a DUPLICATE is
                # worse than both.
                free = [_generated_colour(i, used) for i in range(len(reg) + 1)]
                free = [c for c in free if c not in used] or [PALETTE[0]]
                log(f"palette exhausted at {len(reg)} tags; {key!r} takes generated {free[0]}")
            reg[key] = free[0]
            _save_registry(reg)
        return reg[key]
    finally:
        if lf is not None:
            try:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
                lf.close()
            except OSError:
                pass


# ---------------------------------------------------------------- logging
def distinct_colours(keys, leader=None):
    """{key: colour} for one figure, with NO two keys sharing a colour.

    The registry alone is not enough: it accumulated duplicate assignments while the palette was
    exhausted, and those entries persist after the palette is extended. Classes 3 and 4 of the
    silent-failure register are both "two rows drew in one colour"; this is the defence at the
    point that actually matters, which is a single figure, rather than a property the registry is
    trusted to have. A reassignment is persisted so the tag keeps its new colour from then on.
    """
    # WP-N: exact-string uniqueness is not enough, and scoping the search to the whole registry is
    # the wrong set. My own class-14 repair maximised separation against all 29 registry entries --
    # most of which never share a figure -- and handed a-pma-rep20 #91B745 next to
    # planner-maskfix-stockckpt #79B745, 12.6 degrees apart and both plainly yellow-green on the
    # same panel. Caught by looking at the render, not by any check. The registry already held a
    # 0.5-degree pair (#E69F00 / #DDAA33) that exact-match uniqueness passed for the same reason.
    # So the invariant belongs HERE, where the co-occurring set is known: minimum perceptual gap
    # between the rows of ONE figure.
    MIN_GAP_DEG = 15.0
    out, used = {}, set()
    for k in keys:
        c = color_for(k, leader=leader)
        too_close = next((u for u in used if _perceptual_gap(c, u) < MIN_GAP_DEG), None)
        if c not in used and too_close is not None and k != ANCHOR_TAG and k != leader:
            reg = _load_registry()
            taken = set(reg.values()) | used
            pool = list(PALETTE) + [_generated_colour(i, taken)
                                    for i in range(len(reg) + len(used) + 48)]
            # maximise separation against THIS FIGURE's rows, not against the whole registry
            alt, best = None, -1.0
            for cand in pool:
                if cand in used or _near_reserved(cand):
                    continue
                gap = min((_perceptual_gap(cand, u) for u in used), default=360.0)
                if gap > best:
                    alt, best = cand, gap
            if alt is not None and best > _perceptual_gap(c, too_close):
                log(f"colour too close: {k!r} {c} is {_perceptual_gap(c, too_close):.1f} deg from "
                    f"{too_close} already on this figure; reassigned {alt} (gap {best:.1f} deg)")
                reg[k] = alt
                _save_registry(reg)
                c = alt
        if c in used:
            reg = _load_registry()
            taken = set(reg.values()) | used
            # Class 14 again: this path handed out the first free PALETTE entry without checking
            # the reserved bands, so resolving a clash could re-create the confusion it fixes.
            pool = list(PALETTE) + [_generated_colour(i, taken)
                                    for i in range(len(reg) + len(used) + 24)]
            alt = _farthest_colour(taken, pool)
            if alt is None:
                alt = next(_generated_colour(i, taken) for i in range(len(reg) + len(used) + 1)
                           if _generated_colour(i, taken) not in taken)
            log(f"colour clash: {k!r} had {c}, already used on this figure; reassigned to {alt}")
            if k != ANCHOR_TAG and k != leader:
                reg[k] = alt
                _save_registry(reg)
            c = alt
        used.add(c)
        out[k] = c
    return out


def log(msg):
    print(f"[plot {_dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


# ---------------------------------------------------------------- loading
_TS_RE = re.compile(r"_(\d{8})_(\d{6})\.json$")


def wallclock(path, record=None):
    """When the run finished.

    Prefers the record's own `timestamp` field; falls back to the timestamp in the filename.
    E is adding `timestamp` to the official records (it was bench-only); the fallback stays so
    the figures keep working on every JSON written before that lands.
    """
    ts = (record or {}).get("timestamp")
    if isinstance(ts, str):
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y%m%d_%H%M%S"):
            try:
                return _dt.datetime.strptime(ts, fmt)
            except ValueError:
                pass
    m = _TS_RE.search(os.path.basename(path))
    if not m:
        return None
    try:
        return _dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def load_runs(runs_dir=None):
    """(bench runs, official runs). Malformed or partial JSON is skipped with a log line, never fatal."""
    runs_dir = runs_dir or RUNS_DIR      # module-level, so a caller can repoint it for a test
    bench, official = [], []
    for path in sorted(glob.glob(os.path.join(runs_dir, "*.json"))):
        try:
            with open(path) as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            log(f"skipping {os.path.basename(path)}: {exc}")
            continue
        if not isinstance(d, dict):
            continue
        d["_path"] = path
        d["_wallclock"] = wallclock(path, d)
        try:
            if d.get("kind") == "official":
                if "official_area" not in d:
                    log(f"skipping {os.path.basename(path)}: official record without official_area")
                    continue
                if d.get("smoke_test"):
                    log(f"skipping {os.path.basename(path)}: smoke_test record, not a real acceptance run")
                    continue
                official.append(d)
            elif all(k in d for k in ("tag", "families", "mean_area")):
                if not isinstance(d["families"], dict) or not d["families"]:
                    log(f"skipping {os.path.basename(path)}: empty families block")
                    continue
                bench.append(d)
        except Exception as exc:  # a partial file can be any shape at all
            log(f"skipping {os.path.basename(path)}: {exc}")
    return bench, official


def display_key(run):
    """Row identity for colouring and legends. A 20k run and a --full run of one tag are two rows."""
    key = run["tag"]
    if run.get("full"):
        key += " (full)"
    sfx = suite(run)
    if sfx in ("heldout", "colleague", "altdata", "other"):
        key += f" ({sfx})"
    return key



_REP_SUFFIX = __import__("re").compile(r"-rep\d+$")


def strip_rep(tag):
    """'a-pma-rep20' -> 'a-pma'. leader_tag() already treats -repNN as the same model."""
    return _REP_SUFFIX.sub("", tag)


def model_identity(runs):
    """{tag: identity key} — one model is one identity, so it takes one colour.

    Planner ruling: key on the recorded checkpoint basename, falling back to the tag with -repNN
    stripped when no checkpoint is recorded.

    With register class 1 respected, which is the whole difficulty. Grouping on the checkpoint
    ALONE is exactly what merged the anchor with the two mask-fix rows and dropped the anchor from
    every figure: one checkpoint, three genuinely different measurements. So a checkpoint merges
    tags only when it carries the SECOND discriminator too — every tag on it reduces to the same
    base once -repNN is stripped. anchor-stock-baseline and planner-maskfix-stockckpt share a
    checkpoint and do not reduce to a common base, so they stay separate, as they must.
    """
    by_ckpt = {}
    for r in runs:
        if r.get("ckpt"):
            by_ckpt.setdefault(os.path.basename(r["ckpt"]), set()).add(r["tag"])
    ident = {}
    for ck, tags in by_ckpt.items():
        bases = {strip_rep(t) for t in tags}
        if len(bases) == 1:                       # same model, re-benched: merge
            for t in tags:
                ident[t] = next(iter(bases))
        else:                                     # ambiguous: class 1, keep them apart
            log(f"identity: checkpoint {ck} carries {sorted(tags)} which do not share a base tag; "
                f"left separate")
    for r in runs:
        ident.setdefault(r["tag"], strip_rep(r["tag"]))
    return ident


def probe_repeats(run):
    try:
        return int(run.get("probe_repeats") or 1)
    except Exception:
        return 1

def ckpt_colour_donors(runs):
    """{checkpoint basename: the scoring tag whose colour that model already owns}.

    A row measured on another eval file or another suite is the SAME MODEL as its scoring row and
    should carry that row's colour on every figure. Only checkpoints owned by exactly one scoring
    tag become donors: the stock checkpoint is shared by anchor-stock-baseline and both mask-fix
    rows, which are three different measurements, so it is ambiguous and donates nothing. That
    ambiguity check is the whole safety property -- grouping on the checkpoint without it is what
    merged the anchor with the mask-fix rows and dropped the anchor from every figure.
    """
    owners = {}
    for r in runs:
        if not is_scoring(r) or not r.get("ckpt"):
            continue
        owners.setdefault(os.path.basename(r["ckpt"]), set()).add(r["tag"])
    return {ck: next(iter(tags)) for ck, tags in owners.items() if len(tags) == 1}


def family_suite(run):
    """The suite a run's FAMILIES belong to, ignoring which eval file it used.

    An altdata figure holds rows of more than one family set -- the scoring five on the L1T file
    and colleague-group3's three on the same file -- and a mean panel may only average within one.
    """
    stripped = dict(run)
    stripped.pop("data", None)
    return suite(stripped)


def base_tag(run):
    """The tag whose colour this row should take.

    A held-out run is the same model as its scoring run, benched on the other family set, and
    the planner ruled the two must share a colour. Its tag is the scoring tag plus a
    "-heldout" suffix (d-pma0-aug -> d-pma0-aug-heldout), so strip it.
    """
    t = run["tag"]
    for suffix in ("-heldout", "-colleague"):
        if t.endswith(suffix):
            return t[: -len(suffix)]
    return t


# The eval file every comparable number is measured on. E now records `data` in the bench JSON;
# runs written before that field existed all used this file, so a missing field means standard.
STANDARD_DATA = "robust_tagging_eval_small.pt"
# The training file the certified recipe used. A row trained on something else (the L1T train
# file, for the FPGA/HGQ2 line) is answering a different question and must be visible as such.
STANDARD_TRAIN = "robust_tagging_train_data_small.pt"

# What an architecture variant IS, in words, for rows whose class name only means something to
# its author. Keyed on `encoder_class`, which the bench records, rather than on the tag -- a tag
# map would be the naming-convention mistake of register classes 6 and 8. This only changes label
# TEXT; it never decides which figure a row lands on or what it is compared against.
ENCODER_NOTES = {
    "FloatBNPMAEncoder": "LayerNorm->BatchNorm, no quantization",
}


# Rows withdrawn by their own author. This is a documented human ruling, not an inferred rule --
# the same category as RULED_WINNER, and deliberately NOT a heuristic: a figure cannot tell a real
# failure from a harness artifact, so only the author can say which a row is.
#   gB-q-b1e-5-relu-l1t: WP-G, 00:45. The activation was a constructor kwarg, bench_eval builds
#   the encoder from eval.py's signature and passes no kwargs, so 81 ReLU-trained tensors were
#   strict-loaded into a GELU emulator. Every name and shape matched, nothing raised, and the
#   near-chance AUC is exactly what that produces. The distillation rms was a healthy 0.269: the
#   trained model was fine, the thing that benched it was wrong.
RETRACTED = {
    "gB-q-b1e-5-relu-l1t": "withdrawn by WP-G: activation mismatch in the bench harness "
                           "(ReLU weights loaded into a GELU graph), not a property of the model",
    "gD-bits3-l1t": "planner 00:47: a collapsed network (0.5000 +- 0.0000, one distinct latent "
                    "over 500 events, per-dim std exactly zero). Unlike the row above this IS a "
                    "real measurement, but it is not a result to plot on the operating-point "
                    "figure; it belongs on WP-G's accuracy-vs-bit-cap curve",
}

# Which of WP-G's sweep rows appear on the altdata figure. Planner 00:47: altdata shows the FPGA
# OPERATING POINT beside the two float references, not the sweep -- the sweep is WP-G's own
# accuracy-vs-bit-cap figure. Membership in G's sweep table is the recorded property that
# identifies a sweep row; this list is the explicit exception to it.
# WP-G designates cap 6 (gD-bits6-l1t) as the operating point, superseding the cap-8 row the
# planner named at 00:47: same accuracy (0.8080 +- 0.0001 vs 0.8079 +- 0.0011, tied) for 3.81e8
# EBOPs against 5.06e8, a 25% saving. The planner's rule was "show the operating point"; which row
# that IS belongs to the package that owns the line.
G_OPERATING_POINT = {
    "g-stageA-l1t",     # the float-BN reference the quantization is measured against
    "gF-bits6-relu-l1t",  # the operating point, MEASURED as a combination rather than composed
                          # from a 6-bit GELU row and a cap-8 ReLU row -- this project's own
                          # ruling records that ablation arms are not additive
    "gD-relu-l1t",      # activation ablation AT the operating cap -- not a bit-cap point. WP-G:
                        # placing it at x=8 on a bit-cap axis would assert it is one.
}

# Author-supplied descriptions for rows whose tag says nothing to a reader. TEXT ONLY: like
# ENCODER_NOTES this never decides which figure a row lands on, only what its label says. Keyed on
# the tag because these are per-row facts, not per-architecture ones.
TAG_NOTES = {
    "gD-relu-l1t": "activation ablation: ReLU at the operating cap",
    "gF-bits6-relu-l1t": "operating point: 6-bit cap + ReLU, 3.804e8 EBOPs",
}


def tag_note(run):
    return TAG_NOTES.get(run.get("tag"))


def is_g_sweep_row(run):
    """True for a WP-G sweep row that is not the designated operating point.

    Keyed on the CHECKPOINT PATH (every one of G's rows is under rt-g/quant/ckpt), not on
    membership in G's sweep table: the table is regenerated by a separate script and lags, so
    gD-bits4-l1t benched and reached the figure before its table row existed. Membership is kept
    as a second route in case a future row lives elsewhere. This is a recorded property of the
    measurement, not a tag pattern -- the distinction register classes 6 and 8 are about.
    """
    tag = run.get("tag")
    if not tag or tag in G_OPERATING_POINT:
        return False
    ck = run.get("ckpt") or ""
    return "/rt-g/" in ck or tag in _quant_rows()


# WP-G's training-side numbers, keyed by the same tag the bench JSONs use. These never reach a
# bench record, and one of them is the signal that would have caught the harness artifact of
# register class 11 without my having to ask its author.
QUANT_TABLE = os.path.expanduser("~/hackathon-shared/quant/g_sweep_table.json")
HEALTHY_DISTILL_RATIO = 0.05     # distill_rms / teacher_scale; the artifact row was 0.023
CHANCE_AREA = 0.55               # a real collapse sits at ~0.50


def _quant_rows():
    try:
        with open(QUANT_TABLE) as f:
            rows = json.load(f)
        return {r["tag"]: r for r in rows if isinstance(r, dict) and r.get("tag")}
    except (OSError, json.JSONDecodeError, TypeError, KeyError):
        return {}


def suspicious_collapse(run):
    """'This model trained fine but benches at chance' -- the shape of a harness artifact.

    WP-G's rule, and the pair matters: a row may legitimately be either well-distilled OR at
    chance, and only the combination is suspicious. The artifact of class 11 had
    distill_rms/teacher_scale = 0.269/11.87 = 0.023 while benching at 0.5226; the genuinely dead
    3-bit row is 3.003/11.28 = 0.27 and its collapse is real. Rows with no distill_rms are not
    distilled and are exempt rather than flagged.

    Returns a reason string, or None. Says so when it CANNOT run rather than passing silently:
    the whole point is not to mistake "not checked" for "checked and fine".
    """
    if run.get("mean_area") is None or run["mean_area"] >= CHANCE_AREA:
        return None
    q = _quant_rows().get(run.get("tag"))
    if not q:
        return None                                     # not a quantization row at all
    rms = q.get("distill_rms")
    if rms is None:
        return None                                     # not distilled: exempt, per WP-G
    scale = q.get("teacher_scale") or q.get("teacher_latent_scale")
    if not scale:
        return (f"benches at chance ({run['mean_area']:.4f}) and distill_rms is {rms:.3f}, but "
                f"teacher_scale is absent from {os.path.basename(QUANT_TABLE)}, so the health "
                f"ratio COULD NOT BE CHECKED -- treat this row as unverified, not as verified")
    ratio = rms / scale
    if ratio < HEALTHY_DISTILL_RATIO:
        return (f"trained well (distill_rms/teacher_scale = {ratio:.3f}) yet benches at chance "
                f"({run['mean_area']:.4f}) -- the shape of a harness artifact, not a model result")
    return None


def is_retracted(run):
    return run.get("tag") in RETRACTED


def encoder_note(run):
    return ENCODER_NOTES.get(str(run.get("encoder_class") or ""))


def alt_data(run):
    """The eval file's basename when a run used something other than the standard one, else None.

    winner-l1t sweeps the five SCORING families, so nothing about its family set marks it out --
    but it does so on robust_tagging_eval_small_l1t.pt, a different eval file. Its mean_area is
    not comparable to any other row's and it must not enter curves_latest, the ranking or the
    accent. Detected from the recorded data path rather than from the tag, because a tag suffix
    is a naming convention and this is a property of the measurement.
    """
    d = run.get("data")
    if not d:
        return None
    base = os.path.basename(d)
    return None if base == STANDARD_DATA else base


def train_data(run):
    """Basename of the file a row was TRAINED on, or None when the JSON does not say."""
    for k in ("train_data", "train_file", "train_data_path"):
        v = run.get(k)
        if isinstance(v, str) and v:
            return os.path.basename(v)
    return None


def train_state(run):
    """One of 'nonstandard', 'standard', 'unrecorded' — three states, never two.

    E's reasoning, and it is the sharper version of my own: `data` may default to the standard
    file because bench_eval LOADS it, so the recorded value is an observation of what the process
    did. `train_data` is not observable from there at all -- bench_eval sees a checkpoint, not its
    history -- so an unset value is written as null and means "no claim", NOT "the standard file".
    Collapsing null into 'standard' would put a guess in the same field shape as a measurement,
    which is silent-failure classes 6 and 8 relocated from a naming convention into a default.
    """
    recorded = run.get("train_data_recorded")
    t = train_data(run)
    if recorded is False or (recorded is None and t is None):
        return "unrecorded"
    if t is None:
        return "unrecorded"
    return "standard" if t == STANDARD_TRAIN else "nonstandard"


def alt_train(run):
    """The training file when the row RECORDS a non-standard one, else None."""
    return train_data(run) if train_state(run) == "nonstandard" else None


def eta_max(run):
    """The bench's eta_max, but only when the JSON says it shaped the corruption.

    E: eta_max rescales the five SCORING families only. The held-out suite has its own geometry
    and the colleague code hardcodes eta in [-3, 3], so a colleague row carries the flag value
    (5.0) while that number shaped nothing about its corruption. Labelling it "eta_max 5" would
    invite exactly the misreading the suite separation exists to prevent, so the value is reported
    only when `eta_max_applies` is true.
    """
    v = run.get("eta_max")
    if not isinstance(v, (int, float)):
        return None
    flag = run.get("eta_max_applies")
    if flag is not None and not flag:
        return None
    if flag is None and family_suite(run) != "scoring":
        return None          # legacy row: the flag's meaning, derived from the family set
    return float(v)


def eta_max_applies(run):
    """Whether eta_max actually shaped this row's corruption.

    Uses the recorded flag when present. Rows written before E added it carry no flag, so fall
    back to the property the flag encodes: eta_max rescales the five SCORING families only, so it
    applies exactly when the row's families are that set. That is still a recorded property of
    the measurement (the family set), not a guess from the tag.
    """
    if run.get("eta_max") is None:
        return False
    flag = run.get("eta_max_applies")
    if flag is not None:
        return bool(flag)
    return family_suite(run) == "scoring"


def suite(run):
    """Which corruption suite a run belongs to: scoring, heldout, colleague, or other.

    Tag suffix first, because "ellipse" belongs to two suites; family set as the fallback for
    runs written before a suffix convention existed.
    """
    if alt_data(run):
        return "altdata"
    tag = run.get("tag", "")
    for name in ("colleague", "heldout"):
        if tag.endswith("-" + name):
            return name
    fams = set(run.get("families") or {})
    if not fams:
        return "other"
    if fams & (COLLEAGUE - HELDOUT):        # cell_dropout / edge_truncation are unambiguous
        return "colleague"
    if fams.issubset(SCORING):
        return "scoring"
    if fams.issubset(HELDOUT):
        return "heldout"
    return "other"


def is_scoring(run):
    """Only these rows may enter curves_latest, the ranking, the leader, or progress.

    Anything else is a different corruption whose area is not comparable to mean_area. Written as
    an allow-list rather than a list of things to exclude: a new suite added upstream defaults to
    being kept OUT of the scoring figures rather than silently polluting them.
    """
    return suite(run) == "scoring"


def is_heldout(run):
    return suite(run) == "heldout"


def is_colleague(run):
    return suite(run) == "colleague"


def latest_per_key(runs):
    best = {}
    for r in runs:
        k = display_key(r)
        cur = best.get(k)
        if cur is None or r.get("timestamp", "") >= cur.get("timestamp", ""):
            best[k] = r
    return list(best.values())


def run_count(runs, key):
    """How many bench JSONs share this row's display key -- independent repeats of one measurement.

    latest_per_key keeps only the newest for the figure, which is right for a superseded run and
    wrong for an independent redraw: a-pma-rep20 was benched twice at R=20 and the two are draws
    from the same distribution, to be pooled rather than one discarding the other. The figure
    still shows the newest; the legend says how many exist so the pooled row in the table is
    findable from the figure.
    """
    return sum(1 for r in runs if display_key(r) == key)


def rank(runs):
    """Bench runs, newest per row key, sorted by mean_area descending."""
    rows = latest_per_key(runs)
    rows.sort(key=lambda r: r.get("mean_area", 0.0), reverse=True)
    return rows


# The accent marks the CERTIFIED SUBMISSION, which is a ruling, not a ranking. As of 00:19 it is
# no longer the top of the bench: a-pma separably beats it on mean_area at R=20 (ratio 2.56) and
# on held-out (1.63), while the officials tie. The ruling rests on the pre-registered
# official-margin rule, the completed certification chain and the 27.3x size difference, and is
# recorded as a disagreement between the proxy metric and the official one. So no figure may say
# "leader": the accent means "this is what we are shipping". Set to None to go back to
# rank-follows-accent.
RULED_WINNER = "d-pma0-aug-meanpt"


def leader_tag(bench_runs):
    """Tag whose row takes the accent colour: the certified submission, not the bench leader.

    While RULED_WINNER is set this returns it regardless of rank; otherwise the top of the ranking.

    A bare tag, not a display key: a tag's 20k, --full and held-out rows are the same model and
    must carry the same colour on every figure of the night.
    """
    rows = [r for r in rank(bench_runs) if is_scoring(r)]
    if RULED_WINNER:
        # A re-bench of the winner is still the winner: the convention in runs/ is the same tag
        # with a "-repNN" suffix (d-deepsets-aug -> d-deepsets-aug-rep10), so an R=20 re-bench of
        # the ruled arm must not lose the accent to its own re-measurement.
        for r in rows:
            if r["tag"] == RULED_WINNER or r["tag"].startswith(RULED_WINNER + "-rep"):
                return r["tag"]
    return rows[0]["tag"] if rows else None


def repeats_std(run, *what):
    """std of a field if the run carries probe repeats, else None (an (n=1) row keeps no error bar)."""
    if not run.get("probe_repeats") or run.get("probe_repeats", 1) < 2:
        return None
    node = run
    for k in what[:-1]:
        node = (node or {}).get(k) or {}
    v = node.get(what[-1])
    return float(v) if isinstance(v, (int, float)) else None


def area_label(run):
    """'0.8183 +- 0.0008' or '0.8183 (n=1)'."""
    sd = repeats_std(run, "mean_area_std")
    if sd is None:
        return f"{run['mean_area']:.4f} (n=1)"
    return f"{run['mean_area']:.4f} ± {sd:.4f}"


def families_of(runs):
    seen = set()
    for r in runs:
        seen.update(r.get("families") or {})
    cols = [f for f in FAMILY_ORDER if f in seen]
    return cols + sorted(seen - set(cols))


def load_baseline(path=BASELINE_JSON):
    """The organisers' reference curve. Missing is not fatal -- the figure just loses that line."""
    try:
        with open(path) as f:
            d = json.load(f)
        if "severities" in d and "aucs" in d:
            return d
        log(f"{path}: no severities/aucs, ignoring")
    except (OSError, json.JSONDecodeError) as exc:
        log(f"reference curve unavailable ({exc})")
    return None


# ---------------------------------------------------------------- output
def script_version(script_path):
    """Version stamp for the footer.

    ~/hackathon-shared is not a git repo, so the honest identifier for the file that actually ran
    is a hash of its bytes; the wp-f commit of the mirrored copy is appended when available.
    """
    try:
        with open(script_path, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()[:8]
    except OSError:
        sha = "unknown"
    stamp = f"{os.path.basename(script_path)}@{sha}"
    try:
        import subprocess
        h = subprocess.check_output(
            ["git", "-C", os.path.expanduser("~/rt-f"), "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
        stamp += f"  wp-f@{h}"
    except Exception:
        pass
    return stamp


def footer(fig, script_path, extra=""):
    txt = f"generated {_dt.datetime.now():%Y-%m-%d %H:%M} · {script_version(script_path)}"
    if extra:
        txt += " · " + extra
    # The footer grew past the canvas as the caveats accumulated -- caught by check_layout, not by
    # anyone looking. Wrap to the figure width so the provenance line cannot be cut off at the edge.
    import textwrap
    width_px = fig.get_size_inches()[0] * fig.dpi
    per_line = max(60, int(width_px / 4.6))            # measured 4.28 px/char at 6pt, plus margin
    lines = textwrap.wrap(txt, per_line) or [txt]
    fig.text(0.005, 0.004, "\n".join(lines), fontsize=6, color="#555555", ha="left", va="bottom")


def check_layout(fig, stem):
    """Report text that leaves the canvas or collides with other text. Runs on every render.

    WP-G's idea, and their evidence for it: their bit-cap figure had a reference label rendered
    entirely off the canvas, another overflowing its axes by 80px, and two annotations on top of
    each other -- for over an hour, while every number in it was correct. Nobody saw it because
    image reads were failing on hook timeouts for all four of us.

    The point generalises past that one figure. A layout defect is invisible to every check I had:
    the data is right, the file is written, the script exits 0. The only thing that catches it is
    looking, and tonight looking was unavailable. So the check has to be mechanical -- which makes
    it register class 12, and the same shape as class 4: a defence that depends on someone
    noticing is not a defence.
    """
    problems = {"clipped": [], "collision": []}
    try:
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
    except Exception as exc:
        log(f"{stem}: layout check could not run ({exc!r}) -- treat as UNCHECKED, not as clean")
        return problems
    fw, fh = fig.canvas.get_width_height()
    # Two sources of noise had to go first, because a check that cries wolf is register class 4
    # again -- a signal nobody reads is not a defence:
    #   * matplotlib keeps tick-label artists for ticks OUTSIDE the current view limits. They are
    #     never drawn, but they report window extents off the canvas, so every panel produced a
    #     spurious "-0.2 leaves the canvas". Skip any artist whose tick lies outside its axis.
    #   * fig.findobj reaches the same artist by more than one route, so every label collided with
    #     itself. Dedupe on id().
    live_ticks = set()
    for ax in fig.axes:
        for axis, lo, hi in ((ax.xaxis,) + tuple(ax.get_xlim()), (ax.yaxis,) + tuple(ax.get_ylim())):
            lo, hi = min(lo, hi), max(lo, hi)
            for tick, loc in zip(axis.get_major_ticks(), axis.get_majorticklocs()):
                if lo - 1e-9 <= loc <= hi + 1e-9:
                    live_ticks.update({id(tick.label1), id(tick.label2)})
    all_tick_labels = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for tick in axis.get_major_ticks():
                all_tick_labels.update({id(tick.label1), id(tick.label2)})
    boxes, seen_ids = [], set()
    for t in fig.findobj(match=lambda o: hasattr(o, "get_text")):
        if id(t) in seen_ids:
            continue
        seen_ids.add(id(t))
        if id(t) in all_tick_labels and id(t) not in live_ticks:
            continue                                   # tick outside the view: never rendered
        ax = getattr(t, "axes", None)
        # axis("off") sets `axison`, not visibility, so both must be checked.
        if ax is not None and (not ax.get_visible() or not getattr(ax, "axison", True)):
            continue                                   # label on an axes that is switched off
        try:
            txt = (t.get_text() or "").strip()
            if not txt or not t.get_visible():
                continue
            bb = t.get_window_extent(renderer=r)
        except Exception:
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > fw + 1 or bb.y1 > fh + 1:
            log(f"{stem}: LAYOUT text leaves the canvas ({fw}x{fh}): {txt[:60]!r} "
                f"at x {bb.x0:.0f}-{bb.x1:.0f}, y {bb.y0:.0f}-{bb.y1:.0f}")
            problems["clipped"].append(txt[:60])
        boxes.append((txt, bb))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            ox = min(a.x1, b.x1) - max(a.x0, b.x0)
            oy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ox <= 4 or oy <= 4:                     # a few px of kerning overlap is not a clash
                continue
            # Identical text in near-identical boxes is one artist reached twice by findobj, not
            # two labels on top of each other. G's real case was two DIFFERENT strings colliding.
            if boxes[i][0] == boxes[j][0] and abs(a.x0 - b.x0) < 2 and abs(a.y0 - b.y0) < 2:
                continue
            log(f"{stem}: LAYOUT text collision {boxes[i][0][:34]!r} x {boxes[j][0][:34]!r} "
                f"({ox:.0f}x{oy:.0f} px)")
            problems["collision"].append((boxes[i][0][:34], boxes[j][0][:34]))
    return problems


def check_visibility(png_path, colours, stem, floor_frac=0.25):
    """Count each row's own colour in the rendered PNG and report any row that is nearly invisible.

    WP-N, register class 13. check_layout answers "is the text legible"; nothing answered "is the
    DATA visible". plot_curves draws in rank order and matplotlib draws later calls on top, so the
    best rows were buried by the worst: on the 01:28 render the certified submission showed 287 px
    of its own colour against 21164 for the last-placed row, on a figure whose footer tells the
    reader to identify it by colour. Every other check passed -- colours unique, accent applied,
    legend correct, exit 0.

    This has to be measured on the raster, not reasoned about from the code, because occlusion is a
    property of what was drawn and not of what was requested.
    """
    try:
        from PIL import Image
        import collections
        im = Image.open(png_path).convert("RGB")
        cnt = collections.Counter(im.getdata())
    except Exception as exc:
        log(f"{stem}: visibility check could not run ({exc!r}) -- treat as UNCHECKED, not as clean")
        return
    counts = {}
    for key, hexcolor in colours.items():
        try:
            rgb = tuple(int(hexcolor.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        except Exception:
            continue
        counts[key] = cnt.get(rgb, 0)
    if not counts:
        return
    vals = sorted(counts.values())
    med = vals[len(vals) // 2] if len(vals) % 2 else (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) / 2
    if med <= 0:
        log(f"{stem}: visibility check found no row colours in the raster -- UNCHECKED")
        return
    floor = floor_frac * med
    for key, n in sorted(counts.items(), key=lambda kv: kv[1]):
        if n < floor:
            log(f"{stem}: VISIBILITY {key!r} shows {n} px of its own colour, under {floor:.0f} "
                f"({floor_frac:.0%} of the {med:.0f} px median) -- drawn but effectively invisible")
    return counts


INSPECTED_FILE = os.path.join(PLOTS_DIR, "inspected.json")


def inspection_note(stem, content_key=None):
    """Footer fragment recording whether a HUMAN has looked at this figure's current content.

    The planner asked every figure to state whether it was visually inspected. A footer that
    printed "visually inspected" at render time would assert it every time, including the times
    nobody looked -- register class 5, a figure asserting something nobody decided. So the claim is
    recorded out of band by whoever actually looked (WP-N is currently the only session whose image
    reads work), keyed to the content hash of the PNG. Re-rendering changes the hash, which
    silently retires the old claim: an inspection of the previous content is not an inspection of
    this one. Absence prints the honest negative.
    """
    try:
        with open(INSPECTED_FILE) as fh:
            rec = json.load(fh).get(stem)
    except Exception:
        rec = None
    if not rec:
        return "NOT visually inspected"
    if content_key is not None and rec.get("key") != content_key:
        return "NOT visually inspected (content changed since the last look)"
    return f"visually inspected by {rec.get('by', '?')} at {rec.get('at', '?')}"


def record_inspection(stem, content_key, by="N"):
    """Called by the session that actually looked at the figure, after looking."""
    digest = content_key
    try:
        with open(INSPECTED_FILE) as fh:
            rec = json.load(fh)
    except Exception:
        rec = {}
    rec[stem] = {"by": by, "at": time.strftime("%Y-%m-%d %H:%M"), "key": digest}
    with open(INSPECTED_FILE, "w") as fh:
        json.dump(rec, fh, indent=1)
    return rec[stem]

def save(fig, stem, script_path, extra_footer="", content_key=None):
    """Write <stem>_<YYYYMMDD_HHMM>.{png,pdf} and <stem>_latest.{png,pdf}. Returns the paths."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    # The inspection state is read BEFORE this render is written, so it describes the PREVIOUS
    # content; the hash check in inspection_note() retires it as soon as the new file lands, and
    # the next render prints "content changed since the last look" until someone looks again.
    footer(fig, script_path, (extra_footer + " · " if extra_footer else "") + inspection_note(stem, content_key))
    if content_key:
        try:
            with open(os.path.join(PLOTS_DIR, stem + "_key.txt"), "w") as fh:
                fh.write(content_key)
        except OSError:
            pass
    problems = check_layout(fig, stem) or {"clipped": [], "collision": []}
    # WP-N's brief: "no figure is published unless it passes". Nothing enforced that -- the check
    # logged and the render shipped anyway, and I did exactly what the register warns about: ran
    # the check, grepped past its output, and published a legend clipped on both edges. A log line
    # is not a defence (class 4). Clipping DESTROYS information -- a truncated label cannot be read
    # at all -- so it blocks the publish; a collision degrades information without destroying it,
    # so it warns. The timestamped file is still written for diagnosis; only _latest is withheld,
    # because _latest is what everything else consumes.
    ts = time.strftime("%Y%m%d_%H%M")
    written = []
    for ext in ("png", "pdf"):
        p = os.path.join(PLOTS_DIR, f"{stem}_{ts}.{ext}")
        fig.savefig(p, dpi=200)
        latest = os.path.join(PLOTS_DIR, f"{stem}_latest.{ext}")
        if problems["clipped"]:
            log(f"{stem}: NOT PUBLISHED to _latest -- {len(problems['clipped'])} text item(s) "
                f"leave the canvas; {stem}_{ts}.{ext} written for diagnosis only")
            written.append(p)
            continue
        try:
            with open(p, "rb") as src, open(latest, "wb") as dst:
                dst.write(src.read())
        except OSError as exc:
            log(f"could not refresh {latest}: {exc}")
        written.append(p)
    return written


def use_agg():
    import matplotlib
    matplotlib.use("Agg")


# ---------------------------------------------------------------- watch
def runs_fingerprint(runs_dir=None):
    """Set of (path, size) over runs/*.json -- changes when a JSON appears, grows or is rewritten."""
    runs_dir = runs_dir or RUNS_DIR
    out = set()
    for p in glob.glob(os.path.join(runs_dir, "*.json")):
        try:
            out.add((p, os.path.getsize(p)))
        except OSError:
            pass
    return out


def watch(regen, interval=300, runs_dir=None, event_log=None, prime=True):
    """Poll runs/ and call regen() whenever a JSON appears or changes. Never exits on an error.

    prime=True records the current contents as already seen, so a caller that has just drawn the
    figures once does not immediately redraw them and log every existing JSON as \"new\".
    """
    runs_dir = runs_dir or RUNS_DIR
    event_log = event_log or os.path.join(PLOTS_DIR, "watch_events.jsonl")
    seen = runs_fingerprint(runs_dir) if prime else set()
    log(f"watch: polling {runs_dir} every {interval}s (pid {os.getpid()})")
    while True:
        try:
            fp = runs_fingerprint(runs_dir)
            new = {p for p, _ in fp} - {p for p, _ in seen}
            if fp != seen:
                paths = sorted(os.path.basename(p) for p in new)
                reason = ("new: " + ", ".join(paths)) if paths else "a JSON changed on disk"
                log(f"watch: regenerating ({reason})")
                try:
                    written = regen() or []
                except Exception as exc:
                    log(f"watch: regeneration failed, keeping the previous figures: {exc!r}")
                    written = []
                seen = fp
                try:
                    with open(event_log, "a") as f:
                        f.write(json.dumps({
                            "time": _dt.datetime.now().isoformat(timespec="seconds"),
                            "reason": reason, "new": paths,
                            "figures": [os.path.basename(w) for w in written],
                            "latest": os.path.join(PLOTS_DIR, "curves_latest.png"),
                        }) + "\n")
                except OSError as exc:
                    log(f"watch: could not append to {event_log}: {exc}")
        except Exception as exc:
            log(f"watch: poll failed, retrying: {exc!r}")
        time.sleep(interval)


# ---------------------------------------------------------------- E's t-SNE figures
# bench/tsne_latents.py is E's. The planner folded it into this watch loop (00:12) but it must NOT
# run on every JSON: it is a 6000-event t-SNE over two models and three severities. It regenerates
# only when its own inputs change -- the two checkpoints it embeds, and the winner's newest bench
# JSON, which it reads the annotated AUCs from.
#
# It also runs the encoder on CUDA when CUDA is there and takes no lock of its own, and WP-F is
# not permitted to touch the GPU or either lock directly. So it is invoked through gpu_small.sh,
# which serialises it against the other small jobs. That is a deliberate deviation from "run it
# from the watch loop" and is stated in the plots README.
TSNE_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tsne_latents.py")

# Stems E's script writes. A stem listed in TSNE_HOLD is NOT snapshotted, so a figure whose claim
# its author has withdrawn cannot be made citable by an automatic loop. tsne_drift was held from
# 00:23 while E revised a caption that asserted the along-probe component explains the AUC loss --
# which E's own numbers refute. E confirmed the 00:25 render final at 01:20 and the hold is lifted.
# The mechanism stays: automatic snapshotting means "published" now happens with no human in the
# loop, so a way to say "not this one, not yet" is part of the design and not a workaround.
TSNE_STEMS = ("tsne_latents", "tsne_drift")
TSNE_HOLD = set()


def _tsne_inputs():
    """Checkpoints and bench tags to fingerprint, imported from E's script rather than copied.

    A constant duplicated across two files goes stale silently, and this one would go stale in the
    worst direction: the loop would keep reporting "inputs unchanged" while the inputs HAD changed
    -- a watcher that fails closed and looks healthy. E exports them so there is nothing to keep
    in sync.
    """
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("rt_tsne_consts", TSNE_SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return list(getattr(mod, "CHECKPOINT_PATHS", [])), list(getattr(mod, "BENCH_TAGS", []))
    except Exception as exc:
        log(f"tsne: could not import its constants ({exc!r}); treating inputs as unknown")
        return [], []


_tsne_state = {"fingerprint": None}


def _tsne_fingerprint():
    ckpts, tags = _tsne_inputs()
    if not ckpts:
        return None                      # unknown inputs: never claim "unchanged"
    parts = []
    for c in ckpts:
        try:
            st = os.stat(c)
            parts.append((os.path.basename(c), int(st.st_mtime), st.st_size))
        except OSError:
            parts.append((os.path.basename(c), None, None))
    for tag in tags:
        newest = None
        for p in glob.glob(os.path.join(RUNS_DIR, f"{tag}_2*.json")):
            try:
                m = os.path.getmtime(p)
            except OSError:
                continue
            if newest is None or m > newest[1]:
                newest = (os.path.basename(p), m)
        parts.append((tag, newest))
    return tuple(parts)


def refresh_tsne(force=False):
    """Re-run E's t-SNE only when its inputs changed, then keep a timestamped copy of each output."""
    fp = _tsne_fingerprint()
    if fp is None:
        return []                        # inputs unreadable; do not run and do not claim unchanged
    if not force and fp == _tsne_state["fingerprint"]:
        return []
    first = _tsne_state["fingerprint"] is None
    _tsne_state["fingerprint"] = fp
    if first and all(os.path.exists(os.path.join(PLOTS_DIR, f"{s}_latest.png")) for s in TSNE_STEMS):
        log("tsne: inputs unchanged since the existing figures; not re-running")
        return _snapshot_tsne()
    import subprocess
    # E added --device, so this runs entirely on the CPU: no GPU, no lock, nothing to breach.
    log("tsne: inputs changed, re-running with --device cpu (this is slow)")
    try:
        r = subprocess.run(["python", TSNE_SCRIPT, "--device", "cpu"], capture_output=True,
                           text=True, timeout=5400)
        if r.returncode != 0:
            log(f"tsne: failed rc={r.returncode}; keeping the previous figures. "
                f"{(r.stderr or '').strip()[-300:]}")
            return []
    except Exception as exc:
        log(f"tsne: could not run ({exc!r}); keeping the previous figures")
        return []
    return _snapshot_tsne()


def _snapshot_tsne():
    """E's script writes only *_latest; give each output a timestamped copy like every other figure."""
    ts = time.strftime("%Y%m%d_%H%M")
    out = []
    for stem in TSNE_STEMS:
        if stem in TSNE_HOLD:
            log(f"tsne: {stem} is on hold at its author's request; not snapshotting")
            continue
        src = os.path.join(PLOTS_DIR, f"{stem}_latest.png")
        dst = os.path.join(PLOTS_DIR, f"{stem}_{ts}.png")
        if os.path.exists(src) and not os.path.exists(dst):
            try:
                with open(src, "rb") as a, open(dst, "wb") as b:
                    b.write(a.read())
                out.append(dst)
            except OSError as exc:
                log(f"tsne: could not snapshot {stem}: {exc}")
    return out
