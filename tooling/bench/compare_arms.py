#!/usr/bin/env python
"""Three-metric comparison of two arms, with the planner's separability threshold applied.

  python ~/hackathon-shared/bench/compare_arms.py a-pma d-pma0-aug-meanpt

Read-only over runs/. Exists so the decisive comparison is computed the same way every time
rather than by hand in a message: mean_area and held-out get 3*sqrt(sa^2/Ra + sb^2/Rb) from each
row's OWN sigma (the probe floor is architecture-dependent -- 0.0008 transformer, 0.0020 Deep
Sets, 0.0038 PMA per refit -- so a shared constant is wrong), and official areas are reported as
mean with min-max, since with n=2 or 3 there is a spread and not a sigma.

A ratio just over 1.0 is not a result. It is printed so it can be seen for what it is.
"""
import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_common as pc


def bench_row(runs, tag):
    hits = [r for r in runs if r["tag"] == tag and not pc.is_heldout(r)]
    return max(hits, key=lambda r: r.get("timestamp", "")) if hits else None


def heldout_row(runs, tag):
    hits = [r for r in runs if pc.is_heldout(r) and pc.base_tag(r) == tag]
    return max(hits, key=lambda r: r.get("timestamp", "")) if hits else None


def threshold(a, b):
    """The planner's rule, using each row's own sigma. None when either row lacks repeats."""
    for r in (a, b):
        if not r or not r.get("mean_area_std") or not r.get("probe_repeats"):
            return None
    return 3 * math.sqrt(a["mean_area_std"] ** 2 / a["probe_repeats"]
                         + b["mean_area_std"] ** 2 / b["probe_repeats"])


def verdict(a, b, name_a, name_b, label):
    if a is None or b is None:
        return f"  {label:10s} no {label} run for {name_a if a is None else name_b}"
    thr = threshold(a, b)
    d = a["mean_area"] - b["mean_area"]
    line = (f"  {label:10s} {name_a} {a['mean_area']:.6f} ± {a.get('mean_area_std', 0):.6f} "
            f"(R={a.get('probe_repeats')})  vs  {name_b} {b['mean_area']:.6f} "
            f"± {b.get('mean_area_std', 0):.6f} (R={b.get('probe_repeats')})")
    if thr is None:
        return line + f"\n  {'':10s} diff {d:+.6f}, threshold unavailable (a row has no repeats)"
    ratio = abs(d) / thr if thr else float("inf")
    who = name_a if d > 0 else name_b
    call = f"{who} separably ahead" if abs(d) > thr else "TIED"
    edge = "   <-- at the edge, treat as tied" if 1.0 < ratio < 1.3 else ""
    return line + (f"\n  {'':10s} diff {d:+.6f}  threshold {thr:.6f}  ratio {ratio:.2f}"
                   f"  -> {call}{edge}")


def official(tag):
    areas = []
    for p in sorted(glob.glob(os.path.join(pc.RUNS_DIR, f"official_{tag}_*.json"))):
        try:
            d = json.load(open(p))
        except (OSError, json.JSONDecodeError):
            continue
        if d.get("kind") == "official" and not d.get("smoke_test") and "official_area" in d:
            areas.append(d["official_area"])
    return areas


def seed_members(runs, config, heldout=False, regime=None):
    """Every seed of one CONFIGURATION, WITHIN ONE DATA REGIME.

    Regime matters and is not optional: 00-common-v2.md rule 5 says a candidate is compared only
    against a reference trained in the same data regime. Without this filter a configuration whose
    name appears on both the small and the full file pools or silently picks one -- a fixture
    caught exactly that, comparing a full-file candidate against a small-file reference and
    reporting a confident ratio for it. Returns (members, regimes_seen) so the caller can refuse
    an ambiguous comparison rather than quietly resolving it.
    """
    import ablation_table as at
    out, regimes = {}, set()
    for r in runs:
        if pc.is_heldout(r) != heldout:
            continue
        tag = pc.base_tag(r) if heldout else r["tag"]
        cfg, seed_tag = at.config_key(tag)
        if cfg != config:
            continue
        rg = at.regime_of(r)
        regimes.add(rg)
        if regime is not None and rg != regime:
            continue
        seed = r.get("seed", seed_tag)
        if seed is None:
            seed = seed_tag
        key = (rg, seed)
        if key not in out or r.get("timestamp", "") > out[key].get("timestamp", ""):
            out[key] = r
    return out, regimes


def seed_verdict(runs, A, B, heldout=False, regime=None):
    """Campaign-2 seed test: the unit of evidence is the configuration, not the run."""
    import ablation_table as at
    label = "held-out" if heldout else "mean_area"
    ma, ra = seed_members(runs, A, heldout, regime)
    mb, rb = seed_members(runs, B, heldout, regime)
    if not ma or not mb:
        return f"  {label:10s} no seeds found for {A if not ma else B}"
    # Refuse rather than resolve. Two regimes under one config is not a comparison we can make.
    amb = {n: r for n, r in ((A, ra), (B, rb)) if len(r) > 1}
    if regime is None and amb:
        which = "; ".join(f"{n} spans {sorted(r)}" for n, r in amb.items())
        return (f"  {label:10s} AMBIGUOUS -- {which}. Pass --regime <train_data> to choose; "
                f"comparing across data regimes is not a valid comparison.")
    seen = {at_regime for at_regime in
            {k[0] for k in ma} | {k[0] for k in mb}}
    if len(seen) > 1:
        return (f"  {label:10s} AMBIGUOUS -- the two configurations are in different regimes "
                f"{sorted(seen)}; only same-regime comparisons are valid.")
    sa, sb = at.seed_stats(list(ma.values())), at.seed_stats(list(mb.values()))
    d, thr, v = at.seed_separability(sa, sb)
    line = (f"  {label:10s} {A} {sa['mean']:.6f} ± {fmt(sa['seed_std'])} (n={sa['n']}: "
            f"{','.join(str(k[1]) for k in sorted(ma, key=lambda x: (x[1] is None, x[1])))})  vs  "
            f"{B} {sb['mean']:.6f} ± {fmt(sb['seed_std'])} (n={sb['n']}: "
            f"{','.join(str(k[1]) for k in sorted(mb, key=lambda x: (x[1] is None, x[1])))})")
    if thr is None:
        return line + f"\n  {'':10s} diff {d:+.6f}  -> {v}"
    ratio = abs(d) / thr if thr else float("inf")
    who = A if d > 0 else B
    call = f"{who} separably ahead" if abs(d) > thr else "TIED"
    edge = "   <-- at the edge, three seeds do not resolve it" if 1.0 < ratio < 1.3 else ""
    return line + (f"\n  {'':10s} diff {d:+.6f}  threshold {thr:.6f}  ratio {ratio:.2f}"
                   f"  -> {call}{edge}")


def fmt(x):
    return f"{x:.6f}" if x is not None else "— (n=1, unmeasured)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag_a")
    ap.add_argument("tag_b")
    ap.add_argument("--runs_dir", default=None,
                    help="directory of bench JSONs (default the shared runs/). Exists so the seed "
                         "test can be exercised against a fixture before real seeded runs land.")
    ap.add_argument("--regime", default=None,
                    help="train_data basename selecting the data regime, e.g. "
                         "robust_tagging_train_data.pt. Required when a configuration has runs in "
                         "more than one regime; without it such a comparison is refused, not "
                         "guessed.")
    ap.add_argument("--seeds", action="store_true",
                    help="campaign-2 mode: treat the arguments as CONFIGURATIONS (tags without "
                         "their -s<seed> suffix), pool their seeds, and apply the seed test "
                         "3*sqrt(sa^2/na + sb^2/nb) with s the SEED std. The default per-run mode "
                         "uses the probe std instead and answers a different question: whether two "
                         "single runs differ, not whether two configurations do.")
    args = ap.parse_args()
    runs, _ = pc.load_runs(args.runs_dir) if args.runs_dir else pc.load_runs()
    A, B = args.tag_a, args.tag_b
    print(f"\n{A}  vs  {B}\n")
    if args.seeds:
        print("  [seed mode: +- is the SEED std over training runs, not the probe-refit std]")
        print(seed_verdict(runs, A, B, heldout=False, regime=args.regime))
        print(seed_verdict(runs, A, B, heldout=True, regime=args.regime))
        oa, ob = official(A), official(B)
        for name, o in ((A, oa), (B, ob)):
            if o:
                print(f"  {'official':10s} {name} mean {sum(o)/len(o):.6f}  n={len(o)}  "
                      f"min {min(o):.4f}  max {max(o):.4f}  spread {max(o)-min(o):.4f}")
            else:
                print(f"  {'official':10s} no official run for {name}")
        print()
        return
    print(verdict(bench_row(runs, A), bench_row(runs, B), A, B, "mean_area"))
    print(verdict(heldout_row(runs, A), heldout_row(runs, B), A, B, "held-out"))
    oa, ob = official(A), official(B)
    for name, o in ((A, oa), (B, ob)):
        if o:
            print(f"  {'official':10s} {name} mean {sum(o)/len(o):.6f}  n={len(o)}  "
                  f"min {min(o):.4f}  max {max(o):.4f}  spread {max(o)-min(o):.4f}")
        else:
            print(f"  {'official':10s} no official run for {name}")
    if oa and ob:
        ma, mb = sum(oa)/len(oa), sum(ob)/len(ob)
        print(f"  {'':10s} diff {ma-mb:+.6f}; worst {A} {min(oa):.4f} vs best {B} {max(ob):.4f}; "
              f"worst {B} {min(ob):.4f} vs best {A} {max(oa):.4f}")
    print()


if __name__ == "__main__":
    main()
