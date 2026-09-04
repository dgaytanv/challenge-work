"""Shared benchmark: AUC-vs-severity across fixed corruption families.

Mirrors the grader's mechanics (eval.py: clean latents -> probe on 80% -> same probe
scored on degraded latents, grace period dropped) but sweeps several dead-region
families so no one over-fits to a single shape.

Usage (from anywhere):
  python ~/hackathon-shared/bench/bench_eval.py --repo ~/rt-<wp> --ckpt <path.pth> --tag <wp>-<desc>
Add --full for the whole eval set (slow); default uses the first 20000 events.
"""
import argparse
import importlib
import importlib.util
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.expanduser("~/hack-data/C9_robust_tagging")
SEVERITIES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def load_repo_eval(repo):
    sys.path.insert(0, repo)
    sys.path.insert(0, os.path.join(repo, "src"))
    spec = importlib.util.spec_from_file_location("rt_eval", os.path.join(repo, "eval.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def git_hash(repo):
    try:
        h = subprocess.check_output(["git", "-C", repo, "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "-C", repo, "diff", "--quiet"]) != 0
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def _sha256(path):
    """sha256 of a file, or None if it cannot be read. Never fatal: a bench result is still a
    result if the digest could not be taken, and a null digest is honestly 'not recorded'."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True, help="e.g. wpA-maskfix; used in the output filename")
    ap.add_argument("--train_cfg", default=None)
    ap.add_argument("--data_cfg", default=None)
    ap.add_argument("--data", default=os.path.join(DATA_ROOT, "eval", "robust_tagging_eval_small.pt"))
    ap.add_argument("--encoder_class", default="TransformerEncoder")
    ap.add_argument("--families", default="all",
                    help="'all' (five scoring families), 'heldout' (ours: ellipse/annulus/diagonal), "
                         "'colleague' (Group 3 held-out suite), 'colleague-train' (Group 3 training "
                         "suite), or a comma-separated list. Only 'all' produces a scoring mean_area.")
    ap.add_argument("--train_data", default=None,
                    help="basename or path of the file the CHECKPOINT WAS TRAINED ON. bench_eval "
                         "cannot observe this -- it is a claim by the caller, unlike --data which "
                         "is a file this process actually loads. Left unset it records null, "
                         "meaning 'not recorded', which is what a consumer should treat as making "
                         "no claim. Pass it whenever the training file is not the standard "
                         "robust_tagging_train_data_small.pt.")
    ap.add_argument("--eta_max", type=float, default=5.0,
                    help="half-width of the eta plane the five scoring families are painted over. "
                         "Default 5.0 = the PF-candidate acceptance and reproduces every existing "
                         "PF number. Use 3.0 for the L1T/PUPPI file, whose data is eta in [-3,3]: "
                         "at 5.0 the families spend two thirds of the plane where there are no "
                         "candidates, so severity stops meaning the fraction removed (measured on "
                         "winner-l1t: rect removes 0.28/0.52/0.77/0.93/1.00 at s=0.2..1.0 against "
                         "a design target of s). Does not affect the heldout or colleague suites.")
    ap.add_argument("--use_repo_builder", action="store_true",
                    help="build preproc and encoder via the target repo's own eval.py "
                         "build_preproc_and_encoder, so a repo whose encoder needs extra config "
                         "(e.g. a pooling kwarg, or a PFPreProcessorV2) builds correctly")
    ap.add_argument("--max_events", type=int, default=20000)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--grace_period", type=int, default=1000)
    ap.add_argument("--probe_repeats", type=int, default=5,
                    help="refits of the (unseeded) linear probe over one set of embeddings; "
                         "reported as mean +- std so a difference can be judged against the noise")
    ap.add_argument("--seed", type=int, default=None,
                    help="TRAINING seed of the checkpoint, recorded in the JSON so the table can "
                         "group the seeds of one configuration and report a seed std. Like "
                         "--train_data this is a CLAIM by the caller about how the checkpoint was "
                         "made, not something this process observes; left unset it falls back to "
                         "the train config's `seed` hyperparameter, and records null if that is "
                         "absent too. null means 'not recorded', never 'seed 0'. It does NOT seed "
                         "anything here: the probe is deliberately unseeded and its spread is the "
                         "measurement (see writeup/E-probe-noise-floor.md).")
    ap.add_argument("--batch_size", type=int, default=512)  # 1024 used ~12 GB on the shared A10
    ap.add_argument("--outdir", default=os.path.expanduser("~/hackathon-shared/runs"))
    args = ap.parse_args()

    repo = os.path.abspath(os.path.expanduser(args.repo))
    ev = load_repo_eval(repo)
    sys.path.insert(0, BENCH_DIR)
    from bench_degradation import BenchDegradation, FAMILIES
    from heldout_degradation import HELDOUT_FAMILIES, HeldoutDegradation
    from colleague_degradation import (COLLEAGUE_HELDOUT_FAMILIES, COLLEAGUE_TRAIN_FAMILIES,
                                       ColleagueDegradation)
    COLLEAGUE_ALL = COLLEAGUE_HELDOUT_FAMILIES + COLLEAGUE_TRAIN_FAMILIES
    # Colleague families are written with a `c_` prefix. Their suite and our held-out set BOTH
    # contain a family called "ellipse", from different code producing different corruptions; with
    # bare names a JSON cannot be classified from its own contents. The prefix removes that
    # ambiguity at the data level so nothing downstream has to guess.
    CPFX = "c_"

    def make_degradation(fam, sev, sev_idx=0):
        """Resolve a family NAME to the corruption that owns it.

        `ellipse` is a family in BOTH our held-out suite and Group 3's, from different code
        producing different corruptions (runs/plots/README.md says so explicitly). The `c_` prefix
        exists to remove that ambiguity, so it is the ONLY way to reach a colleague family here.

        This order is load-bearing. Until 04:0x the colleague branch was tested first and matched
        the BARE name, so `--families heldout` -- which expands to bare names -- silently built
        Group 3's ellipse instead of ours for every held-out run after colleague_degradation.py
        landed (23:23). Measured: our ellipse drops 0.2311/0.3819/0.5780/0.7716/1.0000 of
        candidates at s=0.2..1.0, theirs 0.0802/0.1589/0.2271/0.2871/0.3343. annulus and diagonal
        are not colleague families, so they were unaffected -- which is exactly why the corruption
        swap looked like one family behaving oddly rather than like a bug.
        """
        if fam.startswith(CPFX):
            bare = fam[len(CPFX):]
            if bare in COLLEAGUE_ALL:
                # Group 3's families, ported verbatim. Their eta convention is [-3, 3] against our
                # [-5, 5], so the dropped fractions printed below are the ones to read, not severity.
                return ColleagueDegradation(bare, sev, severity_index=sev_idx)
            raise ValueError(f"unknown colleague family {fam!r}; known: "
                             f"{[CPFX + f for f in COLLEAGUE_ALL]}")
        if fam in HELDOUT_FAMILIES:
            return HeldoutDegradation(fam, sev)
        if fam in FAMILIES:
            return BenchDegradation(fam, sev, eta_max=args.eta_max)
        if fam in COLLEAGUE_ALL:
            # A bare colleague name is ambiguous by construction. Refuse rather than pick one.
            raise ValueError(
                f"family {fam!r} is ambiguous: it names both one of our suites and a Group 3 "
                f"family, which are different corruptions. Use {CPFX + fam!r} for Group 3's, or "
                f"a name from {list(FAMILIES) + list(HELDOUT_FAMILIES)} for ours.")
        raise ValueError(f"unknown family {fam!r}")

    from embedding.utils.cfg_handler import train_config, data_config
    from embedding.utils.data_utils import load_data

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = train_config(args.train_cfg or os.path.join(repo, "configs", "train_config.yaml"))
    cfg_data = data_config(args.data_cfg or os.path.join(repo, "configs", "data_config_eval.yaml"))
    num_classes = len(cfg_data.get_label_name_map())

    # Seed provenance: explicit flag wins, else the train config's hyperparameter, else nothing.
    # Recorded with its source so a consumer can tell a claimed seed from an inferred one.
    if args.seed is not None:
        seed_recorded, seed_source = int(args.seed), "flag"
    else:
        _cfg_seed = cfg.hp("seed", None)
        seed_recorded, seed_source = ((int(_cfg_seed), "train_cfg") if _cfg_seed is not None
                                      else (None, None))

    ckpt = torch.load(args.ckpt, map_location=device)
    if args.use_repo_builder:
        # Let the target repo construct its own modules. Needed when a repo's encoder takes extra
        # config we do not know about (Group 3's `pooling` kwarg) or ships its own preprocessor
        # (PFPreProcessorV2): reproducing their constructor here would be guesswork, and a wrong
        # guess on a preprocessor is SILENT because the state_dict keys match.
        preproc, encoder, norm_constants = ev.build_preproc_and_encoder(cfg, ckpt, device)
        print(f"[bench] built via {repo}/eval.py: preproc={type(preproc).__name__}, "
              f"encoder={type(encoder).__name__}")
    else:
        preproc_class = getattr(importlib.import_module("embedding.preprocs"), cfg.get_trdata_cfg("preproc_type", "PFPreProcessor"))
        norm_constants = ckpt["norm_constants"]
        preproc = preproc_class(norm_constants).to(device)
        preproc.load_state_dict(ckpt["preproc"]); preproc.eval()
        models = importlib.import_module("embedding.models")
        EncCls = getattr(models, args.encoder_class)
        encoder = EncCls(
            num_features=preproc.num_features,
            embed_size=cfg.hp("embed_size", 128), latent_dim=cfg.hp("latent_dim", 6),
            num_heads=cfg.hp("num_heads", 8), num_layers=cfg.hp("num_layers", 4),
            linear_dim=cfg.hp("linear_dim", None), num_tokens=None,
            pairwise=cfg.get_trdata_cfg("pairwise", False),
        ).to(device)
        encoder.load_state_dict(ckpt["encoder"]); encoder.eval()

    max_events = -1 if args.full else args.max_events
    feats, labels = load_data(args.data, map_location="cpu", max_events=max_events)
    print(f"[bench] {feats.shape[0]} events, encoder={args.encoder_class}, repo@{git_hash(repo)}, "
          f"eta_max={args.eta_max}")

    def embed(deg=None):
        lat, lab, _ = ev.embed_dataset(preproc, encoder, feats, labels, cfg_data, norm_constants, device,
                                       batch_size=args.batch_size, degradation=deg)
        lat, lab, _ = ev.drop_grace_period(lat, lab, torch.zeros(len(lab)), args.grace_period)
        return lat, lab

    if args.families == "all":
        fams = list(FAMILIES)          # unchanged default: the five scoring families
    elif args.families == "heldout":
        fams = list(HELDOUT_FAMILIES)  # honesty check, never mixed into mean_area of a scoring run
    elif args.families == "colleague":
        fams = [CPFX + f for f in COLLEAGUE_HELDOUT_FAMILIES]   # Group 3's HELD-OUT suite
    elif args.families == "colleague-train":
        fams = [CPFX + f for f in COLLEAGUE_TRAIN_FAMILIES]     # Group 3's TRAINING suite
    else:
        fams = args.families.split(",")

    # ---- Embed once. The corruptions are seeded, so these latents are fixed for the whole run;
    # the only thing that varies between repeats below is the probe refit.
    t0 = time.time()
    lat0, lab0 = embed(None)
    frozen, dropfrac, dead_area = {}, {}, {}
    map_sha, deg_cls = {}, {}
    for fam in fams:
        for sev_idx, s in enumerate(SEVERITIES[1:]):
            deg = make_degradation(fam, s, sev_idx).to(device).eval()
            with torch.no_grad():
                probe_batch = deg(feats[:2048].to(device))
                dropfrac[(fam, s)] = float(((probe_batch[..., 0] == 0) & (feats[:2048, :, 0].to(device) > 0)).float().mean())
            dead_area[(fam, s)] = deg.dead_area_fraction()
            # Fingerprint of the CORRUPTION itself, not of the result. The map tensor is the
            # corruption for every map-based family, and its digest is independent of the eval
            # file, the model and the probe -- so it identifies "which ellipse did this row
            # actually apply". Group 3's ellipse was silently substituted for ours for hours and
            # disagreed in no field a consumer reads except dropped_fraction. This makes that
            # checkable instead of merely recorded. None for families with no grid (Group 3's
            # corruptions are procedural), where the dropped-fraction vector is the fallback.
            _grid = getattr(deg, "grid", None)
            map_sha[(fam, s)] = (hashlib.sha256(
                _grid.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                if _grid is not None else None)
            deg_cls[fam] = type(deg).__name__
            frozen[(fam, s)] = embed(deg)
        print(f"[bench] embedded {fam:7s} ({time.time()-t0:.0f}s)")

    # ---- Refit the probe R times over those frozen latents. eval.py's train_linear_probe seeds
    # nothing -- random EvalMLP init, unseeded shuffling -- so a single fit carries noise of
    # unknown size. Repeating it turns that into an error bar instead of a false precision.
    R = max(1, args.probe_repeats)
    rep_clean_test, rep_clean_full, rep_mean_area = [], [], []
    rep_area = {f: [] for f in fams}
    rep_aucs = {(f, s): [] for f in fams for s in SEVERITIES[1:]}
    for r in range(R):
        Xtr, Xte, ytr, yte = train_test_split(lat0, lab0, stratify=lab0, test_size=0.2, random_state=42)
        probe = ev.train_linear_probe(Xtr, ytr, num_classes, device)
        act = float(ev.probe_auc(probe, Xte, yte, num_classes, device))
        acf = float(ev.probe_auc(probe, lat0, lab0, num_classes, device))
        rep_clean_test.append(act); rep_clean_full.append(acf)
        areas = []
        for fam in fams:
            sevs, aucs = [0.0], [acf]
            for s in SEVERITIES[1:]:
                lat, lab = frozen[(fam, s)]
                a = float(ev.probe_auc(probe, lat, lab, num_classes, device))
                rep_aucs[(fam, s)].append(a)
                sevs.append(s); aucs.append(a)
            ar = float(ev.area_under_curve(sevs, aucs))
            rep_area[fam].append(ar); areas.append(ar)
        rep_mean_area.append(float(np.mean(areas)))
        print(f"[bench] probe refit {r+1}/{R}: clean {acf:.4f}  mean_area {rep_mean_area[-1]:.4f}")

    def ms(xs):
        return float(np.mean(xs)), (float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0)

    auc_clean_test, auc_clean_test_std = ms(rep_clean_test)
    auc_clean_full, auc_clean_full_std = ms(rep_clean_full)
    mean_area, mean_area_std = ms(rep_mean_area)
    print(f"[bench] clean AUC: probe test split {auc_clean_test:.4f}, full set {auc_clean_full:.4f} "
          f"+- {auc_clean_full_std:.4f}  ({time.time()-t0:.0f}s)")

    results = {}
    for fam in fams:
        area, area_std = ms(rep_area[fam])
        sevs, aucs, aucs_std, dfs = [0.0], [auc_clean_full], [auc_clean_full_std], [0.0]
        for s in SEVERITIES[1:]:
            m, sd = ms(rep_aucs[(fam, s)])
            sevs.append(s); aucs.append(m); aucs_std.append(sd); dfs.append(dropfrac[(fam, s)])
            print(f"[bench] {fam:7s} s={s:.1f} dead_area={dead_area[(fam, s)]:.2f} "
                  f"dropped={dropfrac[(fam, s)]:.2f}  AUC={m:.4f} +- {sd:.4f}")
        results[fam] = {
            "severities": sevs, "aucs": aucs, "aucs_std": aucs_std, "dropped_fraction": dfs,
            "area": area, "area_std": area_std, "area_repeats": rep_area[fam],
            # Which corruption this row actually applied. `map_sha256` is the digest of the dead
            # map at each swept severity and is independent of the eval file, the model and the
            # probe; `degradation_class` says which implementation produced it. Together they
            # answer "is this our ellipse or Group 3's" without needing anyone to eyeball
            # dropped_fraction.
            "map_sha256": [map_sha[(fam, sv)] for sv in SEVERITIES[1:]],
            "degradation_class": deg_cls.get(fam),
        }
        print(f"[bench] {fam:7s} area = {area:.4f} +- {area_std:.4f}")

    out = {
        "tag": args.tag, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repo": repo, "commit": git_hash(repo), "ckpt": os.path.abspath(args.ckpt),
        "encoder_class": type(encoder).__name__, "encoder_class_arg": args.encoder_class,
        "preproc_class": type(preproc).__name__, "built_via_repo_eval": bool(args.use_repo_builder),
        "families_arg": args.families, "data": os.path.abspath(args.data),
        "eta_max": args.eta_max,
        # eta_max rescales the five SCORING families only. The heldout and colleague suites carry
        # their own geometry (the colleague code hardcodes eta in [-3,3]), so recording the flag
        # without this marker would invite reading it as having shaped those corruptions.
        "eta_max_applies": args.families == "all" or not (
            set(fams) & (set(HELDOUT_FAMILIES) | {CPFX + f for f in COLLEAGUE_ALL})),
        "train_data": (os.path.basename(args.train_data) if args.train_data else None),
        "train_data_recorded": bool(args.train_data),
        "seed": seed_recorded, "seed_source": seed_source,
        # Encoder parameter count, for the "ties go to the smaller model" clause. Counted from the
        # built encoder, so unlike seed/train_data it is OBSERVED here rather than claimed. Excludes
        # the preprocessor (its BatchNorm is 10 numbers) and the projector/classifier, which the
        # grader never runs -- this is the size of the thing that ships.
        # sha256 of the checkpoint file, so a row can be tied to the exact bytes it measured even
        # after the path is gone (reference checkpoints have been living in session scratchpads,
        # which are ephemeral). Observed here, not claimed. Planner ruling 03:56.
        "ckpt_sha256": _sha256(args.ckpt),
        "encoder_params": int(sum(p.numel() for p in encoder.parameters())),
        "encoder_params_trainable": int(sum(p.numel() for p in encoder.parameters()
                                            if p.requires_grad)),
        "events": int(feats.shape[0]), "full": args.full,
        "probe_repeats": R,
        "auc_clean_test": auc_clean_test, "auc_clean_test_std": auc_clean_test_std,
        "auc_clean_full": auc_clean_full, "auc_clean_full_std": auc_clean_full_std,
        "auc_clean_full_repeats": rep_clean_full,
        "families": results,
        "mean_area": mean_area, "mean_area_std": mean_area_std, "mean_area_repeats": rep_mean_area,
    }
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"{args.tag}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("\n[bench] SUMMARY  tag=%s  clean=%.4f+-%.4f  mean_area=%.4f+-%.4f  (probe_repeats=%d)"
          % (args.tag, auc_clean_full, auc_clean_full_std, mean_area, mean_area_std, R))
    print("        " + "  ".join(f"{k}={v['area']:.4f}" for k, v in results.items()))
    print(f"[bench] wrote {path}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
