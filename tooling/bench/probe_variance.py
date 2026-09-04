"""How much of a mean_area difference is just noise?

`eval.py`'s `train_linear_probe` builds an `EvalMLP` with a random init and trains it with an
unseeded shuffling DataLoader. Nothing in `eval.py` or `bench_eval.py` calls `manual_seed`, so
**the probe is refit differently on every run**, and every area we quote inherits that noise --
the organisers' official number and our own five-family `mean_area` alike.

This measures that noise floor, and it isolates it cleanly: the event embeddings are computed
ONCE (BenchDegradation is seeded, so the dead maps and the dropped particles are deterministic),
and then only the probe is refit `--repeats` times. Any spread in the resulting areas is
therefore attributable to the probe alone, not to the corruption sampling.

Usage (through the small-job lock; one embedding pass, so it is cheap):
  ~/hackathon-shared/gpu_small.sh python ~/hackathon-shared/bench/probe_variance.py \
      --repo ~/rt-e --ckpt <ckpt.pth> --repeats 10

Read the output as: a difference between two tags smaller than a few times the reported
mean_area std is not evidence of anything.
"""
import argparse
import importlib
import os
import statistics
import sys
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
SEVERITIES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--families", default="all")
    ap.add_argument("--encoder_class", default="TransformerEncoder")
    ap.add_argument("--max_events", type=int, default=20000)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--grace_period", type=int, default=1000)
    ap.add_argument("--train_cfg", default=None)
    ap.add_argument("--data_cfg", default=None)
    ap.add_argument("--data", default=os.path.expanduser(
        "~/hack-data/C9_robust_tagging/eval/robust_tagging_eval_small.pt"))
    args = ap.parse_args()

    repo = os.path.abspath(os.path.expanduser(args.repo))
    sys.path.insert(0, BENCH_DIR)
    sys.path.insert(0, repo)
    sys.path.insert(0, os.path.join(repo, "src"))

    import importlib.util
    spec = importlib.util.spec_from_file_location("rt_eval", os.path.join(repo, "eval.py"))
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    from bench_degradation import BenchDegradation, FAMILIES
    from embedding.utils.cfg_handler import train_config, data_config
    from embedding.utils.data_utils import load_data

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = train_config(args.train_cfg or os.path.join(repo, "configs", "train_config.yaml"))
    cfg_data = data_config(args.data_cfg or os.path.join(repo, "configs", "data_config_eval.yaml"))
    num_classes = len(cfg_data.get_label_name_map())

    ckpt = torch.load(args.ckpt, map_location=device)
    preproc_class = getattr(importlib.import_module("embedding.preprocs"),
                            cfg.get_trdata_cfg("preproc_type", "PFPreProcessor"))
    norm_constants = ckpt["norm_constants"]
    preproc = preproc_class(norm_constants).to(device)
    preproc.load_state_dict(ckpt["preproc"]); preproc.eval()
    EncCls = getattr(importlib.import_module("embedding.models"), args.encoder_class)
    encoder = EncCls(
        num_features=preproc.num_features,
        embed_size=cfg.hp("embed_size", 128), latent_dim=cfg.hp("latent_dim", 6),
        num_heads=cfg.hp("num_heads", 8), num_layers=cfg.hp("num_layers", 4),
        linear_dim=cfg.hp("linear_dim", None), num_tokens=None,
        pairwise=cfg.get_trdata_cfg("pairwise", False),
    ).to(device)
    encoder.load_state_dict(ckpt["encoder"]); encoder.eval()

    feats, labels = load_data(args.data, map_location="cpu",
                              max_events=-1 if args.max_events <= 0 else args.max_events)
    fams = list(FAMILIES) if args.families == "all" else args.families.split(",")

    def embed(deg=None):
        lat, lab, _ = ev.embed_dataset(preproc, encoder, feats, labels, cfg_data, norm_constants,
                                       device, batch_size=args.batch_size, degradation=deg)
        lat, lab, _ = ev.drop_grace_period(lat, lab, torch.zeros(len(lab)), args.grace_period)
        return lat, lab

    # ---- embed once; the corruption is seeded, so these tensors are fixed for the whole run
    t0 = time.time()
    print(f"[probe-var] embedding {feats.shape[0]} events once "
          f"({1 + len(fams) * (len(SEVERITIES) - 1)} passes)...", flush=True)
    lat0, lab0 = embed(None)
    degraded = {}
    for fam in fams:
        for s in SEVERITIES[1:]:
            degraded[(fam, s)] = embed(BenchDegradation(fam, s).to(device).eval())
    print(f"[probe-var] embedding done in {time.time()-t0:.0f}s; "
          f"refitting the probe {args.repeats}x", flush=True)

    # ---- refit the probe repeatedly; nothing else varies
    clean_aucs, mean_areas, per_family = [], [], {f: [] for f in fams}
    for rep in range(args.repeats):
        Xtr, Xte, ytr, yte = train_test_split(lat0, lab0, stratify=lab0, test_size=0.2, random_state=42)
        probe = ev.train_linear_probe(Xtr, ytr, num_classes, device)
        auc_clean_full = float(ev.probe_auc(probe, lat0, lab0, num_classes, device))
        clean_aucs.append(auc_clean_full)
        areas = []
        for fam in fams:
            sevs, aucs = [0.0], [auc_clean_full]
            for s in SEVERITIES[1:]:
                lat, lab = degraded[(fam, s)]
                sevs.append(s); aucs.append(float(ev.probe_auc(probe, lat, lab, num_classes, device)))
            a = float(ev.area_under_curve(sevs, aucs))
            per_family[fam].append(a); areas.append(a)
        mean_areas.append(float(np.mean(areas)))
        print(f"[probe-var] repeat {rep+1:2d}/{args.repeats}  clean {auc_clean_full:.4f}  "
              f"mean_area {mean_areas[-1]:.4f}", flush=True)

    def summarise(name, xs):
        sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
        print(f"  {name:12s} mean {statistics.mean(xs):.4f}  std {sd:.4f}  "
              f"min {min(xs):.4f}  max {max(xs):.4f}  spread {max(xs)-min(xs):.4f}")

    print(f"\n[probe-var] {args.repeats} probe refits on IDENTICAL embeddings "
          f"({args.encoder_class}, {os.path.basename(args.ckpt)})")
    summarise("clean AUC", clean_aucs)
    for fam in fams:
        summarise(fam, per_family[fam])
    summarise("mean_area", mean_areas)
    sd = statistics.stdev(mean_areas) if len(mean_areas) > 1 else 0.0
    print(f"\n[probe-var] NOISE FLOOR: a mean_area difference below ~{3*sd:.4f} (3 sigma) between "
          f"two tags\n            is not evidence of a real difference. Everything varying here is "
          f"the probe refit alone.")
    print("""
[probe-var] How to read these spreads (they are NOT independent draws):
            The probe is fit on the CLEAN latents once and then applied frozen to every
            severity, so one refit's error is strongly correlated across the severities of a
            family -- a probe that fits the clean split well looks good everywhere. The five
            family areas within a repeat share that same probe and are correlated too.
            Consequence: threshold on the mean_area sigma above. Do NOT propagate the
            per-severity AUC sigmas as if they were independent; that double-counts and
            overstates the floor.""")


if __name__ == "__main__":
    main()
