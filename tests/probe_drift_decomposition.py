"""How much of the degradation displacement can the frozen probe actually see?

Hypothesis (from WP-D and WP-C): Deep Sets scores mean_area 0.80 while its latent moves a large
fraction of the population spread, because most of that motion is in directions the probe does
not read. If true, a consistency loss should penalise the probe-visible component, not the raw
displacement -- and a whitened consistency term is the cheapest label-free analogue.

Two decompositions of dz = z_degraded - z_clean:
  (a) LOCAL: g = grad of the signal decision function w.r.t. the latent, at each clean event,
      using the grader's own EvalMLP probe. Split dz along g/|g| and orthogonal. Per-event g,
      since EvalMLP is nonlinear.
  (b) GLOBAL: w = weight vector of a logistic probe fit on clean latents; project dz on w/|w|.
      Plus the Mahalanobis norm of dz under the clean-latent covariance, which is what a whitened
      consistency term would penalise.

Every fraction is reported against a RANDOM-DIRECTION BASELINE. In d dimensions a random
displacement already has a substantial mean |cos| with any fixed direction (~0.4 at d=6), so
"only 30% along the probe direction" is meaningless until compared with chance. Without this
control the headline number is unreadable.

Read-only. Run:
  cd ~/rt-a && ~/hackathon-shared/gpu_small.sh python tests/probe_drift_decomposition.py \
      --repo ~/rt-a --ckpt <path> --encoder_class TransformerEncoder --tag stock
"""
import argparse
import importlib
import importlib.util
import json
import os
import sys

import numpy as np
import torch

BENCH = os.path.expanduser("~/hackathon-shared/bench")
DATA = os.path.expanduser("~/hack-data/C9_robust_tagging/eval/robust_tagging_eval_small.pt")
SEVERITIES = [0.2, 0.4, 0.6, 0.8, 1.0]


def load_repo(repo):
    sys.path.insert(0, repo)
    sys.path.insert(0, os.path.join(repo, "src"))
    spec = importlib.util.spec_from_file_location("rt_eval", os.path.join(repo, "eval.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--encoder_class", default="TransformerEncoder")
    ap.add_argument("--train_cfg", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--events", type=int, default=8000)
    ap.add_argument("--out", default=os.path.expanduser("~/hackathon-shared/runs"))
    args = ap.parse_args()

    repo = os.path.abspath(os.path.expanduser(args.repo))
    ev = load_repo(repo)
    sys.path.insert(0, BENCH)
    from bench_degradation import BenchDegradation, FAMILIES
    from embedding.utils.cfg_handler import train_config, data_config
    from embedding.utils.data_utils import load_data
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = train_config(args.train_cfg or os.path.join(repo, "configs", "train_config.yaml"))
    cfg_data = data_config(os.path.join(repo, "configs", "data_config_eval.yaml"))
    num_classes = len(cfg_data.get_label_name_map())

    ck = torch.load(args.ckpt, map_location=device)
    preprocs = importlib.import_module("embedding.preprocs")
    preproc = getattr(preprocs, cfg.get_trdata_cfg("preproc_type", "PFPreProcessor"))(ck["norm_constants"]).to(device)
    preproc.load_state_dict(ck["preproc"]); preproc.eval()
    models = importlib.import_module("embedding.models")
    encoder = getattr(models, args.encoder_class)(
        num_features=preproc.num_features, embed_size=cfg.hp("embed_size", 128),
        latent_dim=cfg.hp("latent_dim", 6), num_heads=cfg.hp("num_heads", 8),
        num_layers=cfg.hp("num_layers", 4), linear_dim=cfg.hp("linear_dim", None),
        num_tokens=None, pairwise=cfg.get_trdata_cfg("pairwise", False),
    ).to(device)
    encoder.load_state_dict(ck["encoder"]); encoder.eval()

    feats, labels = load_data(DATA, map_location="cpu", max_events=args.events)

    def embed(deg=None):
        lat, lab, _ = ev.embed_dataset(preproc, encoder, feats, labels, cfg_data,
                                       ck["norm_constants"], device, batch_size=512, degradation=deg)
        return ev.drop_grace_period(lat, lab, torch.zeros(len(lab)), 1000)[:2]

    z0, y0 = embed(None)
    d = z0.shape[1]
    Xtr, Xte, ytr, yte = train_test_split(z0, y0, stratify=y0, test_size=0.2, random_state=42)
    probe = ev.train_linear_probe(Xtr, ytr, num_classes, device)
    auc0 = float(ev.probe_auc(probe, z0, y0, num_classes, device))

    # (a) local: per-event gradient of the signal decision function through the grader's probe
    z = z0.clone().to(device).requires_grad_(True)
    out = probe(z)
    score = out[:, 1] - out[:, 0] if num_classes == 2 else out[:, 1] - out.mean(dim=1)
    g = torch.autograd.grad(score.sum(), z)[0].detach()
    ghat = g / g.norm(dim=1, keepdim=True).clamp(min=1e-12)

    # (b) global: logistic weight vector on clean latents
    # Fit on the SAME train split the EvalMLP probe uses, so the two AUCs are comparable.
    lr = LogisticRegression(max_iter=2000).fit(Xtr.numpy(), (ytr.numpy() == 1).astype(int))
    def lin_auc(z, y):
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score((y.numpy() == 1).astype(int), lr.predict_proba(z.numpy())[:, 1]))
    auc0_lin = lin_auc(z0, y0)
    w = torch.tensor(lr.coef_[0], dtype=torch.float32, device=device)
    what = (w / w.norm()).unsqueeze(0)

    # clean-latent covariance -> Mahalanobis (what a whitened consistency term penalises)
    z0d = z0.to(device)
    mu = z0d.mean(0, keepdim=True)
    cov = torch.cov(z0d.T.double()) + 1e-6 * torch.eye(d, device=device, dtype=torch.float64)
    cov_inv = torch.linalg.inv(cov)
    spread = float((z0d - mu).norm(dim=1).mean())

    # RANDOM-DIRECTION BASELINE: chance level of |cos| against ghat in this dimension
    rnd = torch.randn(z0.shape[0], d, generator=torch.Generator().manual_seed(0)).to(device)
    rnd = rnd / rnd.norm(dim=1, keepdim=True)
    base_g = float((rnd * ghat).sum(1).abs().mean())
    base_w = float((rnd * what).sum(1).abs().mean())

    print(f"tag={args.tag} encoder={args.encoder_class} latent_dim={d} events={z0.shape[0]} "
          f"ckpt_epoch={ck.get('epoch', '?')}")
    # prove which repo's code was actually used (the single editable install is a trap here)
    print(f"models module: {models.__file__}")
    print(f"clean AUC {auc0:.4f}   clean spread (mean ||z-mu||) {spread:.3f}")
    print(f"RANDOM-DIRECTION BASELINE |cos|: vs probe-grad {base_g:.3f}, vs logistic-w {base_w:.3f}")
    print("  (a fraction at or below this means the motion is no more probe-aligned than chance)\n")
    print(f"clean AUC: EvalMLP (grader's, nonlinear) {auc0:.4f}   logistic (linear) {auc0_lin:.4f}\n")
    print(f"{'family':8s} {'sev':>4s} {'|dz|/sprd':>10s} {'alongG':>7s} {'vs rnd':>7s} "
          f"{'alongW':>7s} {'shared':>7s} {'maha':>7s} {'MLP_AUC':>7s} {'dMLP':>8s} "
          f"{'LIN_AUC':>8s} {'dLIN':>9s}")

    rows = []
    for fam in FAMILIES:
        for s in SEVERITIES:
            deg = BenchDegradation(fam, s).to(device).eval()
            zd, yd = embed(deg)
            dz = (zd.to(device) - z0d)
            n = dz.norm(dim=1).clamp(min=1e-12)
            fg = float(((dz * ghat).sum(1).abs() / n).mean())
            fw = float(((dz * what).sum(1).abs() / n).mean())
            maha = float(torch.sqrt(torch.einsum("bi,ij,bj->b", dz.double(), cov_inv, dz.double())
                                    .clamp(min=0)).mean())
            raw = float(n.mean())
            auc = float(ev.probe_auc(probe, zd, yd, num_classes, device))
            aucl = lin_auc(zd, yd)
            # Is dz one shared shift of the whole population, or per-event scatter? A shared
            # shift is a very different failure: the cloud moves off the manifold the probe was
            # fit on, rather than events crossing the boundary individually.
            dzbar = dz.mean(0, keepdim=True)
            shared = float(dzbar.norm() / max(raw, 1e-9))
            cos_bar_w = float((dzbar / dzbar.norm().clamp(min=1e-12) * what).sum())
            rows.append(dict(family=fam, severity=s, rel=raw / spread, along_g=fg, along_w=fw,
                             maha=maha, raw=raw, auc=auc, shared=shared, cos_shift_w=cos_bar_w,
                             auc_linear=aucl))
            print(f"{fam:8s} {s:4.1f} {raw/spread:10.3f} {fg:7.3f} {fg/base_g:7.2f} "
                  f"{fw:7.3f} {shared:7.3f} {maha:7.3f} {auc:7.4f} {auc-auc0:+8.4f} "
                  f"{aucl:8.4f} {aucl-auc0_lin:+9.4f}")

    rank = lambda x: np.argsort(np.argsort(x))

    def corr_over(rs, k):
        # mean(rel*along_g) is already stored on family-averaged rows; recompute only for raw rows
        v = np.array([r.get("vis", r["rel"] * r["along_g"]) if k == "vis" else r[k] for r in rs])
        aa = np.array([r["auc"] for r in rs])
        if np.std(v) < 1e-12 or np.std(aa) < 1e-12:
            return float("nan"), float("nan")
        return float(np.corrcoef(v, aa)[0, 1]), float(np.corrcoef(rank(v), rank(aa))[0, 1])

    MEASURES = [("rel", "raw |dz| / spread"), ("maha", "Mahalanobis |dz|"),
                ("along_g", "fraction along probe grad"), ("vis", "ABSOLUTE probe-visible |dz|"),
                ("shared", "shared-shift fraction")]

    print(f"\nwhich displacement measure predicts AUC? pooled over all {len(rows)} points "
          f"(pearson / spearman)")
    for k, lbl in MEASURES:
        pr, sp = corr_over(rows, k)
        print(f"  {lbl:28s} r = {pr:+.3f}   rho = {sp:+.3f}")

    # cells behaves unlike rect in every earlier measurement, so break it out per family
    print(f"\nper-family (5 severity points each), pearson r:")
    print(f"{'family':8s} " + " ".join(f"{lbl[:16]:>17s}" for _, lbl in MEASURES))
    for fam in FAMILIES:
        rs = [r for r in rows if r["family"] == fam]
        print(f"{fam:8s} " + " ".join(f"{corr_over(rs, k)[0]:+17.3f}" for k, _ in MEASURES))

    print(f"\nfamily-averaged view (mean over severities per family):")
    print(f"{'family':8s} {'|dz|/sprd':>10s} {'alongG':>7s} {'vs rnd':>7s} {'shared':>7s} "
          f"{'maha':>7s} {'vis|dz|':>8s} {'mean AUC':>9s}")
    fam_rows = []
    for fam in FAMILIES:
        rs = [r for r in rows if r["family"] == fam]
        m = {k: float(np.mean([r[k] for r in rs]))
             for k in ("rel", "along_g", "maha", "auc", "shared")}
        m["vis"] = float(np.mean([r["rel"] * r["along_g"] for r in rs]))
        m["family"] = fam
        fam_rows.append(m)
        print(f"{fam:8s} {m['rel']:10.3f} {m['along_g']:7.3f} {m['along_g']/base_g:7.2f} "
              f"{m['shared']:7.3f} {m['maha']:7.3f} {m['vis']:8.3f} {m['auc']:9.4f}")
    print(f"\nacross the 5 family averages, pearson r vs mean AUC:")
    for k, lbl in MEASURES:
        pr, sp = corr_over(fam_rows, k)
        print(f"  {lbl:28s} r = {pr:+.3f}   rho = {sp:+.3f}")

    am = np.array([r["auc"] for r in rows]); al = np.array([r["auc_linear"] for r in rows])
    print(f"\nNONLINEAR vs LINEAR probe on the SAME latents (the along_w question):")
    print(f"  clean:         MLP {auc0:.4f}   linear {auc0_lin:.4f}")
    print(f"  mean degraded: MLP {am.mean():.4f}   linear {al.mean():.4f}")
    print(f"  mean drop:     MLP {auc0-am.mean():+.4f}   linear {auc0_lin-al.mean():+.4f}")
    print(f"  worst point:   MLP {am.min():.4f}   linear {al.min():.4f}")
    print("  dz is near-orthogonal to the linear discriminant (along_w ~0.02 vs 0.34 chance).")
    print("  If the linear probe degrades much LESS, the damage is the nonlinear probe")
    print("  extrapolating outside the clean latent's support, not a class boundary being crossed.")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"probedrift-{args.tag}.json")
    with open(path, "w") as f:
        json.dump(dict(tag=args.tag, encoder_class=args.encoder_class, ckpt=args.ckpt,
                       latent_dim=d, clean_auc=auc0, spread=spread,
                       baseline_cos_grad=base_g, baseline_cos_w=base_w, rows=rows), f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
