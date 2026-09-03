"""Is mean_area actually driven by per-event latent drift?

Every work package assumes it: the probe is frozen on clean latents, so an event whose latent
moves under degradation gets scored in the wrong place. That is an assumption, not a measurement.
This measures both quantities on the stock checkpoint and correlates them, giving a drift proxy
that costs no probe training.

Raw cosine is reported alongside two controls, because raw cosine is not trustworthy on its own:

  * CENTRED cosine. If the latent cloud sits far from the origin, every pair of events looks
    similar and cosine is pinned near 1 by the common offset rather than by per-event agreement.
    WP-C measured exactly that on the trained two-view latent (offset ~30x the per-event spread).
    Centring on the clean-set mean removes it. The offset ratio is printed so the reader can see
    whether this checkpoint has the problem at all.
  * SHUFFLED-PAIR control. Cosine between a degraded event and a DIFFERENT clean event. This is
    the floor: whatever a "matched" pair scores, an unmatched pair scores this. If matched and
    shuffled are close, the metric is measuring global geometry, not per-event identity, and it
    is worthless as a consistency signal no matter how well it correlates with AUC.

Read-only measurement on the shared bench families (the same thing bench_eval.py does); nothing
here trains or tunes on them.

Run:  cd ~/rt-a && ~/hackathon-shared/gpu_small.sh python tests/latent_drift_diagnostic.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.expanduser("~/hackathon-shared/bench"))

import importlib.util
spec = importlib.util.spec_from_file_location("rt_eval", os.path.join(REPO, "eval.py"))
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

from bench_degradation import BenchDegradation, FAMILIES
from embedding.models import TransformerEncoder
from embedding.preprocs import PFPreProcessor
from embedding.utils.cfg_handler import train_config, data_config
from embedding.utils.data_utils import load_data

CKPT = os.path.expanduser("~/hack-data/C9_robust_tagging/checkpoints/robust_tagging_encoder_20260902_212357.pth")
DATA = os.path.expanduser("~/hack-data/C9_robust_tagging/eval/robust_tagging_eval_small.pt")
SEVERITIES = [0.2, 0.4, 0.6, 0.8, 1.0]
N_EVENTS = 8000


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = train_config(os.path.join(REPO, "configs", "train_config.yaml"))
    cfg_data = data_config(os.path.join(REPO, "configs", "data_config_eval.yaml"))
    num_classes = len(cfg_data.get_label_name_map())

    ck = torch.load(CKPT, map_location=device)
    preproc = PFPreProcessor(ck["norm_constants"]).to(device)
    preproc.load_state_dict(ck["preproc"]); preproc.eval()
    encoder = TransformerEncoder(
        num_features=preproc.num_features, embed_size=cfg.hp("embed_size", 128),
        latent_dim=cfg.hp("latent_dim", 6), num_heads=cfg.hp("num_heads", 8),
        num_layers=cfg.hp("num_layers", 4), linear_dim=None, num_tokens=None, pairwise=False,
    ).to(device)
    encoder.load_state_dict(ck["encoder"]); encoder.eval()

    feats, labels = load_data(DATA, map_location="cpu", max_events=N_EVENTS)
    print(f"stock checkpoint, {feats.shape[0]} events, readout=cls\n")

    def embed(deg=None):
        lat, lab, _ = ev.embed_dataset(preproc, encoder, feats, labels, cfg_data,
                                       ck["norm_constants"], device, batch_size=512, degradation=deg)
        return ev.drop_grace_period(lat, lab, torch.zeros(len(lab)), 1000)[:2]

    z0, y0 = embed(None)
    Xtr, Xte, ytr, yte = train_test_split(z0, y0, stratify=y0, test_size=0.2, random_state=42)
    probe = ev.train_linear_probe(Xtr, ytr, num_classes, device)
    auc0 = float(ev.probe_auc(probe, z0, y0, num_classes, device))

    # Is the latent cloud offset from the origin? If ||mean|| dwarfs the per-event spread, raw
    # cosine is measuring the offset, not the event.
    mu = z0.mean(dim=0, keepdim=True)
    spread = float((z0 - mu).norm(dim=1).mean())
    offset = float(mu.norm())
    print(f"clean AUC {auc0:.4f}")
    print(f"latent offset ||mean|| = {offset:.3f}, per-event spread = {spread:.3f}, "
          f"ratio = {offset / max(spread, 1e-9):.2f}x")
    print("  (WP-C saw ~30x on the trained two-view latent; >~3x means raw cosine is "
          "dominated by the offset)\n")

    z0c = z0 - mu
    perm = torch.randperm(z0.shape[0], generator=torch.Generator().manual_seed(0))
    print(f"{'family':8s} {'sev':>4s} {'cos_raw':>8s} {'cos_cen':>8s} {'cos_shuf':>9s} "
          f"{'rel drift':>10s} {'AUC':>7s} {'dAUC':>7s}")

    rows = []
    for fam in FAMILIES:
        for s in SEVERITIES:
            deg = BenchDegradation(fam, s).to(device).eval()
            zd, yd = embed(deg)
            zdc = zd - mu
            cos = float(F.cosine_similarity(zd, z0, dim=1).mean())
            cos_c = float(F.cosine_similarity(zdc, z0c, dim=1).mean())
            cos_s = float(F.cosine_similarity(zdc, z0c[perm], dim=1).mean())   # unmatched floor
            rel = float(((zd - z0).norm(dim=1) / max(spread, 1e-9)).mean())
            auc = float(ev.probe_auc(probe, zd, yd, num_classes, device))
            rows.append((fam, s, cos, rel, auc, cos_c, cos_s))
            print(f"{fam:8s} {s:4.1f} {cos:8.4f} {cos_c:8.4f} {cos_s:9.4f} "
                  f"{rel:10.3f} {auc:7.4f} {auc-auc0:+7.4f}")

    def pearson(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    def spearman(a, b):
        rank = lambda v: np.argsort(np.argsort(v))
        return pearson(rank(a), rank(b))

    print(f"\nper-family correlation (5 severity points each):")
    print(f"{'family':8s} {'r(raw,AUC)':>11s} {'r(cen,AUC)':>11s} {'rho(cen,AUC)':>13s} "
          f"{'r(drift,AUC)':>13s}")
    for fam in FAMILIES:
        f_rows = [r for r in rows if r[0] == fam]
        c = np.array([r[2] for r in f_rows]); d = np.array([r[3] for r in f_rows])
        a = np.array([r[4] for r in f_rows]); cc = np.array([r[5] for r in f_rows])
        print(f"{fam:8s} {pearson(c, a):+11.3f} {pearson(cc, a):+11.3f} "
              f"{spearman(cc, a):+13.3f} {pearson(d, a):+13.3f}")

    cc_all = np.array([r[5] for r in rows]); cs_all = np.array([r[6] for r in rows])
    a_all = np.array([r[4] for r in rows])
    print(f"\ncentred, all 25 points: pearson r = {pearson(cc_all, a_all):+.3f}, "
          f"spearman = {spearman(cc_all, a_all):+.3f}")
    print(f"shuffled-pair control: matched mean {cc_all.mean():+.4f} vs unmatched mean "
          f"{cs_all.mean():+.4f}  (gap {cc_all.mean() - cs_all.mean():+.4f})")
    print("  A gap near zero would mean the metric tracks global geometry, not per-event identity.")

    cos = np.array([r[2] for r in rows]); rel = np.array([r[3] for r in rows])
    auc = np.array([r[4] for r in rows])
    print(f"\ncorrelation across all {len(rows)} (family, severity) points:")
    print(f"  AUC vs cosine(z_clean, z_deg) : pearson r = {np.corrcoef(cos, auc)[0,1]:+.3f}")
    print(f"  AUC vs relative drift         : pearson r = {np.corrcoef(rel, auc)[0,1]:+.3f}")
    print(f"  spearman (rank) AUC vs cosine : r = "
          f"{np.corrcoef(np.argsort(np.argsort(cos)), np.argsort(np.argsort(auc)))[0,1]:+.3f}")


if __name__ == "__main__":
    main()
