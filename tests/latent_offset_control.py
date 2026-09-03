"""Shuffled-pair control on the latent, for whichever checkpoint you point it at.

Cosine on a RAW latent is only meaningful if the latent cloud is not dominated by a
common offset. If ||mean|| >> per-event spread, cos(a, b) is near 1 for ANY pair, so a
high matched cosine says nothing and (1 - cos) carries almost no gradient.

Usage:
  PYTHONPATH=$HOME/rt-d/src python tests/latent_offset_control.py <ckpt> [encoder_class] [train_cfg]
"""
import sys

import torch

from embedding.utils.cfg_handler import train_config, data_config
from embedding.utils.data_utils import load_data
from embedding.degradation import Degradation
import embedding.models as models
import embedding.preprocs as preprocs

ckpt_path = sys.argv[1]
enc_name = sys.argv[2] if len(sys.argv) > 2 else "DeepSetsEncoder"
cfg_path = sys.argv[3] if len(sys.argv) > 3 else "configs/train_config_d_deepsets_clean.yaml"
N_EVENTS = 4096

dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = train_config(cfg_path)
ck = torch.load(ckpt_path, map_location=dev)

preproc = getattr(preprocs, cfg.get_trdata_cfg("preproc_type", "PFPreProcessor"))(ck["norm_constants"]).to(dev)
preproc.load_state_dict(ck["preproc"]); preproc.eval()
enc = getattr(models, enc_name)(
    num_features=preproc.num_features, embed_size=cfg.hp("embed_size", 128),
    latent_dim=cfg.hp("latent_dim", 6), num_heads=cfg.hp("num_heads", 8),
    num_layers=cfg.hp("num_layers", 4), linear_dim=cfg.hp("linear_dim", None),
    num_tokens=None, pairwise=cfg.get_trdata_cfg("pairwise", False),
    **(cfg.hp("encoder_kwargs", {}) or {}),
).to(dev)
enc.load_state_dict(ck["encoder"]); enc.eval()

feats, _ = load_data("/home/jovyan/hack-data/C9_robust_tagging/eval/robust_tagging_eval_small.pt",
                     map_location="cpu", max_events=N_EVENTS)
x = feats[:N_EVENTS, :, :7].to(dev)

deg = Degradation(severity=0.5).to(dev).eval()
with torch.no_grad():
    mask = torch.zeros(x.shape[0], x.shape[1] + 1, dtype=torch.bool, device=dev)
    z_c = enc(preproc(x), None, mask)
    z_d = enc(preproc(deg(x)), None, mask)

mu = z_c.mean(0)
sd = z_c.std(0)
spread = (z_c - mu).norm(dim=-1).mean()
perm = torch.randperm(z_c.shape[0], device=dev)
cos = torch.nn.functional.cosine_similarity

print(f"encoder={enc_name}  ckpt={ckpt_path.split('/')[-1]}  events={z_c.shape[0]}  latent_dim={z_c.shape[1]}")
print(f"per-dim mean {[round(v, 2) for v in mu.tolist()]}")
print(f"per-dim std  {[round(v, 2) for v in sd.tolist()]}")
print(f"||mean|| = {mu.norm():.3f}   mean per-event spread ||z - mu|| = {spread:.3f}"
      f"   ratio = {mu.norm() / spread:.2f}")
print("RAW latent (what a plain cosine loss/log sees):")
print(f"  cos matched  (z_c[i], z_d[i])        {cos(z_c, z_d).mean():.6f}")
print(f"  cos SHUFFLED (z_c[i], z_d[perm(i)])  {cos(z_c, z_d[perm]).mean():.6f}")
print(f"  cos clean-vs-clean, shuffled         {cos(z_c, z_c[perm]).mean():.6f}")
print("CENTRED on the clean-population mean:")
print(f"  cos matched                          {cos(z_c - mu, z_d - mu).mean():.6f}")
print(f"  cos SHUFFLED                         {cos(z_c - mu, (z_d - mu)[perm]).mean():.6f}")
drift = (z_d - z_c).norm(dim=-1).mean()
print(f"drift ||z_d - z_c||                    {drift:.4f}")
print(f"  / ||z_c||        (offset-inflated)   {(drift / z_c.norm(dim=-1).mean()):.4f}")
print(f"  / population spread (probe-relevant) {(drift / spread):.4f}")
