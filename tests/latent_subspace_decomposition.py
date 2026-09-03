"""Split the degradation-induced latent drift into the part the probe reads and the rest.

Motivation: d-deepsets-clean reaches mean_area 0.8017 while its latent moves ~0.86 of the
population spread under degradation. If most of that motion were harmful the AUC would
collapse, so the hypothesis is that the drift lives largely in directions the grader's
probe never reads. This measures that directly.

Method. Train the grader's own probe (EvalMLP, same recipe as eval.py/bench_eval.py) on
clean latents. For each event linearise the probe at z_c: g = d(signal logit - bkg logit)/dz.
That single direction is, to first order, the only thing the probe's decision responds to.
Decompose d = z_d - z_c into the component along g_hat and the component orthogonal to it.

Magnitudes alone can mislead, so also do the operational version: feed the probe
z_c + d_parallel and z_c + d_perp separately and see which one actually costs AUC.

Usage:
  PYTHONPATH=$HOME/rt-d/src python tests/latent_subspace_decomposition.py <ckpt> [encoder_class] [train_cfg] [severity]
"""
import sys

import torch
from sklearn.model_selection import train_test_split

from embedding.utils.cfg_handler import train_config, data_config
from embedding.utils.data_utils import load_data
from embedding.degradation import Degradation
from embedding.models import EvalMLP
import embedding.models as models
import embedding.preprocs as preprocs
from sklearn.metrics import roc_auc_score

ckpt_path = sys.argv[1]
enc_name = sys.argv[2] if len(sys.argv) > 2 else "DeepSetsEncoder"
cfg_path = sys.argv[3] if len(sys.argv) > 3 else "configs/train_config_d_deepsets_clean.yaml"
severity = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
N_EVENTS = 20000

dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = train_config(cfg_path)
cfg_data = data_config("configs/data_config_eval.yaml")
num_classes = len(cfg_data.get_label_name_map())
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

feats, labels = load_data("/home/jovyan/hack-data/C9_robust_tagging/eval/robust_tagging_eval_small.pt",
                          map_location="cpu", max_events=N_EVENTS)


@torch.no_grad()
def embed(deg=None):
    out = []
    for i in range(0, feats.shape[0], 1024):
        xb = feats[i:i + 1024, :, :7].to(dev)
        if deg is not None:
            xb = deg(xb)
        m = torch.zeros(xb.shape[0], xb.shape[1] + 1, dtype=torch.bool, device=dev)
        out.append(enc(preproc(xb), None, m))
    return torch.cat(out)


z_c = embed()
z_d = embed(Degradation(severity=severity).to(dev).eval())
y = labels.to(dev).long()

# the grader's probe, trained on clean latents only
Xtr, Xte, ytr, yte = train_test_split(z_c.cpu(), y.cpu(), stratify=y.cpu(), test_size=0.2, random_state=42)
probe = EvalMLP(input_dim=z_c.shape[1], num_classes=num_classes).to(dev)
opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
lossf = torch.nn.CrossEntropyLoss()
loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xtr, ytr), batch_size=512, shuffle=True)
probe.train()
for _ in range(20):
    for xb, yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)
        opt.zero_grad(); lossf(probe(xb), yb).backward(); opt.step()
probe.eval()


def auc(z):
    with torch.no_grad():
        p = torch.softmax(probe(z), dim=1).cpu().numpy()
    yy = y.cpu().numpy()
    return roc_auc_score(yy, p[:, 1]) if num_classes == 2 else \
        roc_auc_score(yy, p, multi_class="ovr", average="macro")


# linearise the probe: the one direction its decision responds to, per event
zc = z_c.clone().requires_grad_(True)
logits = probe(zc)
margin = logits[:, 1] - logits[:, 0] if num_classes == 2 else \
    logits.max(dim=1).values - logits.mean(dim=1)
g = torch.autograd.grad(margin.sum(), zc)[0]
ghat = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-12)

d = (z_d - z_c).detach()
par_mag = (d * ghat).sum(-1, keepdim=True)
d_par = par_mag * ghat
d_perp = d - d_par
spread = (z_c - z_c.mean(0)).norm(dim=-1).mean()

print(f"encoder={enc_name}  ckpt={ckpt_path.split('/')[-1]}  severity={severity}  events={z_c.shape[0]}")
print(f"population spread ||z_c - mu||            {spread:.4f}")
print(f"total drift      ||d||                    {d.norm(dim=-1).mean():.4f}  ({d.norm(dim=-1).mean()/spread:.3f} x spread)")
print(f"  probe-relevant ||d_parallel||           {d_par.norm(dim=-1).mean():.4f}  ({d_par.norm(dim=-1).mean()/spread:.3f} x spread)")
print(f"  null-space     ||d_perp||               {d_perp.norm(dim=-1).mean():.4f}  ({d_perp.norm(dim=-1).mean()/spread:.3f} x spread)")
frac = (d_par.norm(dim=-1) / d.norm(dim=-1).clamp(min=1e-12)).mean()
sq = (d_par.pow(2).sum(-1) / d.pow(2).sum(-1).clamp(min=1e-12)).mean()
print(f"  fraction of drift the probe can see     {frac:.3f}  (by magnitude)")
print(f"                                          {sq:.3f}  (by SQUARED magnitude - what an MSE consistency term weights)")
print("operational check - feed the probe one component at a time:")
print(f"  AUC on clean z_c                        {auc(z_c):.4f}")
print(f"  AUC on z_c + d_perp   (harmless part)   {auc(z_c + d_perp):.4f}")
print(f"  AUC on z_c + d_par    (harmful part)    {auc(z_c + d_par):.4f}")
print(f"  AUC on full degraded z_d                {auc(z_d):.4f}")
