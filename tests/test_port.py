"""WP-G G1 acceptance: the Keras 3 port reproduces the certified torch encoder.

Acceptance (from prompts/wp-g.md):
  * 2000 real eval events, clean AND degraded with WP-B's generator in eval mode at s=0.5
  * max |latent_keras - latent_torch| < 1e-4
  * the probe AUC on the ported latents equals the torch model's within the probe floor
    (PMA per-refit floor = 0.0038 on mean_area; RULING.md). Here the probe is fit ONCE on
    the torch latents and applied frozen to both latent sets, so the AUC difference
    isolates the port and carries no refit noise at all.

Run:  KERAS_BACKEND=torch PYTHONPATH=$HOME/rt-g/src python ~/rt-g/tests/test_port.py
"""
import json
import os
import sys
import time

os.environ.setdefault('KERAS_BACKEND', 'torch')

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src'))
sys.path.insert(0, os.path.join(REPO, 'quant'))

from embedding.degradation import Degradation
from embedding.models import TransformerEncoder
from embedding.preprocs import PFPreProcessorMeanPt
from embedding.utils.data_utils import load_data
from keras_port import build_keras_encoder, load_torch_weights, preproc_meanpt_np, survivors_np

CKPT = os.path.join(REPO, 'checkpoints', 'rt_d_pma0_aug_meanpt_encoder_20260903_221125.pth')
DATA = os.path.expanduser('~/hack-data/C9_robust_tagging/eval/robust_tagging_eval_small.pt')
N_EVENTS = 2000
TOL = 1e-4


def torch_latents(preproc, encoder, x_raw, device):
    outs, feats = [], []
    with torch.no_grad():
        for i in range(0, len(x_raw), 250):
            xb = x_raw[i:i + 250].to(device)
            f = preproc(xb)
            outs.append(encoder(f).cpu().numpy())
            feats.append(f.cpu().numpy())
    return np.concatenate(outs), np.concatenate(feats)


def _np(t):
    """Keras on the torch backend returns cuda tensors; bring them home."""
    return t.detach().cpu().numpy() if hasattr(t, 'detach') else np.asarray(t)


def keras_latents(kmodel, x_feat, keep):
    outs = []
    for i in range(0, len(x_feat), 250):
        outs.append(_np(kmodel([x_feat[i:i + 250], keep[i:i + 250]], training=False)))
    return np.concatenate(outs)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(CKPT, map_location=device, weights_only=False)

    preproc = PFPreProcessorMeanPt(ck['norm_constants']).to(device)
    preproc.load_state_dict(ck['preproc']); preproc.eval()
    encoder = TransformerEncoder(num_features=14, embed_size=128, latent_dim=6,
                                 num_heads=8, num_layers=0, linear_dim=None,
                                 num_tokens=None, pairwise=False).to(device)
    encoder.load_state_dict(ck['encoder']); encoder.eval()
    n_par = sum(p.numel() for p in encoder.parameters())
    print(f'[port] torch encoder built, {n_par:,} parameters, device={device}')

    kmodel = load_torch_weights(build_keras_encoder(), ck['encoder'])
    print(f'[port] keras model built, {kmodel.count_params():,} parameters')
    assert n_par == kmodel.count_params(), (n_par, kmodel.count_params())

    bn = {k: ck['preproc'][f'batch_norm.{k}'].cpu().numpy().astype(np.float64)
          for k in ('weight', 'bias', 'running_mean', 'running_var')}

    feats_all, labels_all = load_data(DATA, map_location='cpu', max_events=N_EVENTS)
    print(f'[port] {feats_all.shape[0]} eval events, {feats_all.shape[1]} candidates')

    views = {'clean': feats_all, 'degraded_s0.5': Degradation(severity=0.5)(feats_all.clone())}

    results, ok = {}, True
    for name, x_raw in views.items():
        t0 = time.time()
        lat_t, feat_t = torch_latents(preproc, encoder, x_raw, device)

        feat_np = preproc_meanpt_np(x_raw.numpy(), bn)
        keep = survivors_np(feat_np)
        lat_k = keras_latents(kmodel, feat_np, keep)

        d_pre = float(np.abs(feat_np - feat_t).max())
        d_lat = float(np.abs(lat_k - lat_t).max())
        rel = float(np.abs(lat_k - lat_t).max() / (np.abs(lat_t).max() + 1e-12))
        frac_dead = float(1.0 - keep.mean())
        results[name] = dict(max_abs_preproc=d_pre, max_abs_latent=d_lat,
                             rel_latent=rel, dead_fraction=frac_dead,
                             latent_scale=float(np.abs(lat_t).max()))
        passed = d_lat < TOL
        ok &= passed
        print(f'[{name:14s}] preproc max|d|={d_pre:.3e}  latent max|d|={d_lat:.3e} '
              f'(rel {rel:.2e}, scale {np.abs(lat_t).max():.2f})  dead={frac_dead:.3f}  '
              f'{"PASS" if passed else "FAIL"}  [{time.time()-t0:.1f}s]')

    # ---- probe AUC: fit once on the torch CLEAN latents, apply frozen to both sets ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    y = labels_all.numpy().astype(int)

    lat_t_clean, _ = torch_latents(preproc, encoder, views['clean'], device)
    probe = LogisticRegression(max_iter=2000).fit(lat_t_clean, y)
    for name, x_raw in views.items():
        lt, _ = torch_latents(preproc, encoder, x_raw, device)
        fn = preproc_meanpt_np(x_raw.numpy(), bn)
        lk = keras_latents(kmodel, fn, survivors_np(fn))
        a_t = roc_auc_score(y, probe.decision_function(lt))
        a_k = roc_auc_score(y, probe.decision_function(lk))
        results[name].update(auc_torch=float(a_t), auc_keras=float(a_k), auc_delta=float(a_k - a_t))
        print(f'[{name:14s}] frozen-probe AUC torch={a_t:.6f} keras={a_k:.6f} '
              f'delta={a_k-a_t:+.2e} (PMA probe floor 3.8e-3)')

    out = dict(ckpt=os.path.basename(CKPT), n_events=int(feats_all.shape[0]),
               tol=TOL, params=int(n_par), passed=bool(ok), views=results)
    dst = os.path.expanduser('~/hackathon-shared/quant/g1_port_test.json')
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n[port] wrote {dst}')
    print(f'[port] RESULT: {"PASS" if ok else "FAIL"}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
