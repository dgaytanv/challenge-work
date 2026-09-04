"""WP-G G2 pre-check: the two algebraic restructurings (R1, R2) are exact.

R1 fuses the constant seed query into the key projection (q_proj + k_proj -> one 128->32
Dense). R2 drops the redundant `h = phi(x) * keep` multiply. Neither may change the
output, so this compares the restructured float model against the G1 port on the same
2000 eval events, clean and degraded, at the same 1e-4 tolerance.

Run:  KERAS_BACKEND=torch PYTHONPATH=$HOME/rt-g/src python ~/rt-g/tests/test_restructure.py
"""
import json
import os
import sys

os.environ.setdefault('KERAS_BACKEND', 'torch')

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src'))
sys.path.insert(0, os.path.join(REPO, 'quant'))

from embedding.degradation import Degradation
from embedding.utils.data_utils import load_data
from hgq_model import build_model, load_weights, mask_add_from_keep
from keras_port import build_keras_encoder, load_torch_weights, preproc_meanpt_np, survivors_np

CKPT = os.environ.get('G_REF_CKPT',
                      os.path.join(REPO, 'checkpoints', 'rt_d_pma0_aug_meanpt_encoder_20260903_221125.pth'))
DATA = os.environ.get('G_EVAL_DATA',
                      os.path.expanduser('~/hack-data/C9_robust_tagging/eval/robust_tagging_eval_small.pt'))
TAG = os.environ.get('G_TAG', 'PF')
TOL = 1e-4


def run(model, inputs, batch=250):
    outs = []
    n = len(inputs[0])
    for i in range(0, n, batch):
        o = model([a[i:i + batch] for a in inputs], training=False)
        outs.append(o.detach().cpu().numpy() if hasattr(o, 'detach') else np.asarray(o))
    return np.concatenate(outs)


def main():
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    enc = ck['encoder']
    bn = {k: ck['preproc'][f'batch_norm.{k}'].cpu().numpy().astype(np.float64)
          for k in ('weight', 'bias', 'running_mean', 'running_var')}

    ref = load_torch_weights(build_keras_encoder(), enc)
    new = load_weights(build_model(norm='ln', act='gelu', quantized=False), enc, norm='ln')
    print(f'[restructure] port model      {ref.count_params():,} params')
    print(f'[restructure] restructured ln {new.count_params():,} params  '
          f'(R1 removes {ref.count_params()-new.count_params():,})')

    feats, _ = load_data(DATA, map_location='cpu', max_events=2000)
    views = {'clean': feats, 'degraded_s0.5': Degradation(severity=0.5)(feats.clone())}

    ok, res = True, {}
    for name, x_raw in views.items():
        xf = preproc_meanpt_np(x_raw.numpy(), bn)
        keep = survivors_np(xf)
        ma = mask_add_from_keep(keep)
        a = run(ref, [xf, keep])
        b = run(new, [xf, ma])
        d = float(np.abs(a - b).max())
        res[name] = dict(max_abs=d, scale=float(np.abs(a).max()))
        ok &= d < TOL
        print(f'[{name:14s}] restructured vs port  max|d| = {d:.3e}  '
              f'{"PASS" if d < TOL else "FAIL"}')

    out = dict(tag=TAG, ckpt=os.path.basename(CKPT), params_port=int(ref.count_params()), params_restructured=int(new.count_params()),
               tol=TOL, passed=bool(ok), views=res)
    with open(os.path.expanduser('~/hackathon-shared/quant/g2_restructure_test_%s.json' % TAG), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'[restructure] RESULT: {"PASS" if ok else "FAIL"}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
