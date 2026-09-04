"""WP-G G3 acceptance: the torch emulator reproduces the HGQ2 Keras model exactly.

The prompt asks for "emulation-versus-Keras max latent difference is zero or explain the
residual". Anything above float32 round-off means a quantizer, a rounding mode or an
overflow mode was emulated wrongly, so this test is the gate on every G3 bench number.

Run: KERAS_BACKEND=torch PYTHONPATH=$HOME/rt-g/src python ~/rt-g/tests/test_emulation.py
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
from embedding.models import QuantizedPMAEncoder
from embedding.utils.data_utils import load_data
from hgq_model import build_model, collect_ln_stats, load_weights, mask_add_from_keep
from keras_port import preproc_meanpt_np, survivors_np
from train_qat import save_params

CKPT = os.environ.get('G_REF_CKPT',
                      os.path.join(REPO, 'checkpoints', 'rt_d_pma0_aug_meanpt_encoder_20260903_221125.pth'))
DATA = os.environ.get('G_EVAL_DATA',
                      os.path.expanduser('~/hack-data/C9_robust_tagging/eval/robust_tagging_eval_small.pt'))
PARAMS = os.environ.get('G_PARAMS')          # if set, test a TRAINED model instead
ACT = os.environ.get('G_ACT', 'gelu')
N_EVENTS = int(os.environ.get('G_EVENTS', 1000))
SCRATCH = os.environ.get('G_SCRATCH', '/tmp')


def main():
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    bn = {k: ck['preproc'][f'batch_norm.{k}'].numpy().astype(np.float64)
          for k in ('weight', 'bias', 'running_mean', 'running_var')}
    feats, _ = load_data(DATA, map_location='cpu', max_events=N_EVENTS)
    half = N_EVENTS // 2
    mix = torch.cat([feats[:half], Degradation(severity=0.5)(feats[half:].clone())])
    xf = preproc_meanpt_np(mix.numpy(), bn)
    keep = survivors_np(xf)
    ma = mask_add_from_keep(keep)
    N = xf.shape[1]

    q = build_model(norm='bn', act=ACT, quantized=True, n_tokens=N)
    if PARAMS:
        from train_qat import load_params
        _ = q([xf[:2], ma[:2]], training=False)
        load_params(q, PARAMS)
        pfile = PARAMS
    else:
        teacher = load_weights(build_model(norm='ln', quantized=False, n_tokens=N), ck['encoder'], norm='ln')
        stats = collect_ln_stats(teacher, xf, ma, keep)
        load_weights(q, ck['encoder'], stats, norm='bn')
        from hgq.utils import trace_minmax
        trace_minmax(q, [xf, ma], batch_size=200, verbose=0)
        pfile = save_params(q, os.path.join(SCRATCH, 'emu_test.params.npz'))

    kout = np.concatenate([
        q([xf[i:i + 200], ma[i:i + 200]], training=False).detach().cpu().numpy()
        for i in range(0, len(xf), 200)])

    out_pth = os.path.join(SCRATCH, 'emu_test.pth')
    os.system(f'cd {REPO} && KERAS_BACKEND=torch PYTHONPATH={REPO}/src python quant/export_quant.py '
              f'--params {pfile} --ref {CKPT} --out {out_pth} --act {ACT} --n_tokens {N} >/dev/null 2>&1')
    sd = torch.load(out_pth, map_location='cpu', weights_only=False)['encoder']

    enc = QuantizedPMAEncoder(num_features=14, embed_size=128, latent_dim=6,
                              num_heads=8, num_layers=0, act=ACT)
    enc.load_state_dict(sd); enc.eval()
    with torch.no_grad():
        tout = np.concatenate([
            enc(torch.tensor(xf[i:i + 200])).numpy() for i in range(0, len(xf), 200)])

    d = np.abs(kout - tout)
    scale = float(np.abs(kout).max())
    res = dict(max_abs=float(d.max()), rms=float(np.sqrt((d ** 2).mean())),
               latent_scale=scale, rel=float(d.max() / (scale + 1e-12)),
               n_events=int(len(xf)), n_tokens=int(N), act=ACT, params=pfile,
               exact_fraction=float((d == 0).mean()))
    print(f'[emu] keras vs torch emulation over {len(xf)} events x {N} candidates:')
    print(f'      max|d| = {res["max_abs"]:.3e}   rms = {res["rms"]:.3e}   '
          f'latent scale = {scale:.3f}   relative = {res["rel"]:.2e}')
    print(f'      elements identical to the last bit: {res["exact_fraction"]*100:.2f}%')
    ok = res['rel'] < 1e-5
    res['passed'] = bool(ok)
    dst = os.path.expanduser('~/hackathon-shared/quant/g3_emulation_test.json')
    with open(dst, 'w') as f:
        json.dump(res, f, indent=2)
    print(f'[emu] wrote {dst}')
    print(f'[emu] RESULT: {"PASS" if ok else "FAIL"}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
