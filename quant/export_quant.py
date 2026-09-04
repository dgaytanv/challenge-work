"""WP-G G3: export a trained HGQ2 Keras model into the torch emulator's checkpoint.

Weight quantizers are applied ONCE here (they are data-independent, so this is exact);
the data-lane quantizers are exported as (k, i, f) triples for the emulator to apply at
run time. The output .pth has the keys bench_eval.py / accept.sh expect
(`preproc`, `encoder`, `projector`, `classifier`, `norm_constants`), with `preproc` and
`norm_constants` copied verbatim from the float reference checkpoint so the preprocessor
is bit-identical to the one the float row used.

Usage:
  python quant/export_quant.py --params <tag>.params.npz --ref <float ref .pth> \
      --out checkpoints/<tag>.pth --act gelu
"""
import argparse
import os
import sys

os.environ.setdefault('KERAS_BACKEND', 'torch')

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src'))
sys.path.insert(0, os.path.join(REPO, 'quant'))

from hgq_model import build_model

DENSES = ('phi_0', 'phi_1', 'score', 'v', 'out_proj', 'bottleneck')
NORMS = ('nrm0', 'nrm1', 'norm_pooled')
LUTS = ('phi_act0', 'phi_act1')


def npy(t):
    return t.detach().cpu().numpy() if hasattr(t, 'detach') else np.asarray(t)


def kif(q):
    return [npy(v).astype('float32') for v in q.quantizer.kif]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--params', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--act', default='gelu', choices=('gelu', 'relu'))
    ap.add_argument('--n_tokens', type=int, default=400)
    ap.add_argument('--float', dest='is_float', type=int, default=0,
                    help='1 = export a stage-A float BatchNorm model (no quantizers)')
    ap.add_argument('--homogeneous', type=int, default=0,
                    help='the model was trained with per-tensor activation quantizers '
                         '(G6a). The quantizer VARIABLE SHAPES differ, so this must match '
                         'the training run or load_params silently skips every bitwidth.')
    args = ap.parse_args()

    from train_qat import load_params
    m = build_model(norm='bn', act=args.act, quantized=not args.is_float, n_tokens=args.n_tokens,
                    homogeneous=bool(args.homogeneous))
    _ = m([np.zeros((1, args.n_tokens, 14), 'float32'),
           np.zeros((1, args.n_tokens, 32), 'float32')], training=False)
    load_params(m, args.params)

    if args.is_float:
        # No quantizers to extract: kernels and the BatchNorm affine are the whole model.
        # The (k, i, f) buffers still have to exist for load_state_dict(strict=True), so
        # they are written as zeros and FloatBNPMAEncoder never reads them.
        sd = {'act_relu': torch.tensor(1.0 if args.act == 'relu' else 0.0)}
        for name in DENSES:
            L = m.get_layer(name)
            sd[f'w_{name}'] = torch.tensor(npy(L.kernel))
            sd[f'b_{name}'] = torch.tensor(npy(L.bias))
        for name in NORMS:
            L = m.get_layer(name)
            g, b = npy(L.gamma).astype('float64'), npy(L.beta).astype('float64')
            mu, var = npy(L.moving_mean).astype('float64'), npy(L.moving_variance).astype('float64')
            scale = g / np.sqrt(var + L.epsilon)
            sd[f'ns_{name}'] = torch.tensor((scale).astype('float32'))
            sd[f'no_{name}'] = torch.tensor((b - mu * scale).astype('float32'))
        from embedding.models import FloatBNPMAEncoder
        enc = FloatBNPMAEncoder(num_features=14, embed_size=128, latent_dim=6,
                                num_heads=8, num_layers=0, act=args.act)
        ref_sd = enc.state_dict()
        for k in ref_sd:
            if k not in sd:
                sd[k] = torch.zeros_like(ref_sd[k])
        enc.load_state_dict(sd, strict=True)
        ref = torch.load(args.ref, map_location='cpu', weights_only=False)
        out = dict(preproc=ref['preproc'], encoder=sd, projector=ref['projector'],
                   classifier=ref['classifier'], norm_constants=ref['norm_constants'])
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        torch.save(out, args.out)
        print(f'[export] wrote FLOAT {args.out}; FloatBNPMAEncoder strict load OK')
        return

    sd = {'act_relu': torch.tensor(1.0 if args.act == 'relu' else 0.0)}
    for name in DENSES:
        L = m.get_layer(name)
        sd[f'w_{name}'] = torch.tensor(npy(L.qkernel))      # weight quantizer applied
        sd[f'b_{name}'] = torch.tensor(npy(L.qbias))
        k, i, f = kif(L._iq)
        for t, v in zip('kif', (k, i, f)):
            sd[f'q_{name}_iq_{t}'] = torch.tensor(v)
    for name in NORMS:
        L = m.get_layer(name)
        scale, offset = L.qscaler_and_qoffset                # weight quantizer applied
        sd[f'ns_{name}'] = torch.tensor(npy(scale)).reshape(-1)
        sd[f'no_{name}'] = torch.tensor(npy(offset)).reshape(-1)
        for t, v in zip('kif', kif(L._iq)):
            sd[f'q_{name}_iq_{t}'] = torch.tensor(v)
    for name in LUTS:
        L = m.get_layer(name)
        for t, v in zip('kif', kif(L._iq)):
            sd[f'q_{name}_iq_{t}'] = torch.tensor(v)
        for t, v in zip('kif', kif(L._oq)):
            sd[f'q_{name}_oq_{t}'] = torch.tensor(v)
    add = m.get_layer('mask_add_op')
    for j in (0, 1):
        for t, v in zip('kif', kif(add._iq.quantizers[j])):
            sd[f'q_add_iq{j}_{t}'] = torch.tensor(v)
    sm = m.get_layer('softmax')
    for tag, sub in (('exp', sm.exp_table), ('inv', sm.inv_table)):
        for t, v in zip('kif', kif(sub._iq)):
            sd[f'q_{tag}_iq_{t}'] = torch.tensor(v)
        for t, v in zip('kif', kif(sub._oq)):
            sd[f'q_{tag}_oq_{t}'] = torch.tensor(v)
    cb = m.get_layer('combine')
    for j in (0, 1):
        for t, v in zip('kif', kif(cb._iq.quantizers[j])):
            sd[f'q_comb_iq{j}_{t}'] = torch.tensor(v)

    # The emulator's (k, i, f) buffers are declared at the PER-CHANNEL shape. A
    # homogeneous (G6a) model gives one scalar per tensor instead, so strict loading would
    # raise on shape rather than on meaning. Broadcasting a scalar (k, i, f) to every
    # channel is EXACT - the quantizer is elementwise and the triple is the same for all
    # channels by construction - so the emulator stays bit-comparable with Keras. Only
    # widening is allowed; a genuine shape disagreement still raises.
    from embedding.models import QuantizedPMAEncoder
    enc = QuantizedPMAEncoder(num_features=14, embed_size=128, latent_dim=6,
                              num_heads=8, num_layers=0, act=args.act)
    ref_sd = enc.state_dict()
    n_bcast = 0
    for k in list(sd):
        if k not in ref_sd:
            continue
        want, got = tuple(ref_sd[k].shape), tuple(sd[k].shape)
        if want != got:
            sd[k] = sd[k].reshape((1,) * (len(want) - sd[k].dim()) + got).expand(want).contiguous()
            n_bcast += 1
    if n_bcast:
        print(f'[export] broadcast {n_bcast} scalar quantizer triples to per-channel shape '
              f'(homogeneous={bool(args.homogeneous)}); exact, the value is shared anyway')

    ref = torch.load(args.ref, map_location='cpu', weights_only=False)
    out = dict(preproc=ref['preproc'], encoder=sd, projector=ref['projector'],
               classifier=ref['classifier'], norm_constants=ref['norm_constants'])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(out, args.out)
    print(f'[export] wrote {args.out} with {len(sd)} encoder tensors')

    missing, unexpected = enc.load_state_dict(sd, strict=True)  # raises on mismatch
    print('[export] QuantizedPMAEncoder.load_state_dict(strict=True): OK')


if __name__ == '__main__':
    main()
