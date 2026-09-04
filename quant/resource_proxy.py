"""WP-G G4: a resource proxy, since no HLS tool exists on this box to synthesise with.

No Vitis/Vivado HLS is installed (G0), so LUT/FF/DSP/BRAM, latency and II cannot be
measured. What CAN be reported honestly is the quantity HGQ2 optimises and hls4ml's own
cost model is built around: **EBOPs**, the bit-weighted multiply-accumulate count
(sum over each layer of bits(input) x bits(weight)), plus the raw multiply count and the
learned bit widths. EBOPs is the accepted proxy for LUT usage in the HGQ literature; it
is not a substitute for synthesis and is not presented as one.

Usage: python quant/resource_proxy.py --params <tag>.params.npz [--n_tokens 400]
"""
import argparse
import json
import os
import sys

os.environ.setdefault('KERAS_BACKEND', 'torch')

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src'))
sys.path.insert(0, os.path.join(REPO, 'quant'))

# multiplies per event, as a function of the token budget N
MULTS = {
    'phi_0': lambda N: N * 14 * 128,
    'phi_1': lambda N: N * 128 * 128,
    'score': lambda N: N * 128 * 32,
    'v': lambda N: N * 128 * 128,
    'combine': lambda N: N * 8 * 4 * 16,      # attn[n,h,s] * v[n,h,d]
    'out_proj': lambda N: 4 * 128 * 128,      # per seed, N-independent
    'bottleneck': lambda N: 512 * 6,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--params', required=True)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--n_tokens', type=int, default=400)
    ap.add_argument('--act', default='gelu')
    args = ap.parse_args()
    tag = args.tag or os.path.basename(args.params).replace('.params.npz', '')

    from hgq_model import build_model
    from train_qat import load_params

    N = args.n_tokens
    m = build_model(norm='bn', act=args.act, quantized=True, n_tokens=N)
    _ = m([np.zeros((2, N, 14), 'float32'), np.zeros((2, N, 32), 'float32')], training=False)
    load_params(m, args.params)
    # EBOPs are only accumulated on a training-mode pass (hgq/layers/core/base.py)
    _ = m([np.zeros((2, N, 14), 'float32'), np.zeros((2, N, 32), 'float32')], training=True)

    rows, tot_e, tot_m = [], 0.0, 0
    for L in m.layers:
        if not hasattr(L, 'ebops'):
            continue
        e = L.ebops
        e = float(e.detach().cpu() if hasattr(e, 'detach') else np.asarray(e))
        mults = MULTS.get(L.name, lambda n: 0)(N)
        wb = ab = None
        if hasattr(L, '_kq'):
            k, i, f = (np.asarray(t.detach().cpu()) for t in L._kq.quantizer.kif)
            wb = float((k + i + f).mean()) if f.size else float((k + i).mean())
        if hasattr(L, '_iq') and getattr(L, 'enable_iq', False):
            try:
                k, i, f = (np.asarray(t.detach().cpu()) for t in L._iq.quantizer.kif)
                ab = float((k + i + f).mean())
            except Exception:
                pass
        tot_e += e; tot_m += mults
        rows.append(dict(layer=L.name, ebops=e, multiplies=mults,
                         mean_weight_bits=wb, mean_act_bits=ab))

    print(f'{tag}  (N={N} candidates)')
    print(f'{"layer":14s} {"EBOPs":>12s} {"multiplies":>12s} {"w bits":>7s} {"a bits":>7s}')
    for r in rows:
        print(f'{r["layer"]:14s} {r["ebops"]:12.4e} {r["multiplies"]:12,d} '
              f'{r["mean_weight_bits"] if r["mean_weight_bits"] is not None else float("nan"):7.2f} '
              f'{r["mean_act_bits"] if r["mean_act_bits"] is not None else float("nan"):7.2f}')
    print(f'{"TOTAL":14s} {tot_e:12.4e} {tot_m:12,d}')

    out = dict(tag=tag, n_tokens=N, act=args.act, total_ebops=tot_e,
               total_multiplies=tot_m, layers=rows,
               note='EBOPs is a bit-weighted MAC count and a proxy for LUT cost; '
                    'no HLS tool exists on this box so it is NOT a synthesis result.')
    dst = os.path.expanduser(f'~/hackathon-shared/quant/g4_resource_{tag}_N{N}.json')
    json.dump(out, open(dst, 'w'), indent=2)
    print(f'wrote {dst}')


if __name__ == '__main__':
    main()
