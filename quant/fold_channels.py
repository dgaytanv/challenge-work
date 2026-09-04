"""WP-G: fold constant input channels into the first Dense's bias, with an assertion.

Three of the seven pdgId one-hot channels (130 h0, 1 h_HF, 2 egamma_HF) are identically
zero in every file measured, and on L1T `is_pf` is additionally constant 1.0. A channel
that is constant over the data folds into the next layer's bias EXACTLY:

    y = sum_c W[c,:] x[c] + b  =  sum_{c not const} W[c,:] x[c]  +  (b + sum_{c const} W[c,:] v_c)

so the column is deleted and its contribution becomes part of the bias. No retraining, no
approximation, and the folded model agrees with the original to float round-off.

D's caution (00:12), adopted: the fold is exact only WHILE those channels stay constant.
A future data drop containing a 130 or an h_HF would be silently mis-scored by a folded
model. So this asserts constancy over a named data file and records which file was used.

Usage:
  python quant/fold_channels.py --ckpt <float ref .pth> --data <eval .pt> --tag L1T
"""
import argparse
import json
import os
import sys

os.environ.setdefault('KERAS_BACKEND', 'torch')

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src'))
sys.path.insert(0, os.path.join(REPO, 'quant'))

from keras_port import preproc_meanpt_np, survivors_np

NAMES = ['pt', 'eta', 'phi', 'dxy', 'dxysig', 'charge', 'is_pf',
         'pdg211', 'pdg11', 'pdg13', 'pdg22', 'pdg130', 'pdg1', 'pdg2']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--events', type=int, default=20000)
    args = ap.parse_args()

    from embedding.utils.data_utils import load_data
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    bn = {k: ck['preproc'][f'batch_norm.{k}'].numpy().astype(np.float64)
          for k in ('weight', 'bias', 'running_mean', 'running_var')}
    feats, _ = load_data(args.data, map_location='cpu', max_events=args.events)
    xf = preproc_meanpt_np(feats.numpy(), bn)
    keep = survivors_np(xf).astype(bool)
    live = xf[keep]                                     # [M, 14], live candidates only

    lo, hi = live.min(0), live.max(0)
    const = (hi - lo) == 0.0
    values = np.where(const, lo, np.nan)

    # ASSERTION (D, 00:12): every channel we intend to fold really is constant here.
    folded = [NAMES[c] for c in range(14) if const[c]]
    print(f'[fold] {args.tag}: {live.shape[0]:,} live candidates from {os.path.basename(args.data)}')
    for c in range(14):
        flag = f'CONSTANT = {values[c]:+.6g}  -> fold' if const[c] else ''
        print(f'   {NAMES[c]:8s} [{lo[c]:+9.4f}, {hi[c]:+9.4f}]  {flag}')
    assert len(folded) > 0, 'no constant channels found; nothing to fold'

    # Exactness check on the real first Dense of the certified encoder.
    W = ck['encoder']['phi.0.weight'].numpy().astype(np.float64).T      # [14, 128]
    b = ck['encoder']['phi.0.bias'].numpy().astype(np.float64)          # [128]
    kept = ~const
    W_f = W[kept]                                                       # [n_kept, 128]
    b_f = b + values[const] @ W[const]                                  # bias absorbs them

    ref = live @ W + b
    got = live[:, kept] @ W_f + b_f
    err = float(np.abs(ref - got).max())
    print(f'[fold] first Dense {14} -> {int(kept.sum())} inputs; folded {folded}')
    print(f'[fold] max|d| over {live.shape[0]:,} live candidates: {err:.3e} (float64 exact algebra)')
    assert err < 1e-9, f'fold is not exact: {err}'

    mult_before, mult_after = 14 * 128, int(kept.sum()) * 128
    out = dict(tag=args.tag, data=os.path.abspath(args.data), ckpt=os.path.basename(args.ckpt),
               live_candidates=int(live.shape[0]), channels=NAMES,
               constant=[bool(v) for v in const],
               constant_values={NAMES[c]: float(values[c]) for c in range(14) if const[c]},
               folded_channels=folded, n_inputs_after=int(kept.sum()),
               multiplies_before=mult_before, multiplies_after=mult_after,
               multiply_reduction=1 - mult_after / mult_before, max_abs_error=err)
    dst = os.path.expanduser(f'~/hackathon-shared/quant/g_fold_channels_{args.tag}.json')
    with open(dst, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'[fold] first-layer multiplies {mult_before} -> {mult_after} '
          f'({100*(1-mult_after/mult_before):.1f}% fewer).  wrote {dst}')


if __name__ == '__main__':
    main()
