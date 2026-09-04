"""WP-G G7: write REAL-EVENT testbench data for hls4ml csim / C-RTL cosim.

hls4ml's generated testbench reads ONE line per event from `tb_data/tb_input_features.dat`
holding every input tensor concatenated in graph order -- here 400*14 preprocessed
features followed by the 400*32 additive mask -- and compares against one line of 6
latents per event in `tb_data/tb_output_predictions.dat`. Both files are written here from
REAL L1T eval events (not random ones), and the predictions are the HGQ2 Keras model's own
output, so `rtl_cosim_results.log` can be bit-compared against Keras rather than merely
eyeballed.

The preprocessor runs on the host in float (§1 of the writeup: the per-event mean-pt
division is a two-pass reduction over 400 candidates and is not part of the quantized
region), exactly as it does for the bench.
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

from hgq_model import HEADS, NUM_FEATURES, SEEDS, build_model, mask_add_from_keep
from keras_port import preproc_meanpt_np, survivors_np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--params', required=True)
    ap.add_argument('--ref', required=True, help='float reference .pth (for the preproc BatchNorm)')
    ap.add_argument('--data', default=os.path.expanduser(
        '~/hack-data/C9_robust_tagging/eval/robust_tagging_eval_small_l1t.pt'))
    ap.add_argument('--prj', required=True, help='hls4ml project dir')
    ap.add_argument('--n_tokens', type=int, default=400)
    ap.add_argument('--events', type=int, default=16)
    ap.add_argument('--act', default='relu')
    ap.add_argument('--max_bits', type=int, default=6)
    ap.add_argument('--homogeneous', type=int, default=0)
    args = ap.parse_args()

    ck = torch.load(args.ref, map_location='cpu', weights_only=False)
    bn = {k: ck['preproc'][f'batch_norm.{k}'].cpu().numpy().astype(np.float64)
          for k in ('weight', 'bias', 'running_mean', 'running_var')}

    obj = torch.load(args.data, map_location='cpu', weights_only=False)
    raw = obj[0] if isinstance(obj, (tuple, list)) else obj
    raw = raw[:args.events]
    assert raw.shape[1] == args.n_tokens, f'data has {raw.shape[1]} candidates, --n_tokens {args.n_tokens}'
    # the eval file carries the label in the last column; the preprocessor takes 7 raw features
    x = preproc_meanpt_np(raw[..., :7].numpy() if raw.shape[-1] > 7 else raw.numpy(), bn)
    keep = survivors_np(x)
    madd = mask_add_from_keep(keep)

    m = build_model(norm='bn', act=args.act, quantized=True, n_tokens=args.n_tokens,
                    max_bits=args.max_bits, homogeneous=bool(args.homogeneous))
    _ = m([np.zeros((2, args.n_tokens, NUM_FEATURES), 'float32'),
           np.zeros((2, args.n_tokens, HEADS * SEEDS), 'float32')], training=False)
    from train_qat import load_params
    load_params(m, args.params)
    m.trainable = False
    y = m([x.astype('float32'), madd], training=False)
    y = y.detach().cpu().numpy() if hasattr(y, 'detach') else np.asarray(y)

    tb = os.path.join(args.prj, 'tb_data')
    os.makedirs(tb, exist_ok=True)
    flat = np.concatenate([x.reshape(len(x), -1), madd.reshape(len(madd), -1)], axis=1)
    np.savetxt(os.path.join(tb, 'tb_input_features.dat'), flat, fmt='%.9g')
    np.savetxt(os.path.join(tb, 'tb_output_predictions.dat'), y, fmt='%.9g')
    meta = dict(events=int(len(x)), n_tokens=args.n_tokens, cols=int(flat.shape[1]),
                data=os.path.basename(args.data), params=os.path.basename(args.params),
                keep_mean=float(keep.mean()), latent_absmax=float(np.abs(y).max()),
                latent_std=[float(v) for v in y.std(axis=0)])
    with open(os.path.join(tb, 'tb_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    np.save(os.path.join(tb, 'keras_latents.npy'), y)
    print(f'[tb] wrote {len(x)} events x {flat.shape[1]} inputs to {tb}')
    print(f'[tb] keras latent |max|={np.abs(y).max():.4f} per-dim std={y.std(axis=0).round(4)}')
    print(f'[tb] live fraction {keep.mean():.4f}')


if __name__ == '__main__':
    main()
