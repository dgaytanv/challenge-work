"""WP-G G2: quantization-aware distillation of the restructured encoder with HGQ2.

Objective (label-free, as the prompt specifies): MSE between the quantized student's
latent and the FLOAT reference's latent on the same events, over clean AND
generator-degraded views, plus HGQ2's EBOPs resource regulariser. The regulariser is not
added by hand - every HGQ2 layer does `add_loss(ebops * beta)` inside its wrapped `call`
when training is truthy (hgq/layers/core/base.py), so `beta0` IS the sweep knob.

Usage:
  python quant/train_qat.py --ckpt <float ref .pth> --train <train .pt> \
      --beta0 1e-5 --act gelu --epochs 12 --tag g-b1e5-gelu --eta_max 3.0
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault('KERAS_BACKEND', 'torch')

import keras
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src'))
sys.path.insert(0, os.path.join(REPO, 'quant'))

from embedding.degradation import Degradation
from embedding.utils.data_utils import load_data
from hgq_model import (assert_mask_margin, build_model, collect_ln_stats, load_weights,
                       mask_add_from_keep)
from keras_port import preproc_meanpt_np, survivors_np

OUT = os.path.expanduser('~/hackathon-shared/quant')


def bn_dict(ck):
    return {k: ck['preproc'][f'batch_norm.{k}'].cpu().numpy().astype(np.float64)
            for k in ('weight', 'bias', 'running_mean', 'running_var')}


def make_views(raw, bn, deg, chunk=256):
    """Clean + generator-degraded views -> (x_feat, mask_add, keep), concatenated.

    The degradation is applied in chunks of 256 events, matching the batch size the float
    reference was trained at, so the generator's internal curriculum (warmup_calls=600
    train-mode forward passes) advances at the same rate it did during that run rather
    than once per epoch.
    """
    degraded = torch.cat([deg(raw[i:i + chunk].clone()) for i in range(0, len(raw), chunk)])
    xs, ms, ks = [], [], []
    for v in (raw, degraded):
        xf = preproc_meanpt_np(v.numpy(), bn)
        keep = survivors_np(xf)
        xs.append(xf); ks.append(keep); ms.append(mask_add_from_keep(keep))
    return np.concatenate(xs), np.concatenate(ms), np.concatenate(ks)


def predict(model, x, m, batch=256):
    out = []
    for i in range(0, len(x), batch):
        r = model([x[i:i + batch], m[i:i + batch]], training=False)
        out.append(r.detach().cpu().numpy() if hasattr(r, 'detach') else np.asarray(r))
    return np.concatenate(out)


def save_params(model, path):
    """Save EVERY variable of the model, keyed by its keras path.

    Not `save_weights` (h5 cannot transfer between the float and quantized builds, whose
    variable lists differ), and NOT just kernels/biases either: a quantized build's
    per-tensor bitwidths (k, i, f) ARE the quantization, and they are learned during QAT
    and set by `trace_minmax`. An earlier version of this saved only kernels/BN
    parameters, so a reload silently reverted every quantizer to its i0=2 default and the
    exported model wrapped scores of magnitude 42 into [-4, 4). The emulation test caught
    it; keep this saving all variables.
    """
    np.savez(path, **{w.path: np.asarray(w) for w in model.weights})
    return path


def load_params(model, path):
    """Restore by keras variable path. Missing keys are skipped, which is what makes the
    stage A (float) -> stage B (quantized) seeding work: the float build has no quantizer
    variables, so the quantized build keeps its initialisers for those."""
    z = np.load(path)
    n = miss = 0
    for w in model.weights:
        if w.path in z:
            w.assign(z[w.path]); n += 1
        else:
            miss += 1
    if miss:
        print(f'[qat] load_params: {n} variables restored, {miss} left at initialiser')
    return n


def total_ebops(model):
    t = 0.0
    for l in model.layers:
        if hasattr(l, 'ebops'):
            e = l.ebops
            t += float(e.detach().cpu() if hasattr(e, 'detach') else np.asarray(e))
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='float reference checkpoint (.pth)')
    ap.add_argument('--train', required=True, help='training .pt file')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--beta0', type=float, default=1e-5)
    ap.add_argument('--act', default='gelu', choices=('gelu', 'relu'))
    ap.add_argument('--norm', default='bn', choices=('bn', 'ln'))
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--events', type=int, default=20000, help='events resampled per epoch')
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--eta_max', type=float, default=5.0)
    ap.add_argument('--n_tokens', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--quantized', type=int, default=1,
                    help='0 = float BatchNorm student (stage A: cost of the LN->BN swap alone); '
                         '1 = HGQ2 quantized student (stage B)')
    ap.add_argument('--init_params', default=None,
                    help='npz from a previous run to seed from (stage B seeds from stage A)')
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed); keras.utils.set_random_seed(args.seed)
    t_start = time.time()

    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    bn = bn_dict(ck)
    raw_all, _ = load_data(args.train, map_location='cpu', max_events=-1)
    print(f'[qat] {args.tag}: {raw_all.shape[0]} train events of {raw_all.shape[1]} candidates, '
          f'beta0={args.beta0:g} act={args.act} norm={args.norm} eta_max={args.eta_max}')

    N = args.n_tokens
    assert raw_all.shape[1] == N, f'--n_tokens {N} != data {raw_all.shape[1]}'

    # ---- float reference (teacher): the ported certified architecture, LayerNorm intact
    teacher = load_weights(build_model(norm='ln', act='gelu', quantized=False, n_tokens=N),
                           ck['encoder'], norm='ln')

    # ---- calibration sample for the BatchNorm replacement: half clean, half degraded
    deg = Degradation(severity=None, eta_max=args.eta_max)
    cal_raw = raw_all[np.random.choice(len(raw_all), 4000, replace=False)]
    cx, cm, ckeep = make_views(cal_raw, bn, deg)
    stats = collect_ln_stats(teacher, cx, cm, ckeep)
    print(f'[qat] BatchNorm calibration on {len(cx)} views, keep mean {ckeep.mean():.3f}')

    student = build_model(norm=args.norm, act=args.act, quantized=bool(args.quantized),
                          n_tokens=N, beta0=args.beta0)
    if args.init_params:
        n = load_params(student, args.init_params)
        print(f'[qat] seeded {n} layers from {args.init_params}')
    else:
        load_weights(student, ck['encoder'], stats, norm=args.norm)
    student.compile(optimizer=keras.optimizers.Adam(args.lr), loss='mse')

    hist = []
    for ep in range(args.epochs):
        idx = np.random.choice(len(raw_all), min(args.events, len(raw_all)), replace=False)
        x, m, keep = make_views(raw_all[idx], bn, deg)
        y = predict(teacher, x, m)                       # teacher latents, float reference
        h = student.fit([x, m], y, batch_size=args.batch, epochs=1, verbose=0, shuffle=True)
        mse = float(h.history['loss'][-1])
        eb = total_ebops(student)
        pred = predict(student, x[:2000], m[:2000])
        d = np.abs(pred - y[:2000])
        hist.append(dict(epoch=ep, loss=mse, ebops=eb, max_abs=float(d.max()),
                         rms=float(np.sqrt((d ** 2).mean()))))
        print(f'[qat] ep{ep:02d} loss={mse:.5f} EBOPs={eb:.3e} '
              f'max|d|={d.max():.4f} rms={np.sqrt((d**2).mean()):.4f} '
              f'keep={keep.mean():.3f} [{time.time()-t_start:.0f}s]')

    # ---- calibrate the WRAP-overflow datalane quantizers on real data (hgq.utils.trace_minmax)
    x, m, keep = make_views(raw_all[np.random.choice(len(raw_all), 4000, replace=False)], bn, deg)
    if args.quantized:
        from hgq.utils import trace_minmax
        trace_minmax(student, [x, m], batch_size=args.batch, verbose=0)
    eb_final = total_ebops(student)

    # ---- ruling 2: the additive mask must still suppress dead tokens AFTER training
    margin = assert_mask_margin(student, x, keep, m, n_tokens=N)
    print(f'[qat] mask margin: {margin["efolds_margin"]:.1f} e-folds '
          f'(dead score max {margin["score_dead_max"]:+.2f}, live max {margin["live_score_max"]:+.2f})')

    y = predict(teacher, x, m); p = predict(student, x, m)
    d = np.abs(p - y)
    per_layer = {l.name: float(l.ebops.detach().cpu() if hasattr(l.ebops, 'detach') else np.asarray(l.ebops))
                 for l in student.layers if hasattr(l, 'ebops')}

    os.makedirs(OUT, exist_ok=True)
    wpath = save_params(student, os.path.join(OUT, f'{args.tag}.params.npz'))
    res = dict(tag=args.tag, ckpt=os.path.basename(args.ckpt), train=os.path.basename(args.train),
               beta0=args.beta0, act=args.act, norm=args.norm, quantized=bool(args.quantized),
               init_params=args.init_params, epochs=args.epochs,
               events_per_epoch=args.events, lr=args.lr, eta_max=args.eta_max, n_tokens=N,
               ebops=eb_final, ebops_per_layer=per_layer, mask_margin=margin,
               distill_max_abs=float(d.max()), distill_rms=float(np.sqrt((d ** 2).mean())),
               teacher_latent_scale=float(np.abs(y).max()), history=hist,
               weights=wpath, seconds=time.time() - t_start)
    with open(os.path.join(OUT, f'{args.tag}.json'), 'w') as f:
        json.dump(res, f, indent=2)
    print(f'[qat] FINAL EBOPs={eb_final:.4e}  distill max|d|={d.max():.4f} '
          f'rms={np.sqrt((d**2).mean()):.4f} (teacher scale {np.abs(y).max():.2f})')
    print(f'[qat] wrote {OUT}/{args.tag}.json and {wpath}')


if __name__ == '__main__':
    main()
