"""WP-I: latent offset, spread and their ratio, per arm.

Cross-package question from WP-K (03:47): the champion's architecture-plus-objective
inflates the latent offset on its own, with no cosine term to reward it. Does a WIDER
latent inflate faster or slower? My arms answer that as a natural experiment -- I6 is
latent_dim 32 against every other arm's 6, and nothing else differs.

DEFINITIONS. Two spread conventions are reported, not one, because WP-K and campaign 1
use a different one from the obvious choice and the difference is NOT a constant:

    offset    = || mean_i z_i ||                        (norm of the mean latent)
    spread_ms = sqrt( mean_i || z_i - mean z ||^2 )     RMS distance   -> ratio_rms
    spread_mn = mean_i || z_i - mean z ||               mean of norms  -> ratio_mon
    factor    = spread_ms / spread_mn = ratio_mon / ratio_rms

`ratio_mon` is the convention of campaign 1's published numbers (WP-C's writeup defines
spread as mean_i ||z_i - mu||), so the reference values everyone reasons against -- stock
0.17x, the broken two-view run 30.46x / 37.54x / 36.31x -- are mean-of-norms numbers.
Reporting RMS ratios beside them under the same name would shift every campaign-1
comparison. Both columns are emitted and the reference convention is named.

Why the factor cannot simply be divided out: it depends on the DIMENSION, because norms
concentrate as d grows. Measured on Gaussian clouds: 1.128 (d=2), 1.042 (d=6), 1.016
(d=16), 1.008 (d=32), 1.004 (d=64). My arms span latent_dim 6 and 32, so converting
between conventions moves a dim-6 row by ~4.2% and a dim-32 row by ~0.8% -- a ~3.4%
differential that acts in the SAME DIRECTION as the "does a wider latent inflate more"
effect this tool exists to measure. The factor is therefore computed per arm from the
real latents rather than taken from the Gaussian table.

Both offset and spread are norms in the full latent space, so both grow with latent_dim
and only the ratios are comparable across widths -- and even they carry the caveat above.
Computed on clean eval events; no probe, no degradation, so this is a property of the
encoder alone.

Unlike a per-batch measurement this has no sampling floor: WP-K measures per batch of 256
and reports a floor of ~0.0114 on their scan (a true offset of 0 reads as 0.0269 because
norms are non-negative and never average to zero). An eval-set measurement over thousands
of events does not have that, which makes this the better instrument for the low-offset
regime a ||mean z|| penalty pushes into.

Runs on CPU. No GPU slot needed.
"""
import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, 'src')


def geometry(z):
    mu = z.mean(0)
    dev = z - mu
    # Condition number of the latent covariance (WP-K, 04:08): the quantity that actually
    # governs how well a probe can be fit, and on their runs a far larger effect than the
    # offset (champion 6,000-18,000 against a penalised arm near 33). Reported with the
    # same dimensional caveat as the ratios and MORE severely: a 32-dim covariance has more
    # chances to contain a tiny eigenvalue than a 6-dim one, so cond is not comparable
    # between latent widths at all. eigvalsh, covariance is symmetric PSD by construction.
    offset = float(mu.norm())
    spread_ms = float((dev ** 2).sum(1).mean().sqrt())      # RMS distance
    spread_mn = float(dev.norm(dim=1).mean())               # mean of norms (campaign 1)
    cov = (dev.T @ dev) / max(len(dev) - 1, 1)
    ev = torch.linalg.eigvalsh(cov.double()).clamp_min(0)
    lo, hi = float(ev[0]), float(ev[-1])
    cond = hi / lo if lo > 0 else float('inf')
    # Effective rank (participation ratio, (sum l)^2 / sum l^2): how many latent directions
    # actually carry variance. Answers whether a WIDER latent is being used or merely
    # allocated. CAVEAT that must travel with this number: variance is not discriminability.
    # Campaign 1 measured a small probe-aligned latent component carrying ALL the AUC loss
    # while a 3.8x larger null-space component cost almost nothing, so a low-variance
    # direction can still be exactly the one the probe needs. Effective rank bounds how much
    # the latent SPREADS, never how much of it the probe can use.
    tot = float(ev.sum())
    eff_rank = (tot ** 2) / float((ev ** 2).sum()) if tot > 0 else float('nan')
    var_top6 = float(ev.flip(0)[:6].sum()) / tot if tot > 0 else float('nan')
    return dict(offset=offset, spread_rms=spread_ms, spread_meannorm=spread_mn,
                cov_cond=cond, cov_eig_min=lo, cov_eig_max=hi,
                effective_rank=eff_rank, var_frac_top6=var_top6,
                ratio_rms=offset / spread_ms if spread_ms > 0 else float('nan'),
                ratio_meannorm=offset / spread_mn if spread_mn > 0 else float('nan'),
                convention_factor=spread_ms / spread_mn if spread_mn > 0 else float('nan'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='+', default=[])
    ap.add_argument('--pair', nargs='+', default=[], metavar='NAME=CFG:CKPT',
                    help='explicit rows for checkpoints outside my own layout, e.g. another '
                         'package reference. The CKPT should be taken from that run\'s bench '
                         'JSON `ckpt` field, never from a glob: there are three '
                         'rt_c2_champion_s11 checkpoints in three scratchpads (a smoke, the '
                         'reference run, and another session\'s), and only the JSON says '
                         'which one produced the number being compared against.')
    ap.add_argument('--ckptdir', default='checkpoints/phase1')
    ap.add_argument('--data', default=os.path.expanduser(
        '~/hack-data/C9_robust_tagging/eval/robust_tagging_eval_small.pt'))
    ap.add_argument('--events', type=int, default=4000)
    ap.add_argument('--out', default=os.path.expanduser('~/hackathon-shared/quant/i_latent_geometry.json'))
    args = ap.parse_args()

    import embedding.models as models
    from embedding.utils.cfg_handler import train_config
    from embedding.utils.data_utils import load_data
    from embedding.preprocs import PFPreProcessorMeanPt

    feats, _ = load_data(args.data, map_location='cpu', max_events=args.events)
    work = [(os.path.basename(c).replace('.yaml', ''), c, None) for c in sorted(args.configs)]
    for spec in args.pair:
        nm, rest = spec.split('=', 1)
        cfgp, ckp = rest.split(':', 1)
        work.append((nm, cfgp, ckp))

    rows = []
    print(f'{"arm":22s} {"dim":>4s} {"params":>10s} {"offset":>9s} '
          f'{"ratio_mon":>10s} {"ratio_rms":>10s} {"factor":>7s} {"cond":>10s} {"eff_rank":>9s}')
    print(f'{"":22s} {"":4s} {"":10s} {"":9s} {"(campaign 1)":>10s}')
    for name, cfg_path, explicit in work:
        # An explicit path comes from that run's own bench JSON `ckpt` field, never a glob.
        # Select by uniqueness, not by mtime. A newest-mtime glob is the campaign-1
        # "stale checkpoint pick" (WP-H repeated the warning at 03:52): the aux/ directory
        # holds _bestauc and _last, and _last is written LAST, so an mtime rule that ever
        # sees aux/ silently grades the wrong model. This pattern is non-recursive so aux/
        # is already excluded -- the assert makes that a checked property rather than a
        # lucky one, and fails loudly if a rerun ever leaves two primaries side by side.
        if explicit:
            ck = [explicit]
            if not os.path.exists(explicit):
                print(f'{name:22s}  MISSING {explicit}'); continue
        else:
            # A checkpoint existing is NOT a completed run: train.py writes a best-so-far
            # file after every improving epoch. This guard was in the bench driver and NOT
            # here, and as a result an early version of i_latent_geometry_all.json carried
            # a row for i3_embed256_s22 computed on a partial checkpoint from a killed run
            # (empty log, no released line). Same three-state collapse, third location.
            logf = f'{args.ckptdir}/{name}.log'
            done = False
            if os.path.exists(logf):
                done = any('] released' in ln and 'rc=0' in ln for ln in open(logf, errors='ignore'))
            if not done:
                print(f'{name:22s}  SKIP: training did not complete (no released rc=0 in its log)')
                continue
            ck = sorted(glob.glob(f'{args.ckptdir}/{name}/*.pth'))
        if not ck:
            print(f'{name:22s}  no checkpoint'); continue
        assert len(ck) == 1, (f'{name}: expected exactly one primary checkpoint, found '
                              f'{len(ck)}: {[os.path.basename(c) for c in ck]}. Refusing to '
                              f'guess which one the run selected.')
        cfg = train_config(cfg_path)
        ckpt = torch.load(ck[0], map_location='cpu', weights_only=False)
        pre = PFPreProcessorMeanPt(ckpt['norm_constants'])
        pre.load_state_dict(ckpt['preproc']); pre.eval()
        Enc = getattr(models, cfg.hp('encoder_class', 'TransformerEncoder'))
        enc = Enc(num_features=pre.num_features, embed_size=cfg.hp('embed_size', 128),
                  latent_dim=cfg.hp('latent_dim', 6), num_heads=cfg.hp('num_heads', 8),
                  num_layers=cfg.hp('num_layers', 4), linear_dim=cfg.hp('linear_dim', None),
                  num_tokens=None, pairwise=cfg.get_trdata_cfg('pairwise', False))
        enc.load_state_dict(ckpt['encoder']); enc.eval()
        zs = []
        with torch.no_grad():
            for i in range(0, len(feats), 500):
                zs.append(enc(pre(feats[i:i + 500])))
        z = torch.cat(zs).float()
        g = geometry(z)
        n = sum(p.numel() for p in enc.parameters())
        rows.append(dict(arm=name, latent_dim=cfg.hp('latent_dim', 6), params=n,
                         encoder_class=cfg.hp('encoder_class'), events=int(len(z)),
                         ckpt=os.path.basename(ck[0]), **g))
        print(f'{name:22s} {cfg.hp("latent_dim",6):4d} {n:10,d} {g["offset"]:9.4f} '
              f'{g["ratio_meannorm"]:10.3f} {g["ratio_rms"]:10.3f} {g["convention_factor"]:7.4f} '
              f'{g["cov_cond"]:10.1f} {g["effective_rank"]:9.2f}')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(dict(definition='offset=||mean z||. TWO spread conventions: spread_rms='
                              'sqrt(mean||z-mu||^2) and spread_meannorm=mean||z-mu||. '
                              'ratio_meannorm is campaign 1 / WP-K convention (stock 0.17x '
                              'reference); ratio_rms is the RMS one. convention_factor='
                              'spread_rms/spread_meannorm is dimension-dependent (~1.042 at '
                              'dim 6, ~1.008 at dim 32), so the two conventions do NOT differ '
                              'by a constant across arms of different latent width. '
                              'Clean eval events, no degradation, no probe, no batch floor.',
                   data=os.path.basename(args.data), rows=rows), open(args.out, 'w'), indent=2)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
