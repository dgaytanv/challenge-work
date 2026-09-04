"""WP-I: Phase-1 table -- mean over seeds +- seed std, per the measurement standard.

Reports, per configuration: mean_area as mean over the three seeds with the SEED std, the
per-seed probe std beside it (they measure different things and the standard asks for
both), clean AUC, and encoder parameter count.

Separability against a reference uses the seed test from 00-common-v2.md:
    |d| > 3 * sqrt(s_a^2/n_a + s_b^2/n_b)      s = seed std, n = number of seeds
and anything under 0.002 is not called real unless that test says so.
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

RUNS = os.path.expanduser('~/hackathon-shared/runs')
ARMS = {'ff256l1': "I1' 1 block, FFN 256", 'embed256': 'I3 embed_size 256',
        'seeds8': 'I4 8 seed queries', 'lat32': 'I6 latent_dim 32'}


def latest(tag):
    hits = sorted(glob.glob(f'{RUNS}/{tag}_*.json'), key=os.path.getmtime)
    return json.load(open(hits[-1])) if hits else None


def collect(prefix='i'):
    by = defaultdict(list)
    for desc in ARMS:
        for s in (11, 22, 33):
            d = latest(f'{prefix}-{desc}-s{s}')
            if d:
                by[desc].append((s, d))
    return by


def sep(ma, sa, na, mb, sb, nb):
    thr = 3 * np.sqrt(sa ** 2 / max(na, 1) + sb ** 2 / max(nb, 1))
    return ma - mb, thr, abs(ma - mb) > thr


def main():
    ref_tag = sys.argv[1] if len(sys.argv) > 1 else None
    by = collect()
    ref = None
    if ref_tag:
        rows = [latest(f'{ref_tag}-s{s}') for s in (11, 22, 33)]
        rows = [r for r in rows if r]
        if rows:
            m = np.array([r['mean_area'] for r in rows])
            ref = (float(m.mean()), float(m.std(ddof=1)) if len(m) > 1 else 0.0, len(m), ref_tag)

    print(f'{"configuration":26s} {"n":>2s} {"mean_area":>10s} {"seed sd":>8s} '
          f'{"probe sd":>9s} {"clean AUC":>10s} {"params":>10s}')
    print('-' * 82)
    out = []
    for desc, label in ARMS.items():
        rs = by.get(desc, [])
        if not rs:
            print(f'{label:26s} {"0":>2s}  (not benched yet)'); continue
        ma = np.array([d['mean_area'] for _, d in rs])
        ps = np.array([d.get('mean_area_std') or 0.0 for _, d in rs])
        ca = np.array([d.get('auc_clean_full') or np.nan for _, d in rs])
        seed_sd = float(ma.std(ddof=1)) if len(ma) > 1 else float('nan')
        row = dict(arm=desc, label=label, n=len(rs), mean_area=float(ma.mean()),
                   seed_std=seed_sd, probe_std_mean=float(ps.mean()),
                   clean_auc=float(np.nanmean(ca)),
                   seeds={s: d['mean_area'] for s, d in rs})
        out.append(row)
        print(f'{label:26s} {len(rs):2d} {ma.mean():10.4f} {seed_sd:8.4f} '
              f'{ps.mean():9.4f} {np.nanmean(ca):10.4f}')
    if ref:
        print(f'\nreference {ref[3]}: {ref[0]:.4f} +- {ref[1]:.4f} (n={ref[2]})')
        print(f'{"configuration":26s} {"delta":>9s} {"threshold":>10s} {"separable":>10s}')
        for r in out:
            if r['n'] < 2:
                print(f'{r["label"]:26s} {"-":>9s} {"-":>10s} {"n<2":>10s}'); continue
            d, thr, ok = sep(r['mean_area'], r['seed_std'], r['n'], ref[0], ref[1], ref[2])
            print(f'{r["label"]:26s} {d:+9.4f} {thr:10.4f} {("YES" if ok else "no"):>10s}')
    json.dump(out, open(os.path.join(RUNS, 'i_phase1_table.json'), 'w'), indent=2)
    print(f'\nwrote {RUNS}/i_phase1_table.json')


if __name__ == '__main__':
    main()
