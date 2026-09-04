"""WP-G: collect every sweep row into one table.

Columns the planner asked for (00:26): per-row EBOPs, bit widths, the score-path integer
bits (because the live score range grew 5.6x during stage A and that costs integer bits at
the softmax input), and the bench delta against the row's OWN float reference.
"""
import glob
import json
import os

import numpy as np

Q = os.path.expanduser('~/hackathon-shared/quant')
RUNS = os.path.expanduser('~/hackathon-shared/runs')


def bits_of(z, layer):
    """Total bits of every quantizer under one layer, from the saved variable paths.

    Paths look like `phi_0/phi_0_iq/fixed_point_quantizer_kif/{k,i,f}` (KIF) or
    `.../fixed_point_quantizer_kbi/{k,b,i}` (KBI). Bit width is k+i+f for KIF and b (+k)
    for KBI; the integer part is what the score-range question turns on, so both are kept.
    """
    groups = {}
    for key in z.files:
        parts = key.split('/')
        if parts[0] != layer or len(parts) < 3:
            continue
        leaf = parts[-1]
        if leaf not in ('k', 'i', 'f', 'b'):
            continue
        groups.setdefault('/'.join(parts[:-1]), {})[leaf] = np.asarray(z[key], dtype=float)

    out = {}
    for gpath, g in groups.items():
        role = 'iq' if ('_iq' in gpath or 'multiple_quantizers' in gpath) else 'kq' if '_kq' in gpath else 'bq' if '_bq' in gpath \
            else 'oq' if '_oq' in gpath else 'q'
        if {'k', 'i', 'f'} <= set(g):
            bits = g['k'] + g['i'] + g['f']
            ipart = g['i']
        elif {'k', 'b'} <= set(g):
            bits = g['b'] + g['k']
            ipart = g.get('i', np.zeros_like(g['b']))
        else:
            continue
        out.setdefault(role, []).append(dict(bits_max=float(bits.max()), bits_mean=float(bits.mean()),
                                             i_max=float(ipart.max()), i_mean=float(ipart.mean())))
    return {r: dict(bits_max=max(v['bits_max'] for v in vs),
                    bits_mean=float(np.mean([v['bits_mean'] for v in vs])),
                    i_max=max(v['i_max'] for v in vs),
                    i_mean=float(np.mean([v['i_mean'] for v in vs]))) for r, vs in out.items()}


def bench_row(tag):
    """Latest bench JSON for this tag."""
    hits = sorted(glob.glob(os.path.join(RUNS, f'*{tag}*.json')), key=os.path.getmtime)
    if not hits:
        return None
    d = json.load(open(hits[-1]))
    return dict(file=os.path.basename(hits[-1]),
                mean_area=d.get('mean_area'), mean_area_std=d.get('mean_area_std'),
                clean_auc=d.get('auc_clean_full'), clean_auc_std=d.get('auc_clean_full_std'),
                eta_max=d.get('eta_max'), train_data=d.get('train_data'),
                per_family={k: (v.get('area') if isinstance(v, dict) else v)
                            for k, v in (d.get('families') or {}).items()})


def main():
    rows = []
    for f in sorted(glob.glob(f'{Q}/g[ABCD]-*.json')):
        d = json.load(open(f))
        tag = d['tag']
        npz = f.replace('.json', '.params.npz')
        bits = {}
        if os.path.exists(npz):
            z = np.load(npz)
            for lay in ('phi_0', 'phi_1', 'score', 'mask_add_op', 'v', 'out_proj', 'bottleneck'):
                b = bits_of(z, lay)
                if b:
                    bits[lay] = b
        rows.append(dict(tag=tag, quantized=d.get('quantized'), beta0=d.get('beta0'),
                         act=d.get('act'), ebops=d.get('ebops'),
                         distill_rms=d.get('distill_rms'), distill_max=d.get('distill_max_abs'),
                         teacher_scale=d.get('teacher_latent_scale'),
                         margin=d.get('mask_margin', {}).get('efolds_margin'),
                         live_score_max=d.get('mask_margin', {}).get('live_score_max'),
                         bits=bits, bench=bench_row(tag)))

    hdr = (f'{"tag":26s} {"beta0":>7s} {"act":>5s} {"EBOPs":>10s} {"rms":>7s} '
           f'{"score i":>8s} {"margin":>7s} {"mean_area":>18s} {"clean AUC":>10s}')
    print(hdr); print('-' * len(hdr))
    for r in rows:
        si = r['bits'].get('mask_add_op', {}).get('iq', {}).get('i_max')
        b = r['bench'] or {}
        ma = (f"{b['mean_area']:.4f} +- {b.get('mean_area_std') or 0:.4f}"
              if b.get('mean_area') is not None else 'not benched')
        ca = f"{b['clean_auc']:.4f}" if b.get('clean_auc') is not None else '-'
        print(f"{r['tag']:26s} {r['beta0'] or 0:7.0e} {r['act'] or '-':>5s} "
              f"{r['ebops'] or 0:10.3e} {r['distill_rms'] or 0:7.4f} "
              f"{si if si is not None else -1:8.0f} {r['margin'] or 0:7.1f} {ma:>18s} {ca:>10s}")

    # mean weight/activation bit widths across the six Dense layers, for the report table
    for r in rows:
        w = [v['kq']['bits_mean'] for v in r['bits'].values() if 'kq' in v]
        a_ = [v['iq']['bits_mean'] for v in r['bits'].values() if 'iq' in v]
        r['mean_weight_bits'] = float(np.mean(w)) if w else None
        r['mean_act_bits'] = float(np.mean(a_)) if a_ else None
        print(f"   {r['tag']:26s} mean weight bits {r['mean_weight_bits'] or 0:5.2f}  "
              f"mean activation bits {r['mean_act_bits'] or 0:5.2f}")

    json.dump(rows, open(f'{Q}/g_sweep_table.json', 'w'), indent=2)
    print(f'\nwrote {Q}/g_sweep_table.json')


if __name__ == '__main__':
    main()


def markdown_table(path=None):
    """Emit the accuracy-versus-bits table for writeup/G-quantization.md."""
    rows = json.load(open(f'{Q}/g_sweep_table.json'))
    order = {'gD-bits3-l1t': 3, 'gD-bits4-l1t': 4, 'gD-bits6-l1t': 6, 'gD-bits10-l1t': 10}
    lines = ['| cap (bits) | mean w bits | mean act bits | EBOPs (N=200) | distill rms | mean_area | clean AUC | Δ vs stage A |',
             '|---|---|---|---|---|---|---|---|']
    for r in sorted(rows, key=lambda r: order.get(r['tag'], 99)):
        if not r['tag'].startswith('gD-'):
            continue
        b = r['bench'] or {}
        cap = order.get(r['tag'], '—')
        ma = f"{b['mean_area']:.4f} ± {b.get('mean_area_std') or 0:.4f}" if b.get('mean_area') else 'not benched'
        ca = f"{b['clean_auc']:.4f}" if b.get('clean_auc') else '—'
        d = f"{b['mean_area'] - 0.8078:+.4f}" if b.get('mean_area') else '—'
        lines.append(f"| {cap} | {r.get('mean_weight_bits') or 0:.2f} | {r.get('mean_act_bits') or 0:.2f} | "
                     f"{r['ebops']:.3e} | {r['distill_rms']:.3f} | {ma} | {ca} | {d} |")
    out = '\n'.join(lines)
    print(out)
    if path:
        open(path, 'w').write(out)
    return out


if os.environ.get('G_MARKDOWN'):
    markdown_table(os.environ.get('G_MARKDOWN_OUT'))
