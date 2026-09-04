"""WP-G: the accuracy-versus-bit-cap figure for the results page.

Two stacked panels sharing the x axis rather than one dual-axis chart: mean_area and
clean AUC are different measures on different scales, and a second y axis would invite
a comparison between them that does not exist.

Palette: categorical slots 1 (blue) and 2 (orange) from the reference palette, validated
with the skill's validator on the light surface (all six checks PASS, worst adjacent CVD
dE 24.7, normal-vision dE 33.6). Identity is never colour-alone: both series are direct-
labelled and marker shapes differ.
"""
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUNS = os.path.expanduser('~/hackathon-shared/runs')
QUANT = os.path.expanduser('~/hackathon-shared/quant')
PLOTS = os.path.expanduser('~/hackathon-shared/runs/plots')

BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#d8d8d4'

# (tag, bit cap, activation)
ROWS = [('gD-bits3-l1t', 3, 'gelu'), ('gD-bits4-l1t', 4, 'gelu'),
        ('gD-bits6-l1t', 6, 'gelu'), ('gB-q-b1e-6-l1t', 8, 'gelu'),
        ('gD-bits10-l1t', 10, 'gelu'),
        ('gF-bits6-relu-l1t', 6, 'relu'), ('gD-relu-l1t', 8, 'relu')]

FLOAT_REF = ('d-pma0-aug-meanpt-l1t', 0.811735, 0.9067)
STAGE_A = ('g-stageA-l1t', 0.807842, 0.8976)


def bench(tag):
    hits = sorted(glob.glob(f'{RUNS}/{tag}_*.json'), key=os.path.getmtime)
    if not hits:
        return None
    d = json.load(open(hits[-1]))
    return d['mean_area'], d.get('mean_area_std') or 0.0, d.get('auc_clean_full')


def ebops(tag):
    p = f'{QUANT}/{tag}.json'
    return json.load(open(p))['ebops'] if os.path.exists(p) else None


def main():
    pts = []
    for tag, cap, act in ROWS:
        b = bench(tag)
        if b:
            pts.append(dict(tag=tag, cap=cap, act=act, ma=b[0], sd=b[1], auc=b[2], eb=ebops(tag)))
    if not pts:
        print('no benched rows yet'); return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.4, 7.4), sharex=True,
                                   gridspec_kw={'height_ratios': [1.35, 1], 'hspace': 0.12})
    fig.patch.set_facecolor('#fcfcfb')
    for ax in (ax1, ax2):
        ax.set_facecolor('#fcfcfb')
        ax.grid(True, color=GRID, lw=0.6, alpha=0.9)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)

    # reference lines, direct-labelled (no legend entry needed - they are annotations)
    for ax, (name, ma, auc) in ((ax1, FLOAT_REF), (ax2, FLOAT_REF)):
        v = ma if ax is ax1 else auc
        ax.axhline(v, color=MUTED, lw=1.2, ls=(0, (5, 3)), alpha=0.85)
    for ax, (name, ma, auc) in ((ax1, STAGE_A), (ax2, STAGE_A)):
        v = ma if ax is ax1 else auc
        ax.axhline(v, color=MUTED, lw=1.2, ls=(0, (1.5, 2.5)), alpha=0.85)

    gel = sorted([p for p in pts if p['act'] == 'gelu'], key=lambda p: p['cap'])
    rel = sorted([p for p in pts if p['act'] == 'relu'], key=lambda p: p['cap'])
    live = [p for p in gel if p['ma'] > 0.55]
    dead = [p for p in gel if p['ma'] <= 0.55]

    ax1.errorbar([p['cap'] for p in live], [p['ma'] for p in live],
                 yerr=[3 * p['sd'] for p in live], color=BLUE, lw=2, marker='o', ms=8,
                 mfc=BLUE, mec='#fcfcfb', mew=1.6, capsize=3, elinewidth=1.2, zorder=3)
    if dead:
        ax1.plot([p['cap'] for p in dead], [p['ma'] for p in dead], color=BLUE, lw=2,
                 ls=(0, (2, 2)), marker='o', ms=9, mfc='none', mec=BLUE, mew=1.8, zorder=3)
        d = dead[-1]
        ax1.annotate('collapsed\n(1 distinct latent)', (d['cap'], d['ma']),
                     xytext=(9, 14), textcoords='offset points', fontsize=8.5,
                     color=MUTED, ha='left', linespacing=1.3)
        ax1.plot([dead[-1]['cap'], live[0]['cap']], [dead[-1]['ma'], live[0]['ma']],
                 color=BLUE, lw=2, ls=(0, (2, 2)), zorder=2)
    if rel:
        ax1.errorbar([p['cap'] for p in rel], [p['ma'] for p in rel],
                     yerr=[3 * p['sd'] for p in rel], color=ORANGE, lw=0, marker='D',
                     ms=8, mfc=ORANGE, mec='#fcfcfb', mew=1.6, capsize=3,
                     elinewidth=1.2, zorder=4)

    for p in live:
        if p['eb']:
            ax1.annotate(f"{p['eb']/1e8:.2f}e8", (p['cap'], p['ma']), xytext=(0, -17),
                         textcoords='offset points', fontsize=8, color=MUTED, ha='center')
    ax1.set_ylabel('bench mean_area', fontsize=10, color=INK)
    ax1.set_ylim(0.770, 0.816)
    ax1.text(10.15, FLOAT_REF[1], ' float L1T reference', va='center', fontsize=8.5, color=MUTED)
    ax1.text(10.15, STAGE_A[1] - 0.0012, ' stage A (LayerNorm→BatchNorm, unquantized)',
             va='center', fontsize=8.5, color=MUTED)

    ax2.plot([p['cap'] for p in live], [p['auc'] for p in live], color=BLUE, lw=2,
             marker='o', ms=8, mfc=BLUE, mec='#fcfcfb', mew=1.6, zorder=3)
    if rel:
        ax2.plot([p['cap'] for p in rel], [p['auc'] for p in rel], color=ORANGE, lw=0,
                 marker='D', ms=8, mfc=ORANGE, mec='#fcfcfb', mew=1.6, zorder=4)
    ax2.set_ylabel('clean AUC', fontsize=10, color=INK)
    ax2.set_xlabel('learned bit-width cap  (weights: total bits; data lanes: fractional bits)',
                   fontsize=10, color=INK)
    ax2.set_ylim(0.845, 0.912)

    ax1.set_xlim(2.4, 10.6)
    ax1.set_xticks([3, 4, 6, 8, 10])
    ax1.set_xticklabels(['3', '4', '6', '8', '10'])

    from matplotlib.lines import Line2D
    ax1.legend(handles=[
        Line2D([], [], color=BLUE, lw=2, marker='o', ms=7, mec='#fcfcfb', mew=1.4, label='GELU'),
        Line2D([], [], color=ORANGE, lw=0, marker='D', ms=7, mec='#fcfcfb', mew=1.4, label='ReLU'),
    ], loc='lower right', frameon=False, fontsize=9, labelcolor=INK)

    ax1.set_title('Quantized encoder: accuracy versus bit-width cap\n'
                  'L1T eval, eta_max 3.0, R=5 probe refits; error bars 3σ; EBOPs (N=200) below each point',
                  fontsize=11, color=INK, loc='left', pad=12, linespacing=1.5)

    os.makedirs(PLOTS, exist_ok=True)
    import datetime
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    for dst in (f'{PLOTS}/quant_curve_latest.png', f'{PLOTS}/quant_curve_{ts}.png'):
        fig.savefig(dst, dpi=170, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f'wrote {dst}')
    # table view, so identity and values never depend on the figure alone
    with open(f'{PLOTS}/quant_curve_latest.txt', 'w') as f:
        f.write(f'{"tag":24s} {"cap":>4s} {"act":>5s} {"EBOPs":>10s} {"mean_area":>10s} {"3sig":>8s} {"clean":>8s}\n')
        for p in sorted(pts, key=lambda p: (p['act'], p['cap'])):
            f.write(f"{p['tag']:24s} {p['cap']:4d} {p['act']:>5s} {p['eb'] or 0:10.3e} "
                    f"{p['ma']:10.4f} {3*p['sd']:8.4f} {p['auc']:8.4f}\n")
        f.write(f"\nreferences: float L1T {FLOAT_REF[1]:.4f} / {FLOAT_REF[2]:.4f};"
                f"  stage A {STAGE_A[1]:.4f} / {STAGE_A[2]:.4f}\n")
    print(f'wrote {PLOTS}/quant_curve_latest.txt')


if __name__ == '__main__':
    main()
