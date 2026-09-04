"""WP-G G4: convert the HGQ2 model with hls4ml and (if possible) C-simulate it.

Reports honestly what converts and what does not. No Vitis/Vivado HLS exists on this box
(checked in G0: not on PATH, no /opt/Xilinx, /tools/Xilinx, /opt/intelFPGA), so synthesis
numbers are impossible here; hls4ml bundles ap_types and build_lib.sh, so C-simulation
with g++ is possible and is the bit-accuracy evidence.
"""
import argparse
import json
import os
import sys
import traceback

os.environ.setdefault('KERAS_BACKEND', 'torch')

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src'))
sys.path.insert(0, os.path.join(REPO, 'quant'))

from hgq_model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--params', default=None)
    ap.add_argument('--n_tokens', type=int, default=16)
    ap.add_argument('--io', default='io_parallel', choices=('io_parallel', 'io_stream'))
    ap.add_argument('--act', default='gelu')
    ap.add_argument('--outdir', default='/tmp/hls4ml_prj')
    ap.add_argument('--part', default='xcvu13p-flga2577-2-e')
    ap.add_argument('--csim', type=int, default=0)
    args = ap.parse_args()

    import hls4ml
    from hls4ml_patch import apply_patches
    patches = apply_patches()
    m = build_model(norm='bn', act=args.act, quantized=True, n_tokens=args.n_tokens)
    _ = m([np.zeros((2, args.n_tokens, 14), 'float32'),
           np.zeros((2, args.n_tokens, 32), 'float32')], training=False)
    if args.params:
        from train_qat import load_params
        load_params(m, args.params)
    # hls4ml's LUT handler calls .numpy() on variables that still require grad, which
    # only breaks on the torch backend. Freezing the model first avoids patching for it.
    m.trainable = False
    print(f'[hls] keras model ready: N={args.n_tokens}, io={args.io}, act={args.act}; '
          f'hls4ml patches applied: {patches}')

    report = dict(n_tokens=args.n_tokens, io=args.io, act=args.act, part=args.part,
                  hls4ml_patches=patches)
    try:
        hls_model = hls4ml.converters.convert_from_keras_model(
            m, backend='Vitis', io_type=args.io, output_dir=args.outdir,
            part=args.part, project_name='pma0')
        report['convert'] = 'OK'
        print('[hls] convert_from_keras_model: OK')
    except Exception as e:
        report['convert'] = f'{type(e).__name__}: {e}'
        print(f'[hls] convert FAILED: {type(e).__name__}: {e}')
        traceback.print_exc()
        json.dump(report, open(os.path.expanduser(
            f'~/hackathon-shared/quant/g4_hls4ml_N{args.n_tokens}_{args.io}.json'), 'w'), indent=2)
        return 1

    if args.csim:
        try:
            hls_model.compile()
            report['compile'] = 'OK'
            x = np.random.randn(8, args.n_tokens, 14).astype('float32')
            mk = np.zeros((8, args.n_tokens, 32), 'float32')
            ref = m([x, mk], training=False)
            ref = ref.detach().cpu().numpy() if hasattr(ref, 'detach') else np.asarray(ref)
            got = np.asarray(hls_model.predict([x, mk])).reshape(ref.shape)
            d = np.abs(ref - got)
            report.update(csim_max_abs=float(d.max()), csim_rms=float(np.sqrt((d ** 2).mean())),
                          csim_exact=float((d == 0).mean()))
            print(f'[hls] csim vs keras: max|d|={d.max():.3e}  exact={100*(d==0).mean():.2f}%')
        except Exception as e:
            report['compile'] = f'{type(e).__name__}: {e}'
            print(f'[hls] csim FAILED: {type(e).__name__}: {e}')

    json.dump(report, open(os.path.expanduser(
        f'~/hackathon-shared/quant/g4_hls4ml_N{args.n_tokens}_{args.io}.json'), 'w'), indent=2)
    print('[hls] report written')


if __name__ == '__main__':
    sys.exit(main() or 0)
