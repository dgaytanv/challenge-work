"""WP-G campaign 2, G6/G7: build a SYNTHESISABLE hls4ml project at the full token count.

Campaign 1 left one blocker: the fully-unrolled io_parallel design at N=400 never compiled
(time-boxed at 45 min in g++). Two routes were named in the brief; this script builds both,
so they can be chosen on numbers rather than on argument.

G6a  --homogeneous 1 --io io_stream
     hls4ml 1.3.0 refuses io_stream for any model with heterogeneous ACTIVATION
     quantization. Forcing the data-lane quantizers to one (k,i,f) per tensor makes
     io_stream legal; it also throws away the per-channel bit widths HGQ2 exists for, so
     the model must be RE-TRAINED under the constraint and re-benched. The accuracy cost
     is the deliverable, not an assumption.

G6b  --pf 1 --io io_parallel
     Fold the token loop. This needs NO retraining and no model change, so the measured
     accuracy carries over by construction. It is not a hand-written wrapper: hls4ml
     already supports it and it was found by reading the source, not guessed --
       * `hgq.layers.QDense.parallelization_factor` defaults to -1 -> `prod(shape[1:-1])`,
         i.e. one datapath PER TOKEN (400 copies). That default is what campaign 1 hit.
       * `hls4ml/converters/keras_v3/hgq2/_base.py::QDenseHandler` copies the attribute
         into the layer config whenever the input rank > 1.
       * `hls4ml/model/optimizer/passes/multi_dense.py` turns a multidimensional Dense
         into a Conv1D and carries `parallelization_factor` over.
       * `vivado_backend.init_conv1d` then sets `n_partitions = out_width // pf`.
     pf = 1 therefore means 400 partitions of one token each: ONE per-token datapath,
     iterated 400 times. Same weights, same arithmetic, 1/400 of the multipliers.

     Two layers do not fold with pf because they are reductions over the token axis:
       * `softmax` (axis=1) -- has its own `parallelization_factor`, set here too.
       * `combine` (QEinsum, attn.v) -- hls4ml's Einsum has no pf, but its template emits
         `#pragma HLS PIPELINE II = reuse_factor` and
         `#pragma HLS ALLOCATION operation instances=mul limit = total_mults/reuse_factor`,
         so ReuseFactor is the fold knob for it. Set with --einsum_rf.

The script converts and writes the project (and optionally C-simulates). It deliberately
does NOT launch csynth itself: synthesis is long, must run under nohup with at most two at
once on this 8-core box, and is driven from `run_synth.sh` against the written project.
"""
import argparse
import json
import os
import sys
import time
import traceback

os.environ.setdefault('KERAS_BACKEND', 'torch')

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src'))
sys.path.insert(0, os.path.join(REPO, 'quant'))

from hgq_model import HEADS, NUM_FEATURES, SEEDS, build_model

# Layers whose input carries a token axis, i.e. the ones pf applies to.
TOKEN_DENSES = ('phi_0', 'phi_1', 'score', 'v')


def set_parallelization(model, pf, verbose=True):
    """Override every token-axis layer's parallelization_factor.

    QDense.build() sets `self.parallelization_factor = n_parallel` when it was left at -1,
    so this must run AFTER the model is built (it is: build_model calls the layers).
    """
    touched = {}
    for name in TOKEN_DENSES:
        L = model.get_layer(name)
        assert hasattr(L, 'parallelization_factor'), f'{name} has no parallelization_factor'
        touched[name] = (L.parallelization_factor, pf)
        L.parallelization_factor = pf
    L = model.get_layer('softmax')
    if hasattr(L, 'parallelization_factor'):
        touched['softmax'] = (L.parallelization_factor, pf)
        L.parallelization_factor = pf
    if verbose:
        for k, (a, b) in touched.items():
            print(f'[g6] pf {k}: {a} -> {b}')
    return touched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--params', default=None, help='npz of trained parameters')
    ap.add_argument('--n_tokens', type=int, default=400)
    ap.add_argument('--io', default='io_parallel', choices=('io_parallel', 'io_stream'))
    ap.add_argument('--act', default='relu', choices=('gelu', 'relu'))
    ap.add_argument('--max_bits', type=int, default=6)
    ap.add_argument('--homogeneous', type=int, default=0, help='G6a: per-tensor activation quantizers')
    ap.add_argument('--pf', type=int, default=None, help='G6b: token-loop parallelization factor')
    ap.add_argument('--einsum_rf', type=int, default=None, help='ReuseFactor for the attn.v einsum')
    ap.add_argument('--out_proj_pf', type=int, default=None,
                    help='ParallelizationFactor for out_proj. Its input is [4, 128], so hls4ml '
                         'defaults it to n_partitions=1 and unrolls all four seed positions -- '
                         '65,536 multipliers, 61%% of the whole folded design. 1 folds it.')
    ap.add_argument('--dense_rf', type=int, default=None,
                    help='ReuseFactor for phi_1 and v, the two 128x128 per-token Denses. '
                         'rf=4 takes each from 16,384 multipliers to 4,096 at 4x the cycles.')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--part', default='xcvu13p-flga2577-2-e')
    ap.add_argument('--clock', type=float, default=5.0, help='ns; 5.0 = 200 MHz')
    ap.add_argument('--csim', type=int, default=0)
    ap.add_argument('--csim_events', type=int, default=8)
    ap.add_argument('--tag', default=None)
    args = ap.parse_args()

    tag = args.tag or f'{"g6a" if args.homogeneous else "g6b"}_N{args.n_tokens}_{args.io}'
    t0 = time.time()
    import hls4ml
    from hls4ml_patch import apply_patches
    patches = apply_patches()

    m = build_model(norm='bn', act=args.act, quantized=True, n_tokens=args.n_tokens,
                    max_bits=args.max_bits, homogeneous=bool(args.homogeneous))
    _ = m([np.zeros((2, args.n_tokens, NUM_FEATURES), 'float32'),
           np.zeros((2, args.n_tokens, HEADS * SEEDS), 'float32')], training=False)
    if args.params:
        from train_qat import load_params
        n = load_params(m, args.params)
        print(f'[g6] restored {n} variables from {args.params}')
    m.trainable = False   # hls4ml's LUT handler calls .numpy() on grad-carrying vars

    report = dict(tag=tag, n_tokens=args.n_tokens, io=args.io, act=args.act,
                  max_bits=args.max_bits, homogeneous=bool(args.homogeneous),
                  pf=args.pf, einsum_rf=args.einsum_rf, out_proj_pf=args.out_proj_pf,
                  dense_rf=args.dense_rf, part=args.part,
                  clock_ns=args.clock, clock_mhz=1000.0 / args.clock,
                  params=os.path.basename(args.params) if args.params else None,
                  hls4ml_patches=patches, outdir=args.outdir)

    if args.pf is not None:
        report['pf_applied'] = {k: list(v) for k, v in set_parallelization(m, args.pf).items()}

    # Per-layer overrides. hls4ml's hgq2 flow sets precisions itself (bit-exact), so the
    # only knobs touched here are structural.
    hls_cfg = {'Model': {'Precision': 'auto', 'ReuseFactor': 1, 'Strategy': 'Latency'},
               'LayerName': {}}
    if args.einsum_rf:
        hls_cfg['LayerName']['combine'] = {'ReuseFactor': args.einsum_rf}
    if args.pf is not None:
        for name in TOKEN_DENSES:
            hls_cfg['LayerName'].setdefault(name, {})['ParallelizationFactor'] = args.pf
    if args.out_proj_pf is not None:
        hls_cfg['LayerName'].setdefault('out_proj', {})['ParallelizationFactor'] = args.out_proj_pf
        L = m.get_layer('out_proj')
        report['out_proj_pf_applied'] = [L.parallelization_factor, args.out_proj_pf]
        L.parallelization_factor = args.out_proj_pf
    if args.dense_rf is not None:
        for name in ('phi_1', 'v'):
            hls_cfg['LayerName'].setdefault(name, {})['ReuseFactor'] = args.dense_rf

    try:
        hls_model = hls4ml.converters.convert_from_keras_model(
            m, backend='Vitis', io_type=args.io, output_dir=args.outdir,
            part=args.part, clock_period=args.clock, project_name='pma0',
            hls_config=hls_cfg)
        report['convert'] = 'OK'
        print(f'[g6] convert OK in {time.time()-t0:.0f}s')
    except Exception as e:
        report['convert'] = f'{type(e).__name__}: {e}'
        print(f'[g6] convert FAILED: {type(e).__name__}: {e}')
        traceback.print_exc()
        _write(report, tag)
        return 1

    try:
        hls_model.write()
        report['write'] = 'OK'
    except Exception as e:
        report['write'] = f'{type(e).__name__}: {e}'
        print(f'[g6] write FAILED: {type(e).__name__}: {e}')
        traceback.print_exc()
        _write(report, tag)
        return 1

    # what the graph actually became, so the fold is evidenced rather than asserted
    try:
        report['graph'] = [
            dict(name=n.name, cls=n.class_name,
                 n_partitions=n.get_attr('n_partitions'),
                 pf=n.get_attr('parallelization_factor'),
                 rf=n.get_attr('reuse_factor'))
            for n in hls_model.graph.values()]
    except Exception as e:
        report['graph'] = f'{type(e).__name__}: {e}'

    if args.csim:
        try:
            t1 = time.time()
            hls_model.compile()
            report['compile'] = 'OK'
            report['compile_seconds'] = time.time() - t1
            rng = np.random.default_rng(0)
            B = args.csim_events
            x = rng.standard_normal((B, args.n_tokens, NUM_FEATURES)).astype('float32')
            mk = np.zeros((B, args.n_tokens, HEADS * SEEDS), 'float32')
            ref = m([x, mk], training=False)
            ref = ref.detach().cpu().numpy() if hasattr(ref, 'detach') else np.asarray(ref)
            got = np.asarray(hls_model.predict([x, mk])).reshape(ref.shape)
            d = np.abs(ref - got)
            report.update(csim_max_abs=float(d.max()), csim_rms=float(np.sqrt((d ** 2).mean())),
                          csim_exact_frac=float((d == 0).mean()))
            print(f'[g6] csim vs keras: max|d|={d.max():.3e} exact={100*(d==0).mean():.2f}%')
        except Exception as e:
            report['compile'] = f'{type(e).__name__}: {e}'
            print(f'[g6] csim FAILED: {type(e).__name__}: {e}')
            traceback.print_exc()

    report['seconds'] = time.time() - t0
    _write(report, tag)
    return 0


def _write(report, tag):
    out = os.path.expanduser('~/hackathon-shared/quant')
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, f'g6_{tag}.json')
    with open(p, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f'[g6] wrote {p}')


if __name__ == '__main__':
    sys.exit(main())
