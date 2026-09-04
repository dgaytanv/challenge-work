"""WP-G G4: minimal, documented runtime patches to hls4ml 1.3.0 for this model.

Both are hls4ml bugs, not model problems, and both are reported in writeup/G-quantization.md
rather than hidden. Nothing here changes what the model computes.

P1. `converters/keras_v3/hgq2/unary_lut.py` calls `.numpy()` on layer variables that still
    require grad. That is fine on the TF/JAX backends but raises on the torch backend.
    Avoided WITHOUT patching by setting `model.trainable = False` before conversion
    (kept here as a documented step, not a patch).

P3. `backends/fpga/passes/hgq_proxy_model.py::generate_mask_fn` emits ONE LINE OF C++ PER
    ELEMENT of the quantized tensor, unconditionally: it broadcasts the (k, b, i) masks to
    the full shape and writes `out[idx] = ap_fixed<...>(inp[idx]);` 51,200 times for a
    [400, 128] tensor. Six such quantizers is ~300k lines of generated C++, and Vitis HLS
    reported **56,931,767 instructions after Compile/Link** for the N=400 design (its own
    design-size warning threshold is 50M) after 37 minutes of source analysis. This is the
    real N=400 wall, and it is NOT the arithmetic: with the token loop folded (pf=1) the
    four Dense layers are only 29k-258k instructions each, while the quantizers are
    6.1M-7.8M each.

    Our data-lane quantizers are homogeneous over `(batch, token)` by construction (that is
    what `homogeneous_axis=(0, 1)` in `hgq_model.build_model` means, and it has to be: one
    datapath processes every token in hardware). So every token repeats the SAME 128-entry
    mask, and the generated function can be a loop over tokens around a 128-line body.
    P3 detects that case -- it verifies the broadcast masks really are identical across the
    leading axes rather than assuming it -- and emits the loop. Where the masks do vary it
    falls back to hls4ml's own code verbatim, so nothing is silently changed. The emitted
    arithmetic is identical either way; only the source is smaller.

P2. Same file, line 55: `table = layer.oq(table[None, ...])[0]` assumes the LUT's output
    quantizer was built on a RANK-2 tensor. Our activation sits on a per-token tensor
    [B, N, C], so its quantizer is rank 3 and the expand fails with
    `expand(FloatTensor{[1,1,1]}, size=[1,512])`. The table itself is homogeneous (we force
    allow_heterogeneous_table=False, which hls4ml requires anyway), so the quantizer is
    scalar and the rank is pure bookkeeping. The patch adds the missing axes and removes
    them again.
"""
import numpy as np


def apply_patches():
    """Replace hls4ml's QUnaryLUT handler with the same logic, rank-generalised.

    Swapping `layer._oq` for a shim is not possible: on the torch backend a keras Layer is
    an nn.Module AND keras locks state assignment on a built layer, so the attribute cannot
    be replaced from outside. The handler body is therefore reproduced here with the one
    line that assumes rank 2 fixed. Everything else is hls4ml 1.3.0's code path.
    """
    from decimal import Decimal

    import numpy as np
    from quantizers import get_fixed_quantizer_np

    from hls4ml.converters.keras_v3.hgq2 import unary_lut as UL
    from hls4ml.model.types import FixedPrecisionType

    if getattr(UL, '_wpg_patched', False):
        return ['unary_lut rank-2 table assumption (already applied)']

    def handle(self, layer, in_tensors, out_tensors):
        from hgq.quantizer.internal import FixedPointQuantizerBase
        from keras import ops

        if not layer.enable_iq and not layer.enable_oq:
            raise ValueError('Only input_quantizer-enabled UnaryFunctionLUT is supported')
        assert not layer._allow_heterogeneous_table, 'Heterogeneous table is not supported'

        iq = layer.iq.quantizer
        assert isinstance(iq, FixedPointQuantizerBase), 'Only fixed-point quantizers supported'
        k, i, f = iq.kif
        mask = ops.convert_to_numpy(k + i + f) > 0
        i = np.where(mask, ops.convert_to_numpy(i), -32)
        f = np.where(mask, ops.convert_to_numpy(f), -32)
        k = Decimal(int(ops.convert_to_numpy(ops.max(k))))
        i, f = Decimal(int(i.max())), Decimal(int(f.max()))
        _min, _eps = -k * 2**i, 2**-f
        _max = 2**i - _eps
        N = (_max - _min) / _eps + 1
        assert float(N).is_integer(), 'Invalid quantizer range'
        N = int(N)
        assert N <= 1e6, f'Activation LUT would need {N} entries; too large'
        assert np.log2(N).is_integer(), f'N must be a power of 2, got {N}'

        all_inputs = np.linspace(float(_min), float(_max), N, dtype=np.float32)
        table = layer.activation(all_inputs)
        if layer.enable_oq:
            # ---- THE FIX. hls4ml does `layer.oq(table[None, ...])[0]`, which assumes the
            # output quantizer was built on a rank-2 tensor. Our activation sits on a
            # per-token tensor [B, N, C], so its quantizer is rank 3 and the expand fails.
            ndim = len(layer.oq.quantizer.kif[0].shape)
            t = table
            for _ in range(max(ndim - 1, 1)):
                t = t[None, ...]
            t = layer.oq(t)
            for _ in range(max(ndim - 1, 1)):
                t = t[0]
            table = t
        table = ops.convert_to_numpy(table)
        if k:
            table = np.concatenate([table[N // 2:], table[: N // 2]])

        oq = layer.oq.quantizer
        assert isinstance(oq, FixedPointQuantizerBase)
        round_mode = oq.round_mode[2:] if oq.round_mode.startswith('S_') else oq.round_mode
        fixed_q = get_fixed_quantizer_np(round_mode, oq.overflow_mode)
        ko, io, fo = (ops.convert_to_numpy(x).ravel().item() for x in oq.kif)
        table = fixed_q(table, ko, io, fo)
        table_t = FixedPrecisionType(int(ko + io + fo), int(ko + io), bool(ko))

        config = {}
        config.update(self.default_config)
        config.update({
            'class_name': 'UnaryLUT',
            'table_data': table,
            'table_t': table_t,
            'activation': 'unary_lut',
            'n_in': int(np.prod(in_tensors[0].shape[1:])),
        })
        return (config,)

    UL.QUnaryLUTHandler.handle = handle
    UL._wpg_patched = True
    return ['unary_lut rank-2 table assumption'] + _patch_mask_fn()


def _patch_mask_fn():
    """P3: fold the per-element quantizer mask function over the token axis.

    See the module docstring. Returns a list naming what was patched.
    """
    import os as _os

    from hls4ml.backends.fpga.passes import hgq_proxy_model as HP

    # WPG_NO_P3=1 disables the fold, so "stock hls4ml" and "with P3" can be synthesised as
    # a controlled pair. Used to attribute Vitis's long Unroll/Inline phase to one or the
    # other rather than guessing.
    if _os.environ.get('WPG_NO_P3') == '1':
        return ['P3 DISABLED by WPG_NO_P3=1 (stock hls4ml mask codegen)']

    if getattr(HP, '_wpg_mask_patched', False):
        return ['fixed-point quantizer mask folded over the token axis (already applied)']

    orig = HP.generate_mask_fn

    def folded(name, shape, k, b, i, RND, SAT, backend):
        # Only io_parallel reaches here at all (hls4ml raises for io_stream), and only the
        # ap_fixed backends are handled below; anything else falls through untouched.
        if len(shape) < 2 or backend.lower() not in ('vivado', 'vitis'):
            return orig(name, shape, k, b, i, RND, SAT, backend)
        try:
            full = [np.broadcast_to(v[0], shape) for v in (k, b, i)]
        except Exception:
            return orig(name, shape, k, b, i, RND, SAT, backend)
        # Fold over the LARGEST leading block the mask is constant across, not just the
        # first axis. `combine`'s inputs are [400, 8, 4] and [400, 8, 16]: the mask varies
        # over the head axis but not over tokens, so the fold is 400 x 32 and 400 x 128,
        # which the naive shape[-1] split would have missed and silently not folded.
        best = 0
        for p_ in range(len(shape) - 1, 0, -1):
            o_ = int(np.prod(shape[:p_]))
            if o_ < 2:
                continue
            if all(bool((v.reshape(o_, -1) == v.reshape(o_, -1)[0]).all()) for v in full):
                best = p_
                break
        if not best:
            return orig(name, shape, k, b, i, RND, SAT, backend)
        outer = int(np.prod(shape[:best]))
        inner = int(np.prod(shape[best:]))
        Ks, Bs, Is = (v.reshape(outer, inner) for v in full)

        lines = []
        for c in range(inner):
            kk, bb, ii = int(Ks[0, c]), int(Bs[0, c]), int(Is[0, c])
            if bb == 0:
                lines.append(f'        out[o + {c}] = 0;')
            else:
                lines.append(f'        out[o + {c}] = {to_apfixed(kk, bb, ii, RND, SAT)}(inp[o + {c}]);')
        body = chr(10).join(lines)
        return f"""
template<typename input_t, typename output_t>
void {name}(input_t *inp, output_t *out) {{
    // WP-G P3: hls4ml emits one line per element ({outer * inner} of them here). The mask is
    // identical for every one of the {outer} tokens (verified), so it is emitted once and
    // iterated instead. Same arithmetic, {outer}x less generated source.
    {name}_tok: for (unsigned t = 0; t < {outer}; t++) {{
        #pragma HLS PIPELINE II=1
        const unsigned o = t * {inner};
{body}
    }}
}}
"""

    def to_apfixed(kk, bb, ii, RND, SAT):
        u = 'u' if kk == 0 else ''
        return f'ap_{u}fixed<{bb},{ii},AP_{RND},AP_{SAT}>'

    HP.generate_mask_fn = folded
    HP._wpg_mask_patched = True
    return ['fixed-point quantizer mask folded over the token axis']
