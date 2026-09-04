"""WP-G G4: minimal, documented runtime patches to hls4ml 1.3.0 for this model.

Both are hls4ml bugs, not model problems, and both are reported in writeup/G-quantization.md
rather than hidden. Nothing here changes what the model computes.

P1. `converters/keras_v3/hgq2/unary_lut.py` calls `.numpy()` on layer variables that still
    require grad. That is fine on the TF/JAX backends but raises on the torch backend.
    Avoided WITHOUT patching by setting `model.trainable = False` before conversion
    (kept here as a documented step, not a patch).

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
    return ['unary_lut rank-2 table assumption']
