"""WP-G G2: the certified encoder restructured into a hardware shape, in HGQ2.

Two exact algebraic restructurings are applied first. Neither changes what the model
computes (both are verified to float32 round-off in tests/test_restructure.py), so they
cost nothing on the bench and are done before any quantization.

R1. The seed queries are CONSTANT parameters (num_layers=0, so nothing per-event feeds
    them), therefore `q = q_proj(seeds)` is a compile-time constant and the score
    computation collapses:

        scores[b,h,s,n] = <q[h,s,:], k[b,h,n,:]> / sqrt(D)
                        = A[(h,s),:] . h[b,n] + c[(h,s)]
        A[(h,s),:] = sum_d q_flat[s, h*D+d] * W_k[h*D+d, :] / sqrt(D)
        c[(h,s)]   = sum_d q_flat[s, h*D+d] * b_k[h*D+d]   / sqrt(D)

    So `q_proj` (16,512 params) and `k_proj` (16,512 params) fuse into ONE Dense
    128 -> 32, i.e. 4,128 params. 28,896 parameters and their multiplies disappear
    exactly. This is the single biggest hardware win in the whole package and it is free.

R2. `h = phi(x) * keep` is redundant. Dead tokens are removed by the masked softmax
    (attn = 0 exactly), and out = sum_n attn[n] * v[n], so v[dead] never reaches the
    output whatever it is. Dropping the multiply removes a 128-wide elementwise
    multiplier per token.

Masking. `QSoftmax` does support a mask, but **hls4ml 1.3.0 cannot convert it**:
`hls4ml/converters/keras_v3/hgq2/softmax.py` raises `NotImplementedError('Masked softmax
not supported yet')` for a 2-input softmax. So the mask is applied as an ADDITIVE bias on
the scores before a plain 1-input softmax:

        scores' = scores + mask_add,   mask_add = 0 (alive) or -MASK_BIG (dead)

MASK_BIG must be checked, not assumed, and `assert_mask_margin` below does it (planner
ruling 2, 23:47). The reason it is not obvious: R2 leaves a dead token carrying phi(0),
NOT zero, so its score is A.phi(0) + c, a fixed 32-vector measured at [-2.79, +14.03] on
the certified model. In one of the 32 channels the best LIVE score is only +0.35 above
the dead score, so the entire safety margin is the bias itself. At MASK_BIG = 32 the worst
gap was -32.35: one dead token carries exp(-32.35) = 8.9e-15 of the best live token's
weight, and 400 of them carry 3.6e-12 - invisible in float32, and exactly 0 once the exp
table quantizes it. That is safe but it is only 32 e-folds, and nothing in QAT constrains
the scores from growing. MASK_BIG is therefore 64, which costs ONE extra integer bit of
score range (7 -> 8, since the range becomes [min_live - 64, max_live] = [-84.5, +59.1])
and doubles the margin. Re-measure with `assert_mask_margin` after every training run. `mask_add` is a model INPUT: the validity
of a candidate is known to the data path, it is not something the network should infer
(and it cannot - 61% of real candidates in this dataset carry pdgId 0, so the one-hot
block is all-zero for them and does not identify a live candidate).

Normalisation. HGQ2 has NO quantized LayerNorm (grep the package: zero hits). Two builds:
  * norm='ln'     - keras LayerNormalization left in float. CONTROL: measures the cost of
                    quantization alone. Not convertible by hls4ml.
  * norm='affine' - each LayerNorm replaced by a per-channel affine map calibrated on data
                    and then FOLDED into the preceding Dense, so no norm layer survives at
                    all. This is the primary hardware path (planner, 23:19).
"""

import os

os.environ.setdefault('KERAS_BACKEND', 'torch')

import keras
import numpy as np
from keras import ops

MASK_BIG = 64.0
LN_EPS = 1e-5
NUM_FEATURES = 14
EMBED = 128
HEADS = 8
SEEDS = 4
HEAD_DIM = EMBED // HEADS
LATENT = 6


# ======================================================================================
# R1: fuse the constant seed query into the key projection
# ======================================================================================
def fuse_qk(enc_sd) -> tuple[np.ndarray, np.ndarray]:
    """-> A [H*S, E] and c [H*S], channel index h*S + s (matches the attn reshape)."""
    g = lambda k: np.asarray(enc_sd[k].detach().cpu().numpy() if hasattr(enc_sd[k], 'detach') else enc_sd[k],
                             dtype=np.float64)
    seeds = g('pma.seeds')[0]                              # [S, E]
    Wq, bq = g('pma.q_proj.weight'), g('pma.q_proj.bias')  # torch (out, in)
    Wk, bk = g('pma.k_proj.weight'), g('pma.k_proj.bias')

    q_flat = seeds @ Wq.T + bq                             # [S, E]
    scale = 1.0 / np.sqrt(HEAD_DIM)

    A = np.zeros((HEADS * SEEDS, EMBED))
    c = np.zeros((HEADS * SEEDS,))
    for h in range(HEADS):
        sl = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
        for s in range(SEEDS):
            qhs = q_flat[s, sl]                            # [D]
            A[h * SEEDS + s] = (qhs @ Wk[sl, :]) * scale   # [E]
            c[h * SEEDS + s] = (qhs @ bk[sl]) * scale
    return A, c


# ======================================================================================
# LayerNorm -> per-channel affine, calibrated then folded into the preceding Dense
# ======================================================================================
def ln_to_affine(gamma, beta, mean, var, eps=LN_EPS):
    """LayerNorm(gamma, beta) replaced by the population affine y = scale*x + shift."""
    scale = gamma / np.sqrt(var + eps)
    return scale, beta - mean * scale


def fold_affine_into_dense_out(W, b, scale, shift):
    """Affine applied AFTER the Dense (phi_0 -> LN -> act). W [in,out], scale/shift [out]."""
    return W * scale[None, :], b * scale + shift


def fold_affine_into_dense_in(W, b, scale, shift):
    """Affine applied BEFORE the Dense (LN -> bottleneck). W [in,out], scale/shift [in].

    y = W^T (scale*x + shift) + b = (diag(scale) W)^T x + (shift^T W + b)
    """
    return W * scale[:, None], b + shift @ W


# ======================================================================================
# Model
# ======================================================================================
def build_model(norm='bn', act='gelu', quantized=False, n_tokens=400,
                beta0=1e-5, name=None):
    """(x_feat [N,14], mask_add [N,32]) -> latent [6].

    norm      : 'bn' (QBatchNormalization, hardware path) or 'ln' (float LayerNorm control)
    act       : 'gelu' or 'relu'
    quantized : HGQ2 layers with EBOPs regularisation, or the plain float twin
    n_tokens  : MUST be static. `QDense.build` computes its parallelization factor from
                `prod(input_shape[1:-1])`, and hls4ml's einsum/reshape handlers assert
                fully-known shapes. Train at 200 (training events), evaluate at 400.
                The weights are N-independent - every quantizer on a token-axis tensor
                is homogeneous over (batch, token) - so the same weights load into either.
    """
    from hgq.config import LayerConfigScope, QuantizerConfigScope
    from hgq.layers import QAdd, QBatchNormalization, QDense, QEinsum, QSoftmax, QUnaryFunctionLUT

    act_fn = keras.activations.gelu if act == 'gelu' else keras.activations.relu
    name = name or f'pma0_{norm}_{act}_{"q" if quantized else "f"}_{n_tokens}'
    N = n_tokens

    def _build():
        x = keras.Input(shape=(N, NUM_FEATURES), name='x_feat')
        m = keras.Input(shape=(N, HEADS * SEEDS), name='mask_add')

        if quantized:
            D = lambda u, n: QDense(u, name=n)
            A_ = lambda n: QUnaryFunctionLUT(act_fn, name=n)
            NRM = lambda n: QBatchNormalization(epsilon=LN_EPS, name=n)
            ADD, SM = QAdd, lambda n: QSoftmax(axis=1, name=n)
        else:
            D = lambda u, n: keras.layers.Dense(u, name=n)
            A_ = lambda n: keras.layers.Activation(act_fn, name=n)
            NRM = lambda n: keras.layers.BatchNormalization(epsilon=LN_EPS, name=n)
            ADD, SM = keras.layers.Add, lambda n: keras.layers.Softmax(axis=1, name=n)

        def norm_layer(t, tag):
            if norm == 'ln':
                return keras.layers.LayerNormalization(epsilon=LN_EPS, name=tag)(t)
            return NRM(tag)(t)

        h = D(EMBED, 'phi_0')(x)
        h = norm_layer(h, 'nrm0')
        h = A_('phi_act0')(h)
        h = D(EMBED, 'phi_1')(h)
        h = norm_layer(h, 'nrm1')
        h = A_('phi_act1')(h)

        scores = D(HEADS * SEEDS, 'score')(h)                        # R1: fused q.k
        scores = ADD(name='mask_add_op')([scores, m])
        attn = SM('softmax')(scores)                                 # over the token axis

        v = D(EMBED, 'v')(h)
        a4 = keras.layers.Reshape((N, HEADS, SEEDS), name='attn_rs')(attn)
        v4 = keras.layers.Reshape((N, HEADS, HEAD_DIM), name='v_rs')(v)
        if quantized:
            pooled = QEinsum('bnhs,bnhd->bshd', name='combine')([a4, v4])
        else:
            pooled = keras.layers.Lambda(
                lambda t: ops.einsum('bnhs,bnhd->bshd', t[0], t[1]),
                output_shape=(SEEDS, HEADS, HEAD_DIM), name='combine')([a4, v4])
        pooled = keras.layers.Reshape((SEEDS, EMBED), name='pool_rs')(pooled)
        pooled = D(EMBED, 'out_proj')(pooled)
        pooled = keras.layers.Reshape((SEEDS * EMBED,), name='flatten')(pooled)
        # The tail is 2-D [B, 512]: axis 1 is the CHANNEL axis here, not a token axis, so
        # the outer homogeneous_axis=(0,1) would collapse all 512 channels onto one shared
        # bitwidth and throw away exactly the per-channel granularity HGQ2 exists for.
        # Re-open the scope with homogeneous_axis=(0,) for these two layers only.
        if quantized:
            from hgq.config import QuantizerConfigScope as _QCS
            with _QCS(q_type='kif', place='datalane', homogeneous_axis=(0,)), \
                 _QCS(q_type='kbi', place='datalane', homogeneous_axis=(0,)):
                pooled = norm_layer(pooled, 'norm_pooled')
                latent = D(LATENT, 'bottleneck')(pooled)
        else:
            pooled = norm_layer(pooled, 'norm_pooled')
            latent = D(LATENT, 'bottleneck')(pooled)
        return keras.Model([x, m], latent, name=name)

    if not quantized:
        return _build()
    # Datalane bitwidths must be per-CHANNEL only: the token axis is 200 in training and
    # 400 at eval, so a per-token bitwidth would neither transfer nor mean anything in
    # hardware (one datapath processes every token). homogeneous_axis=(0,1)=(batch,token).
    with QuantizerConfigScope(q_type='kif', place='datalane', homogeneous_axis=(0, 1)), \
         QuantizerConfigScope(q_type='kbi', place='datalane', homogeneous_axis=(0, 1)), \
         LayerConfigScope(beta0=beta0, enable_ebops=True):
        return _build()


# ======================================================================================
# Weight transfer: certified torch checkpoint -> restructured model
# ======================================================================================
def load_weights(model, enc_sd, ln_stats=None, norm='bn'):
    """Initialise the restructured model from the certified encoder.

    For norm='bn' the BatchNormalization layers are seeded so that in INFERENCE mode they
    reproduce the calibrated affine replacement of each LayerNorm (gamma/beta copied from
    the LayerNorm, moving mean/variance set to the measured statistics of that norm's
    input). In TRAINING mode they use batch statistics instead, which is what makes the
    swap trainable at all: the frozen-affine initialisation alone is off by ~6800 on a
    latent of scale 12, because LayerNorm is per-sample and no population affine can
    imitate it. `ln_stats` comes from `collect_ln_stats`.
    """
    g = lambda k: np.asarray(enc_sd[k].detach().cpu().numpy() if hasattr(enc_sd[k], 'detach') else enc_sd[k],
                             dtype=np.float64)
    TAGS = {'nrm0': 'phi.1', 'nrm1': 'phi.4', 'norm_pooled': 'norm_pooled'}

    def dense(lyr, W, b):
        """Assign kernel/bias directly. A QDense also owns its quantizers' bitwidth
        variables, beta and ebops, so `set_weights` (which wants the full list) fails."""
        L = model.get_layer(lyr)
        L._kernel.assign(W.astype('float32'))
        L.bias.assign(b.astype('float32'))

    def normw(lyr, gamma, beta, mean=None, var=None):
        """QBatchNormalization renames gamma/beta to bn_gamma/bn_beta - its own `beta`
        property is the EBOPs regularisation strength, not the norm shift."""
        L = model.get_layer(lyr)
        if mean is None:                                   # LayerNormalization
            L.gamma.assign(gamma); L.beta.assign(beta)
            return
        (L.bn_gamma if hasattr(L, 'bn_gamma') else L.gamma).assign(gamma)
        (L.bn_beta if hasattr(L, 'bn_beta') else L.beta).assign(beta)
        L.moving_mean.assign(mean)
        L.moving_variance.assign(var)

    dense('phi_0', g('phi.0.weight').T, g('phi.0.bias'))
    dense('phi_1', g('phi.3.weight').T, g('phi.3.bias'))
    A, c = fuse_qk(enc_sd)                                   # R1
    dense('score', A.T, c)
    dense('v', g('pma.v_proj.weight').T, g('pma.v_proj.bias'))
    dense('out_proj', g('pma.out_proj.weight').T, g('pma.out_proj.bias'))
    dense('bottleneck', g('bottleneck.weight').T, g('bottleneck.bias'))

    for lyr, tag in TAGS.items():
        gamma, beta = g(f'{tag}.weight').astype('float32'), g(f'{tag}.bias').astype('float32')
        if norm == 'ln':
            normw(lyr, gamma, beta)
        else:
            mean, var = ln_stats[tag]
            normw(lyr, gamma, beta, mean.astype('float32'), var.astype('float32'))
    return model


def assert_mask_margin(model, x_feat, keep, mask_add, n_tokens, batch=500, min_efolds=20.0):
    """Planner ruling 2: verify the additive mask really does suppress dead tokens.

    A dead token carries phi(0), so its score is a fixed vector s_dead = A.phi(0) + c.
    The requirement is that ALL dead tokens together stay negligible against the live
    softmax mass, per (head, seed) channel:

        gap = (max_c s_dead[c] - MASK_BIG) - max over live tokens of score[.., c]
        total dead weight / best live weight  <=  n_tokens * exp(gap)

    Returns a dict; raises AssertionError if the margin is under `min_efolds` e-folds.
    """
    probe = keras.Model(model.inputs, model.get_layer('score').output)
    z = np.zeros((1, n_tokens, NUM_FEATURES), np.float32)
    s_dead = probe([z, mask_add_from_keep(np.zeros((1, n_tokens), np.float32))],
                   training=False)
    s_dead = (s_dead.detach().cpu().numpy() if hasattr(s_dead, 'detach') else np.asarray(s_dead))[0, 0]

    ch_max = np.full(HEADS * SEEDS, -np.inf)
    for i in range(0, len(x_feat), batch):
        s = probe([x_feat[i:i + batch], mask_add[i:i + batch]], training=False)
        s = s.detach().cpu().numpy() if hasattr(s, 'detach') else np.asarray(s)
        kb = keep[i:i + batch].astype(bool)
        for c in range(HEADS * SEEDS):
            v = s[..., c][kb]
            if v.size:
                ch_max[c] = max(ch_max[c], float(v.max()))

    gap = (s_dead - MASK_BIG) - ch_max
    worst = float(gap.max())
    efolds = -worst - np.log(n_tokens)
    out = dict(mask_big=MASK_BIG, score_dead_min=float(s_dead.min()), score_dead_max=float(s_dead.max()),
               live_score_max=float(ch_max.max()), worst_gap=worst,
               worst_channel=int(gap.argmax()), efolds_margin=float(efolds),
               total_dead_weight_ratio=float(n_tokens * np.exp(worst)))
    assert efolds >= min_efolds, (
        f'mask margin too thin: {efolds:.1f} e-folds after allowing for {n_tokens} dead '
        f'tokens (need >= {min_efolds}). Raise MASK_BIG. {out}')
    return out


def mask_add_from_keep(keep: np.ndarray) -> np.ndarray:
    """keep [B, N] float 1/0 -> mask_add [B, N, H*S] of 0 / -MASK_BIG."""
    return ((keep - 1.0) * MASK_BIG)[..., None].repeat(HEADS * SEEDS, axis=-1).astype('float32')


def collect_ln_stats(ln_model, x_feat, mask_add, keep, batch=250):
    """Population statistics that best imitate each LayerNorm, for the BatchNorm seed.

    The subtlety that matters: **LayerNorm normalises per TOKEN across channels**, so the
    quantity it divides by is the spread of the 128 (or 512) channels within one token,
    not the spread of one channel across tokens. Seeding a BatchNorm with per-CHANNEL
    statistics therefore uses the wrong denominator whenever the channel MEANS are spread
    out. Measured on the certified model:

        norm            per-channel var   per-token var   ratio   spread of channel means
        phi.1                    0.4217          0.4754    1.13                    0.2334
        phi.4                    0.1802          0.3760    2.09                    0.4438
        norm_pooled              0.0349          0.7843   22.50                    0.8657

    and the resulting initialisation error on the latent is 5165 (per-channel) vs 879
    (per-token), against a latent scale of 10.8. So this returns the per-TOKEN statistics
    (E[mu_row], E[var_row]) broadcast across channels. Fine-tuning does the rest; the
    remaining 879 is genuine per-token variation that no population affine can absorb,
    which is the cost the LayerNorm -> BatchNorm swap is measured for.

    The two per-candidate norms are calibrated over SURVIVING candidates only: dead rows
    are all-zero and are a severity-dependent fraction of the tokens, so including them
    would make the calibration depend on how degraded the calibration set happened to be.

    Caveat carried into the report: at TRAINING time keras BatchNormalization cannot
    exclude dead rows from its batch statistics, so the running statistics it learns are
    taken over live and dead tokens together. That cost shows up in G3, not assumed away.
    """
    taps = {'phi.1': 'phi_0', 'phi.4': 'phi_1', 'norm_pooled': 'flatten'}
    probe = keras.Model(ln_model.inputs, {t: ln_model.get_layer(l).output for t, l in taps.items()})

    acc = {t: [0.0, 0.0, 0, 0] for t in taps}   # sum(mu_row), sum(var_row), count, n_channels
    for i in range(0, len(x_feat), batch):
        out = probe([x_feat[i:i + batch], mask_add[i:i + batch]], training=False)
        kb = keep[i:i + batch].astype(bool)
        for t in taps:
            a = out[t]
            a = a.detach().cpu().numpy() if hasattr(a, 'detach') else np.asarray(a)
            a = a.astype(np.float64)
            sel = a[kb] if a.ndim == 3 else a            # [M, C]
            acc[t][0] += sel.mean(axis=1).sum()
            acc[t][1] += sel.var(axis=1).sum()
            acc[t][2] += sel.shape[0]
            acc[t][3] = sel.shape[1]

    stats = {}
    for t, (smu, svar, n, C) in acc.items():
        stats[t] = (np.full(C, smu / n), np.full(C, max(svar / n, 1e-12)))
    return stats


def transfer_weights(src, dst):
    """Copy kernel/bias and normalisation parameters between two builds of this model.

    Used to seed the quantized student from the trained float-BatchNorm student, so the
    LayerNorm->BatchNorm cost and the quantization cost are measured as separate stages
    rather than as one blended number. Quantizer bitwidth variables are left at their
    initialisers - they are the thing QAT is meant to learn.
    """
    n = 0
    for L in dst.layers:
        try:
            S = src.get_layer(L.name)
        except ValueError:
            continue
        if hasattr(L, '_kernel') and hasattr(S, '_kernel'):
            L._kernel.assign(np.asarray(S._kernel))
            if getattr(L, 'bias', None) is not None:
                L.bias.assign(np.asarray(S.bias))
            n += 1
        elif hasattr(L, 'moving_mean') and hasattr(S, 'moving_mean'):
            (L.bn_gamma if hasattr(L, 'bn_gamma') else L.gamma).assign(
                np.asarray(S.bn_gamma if hasattr(S, 'bn_gamma') else S.gamma))
            (L.bn_beta if hasattr(L, 'bn_beta') else L.beta).assign(
                np.asarray(S.bn_beta if hasattr(S, 'bn_beta') else S.beta))
            L.moving_mean.assign(np.asarray(S.moving_mean))
            L.moving_variance.assign(np.asarray(S.moving_variance))
            n += 1
    return n
