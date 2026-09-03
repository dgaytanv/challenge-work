"""WP-G G1: Keras 3 re-implementation of the certified encoder, layer by layer.

Ports `PFPreProcessorMeanPt` + `PMAEncoder(num_layers=0)` (aliased to
`TransformerEncoder` on `submission-pma0-meanpt`) into Keras 3 so HGQ2 can quantize it.

Numerics that have to be got right, and are (see tests/test_port.py):
  * torch `nn.GELU()` is the exact erf form; Keras `keras.activations.gelu` defaults to
    `approximate=False`, which is the same. Never pass approximate=True.
  * torch `nn.LayerNorm` eps defaults to 1e-5; Keras `LayerNormalization` defaults to
    **1e-3**. Every LayerNorm here is built with epsilon=1e-5 explicitly.
  * torch `nn.Linear` holds the kernel transposed relative to Keras `Dense`
    (out,in) vs (in,out), so every kernel is transposed on load.
  * The masked softmax fills dead columns with -1e4 (`_MASK_FILL` in models.py), not
    -inf. exp(-1e4 - rowmax) underflows to exactly 0 in float32, so the choice of a
    finite fill is invisible in the output as long as at least one column survives -
    which `_survivors`' all-dead guard ensures.

Input contract. The model takes TWO inputs and does not re-derive the mask itself:
    x_feat : [B, N, 14]  preprocessed features (dead candidates are all-zero rows)
    keep   : [B, N]      1.0 for a surviving candidate, 0.0 otherwise
`keep` is computed on the host by `survivors_np`, which reproduces `models._survivors`
exactly (all-zero row OR the dataloader mask, then the all-dead guard). Splitting it out
keeps the arithmetic graph free of the boolean reduction and gives hls4ml a plain
numeric mask input, which is what the masked softmax wants anyway.
"""

import os

os.environ.setdefault('KERAS_BACKEND', 'torch')

import keras
import numpy as np
from keras import ops

EPS = 1e-4          # embedding.utils.data_utils.EPS
MASK_FILL = -1e4    # embedding.models._MASK_FILL
LN_EPS = 1e-5       # torch nn.LayerNorm default
BN_EPS = 1e-5       # torch nn.BatchNorm1d default

PDGIDS = (211, 11, 13, 22, 130, 1, 2)
NUM_CONT = 5        # pt, eta, phi, dxy, dxysig go through the batch norm
NUM_FEATURES = 14


# --------------------------------------------------------------------------------------
# Preprocessor (host side, numpy) - see the module docstring of quant/README for why the
# per-event mean-pt reduction is kept here rather than pushed into the Keras graph.
# --------------------------------------------------------------------------------------
def preproc_meanpt_np(x_raw: np.ndarray, bn: dict) -> np.ndarray:
    """`PFPreProcessorMeanPt.forward` in numpy, eval mode (frozen batch-norm statistics).

    x_raw : [B, N, >=7] = [pt, eta, phi, dxy, dxysig, is_pf, pdgId, ...]
    bn    : dict with weight/bias/running_mean/running_var, each [5]

    In eval mode the batch norm is a frozen affine map, so applying it to every row and
    then zeroing the invalid rows is identical to torch's `flat[valid] = bn(flat[valid])`
    followed by `where(valid, x_proc, 0)`. That equivalence is what makes the whole
    preprocessor elementwise apart from the single per-event mean.
    """
    x_raw = np.asarray(x_raw, dtype=np.float64)
    pt, eta, phi = x_raw[..., 0], x_raw[..., 1], x_raw[..., 2]
    dxy, dxysig, is_pf, pdgid = x_raw[..., 3], x_raw[..., 4], x_raw[..., 5], x_raw[..., 6]

    valid = pt > 0

    # pt -> log(pt_i / mean surviving pt). The only per-event (non-elementwise) step.
    ptv = np.where(valid, pt, 0.0)
    n = np.maximum(valid.sum(axis=-1, keepdims=True), 1).astype(np.float64)
    mean = np.maximum(ptv.sum(axis=-1, keepdims=True) / n, EPS)
    pt_feat = np.log(np.maximum(pt / mean, EPS))
    pt_feat = np.where(valid, pt_feat, 0.0)

    dxy_feat = np.where(valid, np.tanh(dxy), 0.0)

    pos = (pdgid == 11) | (pdgid == 13) | (pdgid == 211)
    neg = (pdgid == -11) | (pdgid == -13) | (pdgid == -211)
    charge = np.where(valid & pos, 1.0, np.where(valid & neg, -1.0, 0.0))

    onehot = (np.abs(pdgid)[..., None] == np.asarray(PDGIDS, dtype=np.float64)).astype(np.float64)

    cont = np.stack([pt_feat, eta, phi, dxy_feat, dxysig], axis=-1)
    scale = bn['weight'] / np.sqrt(bn['running_var'] + BN_EPS)
    shift = bn['bias'] - bn['running_mean'] * scale
    cont = cont * scale + shift

    disc = np.stack([charge, is_pf], axis=-1)
    out = np.concatenate([cont, disc, onehot], axis=-1)
    return np.where(valid[..., None], out, 0.0).astype(np.float32)


def survivors_np(x_feat: np.ndarray, mask=None) -> np.ndarray:
    """`models._survivors` in numpy -> float32 [B, N], 1.0 = surviving candidate."""
    dead = np.abs(x_feat).sum(axis=-1) == 0
    if mask is not None:
        dead = dead | np.asarray(mask)[:, 1:].astype(bool)
    keep = ~dead
    empty = ~keep.any(axis=1)
    if empty.any():
        keep = keep.copy()
        keep[empty, 0] = True
    return keep.astype(np.float32)


# --------------------------------------------------------------------------------------
# Keras encoder
# --------------------------------------------------------------------------------------
class MaskedPMAPooling(keras.layers.Layer):
    """`models.PMAPooling` for float parity: 4 learned seeds, 8 heads, masked softmax.

    Kept as one layer (rather than keras MultiHeadAttention) so the seed queries, the
    -1e4 mask fill and the 1/sqrt(head_dim) scale sit exactly where torch puts them.
    The HGQ2 version in quant/hgq_model.py mirrors this structure with Q* layers.
    """

    def __init__(self, embed_dim=128, num_heads=8, num_seeds=4, **kw):
        super().__init__(**kw)
        self.embed_dim, self.num_heads, self.num_seeds = embed_dim, num_heads, num_seeds
        self.head_dim = embed_dim // num_heads
        self.q_proj = keras.layers.Dense(embed_dim, name='q_proj')
        self.k_proj = keras.layers.Dense(embed_dim, name='k_proj')
        self.v_proj = keras.layers.Dense(embed_dim, name='v_proj')
        self.out_proj = keras.layers.Dense(embed_dim, name='out_proj')

    def build(self, input_shape):
        self.seeds = self.add_weight(
            shape=(1, self.num_seeds, self.embed_dim), name='seeds', initializer='zeros')
        for lyr in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            lyr.build((None, None, self.embed_dim))
        super().build(input_shape)

    def call(self, h, keep):
        B = ops.shape(h)[0]
        N = ops.shape(h)[1]
        H, S, D = self.num_heads, self.num_seeds, self.head_dim

        seeds = ops.broadcast_to(self.seeds, (B, S, self.embed_dim))
        q = ops.transpose(ops.reshape(self.q_proj(seeds), (B, S, H, D)), (0, 2, 1, 3))
        k = ops.transpose(ops.reshape(self.k_proj(h), (B, N, H, D)), (0, 2, 1, 3))
        v = ops.transpose(ops.reshape(self.v_proj(h), (B, N, H, D)), (0, 2, 1, 3))

        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) / np.sqrt(D).astype('float32')
        m = ops.reshape(keep, (B, 1, 1, N))
        scores = scores * m + (1.0 - m) * MASK_FILL
        attn = ops.softmax(scores, axis=-1)

        out = ops.matmul(attn, v)                                   # B,H,S,D
        out = ops.reshape(ops.transpose(out, (0, 2, 1, 3)), (B, S, self.embed_dim))
        return ops.reshape(self.out_proj(out), (B, S * self.embed_dim))


def build_keras_encoder(num_features=NUM_FEATURES, embed_size=128, latent_dim=6,
                        num_heads=8, num_seeds=4, name='pma0_meanpt'):
    """Functional Keras model: (x_feat [N,F], keep [N]) -> latent [latent_dim]."""
    x_in = keras.Input(shape=(None, num_features), name='x_feat')
    keep_in = keras.Input(shape=(None,), name='keep')

    h = keras.layers.Dense(embed_size, name='phi_0')(x_in)
    h = keras.layers.LayerNormalization(epsilon=LN_EPS, name='phi_1')(h)
    h = keras.layers.Activation(keras.activations.gelu, name='phi_2')(h)
    h = keras.layers.Dense(embed_size, name='phi_3')(h)
    h = keras.layers.LayerNormalization(epsilon=LN_EPS, name='phi_4')(h)
    h = keras.layers.Activation(keras.activations.gelu, name='phi_5')(h)
    h = keras.layers.Multiply(name='phi_mask')([h, keras.layers.Reshape((-1, 1))(keep_in)])

    pooled = MaskedPMAPooling(embed_size, num_heads, num_seeds, name='pma')(h, keep_in)
    pooled = keras.layers.LayerNormalization(epsilon=LN_EPS, name='norm_pooled')(pooled)
    latent = keras.layers.Dense(latent_dim, name='bottleneck')(pooled)
    return keras.Model([x_in, keep_in], latent, name=name)


def load_torch_weights(model: keras.Model, enc_sd: dict):
    """Copy the PyTorch encoder state_dict into the Keras model.

    torch Linear stores (out_features, in_features); Keras Dense stores (in, out).
    """
    t = lambda k: np.asarray(enc_sd[k].detach().cpu().numpy() if hasattr(enc_sd[k], 'detach') else enc_sd[k])

    def dense(layer, w, b):
        layer.set_weights([t(w).T, t(b)])

    def ln(layer, w, b):
        layer.set_weights([t(w), t(b)])

    dense(model.get_layer('phi_0'), 'phi.0.weight', 'phi.0.bias')
    ln(model.get_layer('phi_1'), 'phi.1.weight', 'phi.1.bias')
    dense(model.get_layer('phi_3'), 'phi.3.weight', 'phi.3.bias')
    ln(model.get_layer('phi_4'), 'phi.4.weight', 'phi.4.bias')

    pma = model.get_layer('pma')
    pma.seeds.assign(t('pma.seeds'))
    dense(pma.q_proj, 'pma.q_proj.weight', 'pma.q_proj.bias')
    dense(pma.k_proj, 'pma.k_proj.weight', 'pma.k_proj.bias')
    dense(pma.v_proj, 'pma.v_proj.weight', 'pma.v_proj.bias')
    dense(pma.out_proj, 'pma.out_proj.weight', 'pma.out_proj.bias')

    ln(model.get_layer('norm_pooled'), 'norm_pooled.weight', 'norm_pooled.bias')
    dense(model.get_layer('bottleneck'), 'bottleneck.weight', 'bottleneck.bias')
    return model
