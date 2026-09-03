from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, pairwise: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0
        self.pairwise = pairwise

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.bias_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, num_heads)
        )

    def forward(self, x: torch.Tensor, pairwise_feats: Union[None, torch.Tensor] = None, key_padding_mask: Union[None, torch.Tensor] = None):
        B, N, E = x.shape

        Q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # B,H,N,head_dim
        K = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # B,H,N,head_dim
        V = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # B,H,N,head_dim

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # B,H,N,N

        if self.pairwise: # add pairwise bias only if enabled
            if pairwise_feats is None:
                raise ValueError("pairwise_feats must be provided when pairwise is True")
            bias_logits = self.bias_mlp(pairwise_feats)  # (B, N, N, H)
            bias_logits = bias_logits.permute(0, 3, 1, 2)  # (B, H, N, N)
            scores = scores + bias_logits
        
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # B,1,1,N
            scores = scores.masked_fill(mask == True, float('-inf'))

        attn = torch.softmax(scores, dim=-1)  # B,H,N,N
        out = torch.matmul(attn, V)  # B,H,N,head_dim

        out = out.transpose(1, 2).contiguous().view(B, N, E)
        out = self.out_proj(out)
        return out

class LinearAttentionLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, linear_dim: int, num_tokens: int, pairwise: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0
        self.pairwise = pairwise

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Linformer projection matrices
        self.f_proj = nn.Linear(num_tokens, linear_dim, bias=False)
        self.e_proj = nn.Linear(num_tokens, linear_dim, bias=False)

        self.bias_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, num_heads)
        )

    def forward(self, x: torch.Tensor, pairwise_feats: Union[None, torch.Tensor] = None, key_padding_mask: Union[None, torch.Tensor] = None):
        B, N, E = x.shape

        Q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # B,H,N,head_dim
        K = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # B,H,N,head_dim
        V = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # B,H,N,head_dim
        
        if key_padding_mask is not None:
            expanded_mask = key_padding_mask.unsqueeze(1).unsqueeze(-1).expand(B, 1, N, self.head_dim)  # B,1,N,head_dim
            K = K.masked_fill(expanded_mask, 0.0)
            V = V.masked_fill(expanded_mask, 0.0)

        K_prime = self.e_proj(K.transpose(2, 3)).transpose(2, 3) # B,H,linear_dim,head_dim
        V_prime = self.f_proj(V.transpose(2, 3)).transpose(2, 3) # B,H,linear_dim,head_dim
        
        scores = torch.matmul(Q, K_prime.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,H,N,head_dim)x(B,H,head_dim,linear_dim) => B,H,N,linear_dim
        
        if self.pairwise: # add pairwise bias only if enabled
            if pairwise_feats is None:
                raise ValueError("pairwise_feats must be provided when pairwise is True")
            bias_logits = self.bias_mlp(pairwise_feats)  # (B, N, N, H)
            bias_logits = bias_logits.permute(0, 3, 1, 2)  # (B, H, N, N)
            bias_logits_prime = self.e_proj(bias_logits) # NEW. B,H,N,linear_dim

            scores = scores + bias_logits_prime

        attn = torch.softmax(scores, dim=-1)  # B,H,N,linear_dim
        out = torch.matmul(attn, V_prime)  # (B,H,N,linear_dim)x(B,H,linear_dim,head_dim) => B,H,N,head_dim

        out = out.transpose(1, 2).contiguous().view(B, N, E)
        out = self.out_proj(out)
        return out

class TransformerEncoderBlock(nn.Module):
    def __init__(
            self, 
            embed_dim: int, 
            num_heads: int, 
            dim_feedforward: int = 2048, 
            dropout: float = 0.1, 
            linear_dim: Union[int, None] = None, 
            num_tokens: Union[int, None] = None,
            pairwise: bool = False
        ):
        super().__init__()
        if linear_dim is not None and num_tokens is None:
            raise ValueError("num_tokens must be provided if linear_dim is specified")
        self.self_attn = AttentionLayer(embed_dim, num_heads, pairwise) if linear_dim is None else LinearAttentionLayer(embed_dim, num_heads, linear_dim, num_tokens, pairwise)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(
            self, 
            src: torch.Tensor, 
            pairwise_feats: Union[None, torch.Tensor] = None, 
            src_key_padding_mask: Union[None, torch.Tensor] = None
        ):
        src2 = self.self_attn(src, pairwise_feats, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerEncoder(nn.Module):
    """
    Transformer encoder dimension parameters:
    - num_features: input feature dimension
    - embed_size: dimension of the token embeddings and the CLS token
    - latent_dim: dimension of the output latent representation (after bottleneck)
    - num_heads: number of attention heads per layer
    - num_layers: number of transformer layers
    - linear_dim: if specified, use linear attention with this projection dimension
    - num_tokens: if using linear attention, the maximum number of tokens (including CLS) for projection
    - pairwise: whether to use pairwise bias in attention layers (requires pairwise_feats input); resource intensive!
    """
    def __init__(
            self, 
            num_features: int, 
            embed_size: int, 
            latent_dim: int, 
            num_heads: int = 8, 
            num_layers: int = 4,
            linear_dim: Union[int, None] = None,
            num_tokens: Union[int, None] = None,
            pairwise: bool = False,
        ):
        super().__init__()
        self.input_proj = nn.Linear(num_features, embed_size)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_size, 
                    num_heads, 
                    linear_dim=linear_dim, 
                    num_tokens=num_tokens+1 if num_tokens is not None else None,
                    pairwise=pairwise
                ) for _ in range(num_layers)
            ]
        )
        self.norm_cls_embedding = nn.LayerNorm(embed_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))
        self.bottleneck = nn.Linear(embed_size, latent_dim)
        self.pairwise = pairwise # bool

    def forward(self, x: torch.Tensor, pairwise_feats: Union[None, torch.Tensor] = None, mask: Union[None, torch.Tensor] = None):
        B, N, F = x.shape
        # Rows zeroed by degradation (or padding) arrive as all-zero feature rows but the
        # dataloader's mask was built BEFORE degradation. Re-derive the mask from the input so
        # dead candidates are excluded from attention instead of entering as constant tokens.
        dead = (x.abs().sum(dim=-1) == 0)  # [B, N]
        if mask is None:
            mask = torch.zeros(B, N + 1, dtype=torch.bool, device=x.device)
        mask = mask.clone()
        mask[:, 1:] |= dead
        x = self.input_proj(x) # [B, N, E]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1) 
        device = x.device

        if self.pairwise != (pairwise_feats is not None):
            raise ValueError(
                f"Pairwise mode is {self.pairwise}, but pairwise_feats was "
                f"{'provided' if pairwise_feats is not None else 'not provided'}"
            )
        
        pairwise_bias = None
        if self.pairwise:
            N = x.size(1)
            B = x.size(0)
            pairwise_bias = torch.zeros(B, N, N, 1, device=device)
            pairwise_bias[:, 1:, 1:, 0] = pairwise_feats[..., 0]
            
        for layer in self.layers:
            x = layer(x, pairwise_bias, src_key_padding_mask=mask)

        cls_embedding = x[:, 0, :] # CLS token embedding
        latent = self.bottleneck(self.norm_cls_embedding(cls_embedding))
        return latent
    
class Projector(nn.Module):
    def __init__(self, input_dim, proj_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, z):
        z = self.net(z)
        z = F.normalize(z, dim=-1)
        return z
    
class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1
    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)

class MixerBlock(nn.Module):
    def __init__(self, num_tokens, hidden_dim, token_mlp_dim, channel_mlp_dim, dropout=0.1):
        super().__init__()
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.token_mlp_1 = nn.Linear(num_tokens, token_mlp_dim)
        self.token_act = nn.ReLU()
        self.token_mlp_2 = nn.Linear(token_mlp_dim, num_tokens)
        self.token_dropout = nn.Dropout(dropout)

        self.channel_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, channel_mlp_dim),
            nn.ReLU(),
            nn.Linear(channel_mlp_dim, hidden_dim)
        )
        self.channel_dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Token-mixing
        y = self.token_norm(x)
        y = y.transpose(1, 2)
        y = self.token_mlp_1(y)
        y = self.token_act(y)
        y = self.token_mlp_2(y)
        y = self.token_dropout(y)
        y = y.transpose(1, 2)
        x = x + y

        # Channel-mixing
        y = self.channel_mlp(x)
        y = self.channel_dropout(y)
        x = x + y
        return x

class MLPMixer(nn.Module):
    def __init__(self, num_particles, num_features, num_classes, num_blocks, token_mlp_dim, channel_mlp_dim, dropout=0.1):
        super().__init__()
        hidden_dim = num_features
        self.mixer_blocks = nn.ModuleList([
            MixerBlock(num_particles, hidden_dim, token_mlp_dim, channel_mlp_dim, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.avgpool = nn.AvgPool1d(num_particles)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        for block in self.mixer_blocks:
            x = block(x)
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x

class EvalMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class RegressionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# ---------------------------------------------------------------------------
# WP-D: permutation-symmetric set encoders.
#
# Both take the SAME constructor signature and forward contract as
# TransformerEncoder, so eval.py / bench_eval.py can build them unchanged:
#   forward(x, pairwise_feats=None, mask=None) -> latent [B, latent_dim]
# with mask [B, N+1] (CLS slot first, True = padded). Dead rows (all-zero
# features, left behind by the degradation) are OR'd into the mask, exactly as
# TransformerEncoder does, so pooling sees only surviving candidates.
# ---------------------------------------------------------------------------

_MASK_FILL = -1e4  # finite so it survives fp16 autocast


def _survivors(x: torch.Tensor, mask: Union[None, torch.Tensor]) -> torch.Tensor:
    """[B, N] bool, True where the candidate is real and not zeroed."""
    B, N, _ = x.shape
    dead = (x.abs().sum(dim=-1) == 0)
    if mask is not None:
        dead = dead | mask[:, 1:].bool()
    keep = ~dead
    # Guard the all-dead event: keep one slot so pooling and attention stay finite.
    empty = ~keep.any(dim=1)
    if empty.any():
        keep = keep.clone()
        keep[empty, 0] = True
    return keep


def _masked_pool(h: torch.Tensor, keep: torch.Tensor, mode: str = "mean+max"):
    """Masked pooling over the token axis. Always divides by the surviving count.

    ``mean`` is smooth under deletion: dropping a candidate removes its term and
    rescales by the new count. ``max`` is not - if the arg-max candidate falls inside
    the dead region the pooled value jumps to the runner-up, which is exactly the
    discontinuity the metric punishes. ``lse`` (log-sum-exp, count-normalised) is the
    smooth stand-in for max. Default stays ``mean+max``: that is what d-deepsets-clean
    measured, so it remains the benchmarked reference until an arm says otherwise.
    """
    k = keep.unsqueeze(-1).to(h.dtype)
    count = k.sum(dim=1).clamp(min=1.0)                       # [B, 1]
    mean = (h * k).sum(dim=1) / count
    if mode == "mean":
        return mean, count
    if mode == "mean+max":
        mx = h.masked_fill(~keep.unsqueeze(-1), _MASK_FILL).max(dim=1).values
        return torch.cat([mean, mx], dim=-1), count
    if mode == "mean+lse":
        z = h.masked_fill(~keep.unsqueeze(-1), _MASK_FILL)
        lse = torch.logsumexp(z.float(), dim=1) - torch.log(count.float())
        return torch.cat([mean, lse.to(h.dtype)], dim=-1), count
    raise ValueError(f"unknown pooling mode: {mode!r}")


_POOL_WIDTH = {"mean": 1, "mean+max": 2, "mean+lse": 2}


def _phi_mlp(num_features: int, embed_size: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(num_features, embed_size),
        nn.LayerNorm(embed_size),
        nn.GELU(),
        nn.Linear(embed_size, embed_size),
        nn.LayerNorm(embed_size),
        nn.GELU(),
    )


class DeepSetsEncoder(nn.Module):
    """Deep Sets: per-particle MLP, masked mean+max pooling, MLP head.

    Deleting candidates changes the pooled summary only through the terms they
    contributed, so the latent moves smoothly as a dead region grows. O(N).
    `count_feature` appends log(surviving count) to the pooled vector.
    """

    def __init__(
            self,
            num_features: int,
            embed_size: int,
            latent_dim: int,
            num_heads: int = 8,
            num_layers: int = 4,
            linear_dim: Union[int, None] = None,
            num_tokens: Union[int, None] = None,
            pairwise: bool = False,
            count_feature: bool = False,
            pooling: str = "mean+max",
        ):
        super().__init__()
        self.pairwise = pairwise
        self.count_feature = count_feature
        self.pooling = pooling
        self.phi = _phi_mlp(num_features, embed_size)
        pooled_dim = _POOL_WIDTH[pooling] * embed_size + (1 if count_feature else 0)
        self.norm_pooled = nn.LayerNorm(pooled_dim)
        self.rho = nn.Sequential(
            nn.Linear(pooled_dim, embed_size),
            nn.GELU(),
            nn.Linear(embed_size, latent_dim),
        )

    def forward(self, x: torch.Tensor, pairwise_feats: Union[None, torch.Tensor] = None, mask: Union[None, torch.Tensor] = None):
        keep = _survivors(x, mask)
        h = self.phi(x) * keep.unsqueeze(-1).to(x.dtype)
        pooled, count = _masked_pool(h, keep, self.pooling)
        if self.count_feature:
            pooled = torch.cat([pooled, torch.log(count)], dim=-1)
        return self.rho(self.norm_pooled(pooled))


class PMAPooling(nn.Module):
    """Pooling by multihead attention (Set Transformer): k learned seed queries."""

    def __init__(self, embed_dim: int, num_heads: int, num_seeds: int = 4):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_seeds = num_seeds
        self.seeds = nn.Parameter(torch.randn(1, num_seeds, embed_dim) * embed_dim ** -0.5)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, h: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        B, N, E = h.shape
        S = self.num_seeds
        q = self.q_proj(self.seeds.expand(B, -1, -1)).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)   # B,H,S,N
        scores = scores.masked_fill(~keep.view(B, 1, 1, N), _MASK_FILL)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, E)
        return self.out_proj(out).reshape(B, S * E)


class PMAEncoder(nn.Module):
    """Per-particle MLP, optional masked self-attention blocks, then PMA readout.

    num_layers=0 is Deep Sets with an attention-weighted readout; num_layers>0 is
    the transformer with the CLS token replaced by k learned seed queries, so no
    single token carries the whole summary.
    """

    def __init__(
            self,
            num_features: int,
            embed_size: int,
            latent_dim: int,
            num_heads: int = 8,
            num_layers: int = 4,
            linear_dim: Union[int, None] = None,
            num_tokens: Union[int, None] = None,
            pairwise: bool = False,
            num_seeds: int = 4,
        ):
        super().__init__()
        self.pairwise = pairwise
        self.phi = _phi_mlp(num_features, embed_size)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                embed_size,
                num_heads,
                linear_dim=linear_dim,
                num_tokens=num_tokens if num_tokens is not None else None,
                pairwise=False,
            ) for _ in range(num_layers)
        ])
        self.pma = PMAPooling(embed_size, num_heads, num_seeds=num_seeds)
        self.norm_pooled = nn.LayerNorm(num_seeds * embed_size)
        self.bottleneck = nn.Linear(num_seeds * embed_size, latent_dim)

    def forward(self, x: torch.Tensor, pairwise_feats: Union[None, torch.Tensor] = None, mask: Union[None, torch.Tensor] = None):
        keep = _survivors(x, mask)
        h = self.phi(x) * keep.unsqueeze(-1).to(x.dtype)
        for layer in self.layers:
            h = layer(h, None, src_key_padding_mask=~keep)
            h = h * keep.unsqueeze(-1).to(h.dtype)
        pooled = self.pma(h, keep)
        return self.bottleneck(self.norm_pooled(pooled))

# --- SUBMISSION ALIAS (scaffold for submission-pma0) ------------------------------------
# eval.py constructs TransformerEncoder(...) and passes NO keyword arguments, so every
# option must be a constructor default. PMAEncoder takes the same signature and forward contract.
AttentionTransformerEncoder = TransformerEncoder
TransformerEncoder = PMAEncoder
