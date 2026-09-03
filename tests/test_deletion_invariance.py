"""Acceptance test for WP-A.

A candidate that degradation zeroed must be worth exactly as much to the encoder as a
candidate that was never there. So: zero 50% of the candidates in place, and separately
build the same events with those candidates physically removed (shorter N, zero-padded);
the latents must agree. This is what the mask fix buys, and every readout must preserve it.

Run:  cd ~/rt-a && python tests/test_deletion_invariance.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from embedding.models import TransformerEncoder
from embedding.preprocs import PFPreProcessor

DATA = os.path.expanduser("~/hack-data/C9_robust_tagging/eval/robust_tagging_eval_small.pt")
CKPT = os.path.expanduser("~/hack-data/C9_robust_tagging/checkpoints/robust_tagging_encoder_20260902_212357.pth")
TOL = 1e-5


def pack_survivors(x, keep):
    """Rebuild the batch with only the kept candidates, left-packed and zero-padded."""
    B, N, F = x.shape
    n_keep = int(keep.sum(1).max())
    out = torch.zeros(B, max(n_keep, 1), F, dtype=x.dtype, device=x.device)
    # stable order: kept candidates keep their relative order
    order = torch.argsort((~keep).long(), dim=1, stable=True)
    xs = torch.gather(x, 1, order.unsqueeze(-1).expand(-1, -1, F))
    ks = torch.gather(keep, 1, order)
    out[:, :n_keep] = xs[:, :n_keep] * ks[:, :n_keep].unsqueeze(-1)
    return out


def run(readout, x, dtype, device):
    torch.manual_seed(0)
    preproc = PFPreProcessor({}).to(device=device, dtype=dtype).eval()
    encoder = TransformerEncoder(
        num_features=preproc.num_features, embed_size=128, latent_dim=6,
        num_heads=8, num_layers=4, linear_dim=None, num_tokens=None,
        pairwise=False, readout=readout,
    ).to(device=device, dtype=dtype).eval()
    if readout == "cls" and os.path.isfile(CKPT):
        ck = torch.load(CKPT, map_location=device)
        preproc.load_state_dict(ck["preproc"]); encoder.load_state_dict(ck["encoder"])
        preproc.to(dtype).eval(); encoder.to(dtype).eval()
        source = "stock checkpoint"
    else:
        source = "random init"

    B, N, _ = x.shape
    g = torch.Generator(device="cpu").manual_seed(1)
    keep = (torch.rand(B, N, generator=g) > 0.5).to(device)   # drop ~50%

    x_zeroed = x * keep.unsqueeze(-1)
    x_removed = pack_survivors(x, keep)

    # The dataloader builds its mask from the RAW pt column before degradation, so for the
    # zeroed batch it is all-False here (this data has no padding). The encoder must
    # re-derive the dead rows itself.
    def cls_mask(raw):
        m = raw[..., 0] == 0
        return torch.cat([torch.zeros(raw.size(0), 1, dtype=torch.bool, device=raw.device), m], dim=1)

    with torch.no_grad():
        z_zeroed = encoder(preproc(x_zeroed), None, cls_mask(x))         # pre-degradation mask
        z_removed = encoder(preproc(x_removed), None, cls_mask(x_removed))

    diff = (z_zeroed - z_removed).abs().max().item()
    scale = z_zeroed.abs().max().item()
    print(f"  {readout:14s} [{str(dtype).split('.')[-1]:7s}, {source:17s}] "
          f"N {N} -> {x_removed.shape[1]}  max|dz| = {diff:.2e}  (latent scale {scale:.2f})")
    return diff


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feats = torch.load(DATA, map_location="cpu", weights_only=False)[:64, :, :7].to(device)
    print(f"deletion-invariance test on {feats.shape[0]} events, {feats.shape[1]} candidates, device={device}")

    failures = []
    for dtype in (torch.float32, torch.float64):
        for readout in ("cls", "cls+mean", "cls+mean+max", "pma"):
            d = run(readout, feats.to(dtype), dtype, device)
            if dtype is torch.float64 and d > TOL:
                failures.append((readout, d))

    # all-dead event must still produce a finite latent
    for readout in ("cls", "cls+mean", "cls+mean+max", "pma"):
        torch.manual_seed(0)
        preproc = PFPreProcessor({}).to(device).eval()
        enc = TransformerEncoder(
            num_features=preproc.num_features, embed_size=128, latent_dim=6, num_heads=8,
            num_layers=4, linear_dim=None, num_tokens=None, pairwise=False, readout=readout,
        ).to(device).eval()
        dead_batch = torch.zeros(4, 200, 7, device=device)
        with torch.no_grad():
            z = enc(preproc(dead_batch), None, None)
        ok = torch.isfinite(z).all().item()
        print(f"  {readout:14s} all-dead event -> finite latent: {ok}")
        if not ok:
            failures.append((readout, float("nan")))

    assert not failures, f"FAILED (float64 tol {TOL}): {failures}"
    print(f"\nPASS: every readout is invariant to zeroing vs deleting (float64 max|dz| < {TOL})")


if __name__ == "__main__":
    main()
