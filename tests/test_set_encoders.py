"""WP-D acceptance tests for the set encoders.

Run:  PYTHONPATH=$HOME/rt-d/src python tests/test_set_encoders.py
"""
import time

import torch

from embedding.models import DeepSetsEncoder, PMAEncoder, TransformerEncoder

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NF, E, LAT, H = 14, 128, 6, 8


def build(cls, num_layers, **kw):
    return cls(NF, E, LAT, num_heads=H, num_layers=num_layers,
               linear_dim=None, num_tokens=None, pairwise=False, **kw).to(DEV).eval()


def test_deletion_equivalence():
    """Zeroing candidates must equal physically removing them (within 1e-5)."""
    B, N = 16, 120
    x = torch.randn(B, N, NF, device=DEV)
    drop = torch.zeros(N, dtype=torch.bool, device=DEV)
    drop[torch.randperm(N, device=DEV)[: N // 2]] = True  # same 50% in every event

    x_zeroed = x.clone()
    x_zeroed[:, drop] = 0.0
    x_removed = x[:, ~drop]

    mask_z = torch.zeros(B, N + 1, dtype=torch.bool, device=DEV)
    mask_r = torch.zeros(B, x_removed.shape[1] + 1, dtype=torch.bool, device=DEV)

    for name, model in [
        ("DeepSets", build(DeepSetsEncoder, 0)),
        ("DeepSets+count", build(DeepSetsEncoder, 0, count_feature=True)),
        ("PMA(L=0)", build(PMAEncoder, 0)),
        ("PMA(L=2)", build(PMAEncoder, 2)),
    ]:
        with torch.no_grad():
            a = model(x_zeroed, None, mask_z)
            b = model(x_removed, None, mask_r)
        d = (a - b).abs().max().item()
        print(f"  deletion equivalence {name:16s} max|dz| = {d:.3e}")
        assert d < 1e-5, f"{name}: {d}"


def test_permutation_invariance():
    B, N = 8, 60
    x = torch.randn(B, N, NF, device=DEV)
    perm = torch.randperm(N, device=DEV)
    mask = torch.zeros(B, N + 1, dtype=torch.bool, device=DEV)
    for name, model in [("DeepSets", build(DeepSetsEncoder, 0)), ("PMA(L=2)", build(PMAEncoder, 2))]:
        with torch.no_grad():
            d = (model(x, None, mask) - model(x[:, perm], None, mask)).abs().max().item()
        print(f"  permutation invariance {name:16s} max|dz| = {d:.3e}")
        assert d < 1e-5, f"{name}: {d}"


def test_all_dead_is_finite():
    B, N = 4, 40
    x = torch.randn(B, N, NF, device=DEV)
    x[0] = 0.0
    mask = torch.zeros(B, N + 1, dtype=torch.bool, device=DEV)
    for name, model in [("DeepSets", build(DeepSetsEncoder, 0)), ("PMA(L=2)", build(PMAEncoder, 2))]:
        with torch.no_grad():
            z = model(x, None, mask)
        print(f"  all-dead event finite  {name:16s} {bool(torch.isfinite(z).all())}")
        assert torch.isfinite(z).all()


def test_variable_token_count():
    """Train has N=200, eval N=400: nothing may depend on the token count."""
    mask200 = torch.zeros(4, 201, dtype=torch.bool, device=DEV)
    mask400 = torch.zeros(4, 401, dtype=torch.bool, device=DEV)
    for name, model in [("DeepSets", build(DeepSetsEncoder, 0)), ("PMA(L=2)", build(PMAEncoder, 2))]:
        with torch.no_grad():
            model(torch.randn(4, 200, NF, device=DEV), None, mask200)
            model(torch.randn(4, 400, NF, device=DEV), None, mask400)
        print(f"  N=200 and N=400 both run {name}")


def test_step_time():
    """ms per training step (fwd+bwd) at batch 128, N=200."""
    B, N = 128, 200
    x = torch.randn(B, N, NF, device=DEV)
    mask = torch.zeros(B, N + 1, dtype=torch.bool, device=DEV)
    for name, model in [
        ("TransformerEncoder", build(TransformerEncoder, 4)),
        ("DeepSetsEncoder", build(DeepSetsEncoder, 0)),
        ("PMAEncoder(L=0)", build(PMAEncoder, 0)),
        ("PMAEncoder(L=2)", build(PMAEncoder, 2)),
        ("PMAEncoder(L=4)", build(PMAEncoder, 4)),
    ]:
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        for i in range(15):
            if i == 5:
                if DEV == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
            opt.zero_grad()
            model(x, None, mask).square().mean().backward()
            opt.step()
        if DEV == "cuda":
            torch.cuda.synchronize()
        nparam = sum(p.numel() for p in model.parameters())
        print(f"  {name:20s} {(time.time() - t0) / 10 * 1000:7.1f} ms/step   {nparam/1e6:.2f}M params")


if __name__ == "__main__":
    print(f"device = {DEV}")
    for fn in [test_deletion_equivalence, test_permutation_invariance, test_all_dead_is_finite,
               test_variable_token_count, test_step_time]:
        print(f"\n[{fn.__name__}]")
        fn()
    print("\nALL WP-D ACCEPTANCE TESTS PASSED")
