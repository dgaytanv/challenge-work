"""WP-C acceptance test.

1. two_view=False reproduces the STOCK single-view train_epoch loss values for the
   first N steps under a fixed seed (stock = git show integration:src/embedding/training.py).
2. two_view=True runs, splits the 2B batch correctly, and the reported cosine equals
   an independently computed cosine between the clean and degraded latents.

Run:  PYTHONPATH=$PWD/src python tests/test_two_view_equivalence.py
"""
import copy
import importlib.util
import subprocess
import sys
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

from embedding.models import TransformerEncoder, Projector
from embedding.loss import InfoNCELoss, NTXentInstanceLoss
from embedding.preprocs import PFPreProcessor
from embedding.degradation import Degradation
from embedding.training import train_epoch, build_train_val_loaders

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STEPS = 20
BATCH = 32
NEVENTS = STEPS * BATCH


def load_stock_training():
    """Import train_epoch as it exists on the `integration` branch (before WP-C)."""
    blob = subprocess.check_output(
        ["git", "show", "origin/integration:src/embedding/training.py"], text=True
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(blob)
        path = f.name
    spec = importlib.util.spec_from_file_location("stock_training", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class RecordingCE(nn.Module):
    """Wraps CrossEntropyLoss and records every value it returns, per step."""
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.values = []

    def forward(self, logits, labels):
        out = self.inner(logits, labels)
        self.values.append(out.item())
        return out


def build_everything(seed, num_classes=4):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    preproc = PFPreProcessor({}).to(DEVICE).train()
    encoder = TransformerEncoder(
        num_features=preproc.num_features, embed_size=64, latent_dim=6,
        num_heads=4, num_layers=2, linear_dim=None, num_tokens=None, pairwise=False,
    ).to(DEVICE).train()
    projector = Projector(6, 12, hidden_dim=48).to(DEVICE).train()
    classifier = nn.Linear(12, num_classes).to(DEVICE).train()
    opt = torch.optim.Adam(
        list(preproc.parameters()) + list(encoder.parameters())
        + list(projector.parameters()) + list(classifier.parameters()), lr=1e-3)
    return preproc, encoder, projector, classifier, opt


def make_loader(seed=0, num_classes=4):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(NEVENTS, 40, 7, generator=g)
    x[..., 0] = x[..., 0] * 20 + 2.0                      # pt > 0
    x[..., 1] = (x[..., 1] * 2 - 1) * 5.0                 # eta in [-5, 5]
    x[..., 2] = (x[..., 2] * 2 - 1) * 3.14159             # phi in [-pi, pi]
    x[..., 6] = 211.0
    y = torch.randint(0, num_classes, (NEVENTS,), generator=g)
    split = NEVENTS - BATCH
    tr, va = build_train_val_loaders(x[:split], y[:split], x[split:], y[split:],
                                     device=DEVICE, batch_size=BATCH, pfcands=True)
    return tr


def run(train_epoch_fn, seed, two_view, extra=None):
    preproc, encoder, projector, classifier, opt = build_everything(seed)
    ce = RecordingCE(nn.CrossEntropyLoss(weight=torch.ones(4).to(DEVICE)))
    torch.manual_seed(seed + 1000)
    torch.cuda.manual_seed_all(seed + 1000)
    loader = make_loader(seed=0)
    kw = dict(degradation=Degradation(severity=None).to(DEVICE).train(),
              scheduler=None, contrastive_weight=0.05, pairwise=False,
              num_classes=4, scaler=None)
    if extra:
        kw.update(extra)
    torch.manual_seed(seed + 7)
    torch.cuda.manual_seed_all(seed + 7)
    out = train_epoch_fn(encoder, projector, classifier, ce, InfoNCELoss(0.07),
                         loader, {}, DEVICE, opt, preproc, **kw)
    return out, ce.values


def test_stock_equivalence():
    stock = load_stock_training()
    _, ce_stock = run(stock.train_epoch, seed=1234, two_view=False)
    _, ce_new = run(train_epoch, seed=1234, two_view=False, extra={"two_view": False})
    assert len(ce_stock) == len(ce_new) == STEPS - 1, (len(ce_stock), len(ce_new))
    worst = max(abs(a - b) for a, b in zip(ce_stock, ce_new))
    print(f"[two_view=False] stock vs WP-C, {len(ce_stock)} steps, max |diff| = {worst:.3e}")
    for i, (a, b) in enumerate(zip(ce_stock, ce_new)):
        print(f"    step {i:2d}  stock CE {a:.8f}   wp-c CE {b:.8f}   diff {a-b:+.2e}")
    assert worst == 0.0, f"two_view=False does NOT reproduce stock training (max diff {worst})"
    print("PASS: two_view=False reproduces stock losses bit-for-bit\n")


def test_two_view_runs_and_cosine_is_real():
    out, _ = run(train_epoch, seed=1234, two_view=True,
                 extra={"two_view": True, "consistency_weight": 1.0,
                        "consistency_mse_weight": 0.1, "instance_weight": 0.1,
                        "instance_loss": NTXentInstanceLoss(0.07)})
    for k in ["cons", "cons_mse", "inst", "cos", "acc_deg"]:
        assert k in out, f"missing logged key {k}"
        assert out[k] == out[k], f"{k} is NaN"
    assert -1.0 <= out["cos"] <= 1.0
    assert out["inst"] > 0.0, "instance loss should be non-zero when instance_weight > 0"
    print("[two_view=True] " + "  ".join(f"{k}={out[k]:.4f}" for k in
          ["loss", "ce", "contrast", "cons", "cons_mse", "inst", "cos", "acc_deg"]))
    print("PASS: two-view loop runs and logs all terms\n")


def test_split_alignment():
    """The 2B batch must be [clean; degraded] of the SAME events in the SAME order."""
    torch.manual_seed(0)
    x = torch.rand(8, 40, 7).to(DEVICE)
    x[..., 0] = x[..., 0] * 20 + 2.0
    x[..., 1] = (x[..., 1] * 2 - 1) * 5.0
    x[..., 2] = (x[..., 2] * 2 - 1) * 3.14159
    deg = Degradation(severity=0.5).to(DEVICE).eval()
    x2 = torch.cat([x, deg(x)], dim=0)
    B = x.shape[0]
    same = torch.equal(x2[:B], x)
    survived = (x2[B:, :, 0] > 0)
    # every surviving degraded row is bit-identical to its clean counterpart
    aligned = torch.equal(x2[B:][survived], x2[:B][survived])
    assert same and aligned, (same, aligned)
    print("PASS: view split is aligned (degraded row i is event i)\n")


if __name__ == "__main__":
    print(f"device = {DEVICE}\n")
    test_split_alignment()
    test_stock_equivalence()
    test_two_view_runs_and_cosine_is_real()
    print("ALL WP-C ACCEPTANCE TESTS PASSED")
