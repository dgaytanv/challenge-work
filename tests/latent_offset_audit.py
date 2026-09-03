"""Where does the latent offset come from?

WP-C measured an offset ~30x the per-event spread on the trained two-view latent; the stock
latent is only 0.49x. The proposed explanation was the bottleneck bias drifting under the raw
MSE term. This tests that directly by decomposing the mean latent:

    E[z] = W @ E[h] + b        (h = the bottleneck's input, after norm_cls_embedding)

so the bias contributes exactly ||b|| and everything else is the feature term. If ||b|| is a
small part of ||E[z]||, the offset is not the bias and a bias-targeted fix will not remove it.

Read-only, one clean embedding per checkpoint, no severity sweep.
Run: cd ~/rt-a && ~/hackathon-shared/gpu_small.sh python tests/latent_offset_audit.py \
        --repo ~/rt-c --ckpt <path> [--encoder_class TransformerEncoder]
"""
import argparse, importlib, importlib.util, os, sys
import torch

DATA = os.path.expanduser("~/hack-data/C9_robust_tagging/eval/robust_tagging_eval_small.pt")


def load_repo(repo):
    sys.path.insert(0, repo); sys.path.insert(0, os.path.join(repo, "src"))
    spec = importlib.util.spec_from_file_location("rt_eval", os.path.join(repo, "eval.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--encoder_class", default="TransformerEncoder")
    ap.add_argument("--train_cfg", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--events", type=int, default=6000)
    args = ap.parse_args()

    repo = os.path.abspath(os.path.expanduser(args.repo))
    ev = load_repo(repo)
    from embedding.utils.cfg_handler import train_config, data_config
    from embedding.utils.data_utils import load_data

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = train_config(args.train_cfg or os.path.join(repo, "configs", "train_config.yaml"))
    cfg_data = data_config(os.path.join(repo, "configs", "data_config_eval.yaml"))
    ck = torch.load(args.ckpt, map_location=device)

    preprocs = importlib.import_module("embedding.preprocs")
    preproc = getattr(preprocs, cfg.get_trdata_cfg("preproc_type", "PFPreProcessor"))(ck["norm_constants"]).to(device)
    preproc.load_state_dict(ck["preproc"]); preproc.eval()
    models = importlib.import_module("embedding.models")
    enc = getattr(models, args.encoder_class)(
        num_features=preproc.num_features, embed_size=cfg.hp("embed_size", 128),
        latent_dim=cfg.hp("latent_dim", 6), num_heads=cfg.hp("num_heads", 8),
        num_layers=cfg.hp("num_layers", 4), linear_dim=cfg.hp("linear_dim", None),
        num_tokens=None, pairwise=cfg.get_trdata_cfg("pairwise", False),
    ).to(device)
    enc.load_state_dict(ck["encoder"]); enc.eval()

    feats, labels = load_data(DATA, map_location="cpu", max_events=args.events)
    lat, lab, _ = ev.embed_dataset(preproc, enc, feats, labels, cfg_data, ck["norm_constants"],
                                   device, batch_size=512)
    mu = lat.mean(0)
    spread = float((lat - mu).norm(dim=1).mean())
    offset = float(mu.norm())

    bias = weight = None
    bkey = wkey = None
    for k in ck["encoder"]:
        head = k.split(".")[0]
        if head in ("bottleneck", "rho", "out_proj"):
            if k.endswith("bias"):   bias, bkey = ck["encoder"][k], k
            if k.endswith("weight"): weight, wkey = ck["encoder"][k], k

    print(f"tag={args.tag or os.path.basename(args.ckpt)}  encoder={args.encoder_class}")
    print(f"  models={models.__file__}")
    print(f"  ||E[z]||            = {offset:.4f}      <- offset (numerator)")
    print(f"  per-event spread     = {spread:.4f}      <- mean ||z - E[z]|| (denominator)")
    print(f"  OFFSET RATIO         = {offset/max(spread,1e-9):.2f}x")
    if weight is not None:
        print(f"  {wkey:20s} ||W||_F = {float(weight.norm()):.4f}")
    if bias is not None:
        nb = float(bias.norm())
        print(f"  {bkey:20s} ||b||   = {nb:.4f}  ({100*nb/max(offset,1e-9):.1f}% of the offset)")
        print(f"  feature term ||E[z]-b||          = {float((mu.cpu()-bias.cpu()).norm()):.4f}")
    # spread collapse shows up per dimension, not just in the mean norm
    sd = lat.std(0)
    print(f"  per-dim std          = {[round(float(v),3) for v in sd]}")
    print(f"  per-dim mean         = {[round(float(v),3) for v in mu.cpu()]}")
    print(f"  RATIO-CSV,{args.tag},{offset:.4f},{spread:.4f},{offset/max(spread,1e-9):.3f},"
          f"{float(weight.norm()) if weight is not None else float('nan'):.4f},"
          f"{float(bias.norm()) if bias is not None else float('nan'):.4f}")


if __name__ == "__main__":
    main()
