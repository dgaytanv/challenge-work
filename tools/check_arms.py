"""WP-I: assert that every arm's checkpoint can be rebuilt by the things that must rebuild it.

Campaign 1's register records a run that trained fine and benched at chance because the
harness rebuilt a DIFFERENT model from the same config and strict-loaded the weights
without complaint. The property that failed there is checked here directly, per arm:

  TRAIN path : getattr(models, encoder_class)(..., **encoder_kwargs)   -- what train.py builds
  BENCH path : getattr(models, encoder_class)(...) with NO kwargs      -- what bench_eval.py
               and eval.py:build_preproc_and_encoder both build (neither passes kwargs)

If those two disagree in any parameter name or shape, the arm is not benchable and the
run must not be started. Reported per arm rather than aggregated, so a single bad arm
does not hide behind five good ones.
"""
import sys

import torch

sys.path.insert(0, 'src')
import embedding.models as models  # noqa: E402
from embedding.utils.cfg_handler import train_config  # noqa: E402

NUM_FEATURES = 14  # PFPreProcessorMeanPt: 5 continuous + charge + is_pf + 7-way one-hot


def build(cfg, kwargs):
    cls = getattr(models, cfg.hp('encoder_class', 'TransformerEncoder'))
    return cls(
        num_features=NUM_FEATURES,
        embed_size=cfg.hp('embed_size', 128), latent_dim=cfg.hp('latent_dim', 6),
        num_heads=cfg.hp('num_heads', 8), num_layers=cfg.hp('num_layers', 4),
        linear_dim=cfg.hp('linear_dim', None), num_tokens=None,
        pairwise=cfg.get_trdata_cfg('pairwise', False), **kwargs)


def main(paths):
    x = torch.randn(4, 60, NUM_FEATURES)
    x[:, 45:] = 0.0                      # dead candidates, as degradation leaves them
    ok = True
    print(f'{"config":44s} {"class":14s} {"params":>10s}  rebuild  forward')
    for p in sorted(paths):
        cfg = train_config(p)
        kw = cfg.hp('encoder_kwargs', {}) or {}
        trained = build(cfg, kw)                       # what train.py makes
        rebuilt = build(cfg, {})                       # what the bench / grader makes
        a, b = trained.state_dict(), rebuilt.state_dict()
        same = (a.keys() == b.keys()) and all(a[k].shape == b[k].shape for k in a)
        try:
            rebuilt.load_state_dict(a)                 # the exact call bench_eval.py makes
            rebuilt.eval()
            with torch.no_grad():
                z = rebuilt(x)
            fwd = f'{tuple(z.shape)}'
            assert z.shape == (4, cfg.hp('latent_dim', 6)), z.shape
            assert torch.isfinite(z).all(), 'non-finite latent'
        except Exception as e:
            fwd = f'FAIL {type(e).__name__}: {str(e)[:60]}'
            same = False
        ok &= same
        n = sum(q.numel() for q in trained.parameters())
        print(f'{p.split("/")[-1]:44s} {type(trained).__name__:14s} {n:10,d}  '
              f'{"OK " if same else "MISMATCH":8s} {fwd}')
    print('\nRESULT:', 'all arms rebuildable by the bench/grader path' if ok else 'FAILURES ABOVE')
    return 0 if ok else 1


if __name__ == '__main__':
    import glob
    args = sys.argv[1:] or sorted(glob.glob('configs/train_config_i*_s11.yaml')) + \
        ['configs/train_config_d_pma0_aug_meanpt.yaml']
    sys.exit(main(args))
