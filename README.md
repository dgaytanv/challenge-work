## Robust Tagging Challenge

Train a model whose anomaly-detection performance stays robust as more of the detector drops out. See `hackathon_playground.ipynb` for the full challenge writeup, scoring details, and what you're expected to edit.

Built by Arianna Cox and Ellison Scheuller.

Special thanks to Roy Cruz Candelaria, Maciej Glowacki, and Mehrnoosh Moallemi for the contrastive model baseline used in this challenge.

### Setup in JupyterHub

1. Launch a server in JupyterHub (with the GPU attached) and open a terminal.
2. Clone this repo and `cd` into it:
   ```bash
   git clone https://github.com/ellisonscheuller/robust_tagging_fastml_hackathon.git robust-tagging
   cd robust-tagging
   ```
3. Install the project (installs `embedding` in editable mode plus its dependencies):
   ```bash
   pip install -e .
   ```
4. Open `hackathon_playground.ipynb` in JupyterLab and run the cells top to bottom.
   - The notebook must be run from the root of your repo copy (where this README lives) — that's the default JupyterLab working directory when you open a notebook there.
   - Training/eval data is read directly from the shared PVC at `~/hack-data/C9_robust_tagging/{train,eval}` — you don't need to download or convert anything.
   - Your own checkpoints and plots are written locally into `./checkpoints` and `./eval_plots` inside your repo copy, so they never collide with other participants sharing the same PVC.

### What you'll edit

- **`src/embedding/degradation.py`** — `Degradation.forward()` is a stub. This simulates detector dropout: `train.py` calls it with `severity=None` (implement your own randomized augmentation), `eval.py` calls it with a fixed `severity` for each step of the AUC-vs-severity sweep. Keep the class signature so both call sites keep working.
- **`src/embedding/models.py`** — the baseline architecture (`TransformerEncoder`, `Projector`, ...). Change layers, swap the encoder, add heads — anything, as long as it still produces a latent embedding.
- **`configs/train_config.yaml`** — hyperparameters (lr, embed size, loss weights, etc.).

### Scoring

At eval time the model sees both background and signal events and outputs a per-event anomaly score, scored via AUC vs. degradation severity. A robust model keeps a high AUC as more of the detector goes dark; your score is the area under that curve. `eval.py` overlays a red "(No degradation)" reference curve against your model's ("Your solution") on the same plot.

`eval.py` here uses **your own** `degradation.py` to simulate severity locally — it's for testing your own approach, not the official scoring run. For judging, we'll degrade the eval set ourselves with a method we're not disclosing in advance (conceptually it kills off geometric η–φ regions, similar to real detector dead zones), so solutions aren't tuned to the exact grading procedure.

## How to reproduce (WP-D submission: PMA set encoder + MeanPt)

**Encoder:** `PMAEncoder` with `num_layers=0` — a per-particle MLP followed by pooling by multihead
attention over 4 learned seed queries. No self-attention stack, so cost is O(N) in candidates, and
deleting a candidate removes its terms from a masked pooling rather than perturbing an attention
normalisation over the survivors. It is aliased to `TransformerEncoder` at the end of
`src/embedding/models.py`, so the organisers' `eval.py` builds it with no changes: it takes that
constructor signature and honours that forward contract.

`eval.py` passes **no keyword arguments**, so every option is a constructor default. The one
architectural setting it *does* pass from the config is `num_layers`, so `configs/train_config.yaml`
says `0` **and** the class default is `0`: a missing key cannot silently build a different
architecture, and a wrong value fails loudly at `load_state_dict`.

**Preprocessor:** `PFPreProcessorMeanPt` — pt encoded as `log(pt_i / mean surviving pt)`. This must
match: `PFPreProcessor` and `PFPreProcessorMeanPt` have **identical `state_dict` keys**, so a
mismatched `preproc_type` loads cleanly and silently computes the wrong feature.

**This branch is deliberately training-free.** It ships `src/embedding/models.py`,
`configs/train_config.yaml` and the single checkpoint, which is everything `eval.py` needs. The
training code is on branch `wp-d` (commit `9801bdd`), which carries the `encoder_class` /
`encoder_kwargs` / `use_degradation` hyperparameters this branch's config deliberately omits —
rebuilding the encoder from `eval.py`'s signature must not depend on them.

To retrain, from `wp-d` @ `9801bdd`:

```bash
python train.py \
  --data_cfg configs/data_config_collide1m_small.yaml \
  --train_cfg configs/train_config_d_pma0_aug_meanpt.yaml \
  --data ~/hack-data/C9_robust_tagging/train/robust_tagging_train_data_small.pt \
  --outdir checkpoints
```

Data: `robust_tagging_train_data_small.pt` ([80000, 200, 8], four background classes).
25 epochs, batch 256, patience 5, WP-B's dead-region generator as augmentation.

**Shipped checkpoint:** `checkpoints/rt_d_pma0_aug_meanpt_encoder_20260903_221125.pth` — the only
`.pth` on this branch, via a `.gitignore` exception, because the organisers' notebook selects with
`sorted(glob("checkpoints/*.pth"))[-1]` and asserts if that is empty.

**Measured** (JSONs in `~/hackathon-shared/runs/`):

| metric | value |
|---|---|
| bench `mean_area` (5 families, 20k events, 5 probe refits) | 0.8260 +- 0.0009 |
| official `eval.py` area (70k events, 10 severities) | 0.8790 and 0.8786 |
| held-out families (ellipse, annulus, diagonal) | 0.8021 +- 0.0006 |
| clean AUC | 0.9147 |
| cost | 0.090 M parameters, 11.9 ms per training step |

Stock transformer anchor for comparison: `mean_area` 0.7734, clean AUC 0.8605; the organisers'
no-degradation reference area is 0.8201.

Code, config and checkpoint are byte-identical to `3ac262f`, which passed the full fresh-clone
acceptance run; this commit adds documentation only.
