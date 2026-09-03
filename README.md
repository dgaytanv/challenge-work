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

## How to reproduce (WP-D submission: Deep Sets + MeanPt)

Encoder: `DeepSetsEncoder` (a permutation-symmetric set encoder), aliased to `TransformerEncoder`
at the end of `src/embedding/models.py` so the organisers' `eval.py` builds it unchanged. It takes
that constructor signature and forward contract exactly; `eval.py` passes no keyword arguments, so
`pooling="mean+max"` and `count_feature=False` are constructor DEFAULTS rather than config keys.

Preprocessor: `PFPreProcessorMeanPt` — pt encoded as `log(pt_i / mean surviving pt)`. This matters
and must match: `PFPreProcessor` and `PFPreProcessorMeanPt` have identical `state_dict` keys, so a
mismatched `preproc_type` loads cleanly and silently computes the wrong feature.

**This branch is deliberately training-free.** It ships only what the grader needs:
`src/embedding/models.py`, `configs/train_config.yaml` and the single checkpoint. The
training code lives on branch `wp-d` (commit `9801bdd`), which carries the
`encoder_class` / `encoder_kwargs` / `use_degradation` hyperparameters that this branch's
config deliberately omits — reproducing the encoder from `eval.py` must not depend on them.

To retrain, from `wp-d` @ `9801bdd`:

```bash
python train.py \
  --data_cfg configs/data_config_collide1m_small.yaml \
  --train_cfg configs/train_config_d_deepsets_aug_meanpt.yaml \
  --data ~/hack-data/C9_robust_tagging/train/robust_tagging_train_data_small.pt \
  --outdir checkpoints
```
Data: `robust_tagging_train_data_small.pt` ([80000, 200, 8], four background classes).
Training used WP-B's dead-region generator as augmentation; 25 epochs, batch 256, patience 5.

Shipped checkpoint: `checkpoints/rt_d_deepsets_aug_meanpt_encoder_20260903_214738.pth` (the only
`.pth` on this branch, via a `.gitignore` exception, because the organisers' notebook selects with
`sorted(glob("checkpoints/*.pth"))[-1]`).

Measured (see `~/hackathon-shared/runs/`):
- bench `mean_area` 0.8186 +- 0.0007, clean AUC 0.9066 +- 0.0006 (5 probe refits, 20k events)
- stock anchor for comparison: `mean_area` 0.7734, clean AUC 0.8605
