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

## How to reproduce (submission-a-pma)

**Architecture.** Stock 4-layer transformer body with the CLS readout replaced by pooling with
multihead attention (PMA, k=4 learned seed queries) over the surviving particle tokens.
`TransformerEncoder` defaults to `readout="pma"`, so `eval.py` — which passes no `readout`
kwarg — rebuilds the scored architecture from its signature alone. Preprocessor: `PFPreProcessor`
(stock; **not** MeanPt). The attention mask is re-derived inside `forward` from all-zero input
rows, so candidates zeroed by degradation are excluded from attention rather than entering it as
identical constant tokens.

**Train.**
```bash
~/hackathon-shared/gpu_run.sh python train.py \
  --data_cfg configs/data_config_collide1m_small.yaml \
  --train_cfg configs/train_config_a_pma.yaml \
  --data ~/hack-data/C9_robust_tagging/train/robust_tagging_train_data_small.pt \
  --outdir checkpoints
```
Data: `robust_tagging_train_data_small.pt` ([80000, 200, 8]). 25 epochs, batch 256, patience 5,
mixed precision, WP-B degradation generator. Config differs from `train_config_b_aug_stock.yaml`
in exactly one line (`readout: pma`).

**Shipped checkpoint.** `checkpoints/a_pma_submission.pth` (epoch 25 of 25, best val loss).

**Numbers.**

| measurement | value | source |
|---|---|---|
| bench mean_area, R=5, 20k | 0.8275 +- 0.0006 | `runs/a-pma_20260903_224712.json` |
| bench clean AUC | 0.9228 +- 0.0008 | same |
| per family | rect 0.7679, wedge 0.8589, strip 0.8071, towers 0.8288, cells 0.8746 | same |
| held-out families, R=5 | pending | `runs/a-pma-heldout_*.json` |
| official accept.sh --fast x2 | pending | `runs/official/` |
| bench mean_area, R=20 | pending | `runs/a-pma-rep20_*.json` |

**Provenance of the comparison.** `a-pma` minus `d-pma0-aug` isolates the transformer body with
readout and preprocessor held fixed: +0.0113 against a per-pair threshold of 0.0052 (2.2x), i.e.
the body helps separably. Against the certified winner `d-pma0-aug-meanpt` (0.8260 +- 0.0009) the
gap is +0.0015 against a threshold of 0.00145 — **a tie at the edge, not a win**. Costs O(N^2) in
candidates against the set encoder's O(N).

**Commit.** `1d8cb08` on branch `submission-a-pma`.
