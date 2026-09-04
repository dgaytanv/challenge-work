# Common brief for every implementation instance (read fully before starting)

You are one of several parallel implementation instances on the Fast ML 2026 hackathon "Robust Tagging"
challenge (C9). A planner/reviewer session coordinates the work and makes rulings. You report to it.

**Planner session name:** `jovyan-ff [c5e3fa]`. Use ListAgents to find it, SendMessage to reach it.

## The challenge in one paragraph
Events are sets of particle-flow candidates, each with (pt, eta, phi, dxy, dxy_sig, is_pf, pdgId).
An encoder maps an event to a small latent vector. The grader embeds the clean eval set, trains an MLP probe
on those latents to separate background from signal, then re-embeds the same events with candidates in
random dead eta-phi regions zeroed out, and scores them with the SAME frozen probe. Score = area under the
AUC-vs-severity curve. So the goal is: the latent of a degraded event must stay close to the latent of the
same clean event, for events the encoder has never seen (signal).

## Where things are
- Shared conventions, GPU rules, reporting format: `~/hackathon-shared/README.md` (read it).
- Shared git remote: `~/hackathon-shared/repo.git`. Branch `integration` is the base. Clone, branch `wp-<x>`, push there.
- Data: `~/hack-data/C9_robust_tagging/train/robust_tagging_train_data_small.pt` [80000, 200, 8] (4 background classes),
  `~/hack-data/C9_robust_tagging/eval/robust_tagging_eval_small.pt` [70000, 400, 8] (label 0 bkg / 1 signal).
  Last column is the label. Every slot is a real candidate in these files (no padding).
- Stock reference checkpoint (already trained, stock code): `~/hack-data/C9_robust_tagging/checkpoints/robust_tagging_encoder_20260902_212357.pth`
- Benchmark (the ruler): `python ~/hackathon-shared/bench/bench_eval.py --repo ~/rt-<x> --ckpt <ckpt> --tag <x>-<desc>`
  Anchor for the stock checkpoint: clean AUC 0.8605, mean_area 0.7734
  (rect 0.7283, wedge 0.7982, strip 0.7485, towers 0.7737, cells 0.8185). Beat mean_area.
- One shared A10 GPU (24 GB). Debug with `--test_mode`. Full runs go through `~/hackathon-shared/gpu_run.sh`.
- Literature review (why these choices): https://claude.ai/code/artifact/2086ac84-0b1a-4413-9e83-c9ef04afccf2

## Code map (repo root)
- `train.py` training driver; `src/embedding/training.py` train/val epoch loops; `src/embedding/loss.py` SupCon ("InfoNCELoss").
- `src/embedding/models.py` `TransformerEncoder` (CLS token, 4 post-norm blocks, bottleneck to `latent_dim`=6), `Projector`, classifier head.
- `src/embedding/preprocs.py` `PFPreProcessor`: 7 raw -> 14 features; zeroed candidates become all-zero rows.
- `src/embedding/degradation.py` `Degradation(severity)`: stub. Called on RAW x (before preproc) in train (severity=None) and eval (fixed severity).
- `src/embedding/dataloader.py`: builds the padding mask from pt==0 BEFORE degradation is applied (this is why the mask fix below exists).
- `eval.py` the grader's logic. Do not edit; read it to understand the contract.

## Hard constraints (grader contract)
1. `eval.py` does `TransformerEncoder(num_features, embed_size, latent_dim, num_heads, num_layers, linear_dim, num_tokens=None, pairwise)`
   and `encoder(preproc(x), delta_r, cls_mask)` with `cls_mask` [B, N+1] (CLS slot first, True = padded). Keep name, signature, forward contract.
   Extra constructor kwargs are allowed only with defaults, and the defaults must be the final choice (eval.py will not pass them).
2. Checkpoint keys `preproc`, `encoder`, `projector`, `classifier`, `norm_constants` stay. New state is fine inside `encoder`.
3. Train has 200 candidates per event, eval has 400. Nothing may depend on a fixed token count.
4. Never train on or tune to `~/hackathon-shared/bench/` maps. They are the ruler. Do not read the organisers' private files under `~/hack-data`.

## The mask fix (already merged into `integration`; shown so you know what it does)
In `TransformerEncoder.forward`, right after `B, N, F = x.shape` and before `x = self.input_proj(x)`:
```python
        dead = (x.abs().sum(dim=-1) == 0)  # rows zeroed by degradation/padding
        if mask is None:
            mask = torch.zeros(B, N + 1, dtype=torch.bool, device=x.device)
        mask = mask.clone()
        mask[:, 1:] |= dead
```
Reason: the dataloader builds the mask before degradation, so zeroed candidates otherwise enter attention as
identical constant tokens. Measured on the stock checkpoint with NO retraining: mean_area 0.7734 -> 0.7918, clean AUC unchanged
(runs/planner-maskfix-stockckpt_*.json). Branch from `integration` and you have it.

## Working rules
- Smoke-test every change with `--test_mode` before any full run. A full stock run takes roughly 1-2 h; early stopping usually ends it sooner.
- Commit small and often, push your branch after every milestone. Never push to `main` or `integration`.
- Report at each milestone and at least every 90 minutes: write `~/hackathon-shared/reports/<x>-<HHMM>.md`
  (commit hash, what changed, bench SUMMARY lines, open questions, next step), then SendMessage the planner with the path.
- Ask the planner before: editing files outside your package boundary, changing config keys eval.py reads, changing the
  checkpoint format, starting a run longer than 2 h, or if you are blocked for more than 15 minutes.
- Report numbers as they are. A regression is a result; say so.
