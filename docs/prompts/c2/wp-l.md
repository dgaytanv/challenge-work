# WP-L: preprocessor and optimisation (session jovyan-04)
Read `prompts/c2/00-common-v2.md`, then `prompts/00-common.md`, then `writeup/B-generator-and-pt-rules.md` and `writeup/G-quantization.md` §input audit.

## Why
MeanPt removed the severity-correlated pt shift and won the tie-break. Remaining preprocessor facts from campaign 1: φ enters raw
(the colleague uses sin/cos); dxysig has a 0.1% tail at 65 after BatchNorm; four input channels are constant (pdgId 130/1/2 never occur,
is_pf is constant on both files). Optimisation: the champion selects on best val loss; B showed degraded-val selection is the criterion
that matches the metric; no weight averaging was tried; schedules were 25 epochs with no early stop ever triggering.

## Configurations (champion + ONE change; 3 seeds; small file)
L1 φ → (sin φ, cos φ) (feature count 14 → 15; grader reads `num_features` from the preprocessor, check).
L2 dxysig → tanh(dxysig / 20) before BatchNorm.
L3 EMA of weights (decay 0.999) evaluated instead of raw weights (checkpoint saves the EMA weights; state_dict keys unchanged).
L4 checkpoint selection under a SEEDED validation corruption. Correction (03:47, from WP-L): validation already runs under the training generator, but the corruption is redrawn every epoch; measured draw-to-draw val-loss std 0.0059 vs 0.0005 spread between the champion's best three epochs, so selection was ~11x noise-dominated. L4 seeds the RNG before the validation pass and restores it after.
L5 (if time) 40 epochs with cosine to zero.
Preprocessor changes are allowed by the organisers (confirmed 23:08 on 3 Sep) but every changed preprocessor must load through `eval.py`
from the config alone.

## Rules
`preprocs.py`, `train.py` selection/EMA, configs. Not the encoder, generator or loss.

## Deliverables
Phase 1 table (≤ 5 × 3 seeds), recommendation. Phase 2 after the ruling. Reports `reports/l-<HHMM>.md`.
