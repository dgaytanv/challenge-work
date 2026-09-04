# WP-I tools

* `gen_i_configs.py` — generates the six arm configs x three seeds from
  `configs/train_config_d_pma0_aug_meanpt.yaml`. Generated rather than hand-written so
  "exactly one change from the champion" is true by construction and auditable: the diff
  of any generated config against the champion is the model_name, one hyperparameter and
  the seed. Asserts the changed key already exists in the champion, so a typo in a key
  name fails at generation instead of silently adding a hyperparameter nothing reads.
* `check_arms.py` — asserts every arm's checkpoint is rebuildable by the things that must
  rebuild it (bench_eval.py and eval.py both construct from the grader's fixed signature
  and pass NO kwargs, while train.py passes `encoder_kwargs`). Run before starting any arm.
* `smoke.sh` — CPU `--test_mode`, 1 epoch, no GPU. Proves train.py runs an arm end to end.
