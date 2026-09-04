"""WP-I: generate the capacity-arm configs from the champion config.

Every arm is a COPY of configs/train_config_d_pma0_aug_meanpt.yaml with exactly ONE
hyperparameter changed, plus the seed. Generated rather than hand-written so that "one
change" is enforced by construction and is auditable: the diff of any generated file
against the champion is the seed line plus one other line.
"""
import os

CHAMPION = 'configs/c2/champion.yaml'   # WP-H's canonical campaign-2 reference
SEEDS = (11, 22, 33)

# arm id -> (short description, {config key: value}); the key lives under hyperparameters.
# i1p carries TWO keys and is still ONE behavioural change: at num_layers 0 the class
# PMAEncoderFF256 is bit-identical to PMAEncoder (dim_feedforward is only read when a block
# is built), so neither line alone expresses "add one transformer block of FFN width 256".
ARMS = {
    # Planner ruling 03:38: four arms. I2 (layers2) and I5 (lat16) dropped; I1 at the
    # default ff=2048 replaced by I1' at ff=256, which fills the 1.15x-3.83x hole.
    'i1p': ('ff256l1', {'num_layers': 1, 'encoder_class': 'PMAEncoderFF256'}),
    'i3':  ('embed256', {'embed_size': 256}),
    'i4':  ('seeds8',   {'encoder_class': 'PMAEncoder8'}),
    'i6':  ('lat32',    {'latent_dim': 32}),
}


def main():
    base = open(CHAMPION).read().splitlines()
    os.makedirs('configs', exist_ok=True)
    written = []
    for arm, (desc, changes) in ARMS.items():
        for seed in SEEDS:
            # `seed` is REPLACED, never appended: champion.yaml already carries `seed: 11`
            # (line 11), and appending would emit a duplicate YAML key whose winner depends
            # on the loader rather than on intent.
            full = dict(changes, seed=seed)
            out, applied = [], set()
            for line in base:
                stripped = line.strip()
                if stripped.startswith('model_name:'):
                    out.append(f'model_name: rt_{arm}_{desc}_s{seed}')
                    continue
                key = stripped.split(':', 1)[0] if ':' in stripped else None
                if key in full:
                    indent = line[:len(line) - len(line.lstrip())]
                    out.append(f'{indent}{key}: {full[key]}')
                    applied.add(key)
                    continue
                out.append(line)
            missing = set(full) - applied
            assert not missing, f'{arm}: keys not present in the champion config: {missing}'
            out.insert(1, f'# WP-I arm {arm.upper()} ({desc}): champion config with ONE change, '
                          f'{changes}; seed {seed}.')
            path = f'configs/c2/{arm}_{desc}_s{seed}.yaml'
            open(path, 'w').write('\n'.join(out) + '\n')
            written.append(path)
    print(f'wrote {len(written)} configs')
    return written


if __name__ == '__main__':
    main()
