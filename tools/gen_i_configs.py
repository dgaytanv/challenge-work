"""WP-I: generate the capacity-arm configs from the champion config.

Every arm is a COPY of configs/train_config_d_pma0_aug_meanpt.yaml with exactly ONE
hyperparameter changed, plus the seed. Generated rather than hand-written so that "one
change" is enforced by construction and is auditable: the diff of any generated file
against the champion is the seed line plus one other line.
"""
import os

CHAMPION = 'configs/train_config_d_pma0_aug_meanpt.yaml'
SEEDS = (11, 22, 33)

# arm id -> (short description, {config key: value}); the key lives under hyperparameters:
ARMS = {
    'i1': ('layers1',  {'num_layers': 1}),
    'i2': ('layers2',  {'num_layers': 2}),
    'i3': ('embed256', {'embed_size': 256}),
    'i4': ('seeds8',   {'encoder_class': 'PMAEncoder8'}),
    'i5': ('lat16',    {'latent_dim': 16}),
    'i6': ('lat32',    {'latent_dim': 32}),
}


def main():
    base = open(CHAMPION).read().splitlines()
    os.makedirs('configs', exist_ok=True)
    written = []
    for arm, (desc, changes) in ARMS.items():
        for seed in SEEDS:
            out, applied = [], set()
            for line in base:
                stripped = line.strip()
                if stripped.startswith('model_name:'):
                    out.append(f'model_name: rt_{arm}_{desc}_s{seed}')
                    continue
                key = stripped.split(':', 1)[0] if ':' in stripped else None
                if key in changes:
                    indent = line[:len(line) - len(line.lstrip())]
                    out.append(f'{indent}{key}: {changes[key]}')
                    applied.add(key)
                    continue
                out.append(line)
            missing = set(changes) - applied
            assert not missing, f'{arm}: keys not present in the champion config: {missing}'
            out.insert(1, f'# WP-I arm {arm.upper()} ({desc}): champion config with ONE change, '
                          f'{changes}; seed {seed}.')
            out.append(f'  seed: {seed}')
            path = f'configs/train_config_{arm}_{desc}_s{seed}.yaml'
            open(path, 'w').write('\n'.join(out) + '\n')
            written.append(path)
    print(f'wrote {len(written)} configs')
    return written


if __name__ == '__main__':
    main()
