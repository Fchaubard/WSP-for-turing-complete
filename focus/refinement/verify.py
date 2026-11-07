"""Verify the reconstructed least-squares procedure against every primary Xavier endpoint."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('supplemental_refinement', Path(__file__).with_name('run.py'))
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def main():
    with np.load(study.ROOT / 'extension/optimizer/states.npz') as original_weights, np.load(
        study.ROOT / 'focus/arithmetic/fits.npz') as arithmetic_weights:
        rows = []
        for (task, size, seed, arm), _ in study.fresh_proposals().items():
            if arm != 'examples':
                continue
            x, y = study.observed(task, seed)
            if task in study.arithmetic.TASKS:
                start = study.arithmetic.initial(size, seed)
                archived = arithmetic_weights[f'{task}_{seed}_{size}_trf_xavier']
            else:
                start = study.initial(task, size, seed)
                archived = original_weights[f'{task}_{size}_{seed}_ls_xavier']
            p, _ = study.refine(start, *study.settings(task, size, x, y))
            same = np.array_equal(p, archived)
            rows.append(dict(task=task,size=size,seed=seed,
                             max_parameter_difference=float(max(abs(p-archived))),
                             all_weights_match=bool(same)))
        study.save_csv('solver_replay_check.csv', rows)
        matches = sum(r['all_weights_match'] for r in rows)
        print(f'Identical Xavier solver replays: {matches}/{len(rows)}')
        assert matches == len(rows)


if __name__ == '__main__':
    main()
