"""Independent paired geometry/motion report from closed original arrays."""
import json
from pathlib import Path

import numpy as np

from bind_first_failures import sha

P = Path(__file__).resolve().parent
NEW = Path('/mnt/nas/data/lyf/double_hand/safety_velocity_arrival_20261008_1519/development')
OLD = Path('/mnt/nas/data/lyf/double_hand/safety_native_acceptance_20261007_2307/acceptance')


def small(path, keys=None):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in (data.files if keys is None else keys)}


def exact(a, b):
    assert a.keys() == b.keys()
    for key in a:
        assert a[key].shape == b[key].shape and a[key].dtype == b[key].dtype
        assert a[key].tobytes() == b[key].tobytes(), key


def main():
    sources, checks, cases, summaries = {}, [], [], {}
    all_stats = {mode: [] for mode in ['velocity_arrival', 'adaptive_joint', 'zero_inclusive']}
    keys = ['q', 'q_initial', 'official_margins', 'cmd', 'external_unscaled_cmd',
            'joint_soft_limits', 'initial_violation']
    for block in [0, 1]:
        root = NEW / f'block_{block}' / 'velocity_arrival'
        candidate = small(root / 'cell_001.npz', keys)
        n = small(root / 'native_initial.npz')
        recipe = small(root / 'input_recipe.npz')
        assignment = small(root / 'bank_assignment.npz')
        for mode in all_stats:
            target = root if mode == 'velocity_arrival' else OLD / f'block_{block}' / mode
            cell = candidate if mode == 'velocity_arrival' else small(target / 'cell_001.npz', keys)
            if mode != 'velocity_arrival':
                exact(n, small(target / 'native_initial.npz'))
                exact(recipe, small(target / 'input_recipe.npz'))
                exact(assignment, small(target / 'bank_assignment.npz'))
                exact({k: candidate[k] for k in ['q_initial', 'cmd', 'external_unscaled_cmd']},
                      {k: cell[k] for k in ['q_initial', 'cmd', 'external_unscaled_cmd']})
                checks.append(dict(block=block, baseline=mode, native_initial_all_fields_exact=True,
                                   external_tapes_all_fields_exact=True, risk_assignment_exact=True))
            for name in ['cell_001.npz', 'native_initial.npz', 'input_recipe.npz', 'bank_assignment.npz']:
                path = target / name
                sources[str(path)] = sha(path)
            sequence = np.concatenate([cell['q_initial'][None], cell['q']]).astype(np.float64)
            change = np.diff(sequence, axis=0)
            path_length = np.linalg.norm(change, axis=-1).sum(0)
            ranges = np.ptp(sequence, axis=0)
            moving = np.stack([np.linalg.norm(change[..., part], axis=-1) > .0005
                               for part in [slice(0, 7), slice(7, 14), slice(14, 20), slice(20, 26)]], -1).all(-1).mean(0)
            width = np.diff(cell['joint_soft_limits'], axis=-1)[..., 0]
            assert np.all(width > 0)
            strict = (cell['official_margins'] < 0).any(-1)
            deep = (cell['official_margins'] < -.005).any(-1)
            stat = dict(strict=strict.any(0), deep=deep.any(0), path=path_length,
                        range=ranges, range_fraction=ranges / width, moving=moving,
                        risk=assignment['risk_pair_index'])
            all_stats[mode].append(stat)
            for env in range(64):
                steps = np.flatnonzero(strict[:, env])
                cases.append(dict(block=block, env=env, mode=mode,
                                  initial_negative=bool(cell['initial_violation'][env]),
                                  strict=bool(stat['strict'][env]), deep=bool(stat['deep'][env]),
                                  first_failure_step=int(steps[0]) if len(steps) else None,
                                  path_rad=float(path_length[env])))
    merged = {m: {k: np.concatenate([s[k] for s in blocks]) for k in blocks[0]} for m, blocks in all_stats.items()}
    for mode, s in merged.items():
        summaries[mode] = dict(windows=128, strict_windows=int(s['strict'].sum()),
                               deep_windows=int(s['deep'].sum()), mean_path_rad=float(s['path'].mean()),
                               mean_four_arm_moving_fraction=float(s['moving'].mean()),
                               joint_mean_span_fraction=s['range_fraction'].mean(0).tolist())
    comparisons = {}
    c = merged['velocity_arrival']
    for mode in ['adaptive_joint', 'zero_inclusive']:
        b = merged[mode]
        new_ids = np.flatnonzero(c['strict'] & ~b['strict'])
        rescued = np.flatnonzero(~c['strict'] & b['strict'])
        comparisons[mode] = dict(new_failures=[dict(block=int(i // 64), env=int(i % 64)) for i in new_ids],
                                  rescued=[dict(block=int(i // 64), env=int(i % 64)) for i in rescued],
                                  mean_path_ratio=float(c['path'].mean() / b['path'].mean()),
                                  four_arm_moving_ratio=float(c['moving'].mean() / b['moving'].mean()),
                                  risk_strata_path_ratios=[dict(risk=int(risk), ratio=float(
                                      c['path'][c['risk'] == risk].mean() / b['path'][b['risk'] == risk].mean()))
                                      for risk in np.unique(c['risk'])])
    result = dict(status='COMPLETE_INDEPENDENT_KNOWN_BANK_GEOMETRY_COMPARISON',
                  new_candidate_windows=128, prior_comparator_windows=256, unique_initial_cases=128,
                  summaries=summaries, comparisons=comparisons, exact_pairing=checks,
                  cases=cases, sources=sources, code_sha256=sha(Path(__file__)),
                  acceptance='NOT_ACCEPTED_NEW_PRIMARY_FAILURE',
                  contact_scalar_force_full_trial='UNKNOWN_NOT_RECORDED', fresh_holdout=False,
                  limitations=['Motion spans are marginal coverage; no full joint-space or task coverage claim.',
                               'Paired reused banks are development evidence; no independent confidence bound.',
                               'Geometry comparison does not validate physical contact force or grasp/load safety.'])
    assert comparisons['adaptive_joint']['new_failures'], 'Check decision if no new primary failure'
    (P / 'PAIRED_GEOMETRY.json').write_text(json.dumps(result, indent=2) + '\n')
    print(result['status'], summaries, comparisons)


if __name__ == '__main__':
    main()
