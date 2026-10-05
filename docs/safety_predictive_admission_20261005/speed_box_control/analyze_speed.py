"""Audit all frames and pair the speed-only control with existing pressure runs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'safety_mechanism_20261005'))
sys.path.insert(0, str(HERE.parent / 'safety_predictive_admission_20261005'))
from analyze_mechanism import inspect_mechanism, paired
from analyze_risk import inspect, measured_coverage
from operation_metrics import safe_operation


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--partial', action='store_true'); args = ap.parse_args()
    plan = json.loads((HERE / 'campaign_plan.json').read_text()); root = Path(plan['output_root'])
    campaign = json.loads((root / 'campaign.json').read_text())
    if not args.partial:
        assert campaign['status'] in ('complete', 'complete_with_failures') and len(campaign['jobs']) == 3
    for name, sha in plan['source_sha256'].items():
        assert hashlib.sha256((Path(plan['cwd']) / name).read_bytes()).hexdigest() == sha, name
    for name, sha in plan['research_source_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha, name
    assert hashlib.sha256(Path(plan['checkpoint_path']).read_bytes()).hexdigest() == plan['checkpoint_sha256']
    primary_root = Path('/mnt/nas/data/lyf/double_hand/safety_predictive_admission_20261005/holdout_cells')
    primary = json.loads((primary_root / 'campaign.json').read_text())
    rows, banks, data, matches = [], {}, {}, []
    for job in campaign['jobs']:
        if job['status'] == 'running':
            continue
        row, bank, trace, protocol, _ = inspect(root, job)
        manifest = json.loads((root / job['id'] / 'speed_box_manifest.json').read_text())
        assert manifest['speed_box_only'] and not manifest['actor_gating'] and not manifest['geometry_projection']
        assert manifest['strict_fifo_steps'] == 6 and not manifest['queue_preemption']
        assert manifest['runner_sha256'] == hashlib.sha256((HERE / 'speed_box_runner.py').read_bytes()).hexdigest()
        assert protocol['source_sha256'] == plan['source_sha256']
        assert protocol['checkpoint_sha256'] == plan['checkpoint_sha256']
        row['mode'] = 'speed_box_only'
        if trace is not None:
            box = protocol['effective_backstop']['vmax'] * protocol['dt']
            expected = np.clip(trace['cmd'], -box, box)
            assert np.array_equal(trace['exec'], expected), 'independent speed-only clipping oracle failed'
            previous = np.concatenate([bank['q_initial'][None], trace['controller_target'][:-1]])
            target = np.clip(previous + expected, trace['joint_soft_limits'][..., 0], trace['joint_soft_limits'][..., 1])
            assert np.array_equal(target, trace['controller_target']), 'integral/soft-limit oracle failed'
            assert row['over_nominal_box_env_steps'] == 0
            coverage = measured_coverage(bank['q_initial'], trace['q'], bank['ee_initial'], trace['ee'],
                                         bank['joint_soft_limits'], np.ones(64, dtype=bool))
            row.update(mean_within_window_joint_range=coverage['mean_within_window_joint_range'],
                       mean_joint_path_rad=coverage['mean_joint_path_rad'])
            row.update(safe_operation(bank['q_initial'], trace['q'], trace['official_margins'],
                                     bank['joint_soft_limits'], protocol['dt']))
            row.update(speed_box_oracle_exact=True, integral_soft_target_oracle_exact=True)
        rows.append(row); banks[row['id']] = bank; data[row['id']] = trace
        for mode in ('raw', 'baseline'):
            identifier = f'predictive_{row["seed"]}_{mode}'
            jobs = [j for j in primary['jobs'] if j['id'] == identifier and j['status'] == 'complete']
            if not jobs:
                continue
            other, other_bank, other_data, p, _ = inspect_mechanism(primary_root, {**jobs[0], 'mode': mode})
            for key in ('source_sha256', 'checkpoint_sha256', 'resolved_config', 'effective_backstop', 'effective_coordinator'):
                assert p[key] == protocol[key], key
            banks[other['id']] = other_bank; data[other['id']] = other_data
            matches.append(paired(other, row, banks, data))
    complete = [r for r in rows if r['status'] == 'complete']
    aggregate = []
    if complete:
        aggregate = [dict(mode='speed_box_only', windows=len(complete)*64, planned_windows=192,
                          violations=sum(r['violations'] for r in complete), deep=sum(r['deep'] for r in complete),
                          class_violations=np.sum([r['class_violations'] for r in complete], 0).tolist(),
                          mean_within_window_joint_range=float(np.mean([r['mean_within_window_joint_range'] for r in complete])),
                          mean_joint_path_rad=float(np.mean([r['mean_joint_path_rad'] for r in complete])),
                          mean_safe_duration_s=float(np.mean([r['mean_safe_duration_s'] for r in complete])),
                          mean_safe_prefix_joint_range=float(np.mean([r['mean_safe_prefix_joint_range'] for r in complete])),
                          four_arms_moving_fraction=float(np.mean([r['four_arms_moving_fraction'] for r in complete])),
                          exec_command_l2_ratio=float(np.mean([r['exec_command_l2_ratio'] for r in complete])),
                          all960_frames_audited=True)]
    result = dict(schema='safeduo.speed_box_control.v1', campaign_status=campaign['status'],
                  completed_windows=sum(r['completed_windows'] for r in rows),
                  invalid_windows=sum(r['invalid_windows'] for r in rows), pending_windows=192-len(rows)*64,
                  registered_windows=192, new_unique_random_command_windows=0, new_initial_banks=0,
                  primary_pressure_results_unchanged=True, actor_gating=False, geometry_projection=False,
                  rows=rows, aggregate_rows=aggregate, paired_inputs=matches,
                  actor_sha256=plan['checkpoint_sha256'], analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Registered after partial primary outcomes, before own outcomes; reuse all inputs; no primary reselection')
    if not args.partial:
        assert len(matches) == 6, 'all three seeds must be matched to raw and System0'
    (HERE / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('campaign_status', 'completed_windows', 'invalid_windows', 'pending_windows')}))


if __name__ == '__main__':
    main()
