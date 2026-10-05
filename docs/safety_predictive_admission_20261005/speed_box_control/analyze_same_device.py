"""The primary speed-limit comparison uses only the same physical GPU1."""
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
    plan = json.loads((HERE / 'comparison_plan.json').read_text()); root = Path(plan['output_root'])
    campaign = json.loads((root / 'campaign.json').read_text())
    speed_plan = json.loads((HERE / 'campaign_plan.json').read_text()); speed_root = Path(speed_plan['output_root'])
    speed_campaign = json.loads((speed_root / 'campaign.json').read_text())
    assert campaign['status'] in ('complete', 'complete_with_failures') and len(campaign['jobs']) == 6
    assert speed_campaign['status'] in ('complete', 'complete_with_failures') and len(speed_campaign['jobs']) == 3
    for name, sha in plan['source_sha256'].items():
        assert hashlib.sha256((Path(plan['cwd']) / name).read_bytes()).hexdigest() == sha, name
    for name, sha in plan['research_source_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha, name
    rows, banks, data, protocols = [], {}, {}, []
    specs = {j['id']: j for j in plan['jobs']}
    for job in campaign['jobs']:
        row, bank, trace, p, _ = inspect_mechanism(root, {**job, 'mode': specs[job['id']]['mode']})
        rows.append(row); banks[row['id']] = bank; data[row['id']] = trace; protocols.append(p)
    audited = json.loads((HERE / 'results.json').read_text())
    assert audited['completed_windows'] == 192 and audited['invalid_windows'] == 0
    for job in speed_campaign['jobs']:
        row, bank, trace, p, _ = inspect(speed_root, job)
        row.update(mode='speed_box_only', speed_box_oracle_exact=True)
        assert next(r for r in audited['rows'] if r['id'] == row['id'])['speed_box_oracle_exact']
        rows.append(row); banks[row['id']] = bank; data[row['id']] = trace; protocols.append(p)
    for p in protocols:
        assert p['args']['device'] == 'cuda:1'
        for key in ('source_sha256', 'checkpoint_sha256', 'resolved_config', 'effective_backstop', 'effective_coordinator', 'gpu'):
            assert p[key] == protocols[0][key], key
    aggregate = []
    for mode in ('raw', 'baseline', 'speed_box_only'):
        sel = [r for r in rows if r['mode'] == mode and r['status'] == 'complete']
        assert len(sel) == 3, f'{mode}: incomplete or invalid conditions remain'
        for row in sel:
            b, d = banks[row['id']], data[row['id']]
            c = measured_coverage(b['q_initial'], d['q'], b['ee_initial'], d['ee'], b['joint_soft_limits'], np.ones(64, bool))
            row.update(mean_within_window_joint_range=c['mean_within_window_joint_range'], mean_joint_path_rad=c['mean_joint_path_rad'])
            row.update(safe_operation(b['q_initial'], d['q'], d['official_margins'], b['joint_soft_limits'], protocols[0]['dt']))
        aggregate.append(dict(mode=mode, windows=192, planned_windows=192,
                              violations=sum(r['violations'] for r in sel), deep=sum(r['deep'] for r in sel),
                              class_violations=np.sum([r['class_violations'] for r in sel], 0).tolist(),
                              **{k: float(np.mean([r[k] for r in sel])) for k in
                                 ('mean_within_window_joint_range', 'mean_joint_path_rad', 'mean_safe_duration_s',
                                  'mean_safe_prefix_joint_range', 'four_arms_moving_fraction', 'exec_command_l2_ratio')}))
    matches = []
    for seed in plan['seeds']:
        speed = next(r for r in rows if r['seed'] == seed and r['mode'] == 'speed_box_only')
        for mode in ('raw', 'baseline'):
            left = next(r for r in rows if r['seed'] == seed and r['mode'] == mode)
            matches.append(paired(left, speed, banks, data))
    result = dict(schema='safeduo.speed_box_same_device.v1', completed_windows=192, invalid_windows=0, pending_windows=0,
                  comparison_completed_windows=384, supplementary_total_windows=576,
                  new_unique_random_command_windows=0, new_initial_banks=0, primary_pressure_results_unchanged=True,
                  actor_gating=False, geometry_projection=False, comparison_device='cuda:1', gpu=protocols[0]['gpu'],
                  aggregate_rows=aggregate, rows=rows, paired_inputs=matches,
                  same_device=True, source_frozen=True, all_frames_audited=True,
                  registration_timing='partial primary outcomes seen; own speed-only conditions completed before comparator registration; own speed-only outcomes not analyzed until comparator freeze',
                  initial_and_external_inputs_exact=True, actual_trajectories_not_assumed_exact=True,
                  analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE / 'same_device_results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps([(r['mode'], r['violations'], r['windows']) for r in aggregate]))


if __name__ == '__main__':
    main()
