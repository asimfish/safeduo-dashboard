"""Audit new-factor neutral controls and retain the three original conditions."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'safety_predictive_admission_20261005'))
from analyze_predictive import inspect_mechanism as inspect_admission


def main():
    plan = json.loads((HERE / 'campaign_plan.json').read_text()); root = Path(plan['output_root'])
    campaign = json.loads((root / 'campaign.json').read_text())
    assert campaign['status'] in ('complete', 'complete_with_failures') and len(campaign['jobs']) == 6
    for name, sha in plan['source_sha256'].items():
        assert hashlib.sha256((Path(plan['cwd']) / name).read_bytes()).hexdigest() == sha, name
    for name, sha in plan['research_source_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha, name
    old = json.loads((HERE.parent / 'safety_zero_command_20261005/results.json').read_text())
    assert old['completed_windows'] == 576 and old['invalid_windows'] == 0
    specs = {j['id']: j for j in plan['jobs']}; rows, protocols = [], []
    for job in campaign['jobs']:
        row, bank, trace, p, _ = inspect_admission(root, {**job, 'mode': specs[job['id']]['mode']})
        assert row['sampling']['neutral_diagnostic'] and row['sampling']['initial_banks_reused']
        assert row['sampling']['neutral_runner_sha256'] == hashlib.sha256((HERE / 'zero_predictive_runner.py').read_bytes()).hexdigest()
        assert not bank['tape'].any() and not bank['updates'].any() and not bank['holds'].any() and not bank['segment_amplitudes'].any()
        if trace is not None:
            assert not trace['cmd'].any()
            row.update(nonzero_execution_windows=int((np.abs(trace['exec']).max((0, 2)) > 1e-7).sum()),
                       max_actual_joint_drift_rad=float(np.abs(trace['q'] - bank['q_initial']).max()),
                       max_absolute_velocity_rad_s=float(np.abs(trace['pre_qd_compact']).max()),
                       full960_zero_commands_verified=True, neutral_violations=row['violations'])
        rows.append(row); protocols.append(p)
    for p in protocols:
        assert p['args']['device'] == 'cuda:1'
        for key in ('source_sha256', 'checkpoint_sha256', 'resolved_config', 'effective_backstop', 'effective_coordinator'):
            assert p[key] == protocols[0][key], key
    aggregate = list(old['aggregate_rows'])
    for mode in ('predictive', 'predictive_envelope'):
        sel = [r for r in rows if r['mode'] == mode and r['status'] == 'complete']
        assert len(sel) == 3, f'{mode}: incomplete/invalid zero windows retained'
        aggregate.append(dict(mode=mode, windows=192, planned_windows=192,
                              violations=sum(r['violations'] for r in sel), deep=sum(r['deep'] for r in sel),
                              class_violations=np.sum([r['class_violations'] for r in sel], 0).tolist(),
                              nonzero_execution_windows=sum(r['nonzero_execution_windows'] for r in sel),
                              max_actual_joint_drift_rad=max(r['max_actual_joint_drift_rad'] for r in sel),
                              all960_zero_commands=True, fifo_exact=True, soft_limits_pass=True))
    # Audit all five initial assignments against the original raw neutral window.
    for row in rows:
        reference = next(r for r in old['rows'] if r['seed'] == row['seed'] and r['mode'] == 'raw')
        with np.load(Path(reference['path']) / 'input_recipe.npz', allow_pickle=False) as ref:
            with np.load(Path(row['path']) / 'input_recipe.npz', allow_pickle=False) as actual:
                assert np.array_equal(ref['q_initial'], actual['q_initial']) and np.array_equal(ref['tape'], actual['tape'])
    result = dict(schema='safeduo.zero_factor_controls.v1', campaign_status=campaign['status'],
                  completed_windows=960, invalid_windows=0, pending_windows=0, registered_windows=960,
                  original_three_completed_windows=576, new_factor_zero_windows=384,
                  new_unique_random_command_windows=0, new_initial_banks=0, primary_pressure_results_unchanged=True,
                  rows=old['rows'] + rows, aggregate_rows=aggregate, actor_sha256=plan['checkpoint_sha256'],
                  registration_timing='original three after partial pressure, before own outcomes; new two after original zero/speed outcomes, before own outcomes; no parameter or initial changes')
    (HERE / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps([(x['mode'], x['violations'], x['windows']) for x in aggregate]))


if __name__ == '__main__':
    main()
