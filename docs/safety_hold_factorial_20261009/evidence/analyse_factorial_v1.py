"""Audit two predefined interventions and retain every development input."""
import json
from pathlib import Path
import sys

import numpy as np

import independent_oracle16_v1 as oracle

P = Path(__file__).resolve().parent
H = P.parent
REG = json.loads((P / 'REGISTRATION_V1.json').read_text())
ROOT = Path(REG['raw'])
FACTORS = REG['factors']
ARMS = oracle.ARMS
load = oracle.load
sha = oracle.sha


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def fullpass(row):
    return (not row['primary_failure']
            and row['supplementary_max_all74_hard_violation_rad'] <= 1e-5
            and row['supplementary_max_all74_velocity_exceedance_rad_s'] <= 1e-5)


def audit(batch, factor):
    oracle.audit(batch, factor)
    leaf = ROOT / f'paired_batch{batch}_{factor}_v8'
    p = load(leaf / 'resolved_native_parameters.npz')
    s = load(leaf / 'response_stream.npz')
    bank = load(Path(REG['bank']))
    selected = np.asarray(REG['batch_bank_positions'][batch])
    roots = read(leaf / 'actual_solver_roots.json')
    assert len(roots) == 4
    assert all(r['position_iterations'] == 64 and r['velocity_iterations'] == 0 for r in roots)
    assert np.array_equal(p['global_input_id'], bank['selected_input_id'][selected])
    for key in ['cell_id', 'velocity_fraction', 'command_mode', 'refresh_controls']:
        assert np.array_equal(p[key], bank[key][selected])
    for arm in ARMS:
        idx = p[arm + '_controlled_joint_indices']
        hand = np.ones(p[arm + '_initial_q'].shape[-1], bool)
        hand[idx] = False
        expected_v = bank[arm + '_initial_qd'][selected].copy()
        expected_hold = bank[arm + '_full_hold_target'][selected].copy()
        if factor in ['zero_arm_velocity', 'both']:
            expected_v[:, idx] = 0
        if factor in ['matched_hand_target', 'both']:
            expected_hold[:, hand] = bank[arm + '_initial_q'][selected][:, hand]
        assert np.array_equal(p[arm + '_initial_q'], bank[arm + '_initial_q'][selected])
        assert np.array_equal(p[arm + '_baseline_initial_qd'], bank[arm + '_initial_qd'][selected])
        assert np.array_equal(p[arm + '_initial_qd'], expected_v)
        assert np.array_equal(p[arm + '_full_hold_target'], expected_hold)
        assert np.array_equal(p[arm + '_hard_limits'], bank[arm + '_hard_limits'][selected])
        assert np.array_equal(p[arm + '_max_velocity'], bank[arm + '_vmax'][selected])
    initial = np.concatenate([p[a + '_initial_q'][:, p[a + '_controlled_joint_indices']] for a in ARMS], -1)
    assert np.array_equal(s['issued_target'], np.broadcast_to(initial, (16, 64, 26)))
    assert np.array_equal(s['applied_target'], s['issued_target'])
    assert s['scripted_hold_issued'].all()
    assert np.array_equal(s['reference_target'], bank['reference_target'][:16, selected])
    protocol = read(leaf / 'response_protocol.json')
    assert protocol['factor'] == factor and protocol['controller'] == 'scripted_initial_hold'
    assert protocol['simulation_initial_state_intervention_only'] and not protocol['runtime_braking']
    o = read(P / f'PAIRED_ORACLE_batch{batch}_{factor}_V1.json')
    metrics = load(P / f'PAIRED_METRICS_batch{batch}_{factor}_V1.npz')
    for row in o['states']:
        lane = row['lane']
        row['factor'] = factor
        row['full74_and_primary_pass'] = fullpass(row)
        row['max_controlled_departure_rad'] = float(abs(metrics['controlled_q'][:, lane] - initial[lane]).max())
    o.update(factor_intervention_exact=True, all4_native_roots64_velocity0=True,
             horizon_controls=16, horizon_microsteps=32, source_sha256=sha(Path(__file__)),
             runtime_braking=False, fullSystem0_accepted=False)
    write(P / f'FACTOR_ORACLE_batch{batch}_{factor}_V1.json', o)
    print('FACTOR_AUDIT_PASS', batch, factor, flush=True)


def summary(rows):
    return dict(inputs=len(rows), primary_failures=sum(r['primary_failure'] for r in rows),
                geometry_failures=sum(r['geometry_failure'] for r in rows),
                force_failures=sum(r['force_failure'] for r in rows),
                all74_hard_bad=sum(r['supplementary_max_all74_hard_violation_rad'] > 1e-5 for r in rows),
                all74_speed_bad=sum(r['supplementary_max_all74_velocity_exceedance_rad_s'] > 1e-5 for r in rows),
                joint_and_primary_pass=sum(fullpass(r) for r in rows),
                minimum_gap_m=min(r['minimum_raw_gap_m'] for r in rows),
                peak_normal_N=max(r['peak_all_arm_scalar_N'] for r in rows),
                max_controlled_departure_rad=max(r['max_controlled_departure_rad'] for r in rows),
                total_controlled_path_rad=sum(r['controlled_path_rad'] for r in rows))


def baseline_reproduction(batch):
    leaf = ROOT / f'paired_batch{batch}_baseline_v8'
    oldleaf = Path(REG['baseline_long_root']) / f'paired_batch{batch}_multirow_hold_fallback_v8'
    p, oldp = load(leaf / 'resolved_native_parameters.npz'), load(oldleaf / 'resolved_native_parameters.npz')
    assert set(oldp) <= set(p)
    for key in oldp:
        assert np.array_equal(p[key], oldp[key]), 'baseline parameter ' + key
    s, olds = load(leaf / 'response_stream.npz'), load(oldleaf / 'response_stream.npz')
    physical_fields = ['issued_target', 'applied_target', 'reference_target', 'pre_pending',
                       'substep_min_raw_gap_m', 'scalar_normal_max_N']
    physical_fields += [stage + '_' + a + '_' + f for stage in ['pre', 'post'] for a in ARMS for f in ['q', 'qd']]
    for key in physical_fields:
        assert np.array_equal(s[key], olds[key][:16]), 'baseline physical stream ' + key
    gr = read(leaf / 'all_raw_geometry_receipts.json')
    oldgr = read(oldleaf / 'all_raw_geometry_receipts.json')
    assert len(gr['chunks']) == 1 and gr['chunks'][0]['events'] == 32
    d = load(leaf / gr['chunks'][0]['path'])
    oldd = load(oldleaf / oldgr['chunks'][0]['path'])
    assert np.array_equal(d['distances'], oldd['distances'][:32])
    newm = load(P / f'PAIRED_METRICS_batch{batch}_baseline_V1.npz')
    oldm = load(Path(REG['baseline_long_metrics_root']) / f'PAIRED_METRICS_batch{batch}_multirow_hold_fallback_V1.npz')
    for key in ['global_input_id', 'cell_id']:
        assert np.array_equal(newm[key], oldm[key])
    for key in ['gap_m', 'force_N', 'hard_violation_rad', 'velocity_exceedance_rad_s']:
        assert np.array_equal(newm[key], oldm[key][:32]), 'baseline every-microstep metric ' + key
    assert np.array_equal(newm['controlled_q'], oldm['controlled_q'][:17])
    return dict(batch=batch, status='PASS_EXACT_ORIGINAL_FIRST16_NATIVE_BASELINE',
                all_original_native_parameters_exact=True, all74_q_qd_and_FIFO_targets_exact=True,
                all32x9021_geometry_rows_exact=True, all32_point_force_hard_speed_metrics_exact=True)


def aggregate():
    execution = read(H / 'hold_factorial128_v1_execution.json')
    assert execution['status'] == 'complete' and len(execution['jobs']) == 8
    assert all(j['status'] == 'complete' and j['actual_exit'] == 0 for j in execution['jobs'])
    reproduction = [baseline_reproduction(batch) for batch in range(2)]
    write(P / 'BASELINE_REPRODUCTION_V1.json', dict(status='PASS_ALL128_EXACT_FIRST16_NATIVE_BASELINE', records=reproduction))
    states = {factor: [] for factor in FACTORS}
    for batch in range(2):
        params = {factor: load(ROOT / f'paired_batch{batch}_{factor}_v8/resolved_native_parameters.npz') for factor in FACTORS}
        base = params['baseline']
        for factor, p in params.items():
            assert set(p) == set(base)
            mutable = {a + '_' + f for a in ARMS for f in ['initial_qd', 'full_hold_target']}
            for key in base:
                if key not in mutable:
                    assert np.array_equal(p[key], base[key]), 'unexpected factor parameter change ' + factor + ' ' + key
            o = read(P / f'FACTOR_ORACLE_batch{batch}_{factor}_V1.json')
            assert o['factor_intervention_exact'] and len(o['states']) == 64
            states[factor].extend(o['states'])
    ids = sorted(load(Path(REG['bank']))['selected_input_id'].tolist())
    for factor, rows in states.items():
        rows.sort(key=lambda r: r['input_id'])
        assert len(rows) == 128 and [r['input_id'] for r in rows] == ids
    comparisons = {}
    for factor in FACTORS[1:]:
        comparisons[factor] = dict(
            rescued_primary_ids=[b['input_id'] for b, r in zip(states['baseline'], states[factor]) if b['primary_failure'] and not r['primary_failure']],
            new_primary_failure_ids=[b['input_id'] for b, r in zip(states['baseline'], states[factor]) if not b['primary_failure'] and r['primary_failure']],
            rescued_full74_ids=[b['input_id'] for b, r in zip(states['baseline'], states[factor]) if not fullpass(b) and fullpass(r)],
            new_full74_failure_ids=[b['input_id'] for b, r in zip(states['baseline'], states[factor]) if fullpass(b) and not fullpass(r)])
    original_failures = sorted(r['input_id'] for r in read(Path(REG['baseline_long_result']))['states']['multirow_hold_fallback'] if r['primary_failure'])
    baseline_failures = sorted(r['input_id'] for r in states['baseline'] if r['primary_failure'])
    assert baseline_failures == original_failures, 'all10 onset failures must reproduce within registered16 controls'
    cells = [dict(cell=c, factors={f: summary([r for r in rows if r['cell'] == c]) for f, rows in states.items()}) for c in range(16)]
    assert all(v['inputs'] == 8 for cell in cells for v in cell['factors'].values())
    result = dict(status='CLOSED_MATCHED128_NATIVE_INITIAL_HOLD_FACTORIAL16_DEVELOPMENT',
                  inputs=128, controls_each=16, microsteps_each=32, counts={f: summary(rows) for f, rows in states.items()},
                  states=states, cells=cells, paired_changes=comparisons, original10_onset_failures_reproduced=original_failures,
                  all8_actual_native_exit0=True, all8_independent_point_geometry_FIFO_oracles_pass=True,
                  baseline_exact_reproduction=True, intervention_parameters_exact=True, all128_each_factor_retained=True,
                  registration_sha256=sha(P / 'REGISTRATION_V1.json'), analysis_sha256=sha(Path(__file__)),
                  formal_holdout=False, long_horizon_intervention_tested=False, runtime_braking=False,
                  objects=False, trained_actor=False, production_adoption=False, fullSystem0_accepted=False,
                  scope='Matched simulator development intervention for early hold failures; velocity reset changes initial conditions and is not a runtime safety action. No 480-control intervention result.')
    write(P / 'FACTORIAL128_RESULT_V1.json', result)
    print(result['status'], result['counts'], flush=True)


if __name__ == '__main__':
    audit(int(sys.argv[1]), sys.argv[2]) if len(sys.argv) == 3 else aggregate()
