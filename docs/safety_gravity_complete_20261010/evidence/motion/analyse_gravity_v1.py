"""Independent physical scoring and exact local-effort counterfactual checks."""
import json
from pathlib import Path
import sys

import numpy as np

import independent_oracle16_v1 as oracle

P = Path(__file__).resolve().parent
H = P.parent
REG = json.loads((P / 'REGISTRATION_V1.json').read_text())
ROOT = Path(REG['raw'])
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


def audit(batch, mode):
    oracle.audit(batch, mode)
    leaf = ROOT / f'paired_batch{batch}_{mode}_v8'
    p = load(leaf / 'resolved_native_parameters.npz')
    stream = load(leaf / 'response_stream.npz')
    goals = load(Path(REG['common_goal_tape']))
    events = load(leaf / 'gravity_feedforward_events.npz')
    bank = load(Path(REG['bank']))
    selected = np.asarray(REG['batch_bank_positions'][batch])
    assert np.array_equal(p['global_input_id'], bank['selected_input_id'][selected])
    assert np.array_equal(events['macro'], np.repeat(np.arange(16), 2))
    assert np.array_equal(events['substep'], np.tile(np.arange(2), 16))
    gravity = read(leaf / 'actual_body_gravity_properties.json')
    assert len(gravity) >= 82
    effort_summary = {}
    for arm in ARMS:
        for field in ['initial_q', 'initial_qd', 'full_hold_target']:
            assert np.array_equal(p[arm + '_' + field], bank[arm + '_' + field][selected])
        bounds = p[arm + '_max_force']
        g = events[arm + '_gravity_compensation_getter']
        issued = events[arm + '_issued_extra_effort']
        actual = events[arm + '_native_extra_effort']
        assert g.shape == issued.shape == actual.shape == (32, 64, bounds.shape[-1])
        assert np.isfinite(g).all() and np.isfinite(issued).all() and np.isfinite(actual).all()
        disabled = bool(p[arm + '_disable_gravity_cfg'][0])
        body_rows = [r for r in gravity if r['arm'] == arm]
        assert len(body_rows) and all(r['disable_gravity'] == disabled for r in body_rows)
        expected = np.clip(g, -bounds[None], bounds[None]) if not disabled else np.zeros_like(g)
        assert np.array_equal(issued, expected)
        assert np.array_equal(actual, issued), 'native extra effort readback mismatch ' + arm
        effort_summary[arm] = dict(actual_body_gravity_disabled=disabled,
            inspected_body_count=len(body_rows), maximum_extra_effort_native_units=float(abs(actual).max()),
            all32_events_issued_and_native_exact=True, total_PD_torque_observed=False)
    initial = np.concatenate([p[a + '_initial_q'][:, p[a + '_controlled_joint_indices']] for a in ARMS], -1)
    assert np.array_equal(stream['reference_target'], goals['reference_target'][:, selected])
    expected_centre = np.concatenate([initial[None], stream['issued_target'][:-1]])
    assert np.array_equal(stream['queued_centre'], expected_centre)
    if mode == 'bounded_raw':
        assert np.array_equal(stream['issued_target'], stream['reference_target'])
    else:
        satisfied = ((stream['multi_geometry_velocity_residual'] <= 1e-4)
            & (stream['multi_controlled_velocity_limit_residual'] <= 1e-3)
            & (stream['multi_full_drive_effort_residual'] <= 1e-3)
            & ~stream['multi_missing_direction'] & ~stream['multi_invalid_box']
            & ~stream['multi_uncontrolled_nearby_raw_negative'])
        assert np.array_equal(satisfied, stream['multi_model_constraints_satisfied'])
        assert np.array_equal(stream['fallback_from_model_unsatisfied'], ~satisfied)
        assert np.array_equal(stream['issued_target'], np.where(satisfied[..., None], stream['nominal_issued_target'], expected_centre))
        assert abs(stream['issued_target'] - expected_centre).max() <= .020001
    protocol = read(leaf / 'response_protocol.json')
    assert protocol['gravity_mode'] == 'on' and protocol['action_mode'] == mode and not protocol['initial_velocity_reset']
    assert protocol['all_microstep_native_extra_efforts_exact'] and not protocol['total_native_PD_torque_observed']
    o = read(P / f'PAIRED_ORACLE_batch{batch}_{mode}_V1.json')
    metrics = load(P / f'PAIRED_METRICS_batch{batch}_{mode}_V1.npz')
    for row in o['states']:
        row['gravity_mode'] = mode
        row['full74_and_primary_pass'] = fullpass(row)
        row['max_controlled_departure_rad'] = float(abs(metrics['controlled_q'][:, row['lane']] - initial[row['lane']]).max())
    o.update(initial_positions_velocities_targets_exact=True, actual_body_gravity_inspected=True,
             all32_event_native_extra_efforts_exact=True, effort_summary=effort_summary,
             local_effort_channel_delay_physics_steps=0, position_FIFO_controls=6,
             source_sha256=sha(Path(__file__)), fullSystem0_accepted=False)
    write(P / f'GRAVITY_ORACLE_batch{batch}_{mode}_V1.json', o)
    print('GRAVITY_AUDIT_PASS', batch, mode, effort_summary, flush=True)



def summary(rows):
    return dict(inputs=len(rows), joint_and_primary_pass=sum(fullpass(r) for r in rows),
        primary_failures=sum(r['primary_failure'] for r in rows), geometry_failures=sum(r['geometry_failure'] for r in rows),
        force_failures=sum(r['force_failure'] for r in rows),
        all74_hard_bad=sum(r['supplementary_max_all74_hard_violation_rad'] > 1e-5 for r in rows),
        all74_speed_bad=sum(r['supplementary_max_all74_velocity_exceedance_rad_s'] > 1e-5 for r in rows),
        minimum_gap_m=min(r['minimum_raw_gap_m'] for r in rows), peak_normal_N=max(r['peak_all_arm_scalar_N'] for r in rows),
        total_controlled_path_rad=sum(r['controlled_path_rad'] for r in rows),
        all4_arms_moved=sum(r['four_arms_moved'] for r in rows))


def aggregate():
    execution = read(H / 'queue_response128_v2_execution.json')
    assert execution['status'] == 'complete' and len(execution['jobs']) == 4
    assert all(j['actual_exit'] == 0 for j in execution['jobs'])
    states = {m: [] for m in REG['action_modes']}
    blocked = satisfied = 0
    for batch in range(2):
        ps = {m: load(ROOT / f'paired_batch{batch}_{m}_v8/resolved_native_parameters.npz') for m in states}
        ss = {m: load(ROOT / f'paired_batch{batch}_{m}_v8/response_stream.npz') for m in states}
        assert set(ps['bounded_raw']) == set(ps['queue_projection'])
        assert all(np.array_equal(ps['bounded_raw'][k], ps['queue_projection'][k]) for k in ps['bounded_raw'])
        assert np.array_equal(ss['bounded_raw']['reference_target'], ss['queue_projection']['reference_target'])
        for k in ['applied_target'] + ['post_' + a + '_' + f for a in ARMS for f in ['q', 'qd']]:
            assert np.array_equal(ss['bounded_raw'][k][:6], ss['queue_projection'][k][:6])
        for mode in states:
            o = read(P / f'GRAVITY_ORACLE_batch{batch}_{mode}_V1.json')
            states[mode].extend(o['states'])
        blocked += int(ss['queue_projection']['fallback_from_model_unsatisfied'].sum())
        satisfied += int(ss['queue_projection']['multi_model_constraints_satisfied'].sum())
    ids = sorted(load(Path(REG['bank']))['selected_input_id'].tolist())
    for rows in states.values():
        rows.sort(key=lambda r: r['input_id'])
        assert len(rows) == 128 and [r['input_id'] for r in rows] == ids
    pairs = dict(full74_rescues=[b['input_id'] for b, r in zip(states['bounded_raw'], states['queue_projection']) if not fullpass(b) and fullpass(r)],
        new_full74_failures=[b['input_id'] for b, r in zip(states['bounded_raw'], states['queue_projection']) if fullpass(b) and not fullpass(r)])
    cells = [dict(cell=c, modes={m: summary([r for r in rows if r['cell'] == c]) for m, rows in states.items()}) for c in range(16)]
    assert all(v['inputs'] == 8 for cell in cells for v in cell['modes'].values())
    result = dict(status='CLOSED128_QUEUE_CENTRED_RESPONSE16_DEVELOPMENT',
        counts={m: summary(rows) for m, rows in states.items()}, states=states, cells=cells, paired_changes=pairs,
        fallback_lane_controls=blocked, model_satisfied_lane_controls=satisfied, total_lane_controls=2048,
        all128_each_mode_retained=True, all4_native_and_independent32_oracles_actual_exit0=True,
        controls=16, microsteps=32, position_FIFO_controls=6, local_effort_delay_physics_steps=0,
        same_initial_states_parameters_goal_tape=True, all74_first6_states_exact=True,
        actual_native_extra_efforts_exact=True, objects=False, trained_actor=False,
        useful_motion_acceptance=False, formal_holdout=False, long_horizon_tested=False,
        production_adoption=False, fullSystem0_accepted=False, scope=REG['scope'],
        registration_sha256=sha(P / 'REGISTRATION_V1.json'), analysis_sha256=sha(Path(__file__)))
    write(P / 'QUEUE_RESPONSE128_RESULT_V1.json', result)
    print(result['status'], result['counts'], pairs, blocked, satisfied, flush=True)


if __name__ == '__main__':
    audit(int(sys.argv[1]), sys.argv[2]) if len(sys.argv) == 3 else aggregate()
