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
        expected = np.clip(g, -bounds[None], bounds[None]) if mode == 'on' and not disabled else np.zeros_like(g)
        assert np.array_equal(issued, expected)
        assert np.array_equal(actual, issued), 'native extra effort readback mismatch ' + arm
        effort_summary[arm] = dict(actual_body_gravity_disabled=disabled,
            inspected_body_count=len(body_rows), maximum_extra_effort_native_units=float(abs(actual).max()),
            all32_events_issued_and_native_exact=True, total_PD_torque_observed=False)
    initial = np.concatenate([p[a + '_initial_q'][:, p[a + '_controlled_joint_indices']] for a in ARMS], -1)
    assert np.array_equal(stream['issued_target'], np.broadcast_to(initial, (16, 64, 26)))
    assert np.array_equal(stream['applied_target'], stream['issued_target'])
    assert np.array_equal(stream['reference_target'], bank['reference_target'][:16, selected])
    protocol = read(leaf / 'response_protocol.json')
    assert protocol['gravity_mode'] == mode and not protocol['initial_velocity_reset']
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


def baseline_reproduction(batch):
    leaf = ROOT / f'paired_batch{batch}_off_v8'
    oldleaf = Path(REG['baseline_long_root']) / f'paired_batch{batch}_multirow_hold_fallback_v8'
    p, oldp = load(leaf / 'resolved_native_parameters.npz'), load(oldleaf / 'resolved_native_parameters.npz')
    assert set(oldp) <= set(p)
    assert all(np.array_equal(p[k], oldp[k]) for k in oldp)
    stream, oldstream = load(leaf / 'response_stream.npz'), load(oldleaf / 'response_stream.npz')
    keys = ['issued_target', 'applied_target', 'reference_target', 'pre_pending', 'scalar_normal_max_N', 'substep_min_raw_gap_m']
    keys += [t + '_' + a + '_' + f for t in ['pre', 'post'] for a in ARMS for f in ['q', 'qd']]
    assert all(np.array_equal(stream[k], oldstream[k][:16]) for k in keys)
    metrics = load(P / f'PAIRED_METRICS_batch{batch}_off_V1.npz')
    oldmetrics = load(Path(REG['baseline_long_metrics_root']) / f'PAIRED_METRICS_batch{batch}_multirow_hold_fallback_V1.npz')
    for k in ['gap_m', 'force_N', 'hard_violation_rad', 'velocity_exceedance_rad_s']:
        assert np.array_equal(metrics[k], oldmetrics[k][:32]), 'baseline microstep metric ' + k
    geometry = load(leaf / 'all_raw_geometry_0016.npz')
    oldgeometry = load(oldleaf / 'all_raw_geometry_0016.npz')
    assert np.array_equal(geometry['distances'], oldgeometry['distances'])
    contacts = read(leaf / 'point_contact_receipts.json')
    oldcontacts = read(oldleaf / 'point_contact_receipts.json')
    assert len(contacts['chunks']) == 1
    new = load(leaf / contacts['chunks'][0]['path'])
    old = load(oldleaf / oldcontacts['chunks'][0]['path'])
    for arm in ARMS:
        for f in ['native_q', 'native_qd', 'native_position_targets']:
            assert np.array_equal(new[arm + '_' + f], old[arm + '_' + f][:32])
    return dict(batch=batch, status='PASS_EXACT_OFF_BASELINE_ALL32_MICROSTEP_ALL74_STATES_GEOMETRY_FORCE_AND_FIFO')


def summary(rows):
    return dict(inputs=len(rows), joint_and_primary_pass=sum(fullpass(r) for r in rows),
        primary_failures=sum(r['primary_failure'] for r in rows), geometry_failures=sum(r['geometry_failure'] for r in rows),
        force_failures=sum(r['force_failure'] for r in rows),
        all74_hard_bad=sum(r['supplementary_max_all74_hard_violation_rad'] > 1e-5 for r in rows),
        all74_speed_bad=sum(r['supplementary_max_all74_velocity_exceedance_rad_s'] > 1e-5 for r in rows),
        minimum_gap_m=min(r['minimum_raw_gap_m'] for r in rows), peak_normal_N=max(r['peak_all_arm_scalar_N'] for r in rows),
        max_controlled_departure_rad=max(r['max_controlled_departure_rad'] for r in rows),
        total_controlled_path_rad=sum(r['controlled_path_rad'] for r in rows))


def aggregate():
    execution = read(H / 'gravity_hold128_v1_execution.json')
    assert execution['status'] == 'complete' and len(execution['jobs']) == 4
    assert all(j['actual_exit'] == 0 and j['status'] == 'complete' for j in execution['jobs'])
    reproduction = [baseline_reproduction(b) for b in range(2)]
    write(P / 'OFF_BASELINE_REPRODUCTION_V1.json', dict(status='PASS_ALL128_EXACT_OFF_BASELINE', records=reproduction))
    states = {m: [] for m in ['off', 'on']}
    for batch in range(2):
        p = {m: load(ROOT / f'paired_batch{batch}_{m}_v8/resolved_native_parameters.npz') for m in states}
        assert set(p['off']) == set(p['on'])
        assert all(np.array_equal(p['off'][k], p['on'][k]) for k in p['off'])
        for mode in states:
            o = read(P / f'GRAVITY_ORACLE_batch{batch}_{mode}_V1.json')
            assert o['all32_event_native_extra_efforts_exact'] and len(o['states']) == 64
            states[mode].extend(o['states'])
    for rows in states.values():
        rows.sort(key=lambda r: r['input_id'])
        assert len(rows) == 128
    ids = sorted(load(Path(REG['bank']))['selected_input_id'].tolist())
    assert all([r['input_id'] for r in rows] == ids for rows in states.values())
    paired = dict(primary_rescues=[b['input_id'] for b, r in zip(states['off'], states['on']) if b['primary_failure'] and not r['primary_failure']],
        new_primary_failures=[b['input_id'] for b, r in zip(states['off'], states['on']) if not b['primary_failure'] and r['primary_failure']],
        full74_rescues=[b['input_id'] for b, r in zip(states['off'], states['on']) if not fullpass(b) and fullpass(r)],
        new_full74_failures=[b['input_id'] for b, r in zip(states['off'], states['on']) if fullpass(b) and not fullpass(r)])
    cells = [dict(cell=c, modes={m: summary([r for r in rows if r['cell'] == c]) for m, rows in states.items()}) for c in range(16)]
    assert all(v['inputs'] == 8 for cell in cells for v in cell['modes'].values())
    result = dict(status='CLOSED_MATCHED128_LOCAL_GRAVITY_FEEDFORWARD_HOLD16_DEVELOPMENT',
        counts={m: summary(rows) for m, rows in states.items()}, states=states, cells=cells, paired_changes=paired,
        all128_each_mode_retained=True, all4_native_and_independent32_oracles_pass=True,
        all4_native_actual_exit0=True, off_baseline_exact_all32_microstep_reproduction=True,
        same_initial_positions_velocities_hand_targets_roots_gains_limits=True,
        actual_perbody_gravity_properties_verified=True, all_native_extra_effort_commands_exact=True,
        controls=16, microsteps=32, position_FIFO_controls=6, local_effort_delay_physics_steps=0,
        initial_velocity_reset=False, physics_gravity_disabled_as_intervention=False,
        total_implicit_PD_torque_observed=False, trained_actor=False, objects=False,
        long_horizon_tested=False, formal_holdout=False, production_adoption=False, fullSystem0_accepted=False,
        registration_sha256=sha(P / 'REGISTRATION_V1.json'), analysis_sha256=sha(Path(__file__)),
        scope='Local-effort support intervention under original states; not policy improvement or task success. Must use common support in future paired policy comparisons.')
    write(P / 'GRAVITY_HOLD128_RESULT_V1.json', result)
    print(result['status'], result['counts'], paired, flush=True)


if __name__ == '__main__':
    audit(int(sys.argv[1]), sys.argv[2]) if len(sys.argv) == 3 else aggregate()
