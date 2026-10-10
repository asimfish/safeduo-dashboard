"""Independent physical scoring and exact local-effort counterfactual checks."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, '/home/liyufeng/safeduo/artifacts/safety_physical_qualified_random_20261009_1613/gravity_long128_v1')
import independent_oracle480_v1 as oracle

P = Path('/home/liyufeng/safeduo/artifacts/safety_physical_qualified_random_20261009_1613/gravity_long128_v1')
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
    assert np.array_equal(events['macro'], np.repeat(np.arange(480), 2))
    assert np.array_equal(events['substep'], np.tile(np.arange(2), 480))
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
        assert g.shape == issued.shape == actual.shape == (960, 64, bounds.shape[-1])
        assert np.isfinite(g).all() and np.isfinite(issued).all() and np.isfinite(actual).all()
        disabled = bool(p[arm + '_disable_gravity_cfg'][0])
        body_rows = [r for r in gravity if r['arm'] == arm]
        assert len(body_rows) and all(r['disable_gravity'] == disabled for r in body_rows)
        expected = np.clip(g, -bounds[None], bounds[None]) if mode == 'on' and not disabled else np.zeros_like(g)
        assert np.array_equal(issued, expected)
        assert np.array_equal(actual, issued), 'native extra effort readback mismatch ' + arm
        effort_summary[arm] = dict(actual_body_gravity_disabled=disabled,
            inspected_body_count=len(body_rows), maximum_extra_effort_native_units=float(abs(actual).max()),
            all960_events_issued_and_native_exact=True, total_PD_torque_observed=False)
    initial = np.concatenate([p[a + '_initial_q'][:, p[a + '_controlled_joint_indices']] for a in ARMS], -1)
    assert np.array_equal(stream['issued_target'], np.broadcast_to(initial, (480, 64, 26)))
    assert np.array_equal(stream['applied_target'], stream['issued_target'])
    assert np.array_equal(stream['reference_target'], bank['reference_target'][:, selected])
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
             all960_event_native_extra_efforts_exact=True, effort_summary=effort_summary,
             local_effort_channel_delay_physics_steps=0, position_FIFO_controls=6,
             source_sha256=sha(Path(__file__)), fullSystem0_accepted=False)
    write(P / f'GRAVITY_ORACLE_batch{batch}_{mode}_V1.json', o)
    print('GRAVITY_AUDIT_PASS', batch, mode, effort_summary, flush=True)



def prefix_reproduction(batch):
    leaf = ROOT / f'paired_batch{batch}_on_v8'
    oldleaf = Path(REG['previous_short_root']) / f'paired_batch{batch}_on_v8'
    p, oldp = load(leaf / 'resolved_native_parameters.npz'), load(oldleaf / 'resolved_native_parameters.npz')
    assert set(p) == set(oldp) and all(np.array_equal(p[k], oldp[k]) for k in p)
    stream, oldstream = load(leaf / 'response_stream.npz'), load(oldleaf / 'response_stream.npz')
    assert set(stream) == set(oldstream)
    assert all(np.array_equal(stream[k][:16], oldstream[k]) for k in stream)
    events, oldevents = load(leaf / 'gravity_feedforward_events.npz'), load(oldleaf / 'gravity_feedforward_events.npz')
    assert set(events) == set(oldevents)
    assert all(np.array_equal(events[k][:32], oldevents[k]) for k in events)
    geo, oldgeo = load(leaf / 'all_raw_geometry_0016.npz'), load(oldleaf / 'all_raw_geometry_0016.npz')
    assert set(geo) == set(oldgeo) and all(np.array_equal(geo[k], oldgeo[k]) for k in geo)
    metrics = load(P / f'PAIRED_METRICS_batch{batch}_on_V1.npz')
    oldmetrics = load(H / 'gravity_hold128_v1' / f'PAIRED_METRICS_batch{batch}_on_V1.npz')
    for k in ['gap_m', 'force_N', 'hard_violation_rad', 'velocity_exceedance_rad_s']:
        assert np.array_equal(metrics[k][:32], oldmetrics[k])
    return dict(batch=batch, status='PASS_EXACT_ON_PREFIX_ALL32_MICROSTEPS')


def summary(rows):
    return dict(inputs=len(rows), joint_and_primary_pass=sum(fullpass(r) for r in rows),
        primary_failures=sum(r['primary_failure'] for r in rows), geometry_failures=sum(r['geometry_failure'] for r in rows),
        force_failures=sum(r['force_failure'] for r in rows),
        all74_hard_bad=sum(r['supplementary_max_all74_hard_violation_rad'] > 1e-5 for r in rows),
        all74_speed_bad=sum(r['supplementary_max_all74_velocity_exceedance_rad_s'] > 1e-5 for r in rows),
        minimum_gap_m=min(r['minimum_raw_gap_m'] for r in rows), peak_normal_N=max(r['peak_all_arm_scalar_N'] for r in rows),
        total_controlled_path_rad=sum(r['controlled_path_rad'] for r in rows))


def aggregate():
    first = read(H / 'gravity_long128_v1_execution.json')
    second = read(H / 'gravity_long_completion_v2_execution.json')
    assert first['status'] == 'failed' and len(first['jobs']) == 1
    assert first['jobs'][0]['id'] == 'batch0_on' and first['jobs'][0]['actual_exit'] == 0 and first['jobs'][0]['status'] == 'complete'
    assert first['error'] == 'RuntimeError: selected GPU resource gate failed; no child launched'
    assert second['status'] == 'complete' and len(second['jobs']) == 1
    assert second['jobs'][0]['id'] == 'batch1_on' and second['jobs'][0]['actual_exit'] == 0 and second['jobs'][0]['status'] == 'complete'
    assert read(P / 'PARTIAL_AUDIT_ACTUAL_EXIT_V1.json')['actual_exit'] == 0
    states = dict(off=[], on=[])
    reproduction = []
    for batch in range(2):
        reproduction.append(prefix_reproduction(batch))
        p = load(ROOT / f'paired_batch{batch}_on_v8/resolved_native_parameters.npz')
        oldp = load(Path(REG['baseline_long_root']) / f'paired_batch{batch}_multirow_hold_fallback_v8/resolved_native_parameters.npz')
        assert set(oldp) <= set(p) and all(np.array_equal(p[k], oldp[k]) for k in oldp)
        on = read(P / f'GRAVITY_ORACLE_batch{batch}_on_V1.json')
        assert on['all960_event_native_extra_efforts_exact'] and len(on['states']) == 64
        off = read(Path(REG['baseline_long_metrics_root']) / f'STRONG_ORACLE_batch{batch}_multirow_hold_fallback_V1.json')
        assert len(off['states']) == 64
        states['on'].extend(on['states'])
        states['off'].extend(off['states'])
    for rows in states.values():
        rows.sort(key=lambda r: r['input_id'])
        assert len(rows) == 128
    ids = sorted(load(Path(REG['bank']))['selected_input_id'].tolist())
    assert all([r['input_id'] for r in rows] == ids for rows in states.values())
    paired = dict(full74_rescues=[b['input_id'] for b, r in zip(states['off'], states['on']) if not fullpass(b) and fullpass(r)],
        new_full74_failures=[b['input_id'] for b, r in zip(states['off'], states['on']) if fullpass(b) and not fullpass(r)])
    cells = [dict(cell=c, modes={m: summary([r for r in rows if r['cell'] == c]) for m, rows in states.items()}) for c in range(16)]
    assert all(v['inputs'] == 8 for cell in cells for v in cell['modes'].values())
    result = dict(status='CLOSED128_GRAVITY_SUPPORT_HOLD480_DEVELOPMENT',
        counts={m: summary(rows) for m, rows in states.items()}, states=states, cells=cells, paired_changes=paired,
        all128_each_mode_retained=True, new_native_jobs=2, new_native_actual_exit0=True,
        independent_all960_microsteps=True, exact_on_prefix_reproduction=reproduction,
        off_baseline_reused=True, original_initial_velocities_preserved=True,
        native_completed_across_two_attempts=True, first_attempt_resource_nochild_preserved=True,
        position_FIFO_controls=6, local_effort_delay_physics_steps=0,
        actual_perbody_gravity_properties_verified=True, all_native_extra_effort_commands_exact=True,
        controls=480, microsteps=960, objects=False, trained_actor=False,
        useful_motion_tested=False, formal_holdout=False, production_adoption=False, fullSystem0_accepted=False,
        long_hold_candidate_pass=summary(states['on'])['joint_and_primary_pass'] == 128 and not paired['new_full74_failures'],
        registration_sha256=sha(P / 'REGISTRATION_V1.json'), analysis_sha256=sha(Path(__file__)),
        scope=REG['scope'])
    write(P / 'GRAVITY_LONG128_RESULT_V1.json', result)
    print(result['status'], result['counts'], paired, flush=True)


if __name__ == '__main__':
    audit(int(sys.argv[1]), sys.argv[2]) if len(sys.argv) == 3 else aggregate()
