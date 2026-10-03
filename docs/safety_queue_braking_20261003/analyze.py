"""Audit matched inputs, unchanged FIFO and actual post-delivery movement."""
import hashlib
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
OLD = OUT.parent / 'safety_latency_forensics_20261003'
CORE = ['src/safeduo/safety/backstop.py', 'src/safeduo/envs/duo_env.py',
        'src/safeduo/eval/research_battery.py', 'src/safeduo/eval/perturbations.py']


def load(path):
    with np.load(path) as n:
        keys = ['q', 'pre_q', 'pre_qd', 'pre_target', 'exec', 'cmd', 'official_margins', 'sphere_centers',
                'q_initial', 'initial_violation', 'joint_soft_limits']
        keys += [k for k in n.files if k.startswith('stop_')]
        keys += [k for k in ['controller_target', 'actuator_target'] if k in n.files]
        d = {k: n[k] for k in keys}
        d['meta'] = json.loads(str(n['meta_json']))
        for k in keys:
            assert np.isfinite(d[k]).all(), (path, k)
        if 'controller_target' not in d:
            limits = d['joint_soft_limits']
            d['controller_target'] = np.clip(d['pre_target'] + d['exec'], limits[..., 0], limits[..., 1])
            assert np.array_equal(d['controller_target'][:-1], d['pre_target'][1:])
            d['actuator_target'] = d['controller_target'].copy()
        return d


def fifo_audit(d):
    lag = d['meta']['actuator_delay_steps']
    issued, delivered = d['controller_target'], d['actuator_target']
    expected = np.concatenate([np.repeat(d['pre_target'][0:1], lag, axis=0), issued[:-lag]], 0) if lag else issued
    assert np.array_equal(expected, delivered), 'delivery differs from strict FIFO'
    return dict(steps=lag, exact=True, max_target_error_rad=0.)


def matched(d, reference):
    assert np.array_equal(d['q_initial'], reference['q_initial']), 'unmatched initial pose'
    assert np.array_equal(d['cmd'], reference['cmd']), 'unmatched command tape'
    return dict(initial_exact=True, command_exact=True,
                q_initial_sha256=hashlib.sha256(d['q_initial'].tobytes()).hexdigest(),
                command_sha256=hashlib.sha256(d['cmd'].tobytes()).hexdigest())


def first_bad(d, e, start=0):
    x = np.flatnonzero((d['official_margins'][start:, e] < 0).any(-1))
    return int(x[0] + start) if x.size else None


def main():
    old_protocol = json.loads((OLD / 'delayed_critical/protocol.json').read_text())
    p = json.loads((OUT / 'delay_controls/protocol.json').read_text())
    assert p['status'] == 'complete' and p['completed_cells'] == 3
    assert p['source_sha256'] == old_protocol['source_sha256']
    assert p['resolved_config'] == old_protocol['resolved_config']
    assert p['effective_coordinator'] == old_protocol['effective_coordinator']
    assert p['effective_backstop'] == old_protocol['effective_backstop']
    assert p['checkpoint_sha256'] == old_protocol['checkpoint_sha256']
    for k in CORE:
        assert p['source_sha256'][k] == old_protocol['source_sha256'][k]
    ds = [load(OUT / f'delay_controls/cell_{i:03d}.npz') for i in range(1, 4)]
    reference = ds[2]
    controls = []
    for i, d in enumerate(ds, 1):
        ep = json.loads((OUT / f'delay_controls/cell_{i:03d}.json').read_text())['episodes']
        assert all(e['initial_violation'] is False for e in ep)
        bad_envs = np.flatnonzero((d['official_margins'] < 0).any((0, 2))).tolist()
        assert bad_envs == [e['env_id'] for e in ep if e['violation']]
        controls.append(dict(delay_steps=d['meta']['actuator_delay_steps'],
                             delay_ms=d['meta']['actuator_delay_steps'] * d['meta']['dt'] * 1000,
                             violations=len(bad_envs), episodes=len(ep), initially_safe=len(ep),
                             failed_envs=bad_envs, min_nonexempt_mm=float(d['official_margins'].min() * 1000),
                             mean_joint_path_rad=float(np.mean([e['measured_joint_path_rad'] for e in ep])),
                             mean_joint_range=float(np.mean([e['joint_range_fraction_mean'] for e in ep])),
                             fifo=fifo_audit(d), matched=matched(d, reference),
                             first_failures={str(e): first_bad(d, e) for e in bad_envs}))
    previous = load(OLD / 'delayed_critical/cell_001.npz')
    repeated = dict(**matched(reference, previous),
                    q_bit_exact=np.array_equal(reference['q'], previous['q']),
                    max_q_difference_rad=float(np.abs(reference['q'] - previous['q']).max()),
                    previous_failed_envs=np.flatnonzero((previous['official_margins'] < 0).any((0, 2))).tolist(),
                    new_failed_envs=controls[-1]['failed_envs'], independent_windows_added=0)
    rp = json.loads((OUT / 'reverse_controls/protocol.json').read_text())
    assert rp['status'] == 'complete' and rp['completed_cells'] == 3
    assert rp['source_sha256'] == p['source_sha256']
    assert rp['resolved_config'] == p['resolved_config']
    assert rp['effective_coordinator'] == p['effective_coordinator']
    assert rp['effective_backstop'] == p['effective_backstop']
    assert rp['checkpoint_sha256'] == p['checkpoint_sha256']
    for k in CORE:
        assert rp['source_sha256'][k] == p['source_sha256'][k]
    reverse_controls = []
    for i in range(1, 4):
        d = load(OUT / f'reverse_controls/cell_{i:03d}.npz')
        ep = json.loads((OUT / f'reverse_controls/cell_{i:03d}.json').read_text())['episodes']
        assert not d['initial_violation'].any()
        bad_envs = np.flatnonzero((d['official_margins'] < 0).any((0, 2))).tolist()
        assert bad_envs == [e['env_id'] for e in ep if e['violation']]
        forward = ds[next(j for j, x in enumerate(controls) if x['delay_steps'] == d['meta']['actuator_delay_steps'])]
        reverse_controls.append(dict(delay_steps=d['meta']['actuator_delay_steps'],
            delay_ms=d['meta']['actuator_delay_steps'] * d['meta']['dt'] * 1000,
            violations=len(bad_envs), episodes=len(ep), initially_safe=len(ep), failed_envs=bad_envs,
            min_nonexempt_mm=float(d['official_margins'].min() * 1000),
            mean_joint_path_rad=float(np.mean([e['measured_joint_path_rad'] for e in ep])),
            mean_joint_range=float(np.mean([e['joint_range_fraction_mean'] for e in ep])),
            fifo=fifo_audit(d), matched=matched(d, reference),
            q_bit_exact_vs_forward=bool(np.array_equal(d['q'], forward['q'])),
            max_q_difference_vs_forward_rad=float(np.abs(d['q'] - forward['q']).max())))
    cold_reference = load(OUT / 'reverse_controls/cell_001.npz')
    cold_replay_fields = {k: bool(np.array_equal(cold_reference[k], previous[k]))
                          for k in ['q', 'pre_qd', 'exec', 'official_margins', 'controller_target', 'sphere_centers']}
    cp = json.loads((OUT / 'cold_50/protocol.json').read_text())
    assert cp['status'] == 'complete' and cp['completed_cells'] == 1
    assert cp['source_sha256'] == p['source_sha256']
    assert cp['resolved_config'] == p['resolved_config']
    assert cp['effective_backstop'] == p['effective_backstop']
    assert cp['effective_coordinator'] == p['effective_coordinator']
    assert cp['checkpoint_sha256'] == p['checkpoint_sha256']
    cold50 = load(OUT / 'cold_50/cell_001.npz')
    ep = json.loads((OUT / 'cold_50/cell_001.json').read_text())['episodes']
    assert not cold50['initial_violation'].any()
    cold_bad = np.flatnonzero((cold50['official_margins'] < 0).any((0, 2))).tolist()
    assert cold_bad == [e['env_id'] for e in ep if e['violation']]
    cold_controls = [dict(**controls[0], run='delay_controls', cell_id=1),
                     dict(delay_steps=3, delay_ms=3 * p['dt'] * 1000,
                          violations=len(cold_bad), episodes=len(ep), initially_safe=len(ep), failed_envs=cold_bad,
                          min_nonexempt_mm=float(cold50['official_margins'].min() * 1000),
                          mean_joint_path_rad=float(np.mean([e['measured_joint_path_rad'] for e in ep])),
                          mean_joint_range=float(np.mean([e['joint_range_fraction_mean'] for e in ep])),
                          fifo=fifo_audit(cold50), matched=matched(cold50, reference),run='cold_50',cell_id=1),
                     dict(**reverse_controls[0],run='reverse_controls',cell_id=1)]
    recipe = json.loads((OUT / 'stop_schedule.json').read_text())
    assert hashlib.sha256((OUT / recipe['reference']).read_bytes()).hexdigest() == recipe['reference_sha256']
    diagnostics = []
    for mode in ['stored', 'measured']:
        sp = json.loads((OUT / f'stop_{mode}/protocol.json').read_text())
        assert sp['status'] == 'complete' and sp['effective_backstop'] == p['effective_backstop']
        assert sp['source_sha256'] == p['source_sha256']
        assert sp['resolved_config'] == p['resolved_config']
        assert sp['effective_coordinator'] == p['effective_coordinator']
        assert sp['checkpoint_sha256'] == p['checkpoint_sha256']
        for k in CORE:
            assert sp['source_sha256'][k] == p['source_sha256'][k]
        d = load(OUT / f'stop_{mode}/cell_001.npz')
        assert np.array_equal(d['stop_sent_target'], d['controller_target'])
        trajectories = []
        for e, c in enumerate(recipe['trigger_steps']):
            if c < 0:
                assert not d['stop_active'][:, e].any()
                continue
            arrival = c + 6
            expected = d['pre_target'][c, e] if mode == 'stored' else d['pre_q'][c, e]
            assert np.array_equal(d['stop_active'][:, e], np.arange(len(d['q'])) >= c)
            assert np.all(d['controller_target'][c:, e] == expected)
            assert np.all(d['stop_fixed_target'][c:, e] == expected)
            prefix_delta = np.abs(d['pre_q'][:c + 1, e] - reference['pre_q'][:c + 1, e])
            cold_delta = np.abs(d['pre_q'][:c + 1, e] - cold_reference['pre_q'][:c + 1, e])
            cold_exact = all(np.array_equal(d[k][:c + 1, e], cold_reference[k][:c + 1, e])
                             for k in ['pre_q', 'pre_qd', 'pre_target'])
            cold_exact &= all(np.array_equal(d[k][:c, e], cold_reference[k][:c, e])
                              for k in ['sphere_centers', 'cmd', 'controller_target'])
            cold_exact &= np.array_equal(d['actuator_target'][:arrival, e], cold_reference['actuator_target'][:arrival, e])
            # End-of-period sampled state; arrival pre-state is the final old-target sample.
            h = min(arrival + 12, len(d['q']))
            after = d['q'][arrival:h, e] - d['pre_q'][arrival, e]
            velocity = d['pre_qd'][arrival, e]
            moving_away = (np.abs(velocity) > .05) & (velocity * (expected - d['pre_q'][arrival, e]) < 0)
            away_travel = np.maximum(after[:, moving_away] * np.sign(velocity[moving_away]), 0)
            traj = dict(env=e, trigger_step=c, first_stop_message_delivery_step=arrival,
                        lead_to_reference_failure_steps=recipe['first_failure_steps'][e] - c,
                        intervention_prefix_q_bit_exact=bool(np.all(prefix_delta == 0)),
                        intervention_prefix_max_q_difference_rad=float(prefix_delta.max()),
                        cold_control_observed_prefix_exact=bool(cold_exact),
                        cold_control_prefix_max_q_difference_rad=float(cold_delta.max()),
                        cold_control_first_failure_step=first_bad(cold_reference,e),
                        first_failure_after_trigger=first_bad(d, e, c),
                        violation_before_stop_delivery=bool((d['official_margins'][c:arrival, e] < 0).any()),
                        violation_after_stop_delivery=bool((d['official_margins'][arrival:, e] < 0).any()),
                        min_after_trigger_mm=float(d['official_margins'][c:, e].min() * 1000),
                        min_before_delivery_mm=float(d['official_margins'][c:arrival, e].min() * 1000),
                        min_200ms_after_delivery_mm=float(d['official_margins'][arrival:h, e].min() * 1000),
                        max_arm_joint_displacement_200ms_after_delivery_rad=float(np.abs(after).max()),
                        max_arm_joint_speed_at_delivery_rad_s=float(np.abs(d['pre_qd'][arrival, e]).max()),
                        joints_moving_away_from_fixed_target_at_delivery=int(moving_away.sum()),
                        max_further_travel_away_from_target_200ms_rad=float(away_travel.max(initial=0)),
                        actual_target_jump_at_trigger_rad=float(np.abs(d['controller_target'][c,e]-d['pre_target'][c,e]).max()),
                        ordinary_increment_box_rad=float(sp['effective_backstop']['vmax']*sp['dt']),
                        trigger_target_obeys_ordinary_increment_box=bool(np.abs(d['controller_target'][c,e]-d['pre_target'][c,e]).max()<=sp['effective_backstop']['vmax']*sp['dt']+1e-7),
                        fixed_target_max_abs_offset_at_trigger_rad=float(np.abs(expected - d['pre_q'][c, e]).max()))
            trajectories.append(traj)
        all_bad = np.flatnonzero((d['official_margins'] < 0).any((0, 2))).tolist()
        ep = json.loads((OUT / f'stop_{mode}/cell_001.json').read_text())['episodes']
        assert all_bad == [e['env_id'] for e in ep if e['violation']]
        diagnostics.append(dict(mode=mode, fifo=fifo_audit(d), matched=matched(d, reference),
                                posthoc=True, independent_safety_windows_added=0, trajectories=trajectories,
                                all_environment_violations=len(all_bad), all_environment_windows=len(ep),
                                all_failed_envs=all_bad,
                                non_intervened_failed_envs=[e for e in all_bad if recipe['trigger_steps'][e] < 0]))
    stop_a = load(OUT / 'stop_stored/cell_001.npz')
    stop_b = load(OUT / 'stop_measured/cell_001.npz')
    pairwise = []
    for e, c in enumerate(recipe['trigger_steps']):
        if c < 0:
            continue
        # All observed states/control inputs before intervention, not every hidden PhysX state.
        fields = {k: dict(exact=bool(np.array_equal(stop_a[k][:c + 1, e], stop_b[k][:c + 1, e])),
                          max_abs_difference=float(np.abs(stop_a[k][:c + 1, e] - stop_b[k][:c + 1, e]).max()))
                  for k in ['pre_q', 'pre_qd', 'pre_target']}
        for k in ['sphere_centers', 'cmd', 'controller_target']:
            fields[k] = dict(exact=bool(np.array_equal(stop_a[k][:c, e], stop_b[k][:c, e])),
                             max_abs_difference=float(np.abs(stop_a[k][:c, e] - stop_b[k][:c, e]).max()))
        fields['pending_delivery_through_trigger_plus_5'] = dict(
            exact=bool(np.array_equal(stop_a['actuator_target'][:c + 6, e], stop_b['actuator_target'][:c + 6, e])))
        fields['q_through_stop_delivery_prestate'] = dict(
            exact=bool(np.array_equal(stop_a['pre_q'][:c + 7, e], stop_b['pre_q'][:c + 7, e])))
        pairwise.append(dict(env=e, trigger_step=c, observed_prefix_exact=all(x['exact'] for x in fields.values()),
                             fields=fields, scope='Observed q/qd/targets/spheres/commands; hidden physics solver state unrecorded'))
    result = dict(schema='safeduo.queue_braking_report.v1', controls=controls, reverse_controls=reverse_controls,
                  cold_start_controls=cold_controls, cold_100_replay_exact_fields=cold_replay_fields,
                  repeated_control=repeated, stop_diagnostics=diagnostics,
                  stop_pairwise_prefix=pairwise,
                  actor_sha256=p['checkpoint_sha256'], core_sha256={k: p['source_sha256'][k] for k in CORE},
                  all_production_sources_and_resolved_config_equal_to_previous=True,
                  rule='critical-only 10mm union; 128 rows; no overflow; actor 32 rows; backlog_aware=false',
                  statistics='Within-input delay comparison; repeated 100ms control and post-hoc holds add no independent safety samples',
                  interpretation='No physical safety certification; no queue preemption; linear feasibility does not certify braking')
    (OUT / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
