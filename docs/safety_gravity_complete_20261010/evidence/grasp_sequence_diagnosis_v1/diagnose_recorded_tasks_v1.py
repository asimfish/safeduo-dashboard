"""Read-only phase diagnosis of sealed native_v5 tasks; no new acceptance trial."""
import hashlib
import json
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parent
B = Path('/home/liyufeng/safeduo/artifacts/safety_object_binding_20261006')
RAW = Path('/mnt/nas/data/lyf/double_hand/safety_object_binding_20261006/native_v5')
ARMS = ('F_L', 'F_R', 'U_L', 'U_R')
OBJECTS = ('beam700', 'beam300')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def longest(mask, interval_start, interval_end):
    """Continuous native control intervals, not the sum of disconnected visits."""
    indices = np.flatnonzero(mask)
    if not len(indices):
        return dict(duration_s=0., first_state_time_s=None, last_state_time_s=None, states=0)
    runs = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
    durations = [float(interval_end[r[-1]] - interval_start[r[0]]) for r in runs]
    run = runs[int(np.argmax(durations))]
    return dict(duration_s=max(durations), first_state_time_s=float(interval_end[run[0]]),
                last_state_time_s=float(interval_end[run[-1]]), states=len(run))


def main():
    output = P / 'RECORDED_TASK_DIAGNOSIS_V1.json'
    assert not output.exists(), 'preserve previous diagnosis attempts'
    rec = json.loads((RAW / 'recording_receipt.json').read_text())
    initial = json.loads((RAW / 'native_initial.json').read_text())
    reg = json.loads((B / 'REGISTRATION_V5.json').read_text())
    assert rec['status'] == 'PASS_COMPLETE_NATIVE_RECORDING'
    assert rec['steps'] == 1572 and rec['cases'] == 8
    assert sha(B / 'REGISTRATION_V5.json') == rec['registration_sha256']
    assert sha(RAW / 'native_initial.json') == rec['native_initial_sha256']
    sources = {str(B / 'REGISTRATION_V5.json'): sha(B / 'REGISTRATION_V5.json'),
               str(RAW / 'recording_receipt.json'): sha(RAW / 'recording_receipt.json'),
               str(RAW / 'native_initial.json'): sha(RAW / 'native_initial.json'),
               str(B / 'native_curves.npz'): sha(B / 'native_curves.npz'),
               str(B / 'PHASE_DIAGNOSIS.json'): sha(B / 'PHASE_DIAGNOSIS.json'),
               str(B / 'native_results.json'): sha(B / 'native_results.json'),
               str(Path(__file__).resolve()): sha(Path(__file__).resolve())}
    arrays = {}
    for chunk in rec['chunks']:
        path = RAW / chunk['file']
        assert sha(path) == chunk['sha256']
        sources[str(path)] = chunk['sha256']
        with np.load(path, allow_pickle=False) as z:
            for key in z.files:
                assert np.isfinite(z[key]).all(), key
                arrays.setdefault(key, []).append(z[key])
    d = {key: np.concatenate(values) for key, values in arrays.items()}
    assert np.array_equal(d['step'], np.arange(1572))
    start, end = d['time'][:, 0], d['time'][:, 1]
    assert np.allclose(end - start, reg['native_control_dt_s'], rtol=0, atol=1e-12)
    assert np.allclose(start[1:], end[:-1], rtol=0, atol=1e-12)
    with np.load(B / 'native_curves.npz', allow_pickle=False) as z:
        curves = {key: z[key].copy() for key in z.files}
    assert np.array_equal(curves['time'], end)
    states = d['objects'].astype(np.float64)
    states[..., :3] -= np.asarray(initial['origins'])[None, :, None, :]
    # Existing curves renormalized the quaternion during independent analysis.
    states[..., 3:7] /= np.linalg.norm(states[..., 3:7], axis=-1, keepdims=True)
    assert np.array_equal(states, curves['states'])
    with np.load(Path(reg['trajectory']), allow_pickle=False) as z:
        meta = json.loads(str(z['meta']))['s9_task']
    source_phases = meta['phases']
    old = json.loads((B / 'native_results.json').read_text())
    table_ids = [next(i for i, p in enumerate(initial['partners']) if p.get('table') == table)
                 for table in ('TableF', 'TableU')]
    rows = []
    for e, case in enumerate(reg['cases']):
        objects = {}
        arm_rows = {}
        for arm in ARMS:
            tracking = np.abs(d[arm + ':q'][:, e] - d[arm + ':target'][:, e])
            intended = np.abs(d[arm + ':cmd'][:, e]).sum(-1)
            executed = np.abs(d[arm + ':exec'][:, e]).sum(-1)
            phase_rows = {}
            for phase in source_phases:
                sel = (start >= phase['t0']) & (start < phase['t1'])
                phase_rows[phase['name']] = dict(states=int(sel.sum()),
                    tracking_abs_median_rad=float(np.median(tracking[sel])),
                    tracking_abs_p95_rad=float(np.quantile(tracking[sel], .95)),
                    tracking_abs_max_rad=float(tracking[sel].max()),
                    commanded_L1_rad=float(intended[sel].sum()), executed_L1_rad=float(executed[sel].sum()))
            arm_rows[arm] = phase_rows
        for oi, obj in enumerate(OBJECTS):
            lift = curves['lift'][:, e, oi]
            tilt = curves['tilt'][:, e, oi]
            hand = curves['hand_normal'][:, e, oi]
            own_table = curves['own_table_normal'][:, e, oi]
            both = (hand > .1).all(-1)
            any_hand = (hand > .1).any(-1)
            carry = (start >= 5.7) & (start < 11.7)
            before_open = (start >= 14.5) & (start < 15.5)
            clear = start >= 18.7
            weight = float(np.asarray(initial['native_mass_kg'][obj])[e, 0] * 9.81)
            support = (own_table >= .8 * weight) & (tilt <= 10.)
            linear_speed = np.linalg.norm(states[:, e, oi, 7:10], axis=-1)
            angular_speed = np.linalg.norm(states[:, e, oi, 10:13], axis=-1)
            supported_rest = support & (linear_speed <= .01) & (angular_speed <= .1)
            height = longest(carry & (lift >= .100), start, end)
            lift_contacts = longest(carry & (lift >= .100) & both, start, end)
            all_descriptive_grasp = longest(carry & (lift >= .100) & both & (tilt <= 10.)
                                            & (own_table <= .1), start, end)
            after_open = start >= 16.8
            fully_released = longest(after_open & ~any_hand & supported_rest, start, end)
            phases = {}
            for phase in source_phases:
                sel = (start >= phase['t0']) & (start < phase['t1'])
                phases[phase['name']] = dict(states=int(sel.sum()), max_lift_m=float(lift[sel].max()),
                    max_tilt_deg=float(tilt[sel].max()), both_hand_contact_states=int((sel & both).sum()),
                    any_hand_contact_states=int((sel & any_hand).sum()),
                    table_weight_fraction_median=float(np.median(own_table[sel]) / weight),
                    supported_rest_states=int((sel & supported_rest).sum()),
                    max_linear_speed_m_s=float(linear_speed[sel].max()),
                    max_angular_speed_rad_s=float(angular_speed[sel].max()))
            objects[obj] = dict(original_object_gate_pass=old['cases'][e]['objects'][obj]['pass_task'],
                original_window_max_lift_m=float(lift.max()), planned_max_lift_m=float(lift[carry].max()),
                continuous_100mm_height=height, continuous_100mm_paired_contact=lift_contacts,
                continuous_100mm_upright_paired_contact_without_table=all_descriptive_grasp,
                descriptive_grasp_2s_observed=all_descriptive_grasp['duration_s'] >= 2.,
                before_open_supported_rest_s=float((before_open & supported_rest).sum() * reg['native_control_dt_s']),
                before_open_table_weight_fraction_median=float(np.median(own_table[before_open]) / weight),
                after_open_supported_detached=fully_released,
                clearance_hand_contact_states=int((clear & any_hand).sum()),
                clearance_max_lift_m=float(lift[clear].max()), phase_measurements=phases)
        rows.append(dict(env=e, layout=case['layout'], binding=case['binding'],
                         original_task_pass=old['cases'][e]['pass_task'], objects=objects,
                         arm_joint_tracking_by_phase=arm_rows))
    result = dict(status='COMPLETE_RETROSPECTIVE_PHASE_DIAGNOSIS', new_native_trials=0,
        original_task_verdicts_unchanged=True, full_system0_accepted=False,
        scope='8 sealed development tasks, last-substep normal contacts. Continuous interval descriptions '
              'do not supply missing physics-substep, friction, full-scene, or four-hand common-object evidence.',
        phase_clock='native control interval start assigns phase; measurements at interval end',
        explanatory_thresholds=dict(height_m=.100, continuous_duration_s=2., paired_normal_N=.1,
                                    upright_deg=10., detached_normal_N=.1,
                                    supported_table_weight_fraction=.8, rest_linear_m_s=.01,
                                    rest_angular_rad_s=.1),
        formal_grasp_accepted=False, source_sha256=sources, cases=rows,
        tallies=dict(descriptive_grasp_2s_objects=sum(x['descriptive_grasp_2s_observed']
                    for row in rows for x in row['objects'].values()), objects=16,
                    cases_with_both_descriptive_grasps=sum(all(x['descriptive_grasp_2s_observed']
                    for x in row['objects'].values()) for row in rows)))
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], tallies=result['tallies'],
        cases=[dict(env=r['env'], objects={o:dict(lift=x['planned_max_lift_m'],
              continuous_100mm_s=x['continuous_100mm_paired_contact']['duration_s'],
              pre_open_rest_s=x['before_open_supported_rest_s'],
              post_open_detached_rest_s=x['after_open_supported_detached']['duration_s'],
              clearance_contacts=x['clearance_hand_contact_states']) for o, x in r['objects'].items()})
              for r in rows]), indent=2))


if __name__ == '__main__':
    main()
