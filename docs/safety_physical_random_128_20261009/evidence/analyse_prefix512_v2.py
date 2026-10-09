"""Independent raw-point oracle and prospectively defined 16-cell selection."""
import hashlib
import json
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
ARMS = ('F_L', 'F_R', 'U_L', 'U_R')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def oracle(root, output, inputs):
    receipt = json.loads((root / 'point_contact_receipts.json').read_text())
    identity = json.loads((root / 'native_contact_identity.json').read_text())
    assert receipt['status'] == 'complete' and receipt['physics_events'] == 12
    peaks, hard_errors, velocity_errors = [], [], []
    valid_points = 0
    for chunk in receipt['chunks']:
        assert sha(root / chunk['path']) == chunk['sha256']
        with np.load(root / chunk['path']) as z:
            frames, subs = z['frame'], z['substep']
            env_peak = np.zeros((len(frames), 64))
            hard_error = np.zeros_like(env_peak)
            velocity_error = np.zeros_like(env_peak)
            for arm in ARMS:
                view = next(x for x in identity['views'] if x['arm'] == arm)
                ids = np.asarray(view['env_ids'])
                prefix = arm + '_points_'
                counts, starts = z[prefix + 'counts'], z[prefix + 'starts']
                si, pi, ii = [z[prefix + key] for key in ['sensor_indices', 'partner_indices', 'point_indices']]
                offsets = z[prefix + 'event_offsets']
                force = z[prefix + 'normal_forces'][:, 0].astype(float)
                stored = z[prefix + 'partner_abs_normal_sum_N']
                for event in range(len(frames)):
                    lo, hi = offsets[event:event + 2]
                    valid_points += int(hi - lo)
                    sensor, partner, point = si[lo:hi], pi[lo:hi], ii[lo:hi]
                    flat = sensor * counts.shape[-1] + partner
                    reconstructed = np.bincount(flat, minlength=counts[event].size).reshape(counts[event].shape)
                    assert np.array_equal(reconstructed, counts[event])
                    assert (point >= starts[event, sensor, partner]).all()
                    assert (point < starts[event, sensor, partner] + counts[event, sensor, partner]).all()
                    assert len(np.unique(point)) == len(point)
                    scalar = np.bincount(flat, weights=abs(force[lo:hi]), minlength=counts[event].size).reshape(counts[event].shape)
                    assert np.array_equal(scalar, stored[event])
                    np.maximum.at(env_peak[event], ids, scalar.max(-1))
                q, v = z[arm + '_native_q'], z[arm + '_native_qd']
                target = z[arm + '_native_position_targets']
                assert np.array_equal(target, np.broadcast_to(inputs[arm + '_full_hold_target'], target.shape))
                hard = inputs[arm + '_hard_limits']
                hard_error = np.maximum(hard_error, np.maximum(hard[None, ..., 0] - q, q - hard[None, ..., 1]).max(-1))
                velocity_error = np.maximum(velocity_error, (abs(v) - inputs[arm + '_vmax'][None]).max(-1))
            peaks.append(env_peak)
            hard_errors.append(hard_error.clip(0))
            velocity_errors.append(velocity_error.clip(0))
    peak = np.concatenate(peaks)
    assert np.array_equal(peak, output['scalar_normal_max_N'])
    assert np.array_equal(np.concatenate(hard_errors), output['supplementary_hard_limit_violation_rad'])
    assert np.array_equal(np.concatenate(velocity_errors), output['supplementary_velocity_limit_violation_rad_s'])
    expected = {x['rigid_owner'] for x in identity['inventory'] if x['collision_enabled'] and x['rigid_owner'] and any('/' + a + '/' in x['rigid_owner'] for a in ARMS)}
    observed = {s for v in identity['views'] for s in v['sensors'] if '/env_0/' in s}
    assert expected <= observed and len(observed) == 82
    return dict(status='PASS_INDEPENDENT_ALL12_EVENT_RAW_POINT_BINCOUT', valid_point_observations=valid_points,
                enabled_rigid_owners_covered=82, all_force_q_limits_velocity_limits_exact=True)


def main():
    reg = json.loads((H / 'REGISTRATION_HAND_VALID_V2.json').read_text())
    root = Path(reg['raw'])
    execution = json.loads((H / 'physical_prefix512_v5_execution.json').read_text())
    assert execution['status'] == 'complete' and len(execution['jobs']) == 8
    assert all(j['status'] == 'complete' and j['actual_exit'] == 0 for j in execution['jobs'])
    with np.load(root / 'geometric_bank_v2/bank.npz') as z:
        bank = {k: z[k] for k in z.files}
    with np.load(root / 'pure_random_recipe_v2.npz') as z:
        recipe = {k: z[k] for k in z.files}
    ledger, oracles, parameter_rows = [], [], []
    for batch in range(8):
        path = root / ('prefix_batch_' + str(batch) + '_v5')
        protocol = json.loads((path / 'prefix_protocol.json').read_text())
        assert protocol['status'] == 'complete' and protocol['physics_events'] == 12
        with np.load(path / 'prefix_result.npz') as z:
            output = {k: z[k] for k in z.files}
        with np.load(path / 'prefix_inputs.npz') as z:
            inputs = {k: z[k] for k in z.files}
        parameter_rows.append(inputs)
        oracles.append(dict(batch=batch, **oracle(path, output, inputs)))
        initial_q = []
        for arm in ARMS:
            q, hard = inputs[arm + '_initial_q'], inputs[arm + '_hard_limits']
            tolerance = reg['initial_hard_limit_tolerance_rad']
            assert ((q >= hard[..., 0] - tolerance) & (q <= hard[..., 1] + tolerance)).all()
            initial_q.append(q[:, inputs[arm + '_controlled_joint_indices']])
        assert np.array_equal(np.concatenate(initial_q, -1), bank['accepted_q'][batch * 64:(batch + 1) * 64])
        geom = output['all_raw_geometry']
        assert geom.shape == (12, 64, 9021)
        force = output['scalar_normal_max_N']
        geo_valid = geom.min((0, 2)) >= reg['physical_geometry_min_m']
        force_valid = force.max(0) <= reg['physics_point_force_gate_N']
        assert np.array_equal(geo_valid & force_valid, output['admitted'])
        for lane, global_id in enumerate(output['global_input_id']):
            bad = (geom[:, lane].min(-1) < 0) | (force[:, lane] > .1)
            indices = np.flatnonzero(bad)
            first = int(indices[0]) if len(indices) else None
            ledger.append(dict(input_id=int(global_id), geometric_proposal_id=int(bank['selected_indices'][global_id]),
                batch=batch, lane=lane, cell=int(recipe['cell_id'][global_id]),
                velocity_fraction=float(recipe['velocity_fraction'][global_id]),
                command_amplitude_rad=float(recipe['command_amplitude_rad'][global_id]),
                initial_full_native_hard_limits_valid=True, prefix_geo_qualified=bool(geo_valid[lane]),
                prefix_force_qualified=bool(force_valid[lane]), admitted=bool(geo_valid[lane] and force_valid[lane]),
                minimum_substep_raw_gap_m=float(geom[:, lane].min()), peak_point_normal_N=float(force[:, lane].max()),
                first_primary_failure_macro=first // 2 if first is not None else None,
                first_primary_failure_substep=first % 2 if first is not None else None,
                supplementary_max_hard_limit_violation_rad=float(output['supplementary_hard_limit_violation_rad'][:, lane].max()),
                supplementary_max_velocity_limit_violation_rad_s=float(output['supplementary_velocity_limit_violation_rad_s'][:, lane].max())))
    assert [r['input_id'] for r in ledger] == list(range(512))
    cells, selected = [], []
    for cell in range(16):
        members = [r for r in ledger if r['cell'] == cell]
        eligible = [r for r in members if r['admitted']]
        used = eligible[:reg['desired_qualified_per_cell']]
        selected.extend(r['input_id'] for r in used)
        cells.append(dict(cell=cell, velocity_fraction=members[0]['velocity_fraction'],
            command_amplitude_rad=members[0]['command_amplitude_rad'], prospective_candidates=len(members),
            prefix_geometry_qualified=sum(r['prefix_geo_qualified'] for r in members),
            prefix_force_qualified=sum(r['prefix_force_qualified'] for r in members),
            admitted=len(eligible), selected=len(used),
            sufficient=len(eligible) >= reg['desired_qualified_per_cell']))
    result = dict(status='CLOSED_ALL512_INITIAL_PHYSICAL_PREFIX_QUALIFICATION', fresh_geometry_bank=True,
        candidate_initial_states=512, native_physics_events=96, native_lane_microstep_observations=6144,
        controls_per_state=6, microsteps_per_state=12, future_policy_outcomes_used=False,
        prefix_qualified=sum(r['admitted'] for r in ledger), selected=len(selected),
        required_selected=128, sufficient_full16cell_bank=all(c['sufficient'] for c in cells), cells=cells,
        all512_initial_native_hard_limits_valid=True, all8_raw_point_oracles_passed=True,
        all_native_children_closed_actual_exit0=True, oracles=oracles,
        selection='First8 qualifying initial inputs per16cells, qualification fixed before policy outcomes',
        conditional_distribution=True, full26Dvolume_coverage_claim=False, safety_acceptance=False,
        known_limitations=['Hand/root fixture fixed; hand DOF/random shape/task/object space not covered',
                           'Prefix qualification is geometry and point normal force; post joint limit/velocity supplementary',
                           'No fullfriction/continuous-time/hardware safety certificate'],
        selected_input_ids=selected, rejection_ledger=ledger)
    (H / 'PHYSICAL_PREFIX_RESULT_V1.json').write_text(json.dumps(result, indent=2) + '\n')
    fields = {'selected_input_id': np.asarray(selected, int), 'cell_id': recipe['cell_id'][selected],
              'velocity_fraction': recipe['velocity_fraction'][selected],
              'command_amplitude_rad': recipe['command_amplitude_rad'][selected],
              'command_unit': recipe['command_unit'][:, selected],
              'velocity_unit': recipe['velocity_unit'][selected]}
    for arm in ARMS:
        for field in ['initial_q', 'initial_qd', 'full_hold_target']:
            fields[arm + '_' + field] = np.concatenate([p[arm + '_' + field] for p in parameter_rows])[selected]
        fields[arm + '_joint_names'] = parameter_rows[0][arm + '_joint_names']
        fields[arm + '_controlled_joint_indices'] = parameter_rows[0][arm + '_controlled_joint_indices']
    np.savez_compressed(root / 'qualified_bank_v1.npz', **fields)
    (H / 'QUALIFIED_BANK_BINDING_V1.json').write_text(json.dumps(dict(bank_sha256=sha(root / 'qualified_bank_v1.npz'),
        result_sha256=sha(H / 'PHYSICAL_PREFIX_RESULT_V1.json'), selected=len(selected),
        complete16cells=result['sufficient_full16cell_bank'], controller_outcomes_consulted=False), indent=2) + '\n')
    print(result['status'], result['prefix_qualified'], 'admitted;', len(selected), 'selected; complete16cells', result['sufficient_full16cell_bank'])


if __name__ == '__main__':
    main()
