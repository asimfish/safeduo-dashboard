"""XML mapping, native per-step inputs, independent impulse-form effort oracle."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from audit_coupled_hand_targets_native_v1 import matrix

H = Path(__file__).resolve().parent


def read(file):
    return json.loads(Path(file).read_text())


def sha(file):
    return hashlib.sha256(Path(file).read_bytes()).hexdigest()


def full_mapping(names, asset):
    hand = asset['hand_joint_names']
    roots, A, _ = matrix(hand, asset['original_URDF'])
    handids = [names.index(n) for n in hand]
    handroots = [names.index(n) for n in roots]
    projection = np.eye(len(names))
    projection[handids] = 0
    projection[np.ix_(handroots, handids)] = A.T
    followers = [names.index(r['slave']) for r in asset['relations']]
    rootids = [i for i in range(len(names)) if i not in followers]
    return rootids, followers, projection, projection[rootids, :].T


def rounded_binding(expected, actual):
    assert expected.dtype == actual.dtype == np.float32
    # Independent equivalent equations can straddle one float32 rounding boundary.
    # This is a force binding precision check, never a q/qd/physical gate change.
    bound = 4 * np.abs(np.spacing(np.abs(expected))) + 1e-7
    assert np.isfinite(actual).all()
    assert np.all(np.abs(expected - actual) <= bound), float(abs(expected - actual).max())


def main(version):
    p = H / f'fourhand_native_probe_v{version}'
    reg = read(p / 'REGISTRATION_V1.json')
    audit = read(p / 'INDEPENDENT_CAPTURE_AUDIT_V1.json')
    assert audit['record_integrity_verified'] and reg['full_motor_native_PD_exactly_zero']
    assert reg['original_native_armature_used_in_controller'] and reg['native_coriolis_query_disabled']
    raw = Path(read(p / 'RESOURCE_EXECUTION_V1.json')['out'])
    params = np.load(raw / 'native_parameters.npz', allow_pickle=False)
    meta = read(reg['coupled_assets'])
    rows, sources = [], {}
    dt = reg['physics_dt_s']
    for arm, asset in meta['assets'].items():
        names = params[arm + '_joint_names'].tolist()
        roots, followers, projection, transform = full_mapping(names, asset)
        np.testing.assert_array_equal(projection, params[arm + '_motor_effort_projection'])
        np.testing.assert_array_equal(transform, params[arm + '_full_motor_velocity_mapping'])
        assert not params[arm + '_stiffness'].any() and not params[arm + '_damping'].any()
        cfg = reg['full_motor_effort_controller'][arm]
        stiffness = np.array([cfg[names[i]]['nominal_K_Nm_rad'] for i in roots])
        damping = np.array([cfg[names[i]]['nominal_D_Nm_s_rad'] for i in roots])
        prior_q = prior_v = None
        events, max_pd_error, max_mass_asymmetry = 0, 0., 0.
        for chunk in read(raw / 'physics_chunks.json'):
            file = raw / chunk['path']
            assert sha(file) == chunk['sha256']
            sources[str(file)] = sha(file)
            with np.load(file, allow_pickle=False) as data:
                q, v = data[arm + '_motor_control_q_before'], data[arm + '_motor_control_qd_before']
                after_q, after_v = data[arm + '_native_q'], data[arm + '_native_qd']
                if prior_q is not None:
                    np.testing.assert_array_equal(q[0], prior_q)
                    np.testing.assert_array_equal(v[0], prior_v)
                np.testing.assert_array_equal(q[1:], after_q[:-1])
                np.testing.assert_array_equal(v[1:], after_v[:-1])
                prior_q, prior_v = after_q[-1], after_v[-1]
                target = data[arm + '_full_motor_actual_target_before']
                np.testing.assert_array_equal(target, data[arm + '_actual_full_position_target'])
                matrices = data[arm + '_native_generalized_mass_before']
                assert matrices.shape == (len(q), 1, len(names), len(names))
                armature = data[arm + '_native_dof_armature_before']
                assert armature.shape == q.shape and np.isfinite(armature).all() and (armature >= 0).all()
                np.testing.assert_array_equal(armature, np.broadcast_to(params[arm + '_native_dof_armature'], q.shape))
                assert arm + '_native_coriolis_compensation_before' not in data.files
                pd = np.zeros_like(data[arm + '_gravity_compensation_raw'])
                for i, full in enumerate(matrices[:, 0].astype(float)):
                    assert np.isfinite(full).all()
                    asym = float(abs(full - full.T).max() / abs(full).max())
                    assert asym <= reg['full_motor_mass_relative_asymmetry_tolerance'] == 5e-6
                    max_mass_asymmetry = max(max_mass_asymmetry, asym)
                    native = .5 * (full + full.T)
                    np.linalg.cholesky(native)
                    # Independently reduce existing rotor inertia, without the producer helper.
                    mass = transform.T @ native @ transform + transform.T @ np.diag(armature[i,0].astype(float)) @ transform
                    np.linalg.cholesky(mass)
                    qb, vb, actualtarget = q[i, 0, roots].astype(float), v[i, 0, roots].astype(float), target[i, 0, roots].astype(float)
                    # Solve delta-v with a different RHS, then use impulse form.
                    system = mass + np.diag(dt * damping + dt * dt * stiffness)
                    rhs = dt * (stiffness * (actualtarget - qb) - (dt * stiffness + damping) * vb)
                    delta_v = np.linalg.solve(system, rhs)
                    tau = mass @ delta_v / dt
                    pd[i, 0, roots] = tau.astype(pd.dtype)
                selected = reg['isolated_zero_PD_motor']
                assert selected in [dict(arm='F_L', joint='left_thumb_2_joint'), dict(arm='F_L', joint='left_thumb_1_joint')]
                original_roots, _, _ = matrix(asset['hand_joint_names'], asset['original_URDF'])
                if arm == selected['arm']:
                    assert selected['joint'] in original_roots
                    paused_id = names.index(selected['joint'])
                    assert paused_id in roots and paused_id not in followers
                    pd[..., paused_id] = 0.
                recorded = data[arm + '_software_full_motor_PD_preclip']
                rounded_binding(pd, recorded)
                max_pd_error = max(max_pd_error, float(abs(pd - recorded).max()))
                gravity = data[arm + '_gravity_compensation_raw'] @ projection.T
                if not bool(params[arm + '_gravity_compensation_enabled']):
                    gravity = np.zeros_like(gravity)
                expected = np.clip(gravity.astype(pd.dtype) + recorded,
                                   -params[arm + '_max_force'], params[arm + '_max_force'])
                rounded_binding(expected, data[arm + '_extra_effort'])
                np.testing.assert_array_equal(data[arm + '_extra_effort'], data[arm + '_native_external_actuation_effort'])
                assert not data[arm + '_native_external_actuation_effort'][..., followers].any()
                events += len(q)
        assert events == audit['physics_events']
        rows.append(dict(arm=arm,independent_motors=len(roots),passive_followers=len(followers),
                         physics_events=events,max_PD_binding_error_Nm=max_pd_error,
                         max_native_mass_relative_asymmetry=max_mass_asymmetry,
                         state_epoch_targets_and_native_submission_exact=True))
        sources[asset['original_URDF']] = sha(asset['original_URDF'])
    assert sum(row['independent_motors'] for row in rows) == 50
    files = [Path(__file__), p / 'REGISTRATION_V1.json', raw / 'native_parameters.npz']
    sources.update({str(file): sha(file) for file in files})
    result = dict(status='PASS_FULL50_INDEPENDENT_IMPULSE_ORACLE_WITH_DECLARED_ONE_THUMB_PD_ZERO',
                  version=version,rows=rows,physics_events=audit['physics_events'],
                  independent_delta_velocity_impulse_form_oracle=True,
                  source_SHA256=sources,force_binding_precision='4float32ULP+1e-7Nm',
                  original_q_qd_and_physical_gates_not_replaced=True,fullSystem0_accepted=False)
    target = p / 'INDEPENDENT_SINGLE_THUMB_PD_EFFORT_V1.json'
    assert not target.exists()
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'], audit['physics_events'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('version', type=int)
    main(parser.parse_args().version)
