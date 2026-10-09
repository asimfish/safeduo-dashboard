"""Development fallback attempt with immutable FIFO6; no certified safe backup."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

H = Path(__file__).resolve().parent
OLD = Path('/home/liyufeng/safeduo/artifacts/safety_velocity_arrival_20261008_1519')
HF = Path('/home/liyufeng/safeduo/artifacts/safety_feasible_response_20261009_0316')
sys.path.insert(0, str(OLD))
sys.path.insert(0, str(HF))
ARMS = ('F_L', 'F_R', 'U_L', 'U_R')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


class ZeroSource:
    def reset(self, ids, generator=None):
        pass

    def sample(self, state):
        from safeduo.safety.types import DeltaCmd
        return DeltaCmd({a: torch.zeros_like(state.q[a]) for a in ARMS})


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--batch', type=int, choices=[0, 1], required=True)
    parser.add_argument('--method', choices=['multirow_hold_fallback'], required=True)
    parser.add_argument('--diagnostic-controls', type=int, choices=[1])
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    reg = json.loads((H / 'PAIRED_REGISTRATION_V1.json').read_text())
    if args.diagnostic_controls:
        reg['control_steps'] = args.diagnostic_controls
    root = Path(reg['raw'])
    for p, expected in reg['input_sha256'].items():
        assert sha(Path(p)) == expected, 'input changed ' + p
    with np.load(root / 'qualified_bank_v1.npz') as z:
        bank = {k: z[k] for k in z.files}
    with np.load(root / 'geometric_bank_v2/bank.npz') as z:
        geometry_bank = {k: z[k] for k in z.files}
    with np.load(root / 'hand_root_fixture.npz') as z:
        fixture = {k: z[k] for k in z.files}
    selected = np.asarray(reg['batch_bank_positions'][args.batch])
    ids = bank['selected_input_id'][selected]
    assert len(selected) == 64 and len(np.unique(ids)) == 64
    out = Path(args.out)
    out.mkdir(exist_ok=False)
    app = AppLauncher(args).app
    env, observer = None, None
    protocol = dict(status='running', batch=args.batch, method=args.method,
        prospective_states=64, actor_actions=0, formal_holdout=False,
        physical_qualification_before_controller_outcomes=True,
        hidden_solver_state_restored=False, safety_acceptance=False)
    completed = 0
    try:
        from safeduo.envs.duo_env import DuoEnv, make_duo_env_cfg
        from safeduo.eval.perturbations import TargetDelayQueue
        from safeduo.safety.geometry import IsaacGeometryProvider
        from native_state_capture import array
        from native_all_body_point_contacts import PointContacts
        from safe_hand_opening import install_open_hand_defaults
        from multirow_response_v3 import make_multirow_issue
        sys.path.insert(0, str(H))
        from issue_or_initial_hold_v1 import issue_or_initial_hold
        cfg = make_duo_env_cfg(num_envs=64, device=args.device,
            yaml_name='duo_env_a31_pending_guard.yaml', coordinator=True,
            arm_aware_obs=True, p2_obs=True)
        cfg.seed = 0
        cfg.coordinator['terminate_on_violation'] = False
        cfg.episode_length_s = 21.
        cfg.enable_viz_camera = False
        cfg.sim.use_fabric = True
        # Headless, no camera: physical scoring only.
        env = DuoEnv(cfg)
        opening = install_open_hand_defaults(env)
        env._delta_src = ZeroSource()
        env.reset()
        (out / 'open_hand_initialization.json').write_text(json.dumps(opening, indent=2) + '\n')
        parameters = dict(global_input_id=ids, bank_positions=selected,
            cell_id=bank['cell_id'][selected], velocity_fraction=bank['velocity_fraction'][selected],
            command_amplitude_rad=bank['command_amplitude_rad'][selected])
        tape, pending = {}, {}
        offset = 0
        for arm in ARMS:
            art, idx = env._arms[arm], env._joint_idx[arm]
            view = art.root_physx_view
            assert np.array_equal(np.asarray(art.joint_names), bank[arm + '_joint_names'])
            assert np.array_equal(array(idx), bank[arm + '_controlled_joint_indices'])
            q = torch.as_tensor(bank[arm + '_initial_q'][selected], device=env.device).clone()
            v = torch.as_tensor(bank[arm + '_initial_qd'][selected], device=env.device).clone()
            hold = torch.as_tensor(bank[arm + '_full_hold_target'][selected], device=env.device).clone()
            root_pose = torch.as_tensor(fixture[arm + '_native_root_xyzw'][0], device=env.device)[None].expand(64, -1).clone()
            root_pose[:, :3] += env.scene.env_origins - torch.as_tensor(fixture['source_origin'][0], device=env.device)
            art.write_root_pose_to_sim(torch.cat([root_pose[:, :3], root_pose[:, 6:7], root_pose[:, 3:6]], -1))
            art.write_root_velocity_to_sim(torch.zeros((64, 6), device=env.device))
            art.write_joint_state_to_sim(q, v)
            art.set_joint_position_target(hold)
            assert torch.equal(hold[:, idx], q[:, idx])
            soft = art.data.soft_joint_pos_limits[:, idx]
            unit = torch.as_tensor(bank['command_unit'][:, selected, offset:offset + len(idx)], device=env.device)
            amp = torch.as_tensor(bank['command_amplitude_rad'][selected], device=env.device, dtype=q.dtype)[None, :, None]
            tape[arm] = (q[None, :, idx] + amp * unit).clamp(soft[None, ..., 0], soft[None, ..., 1])
            pending[arm] = q[:, idx].clone()
            offset += len(idx)
            for field, value in [('initial_q', q), ('initial_qd', v), ('full_hold_target', hold),
                                  ('controlled_joint_indices', idx), ('soft_limits', soft)]:
                parameters[arm + '_' + field] = array(value)
            parameters[arm + '_joint_names'] = np.asarray(art.joint_names)
            parameters[arm + '_disable_gravity_cfg'] = np.full(64, bool(art.cfg.spawn.rigid_props.disable_gravity))
            for field, getter in [('stiffness', view.get_dof_stiffnesses), ('damping', view.get_dof_dampings),
                                  ('max_force', view.get_dof_max_forces), ('max_velocity', view.get_dof_max_velocities),
                                  ('armature', view.get_dof_armatures), ('hard_limits', view.get_dof_limits),
                                  ('friction', view.get_dof_friction_properties), ('masses', view.get_masses),
                                  ('inertias', view.get_inertias)]:
                parameters[arm + '_' + field] = array(getter())
        assert offset == 26
        queue = TargetDelayQueue(6)
        queue.pending = deque({a: pending[a].clone() for a in ARMS} for _ in range(6))
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(0.)
        binding = json.loads((H / 'PREFIX_TRANSLATION_BINDING_CONTRACT_V5.json').read_text())
        for arm in ARMS:
            art, view = env._arms[arm], env._arms[arm].root_physx_view
            for f, getter in [('initial_q', view.get_dof_positions), ('initial_qd', view.get_dof_velocities),
                              ('full_hold_target', view.get_dof_position_targets)]:
                assert np.array_equal(array(getter()), parameters[arm + '_' + f])
            hard, q = parameters[arm + '_hard_limits'], parameters[arm + '_initial_q']
            assert ((q >= hard[..., 0] - 1e-6) & (q <= hard[..., 1] + 1e-6)).all()
            local = array(art.data.body_pos_w - env.scene.env_origins[:, None])
            assert np.allclose(local, geometry_bank[arm + '_body_pos_local'][ids],
                atol=binding['body_position_local_absolute_tolerance_m'], rtol=0)
            quat = array(art.data.body_quat_w)
            source = geometry_bank[arm + '_body_quat_wxyz'][ids]
            assert (np.minimum(abs(quat-source).max(-1), abs(quat+source).max(-1)) <= binding['body_orientation_sign_invariant_absolute_tolerance']).all()
        initial = env.compute_dist()
        assert np.allclose(array(initial.dists), geometry_bank['accepted_d'][ids],
            atol=binding['source_geometry_absolute_binding_tolerance_m'], rtol=0)
        assert (array(initial.dists).min(-1) >= .0001).all()
        np.savez_compressed(out / 'initial_geometry.npz', d=array(initial.dists),
            source_full_d=geometry_bank['accepted_d'][ids], global_input_id=ids)
        np.savez_compressed(out / 'resolved_native_parameters.npz', **parameters)
        np.savez_compressed(out / 'actual_random_issue_tape.npz', **{a: array(t) for a, t in tape.items()})
        observer = PointContacts(env, out)
        full_provider = IsaacGeometryProvider(env._arms,
            {a: torch.arange(len(env._arms[a].joint_names), device=env.device) for a in ARMS},
            env._sph, env.device, jacobian_reference='com')
        profiles = torch.full((64,), 0 if args.method == 'raw' else 2, device=env.device)
        history, dynamics, subgeom, geo_receipts = [], [], [], []
        for step in range(reg['control_steps']):
            pre = env.compute_dist()
            env._last_out = pre
            record = dict(pre_min_raw_gap_m=array(pre.dists.min(-1).values),
                pre_pending=np.stack([np.concatenate([array(v[a]) for a in ARMS], -1) for v in queue.pending]))
            model = dict(step=np.asarray(step))
            for arm in ARMS:
                view = env._arms[arm].root_physx_view
                for field, getter in [('q', view.get_dof_positions), ('qd', view.get_dof_velocities),
                                      ('mass_matrix', view.get_generalized_mass_matrices),
                                      ('actual_velocity_target', view.get_dof_velocity_targets),
                                      ('actuation_force', view.get_dof_actuation_forces),
                                      ('gravity', view.get_gravity_compensation_forces),
                                      ('coriolis', view.get_coriolis_and_centrifugal_compensation_forces)]:
                    model[arm + '_' + field] = array(getter())
                record['pre_' + arm + '_q'] = model[arm + '_q']
                record['pre_' + arm + '_qd'] = model[arm + '_qd']
            reference = {a: tape[a][step] for a in ARMS}
            if args.method == 'multirow_hold_fallback':
                issued, info = make_multirow_issue(env, full_provider, pre, profiles, reference, reg['multirow_settings'])
                record.update({'multi_' + k: array(v) for k, v in info.items()})
                record['nominal_issued_target'] = np.concatenate([array(issued[a]) for a in ARMS], -1)
                issued, fallback = issue_or_initial_hold(issued, pending, info['model_constraints_satisfied'])
                record['fallback_from_model_unsatisfied'] = array(fallback)
            else:
                issued = reference
            applied = queue.push(issued)
            record['applied_target'] = np.concatenate([array(applied[a]) for a in ARMS], -1)
            record['issued_target'] = np.concatenate([array(issued[a]) for a in ARMS], -1)
            record['reference_target'] = np.concatenate([array(reference[a]) for a in ARMS], -1)
            for arm in ARMS:
                env._arms[arm].set_joint_position_target(applied[arm], joint_ids=env._joint_idx[arm])
            scalar = np.zeros(64)
            sub_min = []
            for sub in range(cfg.decimation):
                env.scene.write_data_to_sim()
                for arm in ARMS:
                    model[arm + '_actual_position_target'] = array(env._arms[arm].root_physx_view.get_dof_position_targets())
                env.sim.step(render=False)
                env.scene.update(cfg.sim.dt)
                observer.capture(step, sub, float(cfg.sim.dt))
                scalar = np.maximum(scalar, observer.last_env_scalar_max)
                geom = array(env.compute_dist().dists)
                sub_min.append(geom.min(-1))
                subgeom.append(geom)
            record['substep_min_raw_gap_m'] = np.stack(sub_min)
            record['scalar_normal_max_N'] = scalar
            for arm in ARMS:
                view = env._arms[arm].root_physx_view
                record['post_' + arm + '_q'] = array(view.get_dof_positions())
                record['post_' + arm + '_qd'] = array(view.get_dof_velocities())
                record['post_' + arm + '_body_pos_local'] = array(env._arms[arm].data.body_pos_w - env.scene.env_origins[:, None])
            history.append(record)
            dynamics.append(model)
            completed = step + 1
            if completed % 16 == 0 or completed == reg['control_steps']:
                p = out / ('all_raw_geometry_%04d.npz' % completed)
                np.savez_compressed(p, distances=np.stack(subgeom),
                    first_macro=np.asarray(completed - len(subgeom) // cfg.decimation),
                    global_input_id=ids)
                geo_receipts.append(dict(path=p.name, sha256=sha(p), events=len(subgeom)))
                subgeom = []
            if completed % 20 == 0:
                print('PAIRED', args.batch, args.method, completed, '/', reg['control_steps'],
                    'scalar_peak_N', float(scalar.max()), flush=True)
        np.savez_compressed(out / 'response_stream.npz', **{k: np.stack([r[k] for r in history]) for k in history[0]})
        np.savez_compressed(out / 'dynamics_stream.npz', **{k: np.stack([r[k] for r in dynamics]) for k in dynamics[0]})
        (out / 'all_raw_geometry_receipts.json').write_text(json.dumps(dict(status='complete',
            physics_events=2*completed, raw_rows=9021, exemptions=0, chunks=geo_receipts), indent=2) + '\n')
        observer.close(completed)
        observer = None
        protocol.update(status='complete', completed_steps=completed, physics_events=2*completed,
            rendering_transform_transport='No camera in this separate physical-only mechanism diagnostic',
            native_FIFO6_preserved=True, all74initial_q_qd_targets_exact=True,
            all74initial_native_hard_limits_valid=True, all9021raw_geometry_every_microstep=True,
            all82rigid_owner_points_every_microstep=True, pre_generated_recipe_unchanged=True,
            paired_controller='existing nominal multirow_response_v3 plus initial-target hold on model-invalid lanes; unproven backup',
            candidate_registration_sha256=sha(H / 'HOLD_FALLBACK_DEV_REG_V1.json'),
            no_camera_in_this_diagnostic=True, no_safe_backup_proof=True,
            future_prediction_bound_certified=False,
            initial_geometry_binding_max_abs_m=float(abs(array(initial.dists)-geometry_bank['accepted_d'][ids]).max()))
    except BaseException as error:
        import traceback
        protocol.update(status='failed', completed_steps=completed, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        (out / 'response_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
        if env is not None:
            env.close()
        app.close()
    print(json.dumps(protocol), flush=True)


if __name__ == '__main__':
    main()
