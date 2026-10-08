"""Captured-state actuator-response contrasts; no actor or safety promotion."""
import argparse
from collections import deque
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

H = Path(__file__).resolve().parent
OLD = Path('/home/liyufeng/safeduo/artifacts/safety_velocity_arrival_20261008_1519')
sys.path.insert(0, str(OLD))
ARMS = ('F_L', 'F_R', 'U_L', 'U_R')


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
    parser.add_argument('--method', choices=['raw','multirow'], required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    reg = json.loads((H / 'REGISTRATION_DYNAMIC_V1.json').read_text())
    manifest = json.loads((H / 'INPUT_MANIFEST.json').read_text())
    input_path = Path(reg['raw']) / 'captured_inputs.npz'
    assert hashlib.sha256(input_path.read_bytes()).hexdigest() == manifest['inputs_sha256']
    with np.load(input_path, allow_pickle=False) as source:
        inputs = {k: source[k] for k in source.files}
    with np.load(Path(reg['raw']) / 'random_recipe.npz', allow_pickle=False) as source:
        recipe = {k: source[k] for k in source.files}
    assert hashlib.sha256((Path(reg['raw']) / 'random_recipe.npz').read_bytes()).hexdigest() == manifest['recipe_sha256']
    app = AppLauncher(args).app
    env, observer = None, None
    protocol = dict(status='running', known_source_states=0, fresh_random_arm_states=64, native_lanes=64, method=args.method,
                    random_holdout_windows=0, actor_actions=0, safety_acceptance=False,
                    hidden_solver_state_restored=False, steps_planned=reg['control_steps'])
    completed = 0
    try:
        from safeduo.envs.duo_env import DuoEnv, make_duo_env_cfg
        from safeduo.eval.perturbations import TargetDelayQueue
        from safeduo.baselines.base import stack_robot
        from safeduo.safety.geometry import IsaacGeometryProvider
        from native_all_body_point_contacts import PointContacts
        from native_state_capture import array
        from safe_hand_opening import install_open_hand_defaults
        cfg = make_duo_env_cfg(num_envs=64, device=args.device,
                              yaml_name='duo_env_a31_pending_guard.yaml',
                              coordinator=True, arm_aware_obs=True, p2_obs=True)
        cfg.coordinator['terminate_on_violation'] = False
        cfg.episode_length_s = 21.
        cfg.seed = 0
        cfg.enable_viz_camera = True
        env = DuoEnv(cfg)
        assert env.scene.sensors.pop("viz_cam") is env._viz_cam
        env._viz_cam.reset()
        opening = install_open_hand_defaults(env)
        env._delta_src = ZeroSource()
        env.reset()
        (out / 'open_hand_initialization.json').write_text(json.dumps(opening, indent=2) + '\n')
        ids = np.arange(64)
        profiles = np.full(64, 0 if args.method == 'raw' else 2)
        critical = torch.as_tensor(inputs['critical_rows'][ids], device=env.device)
        lanes = torch.arange(64, device=env.device)
        target, initial_q, initial_qd, pending = {}, {}, {}, {}
        parameters, velocity_clips, target_clips = {}, {}, {}
        sampled_tape = {}
        controlled_offset = 0
        for arm in ARMS:
            art = env._arms[arm]
            idx = env._joint_idx[arm]
            assert np.array_equal(np.asarray(art.joint_names), inputs[arm + '_native_joint_names'])
            assert np.array_equal(idx.cpu().numpy(), inputs[arm + '_controlled_joint_indices'])
            q = torch.as_tensor(inputs[arm + '_native_q'][ids], device=env.device).clone()
            qd = torch.zeros_like(q)
            limits = art.data.soft_joint_pos_limits[:, idx]
            width = len(idx)
            unit = torch.as_tensor(recipe['pose_unit'][:, controlled_offset:controlled_offset+width], device=env.device)
            fraction = torch.as_tensor(recipe['strata'][:, 0], device=env.device)[:, None]
            q[:, idx] = torch.as_tensor(inputs[arm + "_native_q"][:, idx.cpu().numpy()], device=env.device)
            vunit = torch.as_tensor(recipe['velocity_unit'][:, controlled_offset:controlled_offset+width], device=env.device)
            vfrac = torch.as_tensor(recipe['strata'][:, 1], device=env.device)[:, None]
            parameters[arm + '_soft_limits'] = array(limits)
            velocity_limit = art.root_physx_view.get_dof_max_velocities().to(env.device)
            qd[:, idx] = vunit * vfrac * velocity_limit[:, idx]
            velocity_clips[arm] = np.zeros(64, dtype=np.int64)
            root = torch.as_tensor(inputs[arm + '_native_root_xyzw'][ids], device=env.device).clone()
            root[:, :3] += env.scene.env_origins - torch.as_tensor(inputs['source_origin'][ids], device=env.device)
            wxyz = torch.cat([root[:, :3], root[:, 6:7], root[:, 3:6]], -1)
            art.write_root_pose_to_sim(wxyz)
            art.write_root_velocity_to_sim(torch.as_tensor(inputs[arm + '_native_root_velocity'][ids], device=env.device))
            art.write_joint_state_to_sim(q, qd)
            current = q[:, idx].clone()
            issued = current.clone()
            noise = torch.as_tensor(recipe['command_unit'][:, :, controlled_offset:controlled_offset+width], device=env.device)
            amp = torch.as_tensor(recipe['strata'][:, 2], device=env.device)[None, :, None]
            sampled_tape[arm] = (current[None] + amp * noise).clamp(limits[None, ..., 0], limits[None, ..., 1])
            controlled_offset += width
            future = issued.clone()
            target_clips[arm] = np.zeros(64, dtype=np.int64)
            slots = current[:, None].expand(-1, 6, -1).clone()
            pending[arm] = slots
            target[arm], initial_q[arm], initial_qd[arm] = future, q.clone(), qd.clone()
            before_target = current.clone()
            art.set_joint_position_target(before_target, joint_ids=idx)
            parameters[arm + '_disable_gravity_cfg'] = np.full(64, bool(art.cfg.spawn.rigid_props.disable_gravity))
            parameters[arm + '_solver_position_iterations_cfg'] = np.full(64, art.cfg.spawn.articulation_props.solver_position_iteration_count)
            parameters[arm + '_solver_velocity_iterations_cfg'] = np.full(64, art.cfg.spawn.articulation_props.solver_velocity_iteration_count)
            parameters[arm + '_controlled_joint_indices'] = array(idx)
            parameters[arm + '_native_joint_names'] = np.asarray(art.joint_names)
            for name, getter in [('stiffness', art.root_physx_view.get_dof_stiffnesses),
                                 ('damping', art.root_physx_view.get_dof_dampings),
                                 ('max_force', art.root_physx_view.get_dof_max_forces),
                                 ('max_velocity', art.root_physx_view.get_dof_max_velocities),
                                 ('armature', art.root_physx_view.get_dof_armatures),
                                 ('hard_limits', art.root_physx_view.get_dof_limits),
                                 ('friction', art.root_physx_view.get_dof_friction_properties),
                                 ('drive_model', art.root_physx_view.get_dof_drive_model_properties),
                                 ('masses', art.root_physx_view.get_masses),
                                 ('inertias', art.root_physx_view.get_inertias)]:
                parameters[arm + '_' + name] = array(getter())
        queue = TargetDelayQueue(6)
        queue.pending = deque({arm: pending[arm][:, slot].clone() for arm in ARMS} for slot in range(6))
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(0.)
        for arm in ARMS:
            assert torch.equal(env._arms[arm].root_physx_view.get_dof_positions(), initial_q[arm])
            assert torch.equal(env._arms[arm].root_physx_view.get_dof_velocities(), initial_qd[arm])
        full = env.compute_dist()
        env._last_out = full
        observed = array(full.dists)
        original = observed.copy()
        gap_error = np.zeros(64)
        parameters['initial_strata'] = recipe['strata']
        np.savez_compressed(out / 'actual_random_issue_tape.npz', **{a: array(t) for a,t in sampled_tape.items()})
        np.savez_compressed(out / 'initial_geometry.npz', d=observed,
                            exempt=array(full.full_viol_exempt), dmin=array(full.full_dmin),
                            source_full_d=original, source_gap_max_abs_error=gap_error,
                            source_state_id=ids, profile_id=profiles,
                            critical_rows=array(critical),
                            **{arm + '_q': array(initial_q[arm]) for arm in ARMS},
                            **{arm + '_qd': array(initial_qd[arm]) for arm in ARMS},
                            **{arm + '_future_target': array(target[arm]) for arm in ARMS},
                            **{arm + '_initial_pending_targets': array(pending[arm]) for arm in ARMS})
        (out / 'lane_assignment.json').write_text(json.dumps([
            dict(lane=i, source_state=int(ids[i]), profile=reg['profiles'][0 if args.method == 'raw' else 1],
                 original=reg['source_states'][int(ids[i])]) for i in range(64)], indent=2) + '\n')
        np.savez_compressed(out / 'resolved_native_parameters.npz', **parameters)
        observer = PointContacts(env, out)
        history, dynamics = [], []
        full_provider = IsaacGeometryProvider(env._arms,
            {a: torch.arange(len(env._arms[a].joint_names), device=env.device) for a in ARMS},
            env._sph, env.device, jacobian_reference="com")

        def geometry():
            data = env.compute_dist()
            env._last_out = data
            selected = critical[:, None]
            one = replace(data, active_idx=selected, active_mask=torch.ones_like(selected, dtype=torch.bool),
                          active_pairs=torch.stack([data.dists[lanes, critical], data.closing[lanes, critical],
                              env._sph.class_id[critical], env._sph.pair_id[critical]], -1)[:, None],
                          active_dmin=data.full_dmin[lanes, critical][:, None],
                          viol_exempt=data.full_viol_exempt[lanes, critical][:, None])
            rows = env._provider.rows_from(one, env._body_pos_cache)
            rate = sum(torch.einsum('nmd,nd->nm', rows.J[r], stack_robot(env.scene_state().qd, r)) for r in ['F', 'U'])[:, 0]
            full_rows = full_provider.rows_from(one, env._body_pos_cache)
            full_rate = sum(torch.einsum("nmd,nd->nm", full_rows.J[r],
                torch.cat([env._arms[a].root_physx_view.get_dof_velocities() for a in aa], -1))
                for r, aa in [("F", ARMS[:2]), ("U", ARMS[2:])])[:, 0]
            for r, aa in [("F", ARMS[:2]), ("U", ARMS[2:])]:
                cols = torch.cat([env._joint_idx[aa[0]], env._joint_idx[aa[1]] + len(env._arms[aa[0]].joint_names)])
                assert torch.allclose(full_rows.J[r][:, :, cols], rows.J[r], atol=2e-6, rtol=0)
            return data, rate, rows, full_rows, full_rate

        from multirow_response_v3 import make_multirow_issue
        from pilot_views import PilotViews
        camera = PilotViews(env, out, reg["camera"])
        np.savez_compressed(out / "body_identity.npz", **{a: np.asarray(env._arms[a].body_names) for a in ARMS})
        camera.capture(-1)
        profile_tensor = torch.as_tensor(profiles, device=env.device)
        for step in range(reg['control_steps']):
            pre, pre_rate, rows, full_rows, full_rate = geometry()
            record = dict(pre_d=array(pre.dists), pre_exempt=array(pre.full_viol_exempt),
                          pre_critical_velocity=array(pre_rate),
                          J_F=array(rows.J['F'][:, 0]), J_U=array(rows.J['U'][:, 0]),
                          full_J_F=array(full_rows.J["F"][:, 0]), full_J_U=array(full_rows.J["U"][:, 0]),
                          pre_full_critical_velocity=array(full_rate),
                          pre_native_critical_closing=array(pre.closing[lanes, critical]),
                          pre_pending=np.stack([np.concatenate([array(v[a]) for a in ARMS], -1) for v in queue.pending]))
            model = dict(step=np.asarray(step))
            for arm in ARMS:
                view = env._arms[arm].root_physx_view
                for name, getter in [('q', view.get_dof_positions), ('qd', view.get_dof_velocities),
                                     ('mass_matrix', view.get_generalized_mass_matrices),
                                     ('actual_velocity_target', view.get_dof_velocity_targets),
                                     ('actuation_force', view.get_dof_actuation_forces),
                                     ('projected_joint_force', view.get_dof_projected_joint_forces),
                                     ('gravity', view.get_gravity_compensation_forces),
                                     ('coriolis', view.get_coriolis_and_centrifugal_compensation_forces)]:
                    model[arm + '_' + name] = array(getter())
                record['pre_' + arm + '_q'] = model[arm + '_q']
                record['pre_' + arm + '_qd'] = model[arm + '_qd']
            # Every synthetic initial state is retained, including invalid geometry.
            reference = {a: sampled_tape[a][step] for a in ARMS}
            if args.method == 'multirow':
                multi, multi_info = make_multirow_issue(env, full_provider, pre, profile_tensor, reference, reg['multirow_settings'])
                target = multi
                record.update({'multi_' + k: array(v) for k, v in multi_info.items()})
            else:
                target = reference
            guard_info = dict(triggered=torch.full((64,), args.method == 'multirow', device=env.device),
                current_gap=pre.dists[lanes, critical], current_full_rate=full_rate)
            record.update({'guard_' + k: array(v) for k, v in guard_info.items()})
            record['unmodified_pre_pending'] = record['pre_pending'].copy()
            oracle_mask = torch.zeros_like(profile_tensor, dtype=torch.bool)
            for slot in queue.pending:
                for a in ARMS: slot[a][oracle_mask] = target[a][oracle_mask]
            record['pre_pending'] = np.stack([np.concatenate([array(v[a]) for a in ARMS], -1) for v in queue.pending])
            applied = queue.push(target)
            record['applied_target'] = np.concatenate([array(applied[a]) for a in ARMS], -1)
            record['issued_target'] = np.concatenate([array(target[a]) for a in ARMS], -1)
            for arm in ARMS:
                env._arms[arm].set_joint_position_target(applied[arm], joint_ids=env._joint_idx[arm])
            scalar, vector = np.zeros(64), np.zeros(64)
            for sub in range(cfg.decimation):
                env.scene.write_data_to_sim()
                for arm in ARMS:
                    model[arm + '_actual_position_target'] = array(env._arms[arm].root_physx_view.get_dof_position_targets())
                env.sim.step(render=False)
                env.scene.update(cfg.sim.dt)
                observer.capture(step, sub, float(cfg.sim.dt))
                scalar = np.maximum(scalar, observer.last_env_scalar_max)
                vector = np.maximum(vector, observer.last_env_normal_max)
            post, rate, _, _, full_rate = geometry()
            record.update(post_d=array(post.dists), post_exempt=array(post.full_viol_exempt),
                          post_critical_velocity=array(rate), post_full_critical_velocity=array(full_rate),
                          post_native_critical_closing=array(post.closing[lanes, critical]), scalar_normal_max_N=scalar, vector_normal_max_N=vector)
            for arm in ARMS:
                view = env._arms[arm].root_physx_view
                record['post_' + arm + '_q'] = array(view.get_dof_positions())
                record['post_' + arm + '_qd'] = array(view.get_dof_velocities())
                record['post_' + arm + '_body_pos_local'] = array(env._arms[arm].data.body_pos_w - env.scene.env_origins[:, None, :])
            history.append(record)
            dynamics.append(model)
            completed = step + 1
            if completed % reg["camera"]["every_controls"] == 0 or completed == reg["control_steps"]:
                camera.capture(step)
            if completed % 10 == 0:
                print('RESPONSE', completed, '/', reg['control_steps'], 'scalar_peak', float(scalar.max()), flush=True)
        np.savez_compressed(out / 'response_stream.npz', **{k: np.stack([r[k] for r in history]) for k in history[0]})
        np.savez_compressed(out / 'dynamics_stream.npz', **{k: np.stack([r[k] for r in dynamics]) for k in dynamics[0]})
        camera.close()
        observer.close(completed)
        observer = None
        protocol.update(status='complete', completed_steps=completed, physics_events=2 * completed,
                        probe_kind="fresh wide pure-random paired development", independent_fresh_arm_states=64,
                        original_issue_tape_used=False, pre_generated_pure_uniform_commands=True,
                        instantaneous_profiles_are_oracles=[], native_FIFO6_preserved=True, protected_critical_rows_only=False, nearby_raw_geometry_rows=24,
                        full_drive_effort_and_controlled_velocity_constrained_nominally=True,
                        future_prediction_bound_certified=False,
                        initial_q_qd_native_exact=True, original_raw_geometry_max_error_m=float(gap_error.max()),
                        controlled_velocity_clip_counts={a: v.tolist() for a, v in velocity_clips.items()},
                        proposed_twice_retreat_clip_counts={a: v.tolist() for a, v in target_clips.items()})
    except BaseException as error:
        import traceback
        protocol.update(status='failed', completed_steps=completed, error=type(error).__name__ + ': ' + str(error), error_traceback=traceback.format_exc())
        raise
    finally:
        (out / 'response_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
        if env is not None:
            env.close()
        app.close()
    print(json.dumps(protocol), flush=True)


if __name__ == '__main__':
    main()
