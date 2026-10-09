"""Policy-independent FIFO6 admission, all native substeps retained."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

H = Path(__file__).resolve().parent
OLD = Path('/home/liyufeng/safeduo/artifacts/safety_velocity_arrival_20261008_1519')
HF = Path('/home/liyufeng/safeduo/artifacts/safety_feasible_response_20261009_0316')
sys.path.insert(0, str(OLD))
sys.path.insert(0, str(HF))
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
    parser.add_argument('--batch', type=int, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    assert 0 <= args.batch < 8
    reg = json.loads((H / 'REGISTRATION_V1.json').read_text())
    bank_path=Path(reg['bank']);recipe_path=Path(reg['recipe']);fixture_path=Path(reg['fixture'])
    for path,key in [(bank_path,'bank_sha256'),(recipe_path,'recipe_sha256'),(fixture_path,'fixture_sha256')]:
        assert hashlib.sha256(path.read_bytes()).hexdigest()==reg[key]
    with np.load(bank_path) as z:bank={k:z[k] for k in z.files}
    with np.load(recipe_path) as z:recipe={k:z[k] for k in z.files}
    with np.load(fixture_path) as z:fixture={k:z[k] for k in z.files}
    ids = np.arange(args.batch * 64, (args.batch + 1) * 64)
    out = Path(args.out)
    out.mkdir(exist_ok=False)
    app = AppLauncher(args).app
    env, observer = None, None
    protocol = dict(status='running', independent_policy_outcomes_consulted=False,
        qualification_batch=args.batch, initial_states=64, control_steps=0, physics_events=0,
        complete_controller_trials=0, hidden_solver_state_restored=False, safety_acceptance=False)
    try:
        from safeduo.envs.duo_env import DuoEnv, make_duo_env_cfg
        from native_state_capture import array
        from native_all_body_point_contacts import PointContacts
        from safe_hand_opening import install_open_hand_defaults
        cfg = make_duo_env_cfg(num_envs=64, device=args.device,
            yaml_name='duo_env_a31_pending_guard.yaml', coordinator=True, arm_aware_obs=True, p2_obs=True)
        cfg.seed = 0
        cfg.enable_viz_camera=False
        cfg.sim.use_fabric=True
        for arm in ARMS:
            props=cfg.robot_cfgs[arm].spawn.articulation_props
            cfg.robot_cfgs[arm].spawn.articulation_props=props.replace(solver_position_iteration_count=64)
        cfg.coordinator['terminate_on_violation'] = False
        env = DuoEnv(cfg)
        from pxr import Usd,UsdPhysics,PhysxSchema
        import omni.usd
        roots=[]
        for prim in omni.usd.get_context().get_stage().Traverse(Usd.TraverseInstanceProxies()):
            if str(prim.GetPath()).startswith('/World/envs/env_0/') and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                api=PhysxSchema.PhysxArticulationAPI(prim);pos=int(api.GetSolverPositionIterationCountAttr().Get());vel=int(api.GetSolverVelocityIterationCountAttr().Get())
                assert pos==64 and vel==0
                roots.append(dict(path=str(prim.GetPath()),position_iterations=pos,velocity_iterations=vel))
        assert len(roots)==4
        (out/'actual_solver_roots.json').write_text(json.dumps(roots,indent=2)+'\n')
        install_open_hand_defaults(env)
        env._delta_src = ZeroSource()
        env.reset()
        parameters = {'global_input_id':bank['selected_indices'][ids], 'bank_position':ids,'cell_id':recipe['cell_id'][ids], 'velocity_fraction':recipe['velocity_fraction'][ids],'command_mode':recipe['command_mode'][ids],'refresh_controls':recipe['refresh_controls'][ids]}
        offset = 0
        for arm in ARMS:
            art = env._arms[arm]
            idx = env._joint_idx[arm]
            assert np.array_equal(np.asarray(art.joint_names), fixture[arm + '_native_joint_names'])
            q = torch.as_tensor(bank[arm + '_native_q'][ids], device=env.device).clone()
            v = torch.zeros_like(q)
            vmax = art.root_physx_view.get_dof_max_velocities().to(env.device)
            assert np.array_equal(array(vmax[:,idx]),np.broadcast_to(recipe['native_vmax_rad_s'][offset:offset+len(idx)],(64,len(idx))))
            v[:,idx]=torch.as_tensor(recipe['controlled_initial_qd'][ids,offset:offset+len(idx)],device=env.device)
            offset += len(idx)
            root_pose = torch.as_tensor(fixture[arm + '_native_root_xyzw'][0], device=env.device)[None].expand(64, -1).clone()
            root_pose[:, :3] += env.scene.env_origins - torch.as_tensor(fixture['source_origin'][0], device=env.device)
            art.write_root_pose_to_sim(torch.cat([root_pose[:, :3], root_pose[:, 6:7], root_pose[:, 3:6]], -1))
            art.write_root_velocity_to_sim(torch.zeros((64, 6), device=env.device))
            art.write_joint_state_to_sim(q, v)
            hold=torch.as_tensor(fixture[arm+'_full_hold_target'][0],device=env.device)[None].expand(64,-1).clone();hold[:,idx]=q[:,idx]
            art.set_joint_position_target(hold)
            parameters[arm + '_initial_q'] = array(q)
            parameters[arm + '_initial_qd'] = array(v)
            parameters[arm + '_joint_names'] = np.asarray(art.joint_names)
            parameters[arm + '_controlled_joint_indices'] = array(idx)
            parameters[arm + '_hard_limits'] = array(art.root_physx_view.get_dof_limits())
            parameters[arm + '_vmax'] = array(vmax)
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(0.)
        for arm in ARMS:
            view = env._arms[arm].root_physx_view
            assert np.array_equal(array(view.get_dof_positions()), parameters[arm + '_initial_q'])
            assert np.array_equal(array(view.get_dof_velocities()), parameters[arm + '_initial_qd'])
            q = parameters[arm + '_initial_q']
            hard = parameters[arm + '_hard_limits']
            tolerance = reg['initial_hard_limit_tolerance_rad']
            assert ((q >= hard[..., 0] - tolerance) & (q <= hard[..., 1] + tolerance)).all()
            parameters[arm + '_full_hold_target'] = array(view.get_dof_position_targets())
        initial = array(env.compute_dist().dists)
        diagnostic = {'initial_raw_geometry': initial, 'source_raw_geometry': bank['accepted_d'][ids],
            'global_input_id': ids, 'current_env_origins': array(env.scene.env_origins)}
        for arm in ARMS:
            view = env._arms[arm].root_physx_view
            diagnostic[arm + '_body_transforms'] = array(view.get_link_transforms())
            diagnostic[arm + '_root'] = array(view.get_root_transforms())
            diagnostic[arm + '_body_pos_cached_local'] = array(env._arms[arm].data.body_pos_w - env.scene.env_origins[:, None])
            diagnostic[arm + '_source_bank_body_pos_local'] = bank[arm + '_body_pos_local'][ids]
            diagnostic[arm + '_source_bank_body_quat_wxyz'] = bank[arm + '_body_quat_wxyz'][ids]
            diagnostic[arm + '_exact_q'] = array(view.get_dof_positions())
            diagnostic[arm + '_exact_qd'] = array(view.get_dof_velocities())
        np.savez_compressed(out / 'initial_geometry_binding_diagnostic.npz', **diagnostic)
        error = np.abs(initial - bank['accepted_d'][ids])
        protocol['initial_geometry_binding_max_abs_m'] = float(error.max())
        protocol['initial_geometry_binding_critical_location'] = np.unravel_index(error.argmax(), error.shape)
        protocol['initial_geometry_binding_critical_location'] = [int(v) for v in protocol['initial_geometry_binding_critical_location']]
        binding = json.loads(Path(reg['binding_contract']).read_text())
        for arm in ARMS:
            assert np.allclose(diagnostic[arm + '_body_pos_cached_local'],
                diagnostic[arm + '_source_bank_body_pos_local'], atol=binding['body_position_local_absolute_tolerance_m'], rtol=0)
            xyzw = diagnostic[arm + '_body_transforms'][..., 3:]
            wxyz = np.concatenate([xyzw[..., -1:], xyzw[..., :3]], -1)
            source_quat = diagnostic[arm + '_source_bank_body_quat_wxyz']
            quaternion_error = np.minimum(abs(wxyz - source_quat).max(-1), abs(wxyz + source_quat).max(-1))
            assert (quaternion_error <= binding['body_orientation_sign_invariant_absolute_tolerance']).all()
        assert np.allclose(initial, bank['accepted_d'][ids], atol=binding['source_geometry_absolute_binding_tolerance_m'], rtol=0)
        protocol['binding_contract_sha256'] = hashlib.sha256(Path(reg['binding_contract']).read_bytes()).hexdigest()


        protocol['initial_geometry_replay_admitted_count']=int((initial.min(-1)>=.0001).sum())
        parameters['initial_all_raw_geometry'] = initial
        np.savez_compressed(out / 'prefix_inputs.npz', **parameters)
        observer = PointContacts(env, out)
        geometry, force, supplementary_hard, supplementary_velocity = [], [], [], []
        for step in range(reg['physics_prefix_controls']):
            for substep in range(cfg.decimation):
                env.scene.write_data_to_sim()
                for arm in ARMS:
                    assert np.array_equal(array(env._arms[arm].root_physx_view.get_dof_position_targets()), parameters[arm + '_full_hold_target'])
                env.sim.step(render=False)
                env.scene.update(cfg.sim.dt)
                observer.capture(step, substep, float(cfg.sim.dt))
                geometry.append(array(env.compute_dist().dists))
                force.append(observer.last_env_scalar_max.copy())
                hard_violation, velocity_violation = np.zeros(64), np.zeros(64)
                for arm in ARMS:
                    q = array(env._arms[arm].root_physx_view.get_dof_positions())
                    v = array(env._arms[arm].root_physx_view.get_dof_velocities())
                    hard = parameters[arm + '_hard_limits']
                    hard_violation = np.maximum(hard_violation, np.maximum(hard[..., 0] - q, q - hard[..., 1]).max(-1))
                    velocity_violation = np.maximum(velocity_violation, (abs(v) - parameters[arm + '_vmax']).max(-1))
                supplementary_hard.append(hard_violation.clip(0))
                supplementary_velocity.append(velocity_violation.clip(0))
        observer.close(reg['physics_prefix_controls'])
        observer = None
        geometry, force = np.stack(geometry), np.stack(force)
        geo_valid = (initial.min(-1)>=reg['initial_geometry_gate_m']) & (geometry.min((0,2))>=reg['physical_geometry_min_m'])
        joint_valid=(np.stack(supplementary_hard).max(0)<=reg['joint_hard_tolerance_rad']) & (np.stack(supplementary_velocity).max(0)<=reg['joint_speed_tolerance_rad_s'])
        force_valid = force.max(0) <= reg['physics_point_force_gate_N']
        np.savez_compressed(out / 'prefix_result.npz', all_raw_geometry=geometry,
            scalar_normal_max_N=force, global_input_id=ids, geometry_qualified=geo_valid,
            force_qualified=force_valid, admitted=geo_valid & force_valid & joint_valid,
            supplementary_hard_limit_violation_rad=np.stack(supplementary_hard),
            supplementary_velocity_limit_violation_rad_s=np.stack(supplementary_velocity))
        protocol.update(status='complete', control_steps=reg['physics_prefix_controls'],
            physics_events=cfg.decimation * reg['physics_prefix_controls'],
            all74native_initial_q_qd_exact=True, initial74native_hard_limits_valid=True,
            all12_substep_9021raw_geometry_recorded=True, all82rigid_owner_normal_contacts_recorded=True,
            qualified_count=int((geo_valid & force_valid & joint_valid).sum()),position_iterations=64,hand_margin_rad=.03,full74joint_gate_part_of_preregistered_admission=True, geometry_qualified_count=int(geo_valid.sum()),
            force_qualified_count=int(force_valid.sum()),
            prefix_result_sha256=hashlib.sha256((out / 'prefix_result.npz').read_bytes()).hexdigest(),
            prefix_inputs_sha256=hashlib.sha256((out / 'prefix_inputs.npz').read_bytes()).hexdigest())
    except BaseException as error:
        import traceback
        protocol.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        (out / 'prefix_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
        if env is not None:
            env.close()
        app.close()
    print(json.dumps(protocol), flush=True)


if __name__ == '__main__':
    main()
