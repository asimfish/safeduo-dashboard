"""One prospective friction-only four-hand common beam probe, native every substep."""
from pathlib import Path
import argparse
from collections import deque
import hashlib
import json
import sys
import time
import traceback
import faulthandler
import signal
import gc
faulthandler.register(signal.SIGUSR1,file=sys.stderr,all_threads=True)
import numpy as np
import torch

P=Path(__file__).resolve().parent
H=P.parent
HF=Path('/home/liyufeng/safeduo/artifacts/safety_feasible_response_20261009_0316')
OLD=Path('/home/liyufeng/safeduo/artifacts/safety_velocity_arrival_20261008_1519')
for root in [HF,OLD,P]:sys.path.insert(0,str(root))
sys.path.append('/home/liyufeng/safeduo/artifacts/safety_object_binding_20261006')
ARMS=('F_L','F_R','U_L','U_R')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def pack_friction(raw,capacity):
    force,point,counts,starts=[v.detach().clone().cpu().numpy().copy() for v in raw]
    counts,starts=counts.astype(np.int64),starts.astype(np.int64)
    assert counts.shape==starts.shape and counts.min()>=0 and counts.sum()<capacity
    fields=dict(counts=counts,starts=starts)
    indices=[];partners=[]
    for sensor,partner in np.argwhere(counts>0):
        start,n=int(starts[sensor,partner]),int(counts[sensor,partner])
        assert start>=0 and start+n<=capacity
        indices.extend(range(start,start+n));partners.extend([int(partner)]*n)
    indices=np.asarray(indices,dtype=np.int64)
    fields.update(raw_indices=indices,partner_indices=np.asarray(partners,dtype=np.int64),
                  forces=force[indices],points=point[indices])
    assert all(np.isfinite(v).all() for v in fields.values())
    return fields


def main():
    from isaaclab.app import AppLauncher
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    AppLauncher.add_app_launcher_args(parser);args=parser.parse_args();args.enable_cameras=True
    reg=json.loads((P/'REGISTRATION_V1.json').read_text())
    from microstep_contract_v1 import microsteps
    substeps=microsteps(reg)
    for f,d in reg['source_sha256'].items():assert sha(f)==d,'source changed: '+f
    out=Path(args.out);out.mkdir(exist_ok=False)
    for binding in reg['private_driver_library_bindings']:
        assert sha(binding['destination'])==binding['source_sha256']
    save(out/'native_private_driver_bindings_v1.json',dict(
        bindings=reg['private_driver_library_bindings'],
        actual_kernel_module_version=Path('/proc/driver/nvidia/version').read_text(),
        graphics_libraries_not_claimed_uniform570=True,
        native_QD_acceptance_not_replaced=True))
    app=AppLauncher(args).app
    env=None;observer=None;completed=0;frames=[];chunks=[];history=[];rows=[]
    protocol=dict(status='running',task_status='NOT_STARTED',method='RAW_PHYSICAL_GRASP_PROBE',
        trained_actor=False,system0_active=False,objects=1,arms=4,hands=4,
        common_object=True,physical_profile='Experimental v7:2FR3/F2+2UR5/DFX',
        maximum_controls=reg['maximum_controls'],physics_events=0,completed_controls=0,
        no_runtime_object_state_writes=True,no_attachment_joints=True,
        fullSystem0_accepted=False)
    save(out/'fourhand_protocol.json',protocol)
    try:
        import isaaclab.sim as sim_utils
        import omni.usd
        import omni.physics.tensors as tensors
        from pxr import Usd,UsdPhysics,PhysxSchema
        from PIL import Image
        from safeduo.envs.duo_env import DuoEnv,make_duo_env_cfg
        from native_all_body_point_contacts import PointContacts
        from native_state_capture import array
        from safe_hand_opening import install_open_hand_defaults
        from point_contact_data import pack_contact_event,pack_contact_events
        from hand_driver import HandDriver
        from feedback_grip_v1 import FeedbackGrip
        from coupled_hand_filter_v1 import coupled_filtered_hand_targets
        from coupled_motor_effort_v1 import motor_effort_projection
        from coupled_implicit_motor_effort_v1 import coupled_backward_euler_torque
        from native_camera_readback_v1 import snapshot as camera_snapshot
        from clock_contract import control_clock
        from fourhand_payload_phases_v1 import Observation,FourHandPayloadPhases
        from grasp_settle_gate_v1 import GraspSettleGate
        torch.set_num_threads(1)

        class ProbeEnv(DuoEnv):
            def _setup_scene(self):
                super()._setup_scene()
                origin=self.scene.env_origins[0].detach().cpu().numpy()
                for i,y in enumerate([-.18,.18]):
                    support=sim_utils.CuboidCfg(size=(.08,.04,.18),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True,disable_gravity=True),
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.7,dynamic_friction=.6),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.18,.22,.28)))
                    support.func(f'/World/envs/env_0/Support_{i}',support,translation=tuple(origin+np.array([.20,y,.89])))
                for prim in omni.usd.get_context().get_stage().Traverse(Usd.TraverseInstanceProxies()):
                    if prim.HasAPI(UsdPhysics.RigidBodyAPI):PhysxSchema.PhysxContactReportAPI.Apply(prim)

        cfg=make_duo_env_cfg(num_envs=1,device=args.device,yaml_name='duo_env_a31_pending_guard.yaml',
                             coordinator=True,arm_aware_obs=True,p2_obs=True)
        cfg.sim.dt=reg['physics_dt_s'];cfg.decimation=substeps;cfg.sim.render_interval=substeps
        coupled_meta=json.loads(Path(reg['coupled_assets']).read_text())
        with np.load(reg['coupled_reference_endpoints']) as z:coupled_refs={k:z[k].copy() for k in z.files}
        for a in ARMS:
            asset=coupled_meta['assets'][a]
            cfg.robot_cfgs[a].spawn=cfg.robot_cfgs[a].spawn.replace(usd_path=asset['layer'])
            motors=asset['independent_motor_joint_names'];followers=[r['slave'] for r in asset['relations']]
            groups=[k for k in cfg.robot_cfgs[a].actuators if 'hand' in k];assert len(groups)==1
            group=groups[0];actuator=cfg.robot_cfgs[a].actuators[group]
            cfg.robot_cfgs[a].actuators[group]=actuator.replace(joint_names_expr=motors,stiffness=0.,damping=0.)
            cfg.robot_cfgs[a].actuators['hand_passive_mimic']=actuator.replace(joint_names_expr=followers,stiffness=0.,damping=0.)
        cfg.enable_viz_camera=True;cfg.seed=0;cfg.sim.use_fabric=True;cfg.episode_length_s=100.
        cfg.coordinator['terminate_on_violation']=False
        for a in ARMS:
            for group, actuator in list(cfg.robot_cfgs[a].actuators.items()):
                cfg.robot_cfgs[a].actuators[group]=actuator.replace(stiffness=0.,damping=0.)
        cfg.sim.physx.solve_articulation_contact_last=reg['solve_articulation_contact_last']
        cfg.sim.physx.enable_external_forces_every_iteration=reg['enable_external_forces_every_iteration']
        assert cfg.sim.physx.solver_type==1 and cfg.sim.physx.enable_external_forces_every_iteration is True
        cfg.table_objects=[dict(name='beam_common',usd=reg['object_usd'],size=[.04,.7,.08],
             pos=[.20,0,1.02],yaw=0.,mass=.45,scale=[1.,1.,1.],static_friction=.7,dynamic_friction=.6,color=[.55,.57,.60])]
        for arm in ARMS:
            props=cfg.robot_cfgs[arm].spawn.articulation_props
            cfg.robot_cfgs[arm].spawn.articulation_props=props.replace(solver_position_iteration_count=128,solver_velocity_iteration_count=128)
        env=ProbeEnv(cfg);install_open_hand_defaults(env)
        for a in ARMS:
            asset=coupled_meta['assets'][a];names=env._arms[a].joint_names
            hi=[names.index(n) for n in asset['hand_joint_names']]
            assert hi==coupled_refs[a+'_hand_indices'].tolist()
            env._arms[a].data.default_joint_pos[:,hi]=torch.tensor(coupled_refs[a+'_hand_open_q'],device=env.device,dtype=env._arms[a].data.default_joint_pos.dtype)
        class ZeroSource:
            def reset(self,*a,**k):pass
            def sample(self,state):
                from safeduo.safety.types import DeltaCmd
                return DeltaCmd({a:torch.zeros_like(state.q[a]) for a in ARMS})
        env._delta_src=ZeroSource();env.reset()
        stage=omni.usd.get_context().get_stage()
        for name in ['right_thumb_3_joint','right_thumb_4_joint']:
            assert stage.GetPrimAtPath('/World/envs/env_0/F_R/f2_hand/joints/'+name).GetAttribute('physics:axis').Get()=='X'
        protocol['invalid_right_F2_axes_explicitly_canonicalized_to_recorded_X']=True
        protocol['last_setup_stage']='environment_reset_complete';save(out/'fourhand_protocol.json',protocol)
        native_mimic={}
        for a in ARMS:
            asset=coupled_meta['assets'][a];records=[]
            prefix='/World/envs/env_0/'+a
            for r in asset['relations']:
                suffix=r['source_USD_joint_path'].split('/',2)[2]
                prim=stage.GetPrimAtPath(prefix+'/'+suffix)
                axis=prim.GetAttribute('physics:axis').Get();token='rot'+axis
                api=PhysxSchema.PhysxMimicJointAPI(prim,token);assert api
                gear=float(api.GetGearingAttr().Get());offset=float(api.GetOffsetAttr().Get())
                assert abs(gear+r['physx_multiplier'])<1e-6 and abs(offset+np.degrees(r['physx_offset_rad']))<1e-5
                rel=api.GetReferenceJointRel().GetTargets();assert len(rel)==1 and rel[0].name==r['physx_master'] and str(rel[0]).startswith(prefix+'/')
                records.append(dict(slave=r['slave'],master=r['physx_master'],original_URDF_master=r['master'],gearing=gear,offset_degrees=offset,prim=str(prim.GetPath()),reference=str(rel[0])))
            native_mimic[a]=records
        save(out/'native_mimic_identity.json',dict(relations=native_mimic,scope='Full-window coupled grasp development; independent load and task audits required'))
        scene_settings=[]
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                setting=prim.GetAttribute('physxScene:solveArticulationContactLast').Get()
                assert setting is reg['solve_articulation_contact_last'], 'Registered contact solver order not authored'
                external=prim.GetAttribute('physxScene:enableExternalForcesEveryIteration').Get()
                assert external is reg['enable_external_forces_every_iteration'], 'Numerical option not authored natively'
                scene_settings.append(dict(path=str(prim.GetPath()),solve_articulation_contact_last=setting,actual_external_forces_every_iteration=external))
        assert len(scene_settings)==1
        save(out/'native_scene_settings.json',dict(scenes=scene_settings,solver_type=cfg.sim.physx.solver_type,
             enable_external_forces_every_iteration=cfg.sim.physx.enable_external_forces_every_iteration,
             physical_profile='Same v12 solver ordering; feedback task controller14'))
        caps=[]
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                api=PhysxSchema.PhysxSceneAPI(prim)
                values=dict(min_position=int(api.GetMinPositionIterationCountAttr().Get()),
                            max_position=int(api.GetMaxPositionIterationCountAttr().Get()),
                            min_velocity=int(api.GetMinVelocityIterationCountAttr().Get()),
                            max_velocity=int(api.GetMaxVelocityIterationCountAttr().Get()))
                assert values==dict(min_position=cfg.sim.physx.min_position_iteration_count,
                    max_position=cfg.sim.physx.max_position_iteration_count,
                    min_velocity=cfg.sim.physx.min_velocity_iteration_count,
                    max_velocity=cfg.sim.physx.max_velocity_iteration_count)
                assert values['min_position']<=128<=values['max_position']
                assert values['min_velocity']<=128<=values['max_velocity']
                caps.append(dict(path=str(prim.GetPath()),**values))
        assert len(caps)==1
        save(out/'native_scene_iteration_caps_v1.json',dict(scenes=caps,
            metadata_caps_allow128=True,actual_binary_iteration_execution_count_not_proven=True))
        actual_roots=[]
        for prim in omni.usd.get_context().get_stage().Traverse(Usd.TraverseInstanceProxies()):
            if str(prim.GetPath()).startswith('/World/envs/env_0/') and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                api=PhysxSchema.PhysxArticulationAPI(prim)
                pi=int(api.GetSolverPositionIterationCountAttr().Get());vi=int(api.GetSolverVelocityIterationCountAttr().Get())
                assert (pi,vi)==(128,128),'native solver iteration profile mismatch'
                actual_roots.append(dict(path=str(prim.GetPath()),position_iterations=pi,velocity_iterations=vi))
        assert len(actual_roots)==4,'exact four native articulations'
        save(out/'native_solver_roots.json',actual_roots)
        protocol['native_solver_profile']=dict(position_iterations=128,velocity_iterations=128,native_verified=True)
        dt=control_clock(env.step_dt,reg['control_dt_s'],1/60)
        assert abs(cfg.sim.dt-reg['physics_dt_s'])<1e-12 and cfg.decimation==substeps, 'registered physics clock mismatch'
        protocol.update(actual_control_dt_s=dt,actual_physics_dt_s=float(cfg.sim.dt),actual_physics_substeps_per_control=cfg.decimation)
        assert env._sph.n_pairs==9021, f'raw_pairs {env._sph.n_pairs}'
        cam=env._viz_cam;assert env.scene.sensors.pop('viz_cam') is cam;cam.reset()
        protocol['last_setup_stage']='clock_geometry_camera_verified';save(out/'fourhand_protocol.json',protocol)
        with np.load(reg['path']) as z:path={k:z[k] for k in z.files if k.startswith('q_')}
        with np.load(reg['pose_solutions']) as z:poses={k:z[k] for k in z.files}
        with np.load(reg['fixture']) as z:fixture={k:z[k] for k in z.files}
        variants=reg['pose_variants'];initial={};endpoints={};parameters={'scene_origin':array(env.scene.env_origins),'actual_control_dt_s':np.asarray(dt),'actual_physics_dt_s':np.asarray(cfg.sim.dt)}
        for arm in ARMS:
            protocol['last_setup_stage']='initial_state_'+arm;save(out/'fourhand_protocol.json',protocol)
            art=env._arms[arm];view=art.root_physx_view;idx=env._joint_idx[arm]
            q=art.data.joint_pos.clone();v=torch.zeros_like(q)
            hard=view.get_dof_limits().to(env.device)
            hand=torch.ones(q.shape[-1],dtype=torch.bool,device=env.device);hand[idx]=False
            hi=coupled_refs[arm+'_hand_indices'].astype(int)
            q[:,hi]=torch.tensor(coupled_refs[arm+'_hand_open_q'],device=env.device,dtype=q.dtype)
            q[:,idx]=torch.tensor(path['q_'+arm][0],device=env.device)
            root=torch.tensor(fixture[arm+'_native_root_xyzw'][0],device=env.device)[None].clone()
            root[:,:3]+=env.scene.env_origins-torch.tensor(fixture['source_origin'][0],device=env.device)
            art.write_root_pose_to_sim(torch.cat([root[:,:3],root[:,6:7],root[:,3:6]],-1));art.write_root_velocity_to_sim(torch.zeros((1,6),device=env.device))
            art.write_joint_state_to_sim(q,v);art.set_joint_position_target(q)
            # Keep the actual safe initial hand pose as the opening reference.
            art.data.default_joint_pos[:,hand]=q[:,hand]
            initial[arm]=q[:,idx].clone();env._targets[arm][:]=initial[arm]
            start=reg['case']*10+variants[arm]*5
            endpoints[arm]=torch.tensor(poses[arm][start:start+5],device=env.device)
            parameters[arm+'_initial_q']=array(q);parameters[arm+'_hard_limits']=array(hard)
            parameters[arm+'_max_velocity']=array(view.get_dof_max_velocities())
            parameters[arm+'_controlled_indices']=array(idx)
            parameters[arm+'_joint_names']=np.asarray(art.joint_names)
            parameters[arm+'_body_names']=np.asarray(art.body_names)
            parameters[arm+'_stiffness']=array(view.get_dof_stiffnesses());parameters[arm+'_damping']=array(view.get_dof_dampings())
            parameters[arm+'_max_force']=array(view.get_dof_max_forces());parameters[arm+'_soft_joint_limits']=array(art.data.soft_joint_pos_limits)
            parameters[arm+'_coupled_hand_far_q']=coupled_refs[arm+'_hand_far_q']
            names=art.joint_names
            slave_ids=[names.index(r['slave']) for r in coupled_meta['assets'][arm]['relations']]
            assert not array(view.get_dof_stiffnesses())[0,slave_ids].any() and not array(view.get_dof_dampings())[0,slave_ids].any()
            parameters[arm+'_link_masses']=array(view.get_masses())
            parameters[arm+'_link_inertias']=array(view.get_inertias())
            parameters[arm+'_native_material_properties']=array(view.get_material_properties())
            assert ((q>=hard[...,0])&(q<=hard[...,1])).all(), 'initial native hard limits '+arm
        env.scene.write_data_to_sim();env.sim.forward();env.scene.update(0.)
        protocol['last_setup_stage']='initial_robot_states_written';save(out/'fourhand_protocol.json',protocol)
        assert set(env._objects)=={'beam_common'}, 'object identities '+str(set(env._objects))
        obj=env._objects['beam_common'];mass=float(obj.root_physx_view.get_masses()[0])
        assert abs(mass-.45)<=1e-6, f'object mass {mass}'
        effort_projections={};motor_control={}
        for a in ARMS:
            names=parameters[a+'_joint_names'].tolist()
            followers={r['slave'] for r in coupled_meta['assets'][a]['relations']}
            root_names=[n for n in names if n not in followers]
            rootids=[names.index(n) for n in root_names]
            assert not parameters[a+'_stiffness'].any() and not parameters[a+'_damping'].any()
            assert env._arms[a].root_physx_view.shared_metatype.fixed_base
            c=reg['full_motor_effort_controller'][a]
            assert set(c)==set(root_names)
            projection=motor_effort_projection(names,coupled_meta['assets'][a]['relations'])
            mapping=projection[rootids,:].T.copy()
            motor_control[a]=dict(rootids=rootids,mapping=mapping,
                stiffness=np.array([c[n]['nominal_K_Nm_rad'] for n in root_names]),
                damping=np.array([c[n]['nominal_D_Nm_s_rad'] for n in root_names]))
            parameters[a+'_full_motor_velocity_mapping']=mapping
            effort_projections[a]=torch.as_tensor(projection,device=env.device,dtype=torch.float32)
            parameters[a+'_motor_effort_projection']=projection
            parameters[a+'_gravity_compensation_enabled']=np.asarray(not env._arms[a].cfg.spawn.rigid_props.disable_gravity)
        parameters['object_native_material_properties']=array(obj.root_physx_view.get_material_properties())
        object_mu=float(parameters['object_native_material_properties'][...,1].min())
        robot_mu=min(float(parameters[a+'_native_material_properties'][...,1].min()) for a in ARMS)
        assert min(object_mu,robot_mu)>=0
        conservative_mu=min(1.,object_mu)*min(1.,robot_mu)
        parameters['prelift_conservative_dynamic_mu']=np.asarray(conservative_mu)
        np.savez_compressed(out/'native_parameters.npz',**parameters)
        observer=PointContacts(env,out)
        identity=json.loads((out/'native_contact_identity.json').read_text())
        assert sum(len(v['sensors']) for v in identity['views'])==82
        from native_collider_representation_v2 import capture as capture_representation
        capture_representation(env,out,reg['complete_hand_extraction'],identity)
        partners=identity['partners_env0'];object_path='/World/envs/env_0/Obj_beam_common'
        assert object_path in partners
        ov=tensors.create_simulation_view('torch');ov.set_subspace_roots('/')
        cv=ov.create_rigid_contact_view([object_path],filter_patterns=[partners],max_contact_data_count=16384)
        assert cv.check() and list(cv.sensor_paths)==[object_path] and list(cv.filter_paths[0])==partners
        supports=[i for i,p in enumerate(partners) if '/Support_' in p or '/Table' in p]
        assert any('/Support_0' in partners[i] for i in supports) and any('/Support_1' in partners[i] for i in supports)
        arm_filters={a:[i for i,p in enumerate(partners) if '/'+a+'/' in p] for a in ARMS}
        save(out/'native_task_identity.json',dict(object_sensor=object_path,partners=partners,support_partner_indices=supports,
            hand_partner_indices=arm_filters,scene_origin=array(env.scene.env_origins).tolist(),native_mass_kg=mass,native_material=array(obj.root_physx_view.get_material_properties()).tolist(),
            registration_sha256=sha(P/'REGISTRATION_V1.json'),fullSystem0_accepted=False))
        hands=HandDriver(env,dt)
        for a in ARMS:
            hi=hands.ids[a][0];assert hi==coupled_refs[a+'_hand_indices'].tolist()
            assert np.max(abs(array(hands.open_q[a])-coupled_refs[a+'_hand_open_q']))<1e-6
            hands.far_q[a]=torch.as_tensor(coupled_refs[a+'_hand_far_q'],device=env.device,dtype=hands.open_q[a].dtype)
        handposes=json.loads(Path(reg['calibrated_hand_poses']).read_text())
        caps={}
        for a in ARMS:
            frame=handposes[a][str(reg['case'])+':'+str(variants[a])]['notes']['hand_frame']
            caps[a]=(frame['f_squeeze'],frame['f_squeeze'] if frame['f_squeeze_thumb'] is None else frame['f_squeeze_thumb'])
        grip=FeedbackGrip(dt,caps,**reg['feedback_grip_settings'])
        grip_decisions={}
        for a in ARMS:
            frame=handposes[a][str(reg['case'])+':'+str(variants[a])]['notes']['hand_frame']
            hands.command(a,0.,.8)
        pending=deque({a:initial[a].clone() for a in ARMS} for _ in range(6))
        phase='APPROACH';phase_start=0.;last_target={a:initial[a].clone() for a in ARMS}
        motion_from={a:initial[a].clone() for a in ARMS};motion_to={a:endpoints[a][0][None].clone() for a in ARMS}
        duration=4.;feedback=None;unsafe=[];friction_packets=[];normal_packets=[]
        settle=GraspSettleGate(**reg['grasp_settle_gate'])
        settle_status=dict(ready=False,timed_out=False,valid=False,continuous_settled_s=0.)
        settle_failure=None
        approach_settle=GraspSettleGate(**reg['grasp_settle_gate'])
        preshape_settle=GraspSettleGate(**reg['grasp_settle_gate'])
        approach_status=dict(ready=False,timed_out=False,valid=False,continuous_settled_s=0.)
        preshape_status=dict(ready=False,timed_out=False,valid=False,continuous_settled_s=0.)
        retreat_settle=GraspSettleGate(**reg['grasp_settle_gate'])
        retreat_status=dict(ready=False,timed_out=False,valid=False,continuous_settled_s=0.)
        # Prefix and every later frame have a fixed native-state timestamp.
        camera_views=reg['camera_views']

        def capture(step):
            before={a:env._arms[a].root_physx_view.get_dof_positions().clone() for a in ARMS}
            before_obj=obj.data.root_state_w.clone()
            for name,view in camera_views.items():
                eye=torch.tensor([view['eye']],device=env.device);target=torch.tensor([view['target']],device=env.device)
                if 'orientation_wxyz' in view:
                    cam.set_world_poses(positions=eye,orientations=torch.tensor([view['orientation_wxyz']],device=env.device),convention='opengl')
                else:cam.set_world_poses_from_view(eye,target)
                cam.reset()
                env.sim.render();cam.update(env.step_dt,force_recompute=True)
                image=array(cam.data.output['rgb'][0,:,:,:3]).astype(np.uint8)
                rel=f'frames/{name}_{step:05d}.png';image_path=out/rel;image_path.parent.mkdir(exist_ok=True)
                Image.fromarray(image).save(image_path)
                frames.append(dict(view=name,control=step,state_time_s=step*dt,path=rel,sha256=sha(image_path),
                    intrinsic_matrix=array(cam.data.intrinsic_matrices)[0].tolist(),
                    camera_pos_world_m=array(cam.data.pos_w)[0].tolist(),camera_quaternion_opengl_wxyz=array(cam.data.quat_w_opengl)[0].tolist(),
                    image_width=int(image.shape[1]),image_height=int(image.shape[0]),
                    **camera_snapshot(cam,env,obj,view,phase,array)))
            assert torch.equal(before_obj,obj.data.root_state_w)
            assert all(torch.equal(before[a],env._arms[a].root_physx_view.get_dof_positions()) for a in ARMS)

        def flush():
            if not rows:return
            stop=int(rows[-1]['physics_event'])+1;start=stop-len(rows)
            p=out/f'physics_{start:05d}_{stop:05d}.npz'
            fields={k:np.stack([r[k] for r in rows]) for k in rows[0]}
            fields.update({'object_normal_'+k:v for k,v in pack_contact_events(normal_packets).items()})
            offsets=np.concatenate([[0],np.cumsum([len(p['raw_indices']) for p in friction_packets])])
            fields['object_friction_event_offsets']=offsets
            for key in ['raw_indices','partner_indices','forces','points']:
                fields['object_friction_'+key]=np.concatenate([p[key] for p in friction_packets])
            fields['object_friction_counts']=np.stack([p['counts'] for p in friction_packets])
            fields['object_friction_starts']=np.stack([p['starts'] for p in friction_packets])
            np.savez_compressed(p,**fields);chunks.append(dict(path=p.name,start=start,stop=stop,sha256=sha(p)))
            rows.clear();normal_packets.clear();friction_packets.clear()

        capture(0)
        for step in range(reg['maximum_controls']):
            clock=step*dt
            u=min(1.,max(0.,(clock-phase_start)/duration));ease=.5-.5*np.cos(np.pi*u)
            target={a:motion_from[a]+ease*(motion_to[a]-motion_from[a]) for a in ARMS}
            issued={a:last_target[a]+(target[a]-last_target[a]).clamp(-.015,.015) for a in ARMS}
            pending.append(issued);applied=pending.popleft();last_target={a:issued[a].clone() for a in ARMS}
            if phase in ['CLOSE','LIFT','CARRY','PLACE']:
                for ai,a in enumerate(ARMS):
                    view=env._arms[a].root_physx_view;handids=hands.ids[a][0]
                    d=grip.update(a,step,clock,clock,array(view.get_dof_positions())[0,handids],
                        array(view.get_dof_velocities())[0,handids],parameters[a+'_hard_limits'][0,handids],
                        array(hands.is_thumb[a]),hands.cur[a],hands.cur_thumb[a],four[ai])
                    grip_decisions[a]=d
                    hands.command(a,d.finger_fraction,1./reg['feedback_grip_settings']['relax_rate'],d.thumb_fraction)
            previous_hand_targets={a:array(env._arms[a].root_physx_view.get_dof_position_targets())[0,hands.ids[a][0]] for a in ARMS}
            hands.step()
            nominal_hand_targets={};filtered_targets={}
            for a in ARMS:
                fraction=torch.where(hands.is_thumb[a],torch.full_like(hands.open_q[a],hands.cur_thumb[a]),torch.full_like(hands.open_q[a],hands.cur[a]))
                nominal=hands.open_q[a]+fraction*(hands.far_q[a]-hands.open_q[a])
                ids=hands.ids[a][0];nominal_hand_targets[a]=array(nominal)
                filtered=nominal_hand_targets[a]
                if phase in ['CLOSE','LIFT','CARRY','PLACE','RELEASE','RETREAT']:
                    filtered=coupled_filtered_hand_targets([parameters[a+'_joint_names'][i] for i in ids],filtered,
                        previous_hand_targets[a],parameters[a+'_hard_limits'][0,ids],dt,
                        phase in ['CLOSE','LIFT','CARRY','PLACE'],coupled_meta['assets'][a]['relations'],**reg['interior_hand_target_settings'])
                    env._arms[a].set_joint_position_target(torch.as_tensor(filtered,device=env.device,dtype=nominal.dtype)[None],joint_ids=ids)
                filtered_targets[a]=filtered
            for a in ARMS:env._arms[a].set_joint_position_target(applied[a],joint_ids=env._joint_idx[a])
            for sub in range(substeps):
                effort={};gravity_raw={};motor_q_before={};motor_v_before={};motor_pd={};native_body_transforms_before={};full_motor_mass_before={};full_motor_target_before={}  
                for a in ARMS:
                    art=env._arms[a];view=art.root_physx_view
                    g=view.get_gravity_compensation_forces().to(env.device).clone();limit=view.get_dof_max_forces().to(env.device).clone()
                    gravity_raw[a]=g.clone()
                    condensed=g@effort_projections[a].to(dtype=g.dtype).T
                    if art.cfg.spawn.rigid_props.disable_gravity:condensed=torch.zeros_like(g)
                    control=motor_control[a];rootids=control['rootids']
                    motor_q_before[a]=array(view.get_dof_positions())
                    motor_v_before[a]=array(view.get_dof_velocities())
                    native_body_transforms_before[a]=array(view.get_link_transforms())
                    full_motor_mass_before[a]=array(view.get_generalized_mass_matrices())
                    full_motor_target_before[a]=array(art.data.joint_pos_target)
                    native_mass=full_motor_mass_before[a][0]
                    assert native_mass.shape==(len(parameters[a+'_joint_names']),)*2
                    target=full_motor_target_before[a][0,rootids]
                    solution=coupled_backward_euler_torque(native_mass,control['mapping'],target,
                        motor_q_before[a][0,rootids],motor_v_before[a][0,rootids],
                        control['stiffness'],control['damping'],cfg.sim.dt,
                        reg['full_motor_mass_relative_asymmetry_tolerance'])
                    tau=solution.torque
                    motor_pd[a]=torch.zeros_like(g)
                    motor_pd[a][:,rootids]=torch.as_tensor(tau,device=env.device,dtype=g.dtype)[None]
                    effort[a]=(condensed+motor_pd[a]).clamp(-limit,limit)
                    art.set_joint_effort_target(effort[a])
                env.scene.write_data_to_sim()
                for a in ARMS:np.testing.assert_array_equal(array(env._arms[a].root_physx_view.get_dof_position_targets()),full_motor_target_before[a])
                for a in ARMS:assert torch.equal(env._arms[a].root_physx_view.get_dof_actuation_forces().to(env.device),effort[a])
                env.sim.step(render=False);env.scene.update(cfg.sim.dt)
                event=substeps*step+sub;now=(event+1)*cfg.sim.dt
                observer.capture(step,sub,cfg.sim.dt)
                raw=[v.detach().clone().cpu().numpy().copy() for v in cv.get_contact_data(cfg.sim.dt)]
                normal=pack_contact_event(*raw,capacity=16384);friction=pack_friction(cv.get_friction_data(cfg.sim.dt),16384)
                normals=normal['partner_abs_normal_sum_N'][0]
                four=[float(normals[arm_filters[a]].sum()) for a in ARMS]
                support=float(normals[supports].sum())
                state=array(obj.data.root_state_w)[0];local=state[:3]-array(env.scene.env_origins)[0]
                opening=[float(abs(env._arms[a].data.joint_pos[:,hands.ids[a][1]]-hands.open_q[a]).max()) for a in ARMS]
                geometry=array(env.compute_dist().dists)
                unsafe_peak=0.;normal_by_hand=[]
                for view in identity['views']:
                    a=view['arm'];matrix=observer.last_packets[a]['partner_abs_normal_sum_N'];allowed=np.zeros(matrix.shape,dtype=bool)
                    if phase in ['CLOSE','LIFT','CARRY','PLACE','RELEASE']:
                        for si,sensor in enumerate(view['sensors']):
                            body=sensor.rsplit('/',1)[-1]
                            ishand='/f2_hand/' in sensor or body=='fr3_link8' or body=='wrist_3_link' or any(n in body for n in ['thumb','index','middle','ring','little','pinky'])
                            if ishand:allowed[si,view['filters'][si].index(object_path)]=True
                    forbidden=np.where(allowed,0,matrix);peak=float(forbidden.max());unsafe_peak=max(unsafe_peak,peak)
                    if peak>.1 and not unsafe:
                        si,pi=np.unravel_index(forbidden.argmax(),forbidden.shape)
                        unsafe.append(dict(event=event,type='UNINTENDED_NATIVE_CONTACT',sensor=view['sensors'][si],partner=view['filters'][si][pi],normal_N=peak))
                hand_margin=float('inf');hand_speed=0.
                for a in ARMS:
                    ids=hands.ids[a][0];view=env._arms[a].root_physx_view
                    handq=array(view.get_dof_positions())[0,ids];handv=array(view.get_dof_velocities())[0,ids]
                    limits=parameters[a+'_hard_limits'][0,ids]
                    hand_margin=min(hand_margin,float(np.minimum(handq-limits[:,0],limits[:,1]-handq).min()))
                    hand_speed=max(hand_speed,float(abs(handv).max()))
                capacity_proxy=conservative_mu*sum(four)
                weight_reserve=reg['prelift_load_reserve_fraction']*mass*9.81
                no_guard=bool(grip_decisions) and not any(d.thumb_guard or d.finger_guard for d in grip_decisions.values())
                prelift_ready=capacity_proxy>=weight_reserve and hand_margin>=.02 and hand_speed<=.05 and no_guard
                hard_peak=0.;speed_peak=0.;data=dict(physics_event=np.asarray(event),state_time_s=np.asarray(now),phase=np.asarray(phase),object_state=state,
                    object_size_m=np.asarray([.04,.7,.08]),four_hand_normal_N=np.asarray(four),support_normal_N=np.asarray(support),hand_open_error_rad=np.asarray(opening),
                    feedback_grip_mode=np.asarray([grip_decisions[a].mode if a in grip_decisions else 'INACTIVE' for a in ARMS]),
                    feedback_grip_finger_fraction=np.asarray([hands.cur[a] for a in ARMS]),
                    feedback_grip_thumb_fraction=np.asarray([hands.cur_thumb[a] for a in ARMS]),
                    feedback_grip_thumb_guard=np.asarray([grip_decisions[a].thumb_guard if a in grip_decisions else False for a in ARMS]),
                    feedback_grip_finger_guard=np.asarray([grip_decisions[a].finger_guard if a in grip_decisions else False for a in ARMS]),
                    all9021_raw_geometry=geometry,unintended_peak_normal_N=np.asarray(unsafe_peak),
                    prelift_friction_capacity_proxy_N=np.asarray(capacity_proxy),prelift_weight_reserve_N=np.asarray(weight_reserve),
                    prelift_native_hand_min_margin_rad=np.asarray(hand_margin),prelift_max_hand_speed_rad_s=np.asarray(hand_speed),
                    prelift_ready=np.asarray(prelift_ready))
                for a in ARMS:
                    view=env._arms[a].root_physx_view;q=array(view.get_dof_positions());qd=array(view.get_dof_velocities())
                    hard=parameters[a+'_hard_limits'];vmax=parameters[a+'_max_velocity']
                    hard_peak=max(hard_peak,float(np.maximum(np.maximum(hard[...,0]-q,q-hard[...,1]),0).max()))
                    speed_peak=max(speed_peak,float(np.maximum(abs(qd)-vmax,0).max()))
                    data[a+'_native_q']=q;data[a+'_native_qd']=qd;data[a+'_issued_target']=array(issued[a]);data[a+'_applied_target']=array(applied[a])
                    data[a+'_actual_full_position_target']=array(view.get_dof_position_targets());data[a+'_extra_effort']=array(effort[a])
                    data[a+'_gravity_compensation_raw']=array(gravity_raw[a])
                    data[a+'_motor_control_q_before']=motor_q_before[a];data[a+'_motor_control_qd_before']=motor_v_before[a]
                    data[a+'_software_full_motor_PD_preclip']=array(motor_pd[a])
                    data[a+'_native_generalized_mass_before']=full_motor_mass_before[a]
                    data[a+'_full_motor_actual_target_before']=full_motor_target_before[a]
                    data[a+'_native_body_transforms_before_xyzw']=native_body_transforms_before[a]
                    data[a+'_native_body_transforms_after_xyzw']=array(view.get_link_transforms())
                    data[a+'_native_body_com_velocity_world']=array(view.get_link_velocities())
                    data[a+'_native_external_actuation_effort']=array(view.get_dof_actuation_forces())
                    data[a+'_hand_nominal_position_target']=nominal_hand_targets[a]
                    data[a+'_hand_filtered_position_target']=filtered_targets[a]
                    data[a+'_native_projected_joint_force']=array(view.get_dof_projected_joint_forces())
                    data[a+'_native_incoming_joint_force']=array(view.get_link_incoming_joint_force())
                    data[a+'_body_position_local']=array(env._arms[a].data.body_pos_w-env.scene.env_origins[:,None]);data[a+'_body_quaternion_wxyz']=array(env._arms[a].data.body_quat_w)
                    names=parameters[a+'_joint_names'].tolist();mimic_residual=[];mimic_v=[]
                    for r in coupled_meta['assets'][a]['relations']:
                        si=names.index(r['slave']);mi=names.index(r['master'])
                        mimic_residual.append(float(q[0,si]-r['urdf_multiplier']*q[0,mi]-r['urdf_offset_rad']))
                        mimic_v.append(float(qd[0,si]-r['urdf_multiplier']*qd[0,mi]))
                    data[a+'_mimic_position_residual_rad']=np.asarray(mimic_residual)
                    data[a+'_mimic_velocity_residual_rad_s']=np.asarray(mimic_v)
                data.update(all74_hard_violation_rad=np.asarray(hard_peak),all74_speed_exceedance_rad_s=np.asarray(speed_peak))
                if not unsafe and (geometry.min()<0 or hard_peak>1e-5 or speed_peak>1e-5 or max(four)>30):
                    unsafe.append(dict(event=event,type='GEOMETRY_JOINT_OR_PAYLOAD_FORCE_GATE',minimum_geometry_m=float(geometry.min()),hard_violation_rad=hard_peak,speed_exceedance_rad_s=speed_peak,peak_hand_payload_normal_N=max(four)))
                q_error=max(float(abs(env._arms[a].root_physx_view.get_dof_positions().to(env.device)[:,env._joint_idx[a]]-endpoints[a][1][None]).max()) for a in ARMS)
                q_speed=max(float(abs(env._arms[a].root_physx_view.get_dof_velocities().to(env.device)[:,env._joint_idx[a]]).max()) for a in ARMS)
                queue_error=max(float(abs(applied[a]-endpoints[a][1][None]).max()) for a in ARMS)
                pre_q_error=max(float(abs(env._arms[a].root_physx_view.get_dof_positions().to(env.device)[:,env._joint_idx[a]]-endpoints[a][0][None]).max()) for a in ARMS)
                pre_queue_error=max(float(abs(applied[a]-endpoints[a][0][None]).max()) for a in ARMS)
                hand_q_error=max(float(abs(env._arms[a].root_physx_view.get_dof_positions().to(env.device)[:,hands.ids[a][1]]-env._arms[a].root_physx_view.get_dof_position_targets().to(env.device)[:,hands.ids[a][1]]).max()) for a in ARMS)
                if phase=='APPROACH':
                    approach_status=approach_settle.observe(now,now-phase_start,u>=1,pre_q_error,q_speed,pre_queue_error,not bool(unsafe),True)
                elif phase=='PRESHAPE':
                    preshape_nominal=u>=1 and all(abs(hands.cur[a]-hands.goal[a])<=1e-6 for a in ARMS) and hand_q_error<=reg['preshape_hand_error_max_rad']
                    preshape_status=preshape_settle.observe(now,now-phase_start,preshape_nominal,pre_q_error,q_speed,pre_queue_error,not bool(unsafe),True)
                data.update(pregrasp_position_error_rad=np.asarray(pre_q_error),pregrasp_applied_endpoint_error_rad=np.asarray(pre_queue_error),
                    preshape_hand_target_error_rad=np.asarray(hand_q_error),pregrasp_settle_dwell_s=np.asarray(approach_status['continuous_settled_s']),
                    preshape_settle_dwell_s=np.asarray(preshape_status['continuous_settled_s']),preshape_ready=np.asarray(preshape_status['ready']))
                if phase=='RETREAT':
                    nominal_retreat=u>=1 and max(opening)<=.05 and max(four)<=.1
                    retreat_status=retreat_settle.observe(now,now-phase_start,nominal_retreat,pre_q_error,q_speed,pre_queue_error,not bool(unsafe),True)
                data.update(retreat_endpoint_error_rad=np.asarray(pre_q_error),retreat_applied_endpoint_error_rad=np.asarray(pre_queue_error),
                    retreat_settle_dwell_s=np.asarray(retreat_status['continuous_settled_s']),retreat_ready=np.asarray(retreat_status['ready']))
                if phase=='GRASP':
                    settle_status=settle.observe(now,now-phase_start,u>=1,q_error,q_speed,queue_error,not bool(unsafe),True)
                data.update(grasp_position_error_rad=np.asarray(q_error),grasp_arm_speed_rad_s=np.asarray(q_speed),
                    grasp_applied_endpoint_error_rad=np.asarray(queue_error),grasp_settle_dwell_s=np.asarray(settle_status['continuous_settled_s']),
                    grasp_close_ready=np.asarray(settle_status['ready']),grasp_settle_timed_out=np.asarray(settle_status['timed_out']))
                if feedback is not None and not unsafe:
                    intent=feedback.update(Observation('beam_common',event,now,now,True,tuple(local),tuple(state[3:7]),tuple(state[7:10]),tuple(state[10:13]),(.04,.7,.08),tuple(four),support,tuple(opening),bool(prelift_ready)))
                    data['feedback_phase']=np.asarray(intent.phase)
                else:data['feedback_phase']=np.asarray('PRE_GRASP' if feedback is None else 'PHYSICAL_GATE_FAILED')
                rows.append(data);normal_packets.append(normal);friction_packets.append(friction)
                if len(rows)==32:flush()
            completed=step+1
            next_phase=phase
            if phase=='APPROACH':
                if approach_status['timed_out']:next_phase='FAILED';settle_failure='PREGRASP_SETTLE_TIMEOUT'
                elif approach_status['ready']:next_phase='PRESHAPE'
            elif phase=='PRESHAPE':
                if preshape_status['timed_out']:next_phase='FAILED';settle_failure='PRESHAPE_SETTLE_TIMEOUT'
                elif preshape_status['ready']:next_phase='GRASP'
            elif phase=='GRASP':
                if settle_status['timed_out']:
                    next_phase='FAILED';settle_failure='GRASP_SETTLE_TIMEOUT'
                elif settle_status['ready']:next_phase='CLOSE'
            elif feedback is not None:
                next_phase=feedback.phase
                if phase=='RETREAT':
                    if retreat_status['timed_out']:next_phase='FAILED';settle_failure='RETREAT_ENDPOINT_SETTLE_TIMEOUT'
                    elif next_phase=='DONE' and not retreat_status['ready']:next_phase='RETREAT'
            if next_phase!=phase:
                history.append(dict(from_phase=phase,to_phase=next_phase,time_s=completed*dt))
                phase=next_phase;phase_start=completed*dt;motion_from={a:last_target[a].clone() for a in ARMS}
                index={'PRESHAPE':0,'GRASP':1,'CLOSE':1,'LIFT':2,'CARRY':3,'PLACE':4,'RELEASE':4,'RETREAT':0}.get(phase)
                motion_to={a:(endpoints[a][index][None].clone() if index is not None else last_target[a].clone()) for a in ARMS}
                duration={'PRESHAPE':1.,'GRASP':1.5,'CLOSE':1.,'LIFT':1.5,'CARRY':2.,'PLACE':2.,'RELEASE':1.,'RETREAT':2.}.get(phase,1.)
                if phase=='PRESHAPE':
                    for a in ARMS:
                        frame=handposes[a][str(reg['case'])+':'+str(variants[a])]['notes']['hand_frame']
                        hands.command(a,frame['f_preshape_finger'],1.,frame['f_preshape_thumb'])
                if phase=='CLOSE':
                    feedback=FourHandPayloadPhases('beam_common',(.20,0,1.02),(.20,.04),(.04,.7,.08),mass)
                    for a in ARMS:
                        frame=handposes[a][str(reg['case'])+':'+str(variants[a])]['notes']['hand_frame']
                        hands.command(a,frame['f_squeeze'],reg['close_hand_ramp_s'],frame['f_squeeze_thumb'])
                if phase in ['RELEASE','RETREAT','DONE']:
                    for a in ARMS:hands.command(a,0.,.8)
            if completed%4==0 or unsafe or phase in ['DONE','FAILED']:capture(completed)
            if completed%60==0:print('FOUR_HAND_NATIVE',completed,phase,'z',float(local[2]),'normal',four,'support',support,flush=True)
            if unsafe or phase in ['DONE','FAILED']:break
        flush();observer.close(completed)
        protocol.update(status='complete',task_status='PHYSICAL_GATE_FAILED' if unsafe else phase,
            completed_controls=completed,physics_events=substeps*completed,unsafe_events=unsafe,
            phase_transitions=history,feedback_failure=settle_failure or (None if feedback is None else feedback.failure),
            grasp_settle_gate=reg['grasp_settle_gate'],last_grasp_settle_status=settle_status,
            pregrasp_settle_status=approach_status,preshape_settle_status=preshape_status,retreat_settle_status=retreat_status,retreat_endpoint_required=True,preshape_hand_error_max_rad=reg['preshape_hand_error_max_rad'],
            four_hand_task_development_pass=not unsafe and phase=='DONE',
            continuous_native_frames=True,video_rate_Hz=1/(4*dt),frames=len(frames),new_native_video=False,
            all74_all9021_and_all82_point_contact_observed_each_microstep=True,
            object_state_input_kind='Fresh native ground truth with full pose, velocity and registered size; vision model not tested',
            learned_safety_or_formal_holdout_tested=False,fullSystem0_accepted=False)
        protocol.update(task_status='COUPLED_MOTOR_CAMERA_LOAD_PREFIX' if not unsafe else protocol['task_status'],
            four_hand_task_development_pass=False,diagnostic_only=True,coupling_loaded_validated=False,canonical_asset_profile_validated=False)
        save(out/'physics_chunks.json',chunks);save(out/'camera_frames.json',frames)
    except BaseException as e:
        protocol.update(status='failed',error=repr(e),traceback=traceback.format_exc(),completed_controls=completed,fullSystem0_accepted=False)
        print(protocol['traceback'],flush=True)
        raise
    finally:
        save(out/'fourhand_protocol.json',protocol)
        if env is not None:env.close()
        # RGB/NPZ writes above are synchronous; there is no Replicator writer.
        # Detach camera resources and retain full framework cleanup.
        cam=None;observer=None;env=None;gc.collect()
        app.close(wait_for_replicator=False,skip_cleanup=False)


if __name__=='__main__':main()
