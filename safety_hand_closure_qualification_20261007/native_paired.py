"""Registered opening-pose counterfactual with native hand pair contacts. No task trial."""
import argparse, json, hashlib, math
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument('--registration', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
r = json.loads(a.registration.read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
for path, digest in r['source_sha256'].items():
    assert sha(path) == digest, path
a.out.mkdir(parents=True, exist_ok=False)
a.enable_cameras = True
app = AppLauncher(a).app
import torch, numpy as np
import omni.usd, omni.physics.tensors as tensors
import isaaclab.sim as sim_utils
from pxr import Usd, UsdPhysics, UsdGeom, PhysxSchema
from PIL import Image
from contact_points import scalar_contacts
from safeduo.envs.duo_env import DuoEnv, make_duo_env_cfg
from safeduo.delta.skill_replay import SkillTrajectory
from safeduo.safety.types import ARM_KEYS
from paired_paths import JointPaths

def save(name, value):
    (a.out/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

class ProbeEnv(DuoEnv):
    def _setup_scene(self):
        super()._setup_scene()
        stage = omni.usd.get_context().get_stage()
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim)

def as_np(value): return value.detach().clone().cpu().numpy()

def main():
    torch.set_num_threads(1)
    cases = r['cases']; n = len(cases)
    legacy_metadata = json.loads(Path(r['legacy_metadata']).read_text())
    cfg = make_duo_env_cfg(num_envs=n, device=a.device, yaml_name=r['env_yaml'], coordinator=True)
    cfg.enable_viz_camera = True; cfg.scene.env_spacing = 5.0
    cfg.seed = r['seed']; cfg.episode_length_s = 30.
    cfg.scene_dressing['table_visible'] = True
    cfg.scene_dressing['ground_visible'] = True
    cfg.scene_dressing['static_props'] = [v for v in cfg.scene_dressing.get('static_props', []) if v.get('collision', True)]
    # Diagnostic counterfactual only, in an independently registered invocation.
    if r['disable_self_collision']:
        for ac in cfg.robot_cfgs.values():
            ac.spawn.articulation_props.enabled_self_collisions = False
    for arm, ac in cfg.robot_cfgs.items():
        if arm.startswith('U'):
            ac.init_state.joint_pos = {**ac.init_state.joint_pos, '.*thumb_1_joint': r['constructor_thumb_rad']}
    print('CONSTRUCT_STATIC_DIAGNOSTIC', n, 'default_thumb', r['constructor_thumb_rad'], flush=True)
    env = ProbeEnv(cfg)
    def init_snapshot(label):
        return dict(label=label, arms={arm:dict(q=as_np(env._arms[arm].root_physx_view.get_dof_positions()).tolist(), qd=as_np(env._arms[arm].root_physx_view.get_dof_velocities()).tolist(), target=as_np(env._arms[arm].root_physx_view.get_dof_position_targets()).tolist()) for arm in ARM_KEYS}, objects={obj:dict(pose=as_np(env._objects[obj].root_physx_view.get_transforms()).tolist(),velocity=as_np(env._objects[obj].root_physx_view.get_velocities()).tolist()) for obj in ['beam700','beam300']})
    init_snapshots = [init_snapshot('after_constructor_before_env_reset')]
    cam = env._viz_cam; env.scene.sensors.pop('viz_cam'); cam.reset()
    env.reset()
    init_snapshots.append(init_snapshot('after_env_reset_before_diagnostic_writes'))
    dt = env.cfg.sim.dt
    assert abs(dt-r['physics_dt_s']) < 1e-9
    traj = SkillTrajectory.load(r['trajectory'])
    arms = list(ARM_KEYS); objects = ['beam700', 'beam300']
    hands = JointPaths(env, r)
    arm_targets = {}
    metadata = {}
    for arm in arms:
        art = env._arms[arm]; ids = env._joint_idx[arm]
        q = art.data.default_joint_pos.clone()
        for e, case in enumerate(cases):
            index = int(round(case['reference_time_s']/traj.dt))
            q[e, ids] = torch.tensor(traj.q[arm][index], device=env.device)
            for jid in hands.ids[arm][0]:
                q[e,jid] = 0.
                if arm.startswith('U') and 'thumb_1_joint' in art.joint_names[jid]: q[e,jid] = case['post_write_thumb_rad']
        art.write_joint_state_to_sim(q, torch.zeros_like(q))
        art.set_joint_position_target(q)
        arm_targets[arm] = q[:, ids].clone()
        hids = hands.ids[arm][0]
        native = art.root_physx_view
        metadata[arm] = dict(joint_names=art.joint_names, body_names=art.body_names, hand_ids=hids, arm_ids=ids.cpu().tolist(),
            hand_default_rad=as_np(art.data.default_joint_pos[:,hids]).tolist(),
            original_hand_open_rad=as_np(hands.original_open[arm]).tolist(),
            proposed_hand_open_rad=as_np(hands.open_q[arm]).tolist(), far_q_rad=as_np(hands.far_q[arm]).tolist(),
            native_limits_rad=as_np(native.get_dof_limits()).tolist(),
            soft_limits_rad=as_np(art.data.soft_joint_pos_limits).tolist(),
            native_stiffness=as_np(native.get_dof_stiffnesses()).tolist(),
            native_damping=as_np(native.get_dof_dampings()).tolist(),
            native_max_force=as_np(native.get_dof_max_forces()).tolist(),
            arm_reference_rad=as_np(arm_targets[arm]).tolist(), usd_path=cfg.robot_cfgs[arm].spawn.usd_path)
    for oi, obj in enumerate(objects):
        body = env._objects[obj]; pose = body.data.root_state_w[:,:7].clone()
        for e, case in enumerate(cases):
            center = case['centers_m'][oi]
            pose[e,:3] = torch.tensor(center, device=env.device)+env.scene.env_origins[e]
            yaw = r['object_yaws_rad'][oi]
            pose[e,3:] = torch.tensor([math.cos(yaw/2),0,0,math.sin(yaw/2)],device=env.device)
        body.write_root_pose_to_sim(pose)
        body.write_root_velocity_to_sim(torch.zeros(n,6,device=env.device))
    # All pose/velocity writes above are initialization. None in the loop below.
    init_snapshots.append(init_snapshot('after_diagnostic_writes_before_loop'))
    save('initial_states.json', init_snapshots)
    stage = omni.usd.get_context().get_stage()
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    inventory = []; rigid_paths = []
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if not (path.startswith('/World/envs/env_0/') or path.startswith('/World/ground')): continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI): rigid_paths.append(path)
        if not prim.HasAPI(UsdPhysics.CollisionAPI): continue
        enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        ancestor = prim
        while ancestor and not ancestor.HasAPI(UsdPhysics.RigidBodyAPI): ancestor = ancestor.GetParent()
        rigid_owner = str(ancestor.GetPath()) if ancestor else None
        bound = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
        inventory.append(dict(path=path, collision_enabled=enabled, instance_proxy=prim.IsInstanceProxy(),
            rigid_owner=rigid_owner, type=prim.GetTypeName(),
            bbox_min_m=list(bound.GetMin()),bbox_max_m=list(bound.GetMax())))
    # Rigid owners cover all their colliders; static colliders are added individually.
    partners = sorted(set(rigid_paths + [x['path'] for x in inventory if x['collision_enabled'] and not x['rigid_owner']]))
    save('native_metadata.json', dict(cases=cases, arms=metadata, collision_inventory=inventory,
        partners_env0=partners, origins=as_np(env.scene.env_origins).tolist(),
        masses_kg={o:as_np(env._objects[o].root_physx_view.get_masses()).tolist() for o in objects},
        physics_dt_s=dt, no_policy_or_system0_step=True,
        control='constant arm references; registered per-slot joint paths; manual physics steps',
        constructor_thumb_rad=r['constructor_thumb_rad'], constructor_internal_contact_forces_measured=False,
        registration_sha256=sha(a.registration), disable_self_collision=r['disable_self_collision']))
    sv = tensors.create_simulation_view('torch'); sv.set_subspace_roots('/')
    obj_views = []; hand_views = []; identities = []
    for e in range(n):
        filters = [x.replace('/env_0/',f'/env_{e}/') for x in partners]
        for obj in objects:
            sensor = f'/World/envs/env_{e}/Obj_{obj}'
            # Exact paths, not overlapping subtree wildcards.
            fp = [x for x in filters if x != sensor]
            cv = sv.create_rigid_contact_view(sensor,filter_patterns=fp,max_contact_data_count=r['contact_capacity'])
            actual = list(cv.filter_paths)
            if len(actual)==1 and isinstance(actual[0],(list,tuple)): actual=list(actual[0])
            identities.append(dict(env=e,object=obj,sensors=list(cv.sensor_paths),filters=actual,requested=fp))
            assert cv.check() and cv.sensor_count==1 and actual==fp, ('contact_identity',e,obj,actual,fp)
            obj_views.append((e,obj,cv,fp))
        for arm in arms:
            paths = [x for x in filters if f'/env_{e}/{arm}/' in x and ('f2_hand/' in x or any(k in x.rsplit('/',1)[-1] for k in ('thumb','index','middle','ring','little','pinky','wrist_3_link')))]
            cv = sv.create_rigid_contact_view(paths,filter_patterns=[filters for _ in paths],max_contact_data_count=r['contact_capacity'])
            assert cv.check() and set(cv.sensor_paths)==set(paths)
            assert all(list(fp)==filters for fp in cv.filter_paths)
            hand_views.append((e,arm,cv,list(cv.sensor_paths)))
    save('contact_identities.json',dict(objects=identities,hands=[dict(env=e,arm=arm,sensors=paths,filters=list(cv.filter_paths)) for e,arm,cv,paths in hand_views]))
    buffers = {}; chunks = []; images = []; checks = []; cap = r['contact_capacity']
    max_normal_count=0; max_friction_count=0
    renderables = [UsdGeom.Imageable(stage.GetPrimAtPath(f'/World/envs/env_{e}')) for e in range(n)]
    def snapshot():
        return {**{arm+':q':as_np(env._arms[arm].root_physx_view.get_dof_positions()) for arm in arms},
            **{arm+':qd':as_np(env._arms[arm].root_physx_view.get_dof_velocities()) for arm in arms},
            **{obj+':pose':as_np(env._objects[obj].root_physx_view.get_transforms()) for obj in objects},
            **{obj+':vel':as_np(env._objects[obj].root_physx_view.get_velocities()) for obj in objects}}
    detailed=[]
    for step in range(r['steps']):
        clock = step*dt
        # Fresh native pre-step state; contact views contain the previous completed physics step.
        for e,arm,cv,paths in hand_views:
            if arm.startswith('U'):
                art=env._arms[arm];ids=hands.ids[arm][0]
                fp=[list(x) for x in cv.filter_paths]
                internal=np.asarray([[x in paths for x in fs] for fs in fp])
                scalar,_,_,_,_=scalar_contacts(cv,dt,r['contact_capacity'])
                force=np.where(internal,scalar,0).reshape(-1)
                hands.native_states[e,arm]=dict(q=as_np(art.root_physx_view.get_dof_positions()[e,ids]),qd=as_np(art.root_physx_view.get_dof_velocities()[e,ids]),self_normal_n=force,step=step-1)
        hands.step(step)
        for arm in arms:
            env._arms[arm].set_joint_position_target(arm_targets[arm], joint_ids=env._joint_idx[arm].tolist())
        env.scene.write_data_to_sim(); env.sim.step(render=False); env.scene.update(dt)
        data=dict(step=np.asarray(step),time_s=np.asarray((step+1)*dt), cycle=np.asarray(step//r['cycle_steps']), profile_index=np.asarray([(step//r['cycle_steps'])*r['profiles_per_cycle']+case['profile_slot'] for case in cases]))
        for arm in arms:
            art=env._arms[arm]; native=art.root_physx_view
            data[arm+':q']=as_np(native.get_dof_positions());data[arm+':qd']=as_np(native.get_dof_velocities())
            data[arm+':target']=as_np(native.get_dof_position_targets())
            data[arm+':cache_q']=as_np(art.data.joint_pos)
            data[arm+':goal']=as_np(hands.goal[arm])
            data[arm+':link_pose']=as_np(native.get_link_transforms())
        for e,arm,cv,paths in hand_views:
            data[f'e{e}:{arm}:hand_net']=as_np(cv.get_net_contact_forces(dt))
            data[f'e{e}:{arm}:hand_partner_normal']=as_np(cv.get_contact_force_matrix(dt))
            # Copy the used count tensor before another accessor can reuse it.
            scalar,normal_count,point_max,point_force,point_start=scalar_contacts(cv,dt,cap)
            max_normal_count=max(max_normal_count,int(normal_count.sum()))
            data[f'e{e}:{arm}:hand_normal_count']=normal_count
            data[f'e{e}:{arm}:hand_partner_scalar_normal']=scalar
            data[f'e{e}:{arm}:hand_point_normal_max']=np.asarray(point_max)
            if arm.startswith('U'):
                data[f'e{e}:{arm}:hand_point_force']=point_force
                data[f'e{e}:{arm}:hand_point_start']=point_start
        for e,obj,cv,fp in obj_views:
            key=f'e{e}:{obj}:'
            data[key+'pose']=as_np(env._objects[obj].root_physx_view.get_transforms()[e])
            data[key+'velocity']=as_np(env._objects[obj].root_physx_view.get_velocities()[e])
            data[key+'normal']=as_np(cv.get_contact_force_matrix(dt)).reshape(len(fp),3)
            data[key+'net']=as_np(cv.get_net_contact_forces(dt)).reshape(3)
            # Copy ALL outputs: the contact and friction accessors reuse count/start buffers.
            norm=[as_np(v) for v in cv.get_contact_data(dt)]
            fric=[as_np(v) for v in cv.get_friction_data(dt)]
            for v in (norm,fric):
                count,start=v[-2:]
                assert int(count.sum())<cap, ('capacity',e,obj,step)
                assert np.all((count==0)|((start>=0)&(start+count<=cap))), ('range',e,obj,step)
            max_normal_count=max(max_normal_count,int(norm[-2].sum()))
            max_friction_count=max(max_friction_count,int(fric[-2].sum()))
            data[key+'normal_count']=norm[-2].reshape(-1)
            data[key+'friction_count']=fric[-2].reshape(-1)
            fv=np.zeros((len(fp),3),np.float32); nv=fv.copy()
            for j in range(len(fp)):
                count=int(fric[-2][0,j]); start=int(fric[-1][0,j])
                if count: fv[j]=fric[0][start:start+count].sum(0)
                count=int(norm[-2][0,j]); start=int(norm[-1][0,j])
                if count: nv[j]=(norm[0][start:start+count]*norm[2][start:start+count]).sum(0)
            data[key+'friction']=fv;data[key+'normal_reconstructed']=nv
            # Sparse original points retained every 120 physics steps for auditing.
            if step%120==0 and cases[e]['kind']=='support':
                def sparse(v):
                    found=[]
                    for j in range(len(fp)):
                        count=int(v[-2][0,j]);start=int(v[-1][0,j])
                        if count: found.append(dict(partner=fp[j],values=[z[start:start+count].tolist() for z in v[:-2]]))
                    return found
                detailed.append(dict(step=step,env=e,object=obj,normal=sparse(norm),friction=sparse(fric)))
        for key,val in data.items():
            assert np.isfinite(val).all(),(step,key)
            buffers.setdefault(key,[]).append(val.copy())
        if step in r['capture_steps']:
            before=snapshot()
            for e in r.get('capture_envs_by_step', {}).get(str(step), r['capture_envs']):
                for i,v in enumerate(renderables):v.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited if i==e else UsdGeom.Tokens.invisible)
                for name,view in r['views'].items():
                    if name not in r.get('capture_views_by_step', {}).get(str(step), list(r['views'])): continue
                    origin=env.scene.env_origins[e]
                    if view.get('hand_focus'):
                        arm=view['hand_focus'];art=env._arms[arm]
                        hand_paths=next(paths for en,ar,cv,paths in hand_views if en==e and ar==arm)
                        bids=[art.body_names.index(x.rsplit('/',1)[-1]) for x in hand_paths]
                        center=art.root_physx_view.get_link_transforms()[e,bids,:3].mean(0)
                        camera_eye=center+torch.tensor(view['eye_offset_m'],device=env.device)
                        camera_target=center
                    else:
                        camera_eye=origin+torch.tensor(view['eye'],device=env.device)
                        camera_target=origin+torch.tensor(view['target'],device=env.device)
                    cam.set_world_poses_from_view(camera_eye[None],camera_target[None])
                    env.sim.render();cam.update(dt,force_recompute=True)
                    rgb=cam.data.output['rgb'][0].cpu().numpy()[:,:,:3]
                    if rgb.dtype!=np.uint8:rgb=(rgb*255).clip(0,255).astype(np.uint8)
                    rel=f'images/e{e}_s{step:04d}_{name}.png';dest=a.out/rel;dest.parent.mkdir(exist_ok=True)
                    Image.fromarray(rgb).save(dest)
                    images.append(dict(file=rel,sha256=sha(dest),env=e,step=step,state_time_s=(step+1)*dt,view=name,camera_eye_world_m=as_np(camera_eye).tolist(),camera_target_world_m=as_np(camera_target).tolist()))
            after=snapshot();diff={key:float(np.max(np.abs(before[key]-after[key]))) for key in before}
            assert all(v==0 for v in diff.values()),('render_motion',step,diff)
            checks.append(dict(step=step,max_abs_diff=diff))
            for v in renderables:v.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
        if (step+1)%120==0 or step==r['steps']-1:
            first=int(buffers['step'][0]);dest=a.out/f'dense_{first:04d}_{step:04d}.npz'
            np.savez_compressed(dest,**{key:np.stack(val) for key,val in buffers.items()});buffers.clear()
            chunks.append(dict(file=dest.name,sha256=sha(dest),first_step=first,last_step=step))
            save('progress.json',dict(steps=step+1,chunks=chunks,images=len(images)))
            print('STATIC_DENSE',step+1,'/',r['steps'],'images',len(images),flush=True)
    save('sparse_contact_points.json',detailed)
    save('admission_records.json',hands.records)
    save('recording_receipt.json',dict(status='PASS_COMPLETE_STATIC_DIAGNOSTIC',steps=r['steps'],cases=n,
        new_task_trials=0, new_final_trials=0, chunks=chunks,images=images,render_checks=checks,
        registration_sha256=sha(a.registration),max_normal_count=max_normal_count,max_friction_count=max_friction_count,
        contact_capacity=cap,metadata_sha256=sha(a.out/'native_metadata.json'),identities_sha256=sha(a.out/'contact_identities.json'),
        sparse_contact_points_sha256=sha(a.out/'sparse_contact_points.json'),initial_states_sha256=sha(a.out/'initial_states.json')))
    env.close()

try: main()
except BaseException:
    import traceback
    (a.out/'failure.txt').write_text(traceback.format_exc())
    print(traceback.format_exc(),flush=True)
    raise
finally:
    sim_utils.SimulationContext.clear_instance()
    app.close()
