"""Prospective stationary-release and reactive clearance contrasts."""
import argparse,json,hashlib,os,math,time
from pathlib import Path
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser();parser.add_argument('--registration',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True);parser.add_argument('--block',type=int,required=True);AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args();R=Path(__file__).resolve().parent
reg=json.loads(args.registration.read_text())
for name,digest in reg['source_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
block=reg['blocks'][args.block];reg['cases']=block['cases'];reg['physics_seed']=block['physics_seed']
args.out.mkdir(parents=True,exist_ok=False);args.enable_cameras=True
app=AppLauncher(args).app
import torch,numpy as np
import isaaclab.sim as sim_utils
import omni.usd,omni.physics.tensors as tensors
from pxr import UsdGeom,Usd,UsdPhysics,PhysxSchema
from PIL import Image
from safeduo.envs.duo_env import DuoEnv,make_duo_env_cfg
from safeduo.delta.skill_replay import SkillTrajectory,SkillReplayDelta,SkillNoiseParams
from safeduo.eval.endurance_eval import ckpt_arm_aware,ckpt_p2_obs
from safeduo.eval.block1_harness import ActorMLP
from hand_driver import HandDriver
from object_binding_v3 import ARMS,FIELDS,pack_task_input,bind_reference
from clock_contract import control_clock
from clearance_gate import ClearanceGate,zero_held_command

OBJECTS=('beam700','beam300')
class ProbeEnv(DuoEnv):
    def _setup_scene(self):
        super()._setup_scene()
        stage=omni.usd.get_context().get_stage()
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):PhysxSchema.PhysxContactReportAPI.Apply(prim)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,value):
    (args.out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def main():
    torch.set_num_threads(1);n=len(reg['cases']);assert n==12
    traj=SkillTrajectory.load(reg['trajectory']);meta=traj.meta['s9_task'];dt=traj.dt
    cfg=make_duo_env_cfg(num_envs=n,device=args.device,yaml_name=reg['env_yaml'],coordinator=True,
        arm_aware_obs=ckpt_arm_aware(reg['checkpoint']),p2_obs=ckpt_p2_obs(reg['checkpoint']))
    cfg.scene.env_spacing=3.5;cfg.enable_viz_camera=True;cfg.seed=reg['physics_seed']
    cfg.coordinator['terminate_on_violation']=False;cfg.coordinator['retreat_passthrough']=True
    cfg.coordinator['retreat_dwell_steps']=10;cfg.episode_length_s=45.
    cfg.scene_dressing['table_visible']=True
    cfg.scene_dressing['ground_visible']=True
    # Explicit visual simplification: preserve all colliding fixtures.
    removed=[p for p in cfg.scene_dressing.get('static_props',[]) if not p.get('collision',True)]
    cfg.scene_dressing['static_props']=[p for p in cfg.scene_dressing.get('static_props',[]) if p.get('collision',True)]
    print('VISUAL_ONLY_REMOVED_NONCOLLIDING_PROPS',len(removed),flush=True)
    print('PHASE scene construction',flush=True)
    env=ProbeEnv(cfg);print('PHASE scene constructed',flush=True)
    dt=control_clock(env.step_dt,reg['native_control_dt_s'],traj.dt)
    cam=env._viz_cam;assert env.scene.sensors.pop('viz_cam') is cam;cam.reset()
    base=SkillReplayDelta(n,[traj],params=SkillNoiseParams.tier(0),device=env.device,auto_advance=False,loop=False)
    debt={a:torch.zeros_like(env._targets[a]) for a in ARMS};original_sample=base.sample
    gate=ClearanceGate([c['method'] for c in reg['cases']],dt,reg['gate'],device=env.device)
    held=torch.zeros(n,dtype=torch.bool,device=env.device);last_gate_input=None
    def sample(state):
        cmd=original_sample(state);cmd._r37_nominal={a:v.clone() for a,v in cmd.delta_q.items()}
        for a in ARMS:cmd.delta_q[a]=(cmd.delta_q[a]+(.10*debt[a]).clamp(-.001,.001)).clamp(-base.p.amp_max,base.p.amp_max)
        return zero_held_command(cmd,held)
    base.sample=sample;env._delta_src=base
    obs,_=env.reset()
    for a in ARMS:
        art=env._arms[a];jp=art.data.joint_pos.clone();q0=torch.tensor(traj.q[a][0],device=env.device)
        jp[:,env._joint_idx[a]]=q0;art.write_joint_state_to_sim(jp,torch.zeros_like(art.data.joint_vel))
        env._targets[a][:]=q0
    # The only object state writes occur here, before the physical trial starts.
    for oi,o in enumerate(OBJECTS):
        body=env._objects[o];pose=body.data.root_state_w[:,:7].clone()
        for e,c in enumerate(reg['cases']):
            par=c['objects'][o];pose[e,:3]=torch.tensor(reg['nominal_centers'][oi],device=env.device)+env.scene.env_origins[e]
            pose[e,0]+=par['dx'];pose[e,1]+=par['dy']
            angle=reg['nominal_yaws_rad'][oi]+math.radians(par['yaw_delta_deg'])
            pose[e,3:]=torch.tensor([math.cos(angle/2),0,0,math.sin(angle/2)],device=env.device)
        body.write_root_pose_to_sim(pose);body.write_root_velocity_to_sim(torch.zeros(n,6,device=env.device))
        actual=body.root_physx_view.get_transforms().detach().clone()
        assert torch.allclose(actual[:,:3],pose[:,:3],atol=1e-6,rtol=0)
        assert torch.allclose(actual[:,3:],pose[:,[4,5,6,3]],atol=1e-6,rtol=0)
    states=torch.stack([env._objects[o].data.root_state_w for o in OBJECTS],1).detach().clone()
    sizes=torch.tensor(reg['sizes_m'],device=env.device);goals=torch.tensor(reg['goals_local_m'],device=env.device)
    task_input=pack_task_input(states,env.scene.env_origins,sizes,goals,0.,0.,torch.ones(n,2,dtype=torch.bool,device=env.device),dt)
    enabled=torch.tensor([c['binding'] for c in reg['cases']],dtype=torch.bool)
    native_nominal=(torch.tensor(reg['nominal_centers'],device=env.device)[None]+env.scene.env_origins[:,None,:])-env.scene.env_origins[:,None,:]
    limits={a:env._arms[a].data.soft_joint_pos_limits[0,env._joint_idx[a]].cpu().tolist() for a in ARMS}
    save('native_limits_before_planning.json',limits)
    expected=json.loads(Path(reg['native_limits_readback']).read_text())
    for a in ARMS:assert np.allclose(limits[a],expected['arms'][a]['soft_limits_rad'],atol=1e-6,rtol=0)
    bound,ik_audit=bind_reference(traj.q,traj.dt,task_input,enabled,reg['nominal_centers'],reg['nominal_yaws_rad'],native_nominal.cpu().numpy(),joint_limits=limits)
    reference_path=args.out/'bound_reference.npz';np.savez_compressed(reference_path,**{'q_'+a:bound[a] for a in ARMS},task_input=task_input.cpu().numpy())
    for a in ARMS:
        base._qlib[a]=torch.tensor(bound[a],device=env.device)
        qlimit=env._arms[a].data.soft_joint_pos_limits[0,env._joint_idx[a]].cpu().numpy()
        assert np.all(bound[a]>=qlimit[:,0]-1e-5) and np.all(bound[a]<=qlimit[:,1]+1e-5),(a,'reference exceeds native soft limits')
        assert np.max(np.abs(np.diff(bound[a],axis=1)))<=.06,(a,'reference step cap')
    # Reset assigned one library only; expand all per-library grid metadata too.
    base._dt_traj=base._dt_traj.repeat(n)
    base._n_steps=base._n_steps.repeat(n)
    base.K=n
    base._traj[:]=torch.arange(n,device=env.device)
    stage=omni.usd.get_context().get_stage();rigid=[p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    partners=[]
    for a in ARMS:
        prefix=f'/World/envs/env_0/{a}/'
        paths=sorted(str(p.GetPath()) for p in rigid if str(p.GetPath()).startswith(prefix))
        for p in paths:partners.append(dict(path=p,arm=a,hand=any(x in p.rsplit('/',1)[-1] for x in ('thumb','index','middle','ring','pinky','little'))))
        assert sum(p['arm']==a and p['hand'] for p in partners)==12
    for t in ('TableF','TableU'):
        p=f'/World/envs/env_0/{t}/geometry/mesh';assert stage.GetPrimAtPath(p).HasAPI(UsdPhysics.CollisionAPI)
        partners.append(dict(path=p,arm=None,hand=False,table=t))
    views=[];identities=[];sv=tensors.create_simulation_view('torch');sv.set_subspace_roots('/')
    for e in range(n):
        slot=[]
        filters=[p['path'].replace('/env_0/',f'/env_{e}/') for p in partners]
        for o in OBJECTS:
            sensor=f'/World/envs/env_{e}/Obj_{o}'
            cv=sv.create_rigid_contact_view(sensor,filter_patterns=filters,max_contact_data_count=2048)
            actual=list(cv.filter_paths)
            if len(actual)==1 and isinstance(actual[0],(list,tuple)):actual=list(actual[0])
            assert cv.check() and list(cv.sensor_paths)==[sensor] and actual==filters
            assert cv.sensor_count==1 and cv.filter_count==len(partners)
            slot.append(cv);identities.append(dict(env=e,object=o,sensor_path=sensor,filter_paths=actual))
        views.append(slot)
    bbox=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_])
    tables={t:list(bbox.ComputeWorldBound(stage.GetPrimAtPath(f'/World/envs/env_0/{t}/geometry/mesh')).ComputeAlignedRange().GetMax()) for t in ('TableF','TableU')}
    initial=dict(cases=reg['cases'],fields=FIELDS,task_input=task_input.cpu().tolist(),origins=env.scene.env_origins.cpu().tolist(),
        native_mass_kg={o:env._objects[o].root_physx_view.get_masses().cpu().tolist() for o in OBJECTS},
        native_material={o:env._objects[o].root_physx_view.get_material_properties().cpu().tolist() for o in OBJECTS},
        native_table_world_max=tables,partners=partners,contact_identities=identities,ik=ik_audit,
        reference_sha256=sha(reference_path),native_joint_limits_rad=limits,removed_noncolliding_props=removed,qualified_trajectory_sha256=sha(reg['trajectory']),no_fixed_grasp_constraint=not any(p.IsA(UsdPhysics.FixedJoint) and 'Obj_' in str(p.GetPath()) for p in stage.Traverse()),
        registration_sha256=sha(args.registration),source_scope='24 development task windows: 4 layouts x 3 methods x 2 seed/slot blocks; original task gates plus explicit abort/completion; no IID or final holdout',block=args.block,static_collision_paths=[str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI) and str(p.GetPath()).startswith('/World/envs/env_0/Prop_')])
    save('native_initial.json',initial)
    print('OBJECT_INPUT_BOUND',json.dumps(dict(cases=n,fields=len(FIELDS),partners=len(partners),max_ik_error_m=max(v['max_position_error_m'] for v in ik_audit))),flush=True)
    hands=HandDriver(env,dt);events=sorted(meta['hand_events'],key=lambda e:e['t'])
    policy=ActorMLP.from_checkpoint(reg['checkpoint']).to(env.device);engaged=torch.ones(n,4,dtype=torch.bool,device=env.device)
    buffers={};chunks=[];images=[];video_frames=[];render_checks=[]
    def fresh_snapshot():
        return {**{o+':pose':env._objects[o].root_physx_view.get_transforms().detach().clone() for o in OBJECTS},
            **{o+':velocity':env._objects[o].root_physx_view.get_velocities().detach().clone() for o in OBJECTS},
            **{a+':q':env._arms[a].root_physx_view.get_dof_positions().detach().clone() for a in ARMS},
            **{a+':qd':env._arms[a].root_physx_view.get_dof_velocities().detach().clone() for a in ARMS}}
    def fresh_objects():
        return torch.stack([torch.cat((env._objects[o].root_physx_view.get_transforms()[:,0:3],env._objects[o].root_physx_view.get_transforms()[:,[6,3,4,5]],env._objects[o].root_physx_view.get_velocities()),-1) for o in OBJECTS],1).detach().clone()
    renderables=[UsdGeom.Imageable(stage.GetPrimAtPath(f'/World/envs/env_{e}')) for e in range(n)]
    capture_steps=set(reg['capture_steps']);capture_envs=reg['capture_envs']
    for step in range(reg['steps']):
        clock=step*dt;held=gate.step(last_gate_input);base.set_time(gate.phase)
        # Common harness correction: every method recomputes from the authoritative phase.
        env._pending_cmd=None
        while events and events[0]['t']<=clock:
            event=events.pop(0);hands.command(event['arm'],event.get('frac',0),event.get('ramp_s',.6),event.get('frac_thumb'))
        hands.step()
        with torch.no_grad():act=policy(obs['policy']).clamp(-1,1)
        alpha=(act[:,:4]+1)*.5;engaged[:]=torch.where(engaged,alpha>=.2,alpha>.5)
        act=torch.cat((engaged.to(act.dtype)*2-1,act[:,4:5]),-1)
        obs,_,term,trunc,_=env.step(act)
        sc=env._step_cache
        for a in ARMS:debt[a]+=sc['cmd']._r37_nominal[a]-sc['exec'].delta_q[a]
        data=dict(step=np.asarray(step),time=np.asarray([clock,(step+1)*dt]),objects=fresh_objects(),phase=gate.phase.clone(),gate_stage=gate.stage.clone(),gate_reason=gate.reason.clone(),command_held=held.clone(),
            violation=sc['violation'],termination=term,truncation=trunc,action=act,
            class_margin=torch.stack([env._last_out.min_margin[k] for k in ('cross','self_F','self_U','table')],-1),
            hand_fraction=np.asarray([[hands.cur[a],hands.cur_thumb[a]] for a in ARMS]))
        for a in ARMS:
            art=env._arms[a];ids=env._joint_idx[a]
            for key,val in [('q',art.data.joint_pos[:,ids]),('qd',art.data.joint_vel[:,ids]),('target',env._targets[a]),
                ('hand_q',art.data.joint_pos[:,hands.ids[a][0]]),('cmd',sc['cmd'].delta_q[a]),('exec',sc['exec'].delta_q[a])]:data[a+':'+key]=val
        matrices=[];counts_all=[];nets=[]
        for slot in views:
            m=[];counts=[];net=[]
            for cv in slot:
                # Clone count/start outputs before any other contact accessor.
                detail=cv.get_contact_data(env.cfg.sim.dt)
                cnt=detail[-2].detach().clone();starts=detail[-1].detach().clone()
                assert int(cnt.sum())<2048 and int(cnt.min())>=0
                assert bool(((cnt==0)|((starts>=0)&(starts+cnt<=2048))).all())
                force=cv.get_contact_force_matrix(env.cfg.sim.dt).detach().clone()
                assert torch.isfinite(force).all();m.append(force.reshape(len(partners),3));counts.append(cnt.reshape(len(partners)))
                net.append(cv.get_net_contact_forces(env.cfg.sim.dt).detach().clone().reshape(3))
            matrices.append(torch.stack(m));counts_all.append(torch.stack(counts));nets.append(torch.stack(net))
        data['object_partner_normal']=torch.stack(matrices);data['contact_counts']=torch.stack(counts_all);data['object_net_contact']=torch.stack(nets)
        normal=data['object_partner_normal'].norm(dim=-1);states=data['objects'];q=states[:,:,3:7]
        half=torch.tensor(reg['sizes_m'],device=env.device)*.5
        qw,qx,qy,qz=q.unbind(-1)
        rowz=torch.stack((2*(qx*qz-qw*qy),2*(qy*qz+qw*qx),1-2*(qx*qx+qy*qy)),-1)
        bottom=states[:,:,2]-env.scene.env_origins[:,None,2]-(rowz.abs()*half[None]).sum(-1)
        last_gate_input=dict(table_n=torch.stack([normal[:,i,next(j for j,p in enumerate(partners) if p.get('table')==t)] for i,t in enumerate(('TableF','TableU'))],-1),
            bottom_delta_m=bottom-.8,speed_m_s=states[:,:,7:10].norm(dim=-1),angular_rad_s=states[:,:,10:13].norm(dim=-1),
            tilt_deg=torch.acos(rowz[:,:,2].clamp(-1,1))*180/math.pi,
            hand_n=torch.stack([torch.stack([normal[:,i,[j for j,p in enumerate(partners) if p['arm']==a and p['hand']]].sum(-1) for a in ARMS[2*i:2*i+2]],-1) for i in range(2)],1),
            hand_open_max_error_rad=torch.stack([(env._arms[a].root_physx_view.get_dof_positions()[:,hands.ids[a][0]]-hands.open_q[a]).abs().max(-1).values for a in ARMS],-1))
        for k,v in last_gate_input.items():data['gate_input:'+k]=v
        data['gate_stable_count']=gate.stable.clone();data['gate_wait_s']=gate.wait.clone()
        for k,v in data.items():
            arr=v.detach().clone().cpu().numpy() if isinstance(v,torch.Tensor) else v.copy()
            assert np.isfinite(arr).all(),(step,k);buffers.setdefault(k,[]).append(arr)
        video_step=args.block==0 and step in reg['video_steps']
        if step in capture_steps or video_step:
            before_o=data['objects'].detach().clone();before_native=fresh_snapshot()
            selected=list(range(n)) if step in capture_steps else [c['env'] for c in reg['cases'] if c['layout']==0]
            for e in selected:
                for i,v in enumerate(renderables):v.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited if i==e else UsdGeom.Tokens.invisible)
                for name,view in reg['views'].items():
                    if step not in capture_steps and name!='front':continue
                    origin=env.scene.env_origins[e];eye=origin+torch.tensor(view['eye'],device=env.device);target=origin+torch.tensor(view['target'],device=env.device)
                    cam.set_world_poses_from_view(eye[None],target[None]);env.sim.render();cam.update(dt,force_recompute=True)
                    rgb=cam.data.output['rgb'][0].cpu().numpy()[:,:,:3]
                    if rgb.dtype!=np.uint8:rgb=(rgb*255).clip(0,255).astype(np.uint8)
                    rel=f'images/e{e}_s{step:04d}_{name}.png';path=args.out/rel;path.parent.mkdir(exist_ok=True)
                    Image.fromarray(rgb).save(path)
                    item=dict(file=rel,sha256=sha(path),env=e,step=step,state_time_s=(step+1)*dt,view=name,objects=before_o[e].cpu().tolist(),phase_s=float(gate.phase[e]),stage=int(gate.stage[e]))
                    images.append(item)
                    if video_step and reg['cases'][e]['layout']==0 and name=='front':video_frames.append(item.copy())
            after_native=fresh_snapshot()
            differences={k:float((before_native[k]-after_native[k]).abs().max()) for k in before_native}
            assert all(v==0 for v in differences.values()),('render mutated physics',step,differences)
            assert torch.equal(before_o,fresh_objects())
            render_checks.append(dict(step=step,native_readback_max_abs_differences=differences))
            for v in renderables:v.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
        if (step+1)%120==0 or step==reg['steps']-1:
            first=int(buffers['step'][0]);path=args.out/f'dense_{first:04d}_{step:04d}.npz'
            np.savez_compressed(path,**{k:np.stack(v) for k,v in buffers.items()});buffers.clear()
            chunks.append(dict(file=path.name,sha256=sha(path),first_step=first,last_step=step))
            save('progress.json',dict(steps=step+1,cases=n,chunks=chunks,images=len(images)))
            print('DENSE',step+1,'/',reg['steps'],'images',len(images),flush=True)
    save('recording_receipt.json',dict(status='PASS_COMPLETE_NATIVE_RECORDING',steps=reg['steps'],cases=n,env_states=n*reg['steps'],chunks=chunks,images=images,
        registration_sha256=sha(args.registration),native_initial_sha256=sha(args.out/'native_initial.json'),scope=initial['source_scope'],new_final_trials=0,block=args.block,video_frames=video_frames,native_render_checks=render_checks))
    env.close()

try:main()
except BaseException:
    import traceback
    failure=traceback.format_exc()
    (args.out/'failure.txt').write_text(failure)
    print(failure,flush=True)
    raise
finally:
    sim_utils.SimulationContext.clear_instance()
    app.close()
