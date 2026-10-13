"""Independent native task stages, adaptive FIFO custody and source geometry replay."""
from pathlib import Path
import ast,hashlib,json,sys
import numpy as np

K=Path(__file__).resolve().parent
H=Path('/home/liyufeng/safeduo/artifacts/safety_physical_qualified_random_20261009_1613')
T=Path('/mnt/nas/data/lyf/double_hand/safeduo_task_foundation_20261012')
sys.path[:0]=[str(T),'/mnt/nas/data/lyf/double_hand/safeduo_full50_geometry_20261013',str(H/'fourhand_native_probe_v119')]
from full74_geometry_v2 import Full74Geometry,ARMS
from episode_judge import judge_pick_carry_place,continuous_window
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    P=H/'fourhand_native_probe_v119';reg=read(P/'REGISTRATION_V1.json');res=read(P/'RESOURCE_EXECUTION_V1.json');raw=Path(res['out'])
    assert res['actual_exit']==res['actual_native_exit']==0 and res['native_owned_process_reaped_and_absent']
    capture=read(P/'INDEPENDENT_CAPTURE_AUDIT_V1.json');model=read(P/'INDEPENDENT_NATIVE_MIMIC_WINDOW_V6.json')
    assert capture['record_integrity_verified']
    for f,h in reg['source_sha256'].items():assert sha(f)==h,f
    spec=reg['pair_task']['spec'];params=dict(np.load(raw/'native_parameters.npz'));seed=dict(np.load(reg['contact_reproducer']['fixture']))
    identity=read(raw/'native_task_identity.json');partners=identity['partners'];pieces=[];sources={};opposition=[]
    geom=Full74Geometry(params,read(reg['coupled_assets'])['assets']);max_geometry_error=0
    for record in read(raw/'physics_chunks.json'):
        file=raw/record['path'];assert sha(file)==record['sha256'];sources[str(file)]=record['sha256']
        with np.load(file) as z:
            flags=np.zeros((len(z['state_time_s']),4),bool)
            for i in range(len(flags)):
                lo,hi=map(int,z['object_normal_event_offsets'][i:i+2])
                for ai,a in enumerate(ARMS):
                    ids=[j for j in range(lo,hi) if '/'+a+'/' in partners[int(z['object_normal_partner_indices'][j])]
                        and any(s in partners[int(z['object_normal_partner_indices'][j])].lower() for s in ['thumb','index','middle','ring','little','pinky','hand_base'])
                        and abs(float(z['object_normal_normal_forces'][j,0]))>.1]
                    for ii,j in enumerate(ids):
                        for k in ids[ii+1:]:
                            if z['object_normal_partner_indices'][j]==z['object_normal_partner_indices'][k]:continue
                            n0=z['object_normal_normals'][j].astype(float);n1=z['object_normal_normals'][k].astype(float)
                            l0=np.linalg.norm(n0);l1=np.linalg.norm(n1)
                            if abs(l0-1)>.02 or abs(l1-1)>.02:continue
                            n0/=l0;n1/=l1;s=z['object_normal_points'][j].astype(float)-z['object_normal_points'][k].astype(float)
                            if n0@n1<=-.5 and s@n0>=.002 and s@n1<=-.002:flags[i,ai]=True
            np.testing.assert_array_equal(flags,z['task_opposed_native_contacts']);opposition.append(flags)
            error=np.max(np.abs(geom.gaps({a:z[a+'_native_q'][:,0] for a in ARMS})-z['all9021_raw_geometry'][:,0]));max_geometry_error=max(max_geometry_error,float(error))
            pieces.append({k:z[k].copy() for k in z.files if not k.startswith('object_normal_') and not k.startswith('object_friction_')})
    data={k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]};n=len(data['physics_event']);sub=reg['physics_substeps_per_control']
    assert n==capture['physics_events'] and n==res.get('physics_events',n)
    roots={a:data[a+'_full50_root_indices'][0].astype(int) for a in ARMS}
    pending=[{a:seed[a+'_actual_full_position_target_seed'][0,roots[a]].copy() for a in ARMS} for _ in range(6)]
    for step in range(n//sub):
        e=step*sub;issued={a:data[a+'_full50_issued_target'][e,0].copy() for a in ARMS};pending.append(issued);actual=pending.pop(0)
        for a in ARMS:
            np.testing.assert_array_equal(data[a+'_full50_fifo_applied_target'][e:e+sub,0],np.repeat(actual[a][None],sub,axis=0))
            np.testing.assert_array_equal(data[a+'_full50_pending_fifo_targets'][e:e+sub],np.repeat(np.stack([v[a] for v in pending])[None],sub,axis=0))
            np.testing.assert_array_equal(data[a+'_full50_effective_target'][e:e+sub],data[a+'_actual_full_position_target'][e:e+sub,:,roots[a]])
    worker=P/'native_common_beam_v1.py';source=worker.read_text();loop_line=source[:source.index("        for step in range(reg['maximum_controls']):")].count('\n')+1
    writes=[]
    for node in ast.walk(ast.parse(source)):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in ['write_joint_state_to_sim','write_root_pose_to_sim','write_root_velocity_to_sim']:
            writes.append(node.lineno);assert node.lineno<loop_line,'post-initialization state mutation'
    state=data['object_state'];position=state[:,:3]-params['scene_origin'][0];q=state[:,3:7];mounts=[]
    for a in ARMS:
        mount=('left_hand_base' if a.endswith('L') else 'right_hand_base') if a.startswith('F') else 'wrist_3_link'
        mounts.append(data[a+'_body_position_local'][:,0,params[a+'_body_names'].tolist().index(mount)])
    mounts=np.stack(mounts,axis=1);release_end=np.flatnonzero(data['phase']=='RETREAT');retreat=np.zeros((n,4))
    if len(release_end):retreat=np.linalg.norm(mounts-mounts[release_end[0]],axis=2)
    trace={'time_s':data['state_time_s'],'phase':data['phase'],'lift_m':position[:,2]-spec['initial_position'][2],
        'support_N':data['support_normal_N'],'hand_force_N':data['four_hand_normal_N'],'opposed_contact':np.concatenate(opposition),
        'linear_speed_m_s':np.linalg.norm(state[:,7:10],axis=1),'angular_speed_rad_s':np.linalg.norm(state[:,10:13],axis=1),
        'orientation_error_deg':np.degrees(np.arccos(np.clip(1-2*(q[:,1]**2+q[:,2]**2),-1,1))),
        'goal_position_error_m':np.linalg.norm(position[:,:2]-spec['goal_position'][:2],axis=1),
        'goal_height_error_m':position[:,2]-spec['goal_position'][2],
        'object_xy_displacement_m':np.linalg.norm(position[:,:2]-spec['initial_position'][:2],axis=1),
        'hand_open_error_rad':data['hand_open_error_rad'],'retreat_distance_m':retreat,
        'hard_violation_rad':data['all74_hard_violation_rad'],'speed_exceedance_rad_s':data['all74_speed_exceedance_rad_s'],
        'proxy_gap_m':data['all9021_raw_geometry'].min((1,2)),'unintended_force_N':data['unintended_peak_normal_N'],
        'mimic_q_error_rad':np.max(np.stack([np.abs(data[a+'_mimic_position_residual_rad']).max(-1) for a in ARMS]),axis=0),
        'mimic_qd_error_rad_s':np.max(np.stack([np.abs(data[a+'_mimic_velocity_residual_rad_s']).max(-1) for a in ARMS]),axis=0),
        'provenance':{'native_capture_and_runtime_custody_verified':True,'physical_state_mutations_after_initialization':0,'object_attachment_constraints':0,'all_original_microsteps_retained':True}}
    verdict=judge_pick_carry_place(trace,spec['roles'],identity['native_mass_kg'],reg['physics_dt_s'])
    protocol=read(raw/'fourhand_protocol.json');max_movement={a:float(np.max(np.abs(data[a+'_native_q'][:,0,params[a+'_controlled_indices']]-seed[a+'_native_q_seed'][0,params[a+'_controlled_indices']]))) for a in ARMS}
    report={'status':'AUDITED_NATIVE_PAIR_COLIFT_DEVELOPMENT_EPISODE','native_version':119,'native_episodes':1,
        'family':spec['family'],'roles':spec['roles'],'physics_events':n,'simulation_duration_s':n*reg['physics_dt_s'],
        'protocol_task_status':protocol['task_status'],'original_failure':protocol['model_qualification_failure'],
        'task_transitions':protocol['pair_task_transitions'],'task_observer_failure':protocol['pair_task_observer_failure'],
        'independent_stage_verdict':verdict,'max_object_lift_m':float(trace['lift_m'].max()),'max_object_xy_displacement_m':float(trace['object_xy_displacement_m'].max()),
        'per_arm_actual_joint_departure_rad':max_movement,'per_arm_opposed_contact_s':{a:continuous_window(trace['time_s'],trace['opposed_contact'][:,i],reg['physics_dt_s']) for i,a in enumerate(ARMS)},
        'all50_FIFO6_reconstructed':True,'native_geometry_max_replay_error_m':max_geometry_error,'native_geometry_replay_pass':max_geometry_error<=1e-4,
        'actual_training_jobs':0,'formal_holdouts':0,'complete_task_successes':int(verdict['safe_task_pass']),
        'approach_tested':False,'fullSystem0_accepted':False,
        'source_sha256':{**sources,str(P/'REGISTRATION_V1.json'):sha(P/'REGISTRATION_V1.json'),str(worker):sha(worker),str(Path(__file__)):sha(__file__)}}
    output=K/'NATIVE119_INDEPENDENT_PAIR_TASK_V1.json';assert not output.exists();output.write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(K/'NATIVE119_INDEPENDENT_TASK_TRACE_V1.npz',**{k:v for k,v in trace.items() if k!='provenance'})
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'}),flush=True)


if __name__=='__main__':main()
