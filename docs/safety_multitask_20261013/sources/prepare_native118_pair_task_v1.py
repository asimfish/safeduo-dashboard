"""Bind a real friction-only dual-holder task to the qualified capture stack."""
from pathlib import Path
import ast,copy,datetime,hashlib,json,shutil
import numpy as np

K=Path(__file__).resolve().parent
H=Path('/home/liyufeng/safeduo/artifacts/safety_physical_qualified_random_20261009_1613')
B=H/'fourhand_native_probe_v117';P=H/'fourhand_native_probe_v118'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert not P.exists();assert read(K/'NATIVE117_CAPTURE_CAPACITY_COMPATIBILITY_V1.json')['all_targets_states_contacts_and_raw_geometry_byte_equal']
    admission=read(K/'PAIR_TASK_SCENE_ADMISSION_V2.json');assert admission['selected_candidate'] is not None
    selected=admission['candidates'][admission['selected_candidate']];assert selected['admitted']
    reg=read(B/'REGISTRATION_V1.json')
    for f,h in reg['source_sha256'].items():assert sha(f)==h,f
    plan={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'candidate':118,'family':'pair_colift',
        'roles':['F_L','F_R'],'object_count':1,'adaptive_task_controller':True,'IID_random_test':False,
        'scene':admission['task'],'start_stage':'CLOSE','source_seed_native_version':105,'source_seed_physics_event':8191,
        'approach_not_tested_in_this_development_episode':True,'destination_support_posts':admission['destination_support_posts'],
        'maximum_controls':1024,'maximum_microsteps':32768,'maximum_physical_time_s':1024/60,
        'maximum_owned_wall_s':1200,'maximum_native_launches':1,'original_physics_profile_and_gates_retained':True,
        'longer_recording_capacity_is_not_physical_duration_qualification':True,'formal_holdouts':0,
        'training_started':False,'fullSystem0_accepted':False}
    planfile=K/'NATIVE118_PAIR_TASK_PROSPECTIVE_V1.json';planfile.write_text(json.dumps(plan,indent=2)+'\n')
    P.mkdir()
    for f in B.iterdir():
        if f.suffix in ['.py','.usda'] or f.name=='PRIVATE_KIT_USER_CONFIG_FROZEN_V1.json':shutil.copyfile(f,P/f.name)
    shutil.copytree(B/'private_kit_user_data',P/'private_kit_user_data')
    for name in ['multitask_phases_v1.py','opposed_contact_v1.py','pair_task_operator_v1.py']:shutil.copyfile(K/name,P/name)
    source=(P/'native_common_beam_v1.py').read_text()
    def replace(old,new):
        nonlocal source
        assert source.count(old)==1,(old[:100],source.count(old));source=source.replace(old,new)
    # Destination support geometry exists before native physics initialization.
    anchor='                for prim in omni.usd.get_context().get_stage().Traverse(Usd.TraverseInstanceProxies()):'
    extra="""                for i,post in enumerate(reg['pair_task']['destination_support_posts'],start=2):
                    support=sim_utils.CuboidCfg(size=tuple(post['size_m']),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True,disable_gravity=True),
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.7,dynamic_friction=.6),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.18,.22,.28)))
                    support.func(f'/World/envs/env_0/Support_{i}',support,translation=tuple(origin+np.asarray(post['center_m'])))
"""
    replace(anchor,extra+anchor)
    replace("protocol.update(method='RANDOM_FULL50_ACTUAL74_FINGER_GEOMETRY_FIFO6_GUARD',target_producer='Fresh developmentIID50 motor intents through actual full50FIFO6; moving-finger actual74 geometry and latched native50 brake; original physics and motor effort; not learnedSystem0',initial_state_source_event=11271,synthetic_operator=True,human_teleoperation_tested=False,random_test=True)",
        "protocol.update(method='FRICTION_ONLY_PAIR_COLIFT_CONTACT_DRIVEN_TASK',target_producer='Native point-contact feedback controls CLOSE/LIFT/CARRY/PLACE/RELEASE/RETREAT through actual full50FIFO6 and actual74 guard',initial_state_source_event=8191,synthetic_operator=True,human_teleoperation_tested=False,random_test=False,task_family='pair_colift',task_roles=['F_L','F_R'],approach_tested=False)")
    anchor='        from emergency_brake_latch_v1 import EmergencyBrakeLatch'
    setup="""        from multitask_phases_v1 import PayloadTask,TaskSpec
        from pair_task_operator_v1 import PairTaskOperator,native_opposed_flags
        payload_task=PayloadTask(TaskSpec(**reg['pair_task']['spec']))
        with np.load(reg['pair_task']['IK_endpoints']) as f:task_endpoints={a:f[a].copy() for a in ARMS}
        task_operator=PairTaskOperator(payload_task,parameters,full50_ids,task_endpoints,coupled_refs,dt=dt)
"""
    replace(anchor,setup+anchor)
    anchor="            issued={a:torch.as_tensor(motion_bank[a+'_full50_issued_target'][step],device=env.device,dtype=initial[a].dtype)[None] for a in ARMS}"
    producer="""            phase=payload_task.phase
            task_request=task_operator.request(guard_previous_target)
            for a in ARMS:
                motion_bank[a+'_full50_issued_target'][step]=task_request[a]
                motion_bank[a+'_raw_requested_target'][step]=task_request[a][arm_in_full50[a]]
            issued={a:torch.as_tensor(task_request[a],device=env.device,dtype=initial[a].dtype)[None] for a in ARMS}
"""
    replace(anchor,producer.rstrip())
    replace("if phase in ['CLOSE','LIFT','CARRY','PLACE','RELEASE']:","if phase in ['CLOSE','LIFT','CARRY','PLACE','RELEASE'] and a in reg['pair_task']['spec']['roles']:")
    replace("motion_family=np.asarray('RANDOM_FULL50_ACTUAL74_FINGER_GEOMETRY')","motion_family=np.asarray('FRICTION_ONLY_PAIR_COLIFT')")
    anchor="                # Abort an unsuccessful qualification after this full control"
    observe="""                opposed=native_opposed_flags(normal,partners)
                mounts=[];hand_speeds=[]
                for a in ARMS:
                    mount=('left_hand_base' if a.endswith('L') else 'right_hand_base') if a.startswith('F') else 'wrist_3_link'
                    mi=parameters[a+'_body_names'].tolist().index(mount)
                    mounts.append(data[a+'_body_position_local'][0,mi])
                    hand_speeds.append(float(np.abs(data[a+'_native_qd'][0,hands.ids[a][0]]).max()))
                task_observation=dict(object_id='beam_common',sequence=event,time_s=now,now_s=now,
                    position=local.copy(),quaternion_wxyz=state[3:7].copy(),linear_velocity=state[7:10].copy(),
                    angular_velocity=state[10:13].copy(),size_m=reg['pair_task']['spec']['size_m'],mass_kg=mass,
                    hand_force_N=np.asarray(four),opposed=opposed,support_N=support,hand_open_error_rad=np.asarray(opening),
                    hand_speed_rad_s=np.asarray(hand_speeds),mounts_m=np.asarray(mounts),conservative_mu=conservative_mu,
                    physical_observation_valid=not bool(unsafe))
                task_intent=task_operator.observe(task_observation)
                data['task_opposed_native_contacts']=opposed
                data['task_holder_hand_speed_rad_s']=np.asarray(hand_speeds)
                data['task_observed_next_phase']=np.asarray(task_intent['phase'])
                data['task_object_target_m']=np.asarray(task_intent['object_target_m'])
"""
    replace(anchor,observe+anchor)
    anchor="            completed=step+1\n            if model_qualification_failure is not None:"
    replace(anchor,"            completed=step+1\n            if payload_task.phase in ['DONE','FAILED']:phase=payload_task.phase\n            if model_qualification_failure is not None:")
    replace("if unsafe or phase=='FAILED':break","if unsafe or phase in ['FAILED','DONE']:break")
    replace("else 'FULL50_RANDOM_WINDOW_COMPLETE'","else ('PAIR_TASK_OBSERVER_FAILED' if payload_task.failure else 'PAIR_TASK_PHASE_WINDOW_COMPLETE')")
    replace("model_qualification_failure=model_qualification_failure,fulltask_phase_ceiling='CLOSE',","model_qualification_failure=model_qualification_failure,fulltask_phase_ceiling=payload_task.phase,\n            pair_task_transitions=payload_task.transitions,pair_task_observer_failure=payload_task.failure,pair_task_producer_done=payload_task.phase=='DONE',")
    ast.parse(source);(P/'native_common_beam_v1.py').write_text(source)
    launcher=H/'launch_native_clock_admission_v118.py';s=(H/'launch_native_clock_admission_v117.py').read_text().replace('v117','v118').replace('V117','V118').replace('NATIVE117','NATIVE118');ast.parse(s);launcher.write_text(s)
    auditor=H/'audit_motion_capture118_v1.py';s=(H/'audit_motion_capture117_v1.py').read_text()
    old="assert version == 117 and reg['contact_reproducer']['maximum_physics_events']==4096 and reg['maximum_controls']==128"
    assert s.count(old)==1;s=s.replace(old,"assert version == 118 and reg['contact_reproducer']['maximum_physics_events']==32768 and reg['maximum_controls']==1024")
    ast.parse(s);auditor.write_text(s)
    seed=dict(np.load(K/'PAIR_TASK_NATIVE_SEED_V1.npz'));baseline=Path(read(B/'RESOURCE_EXECUTION_V1.json')['out']);params=dict(np.load(baseline/'native_parameters.npz'));bank={}
    with np.load(reg['motion_test']['command_bank']) as f:original_bank={a+'_full50_root_indices':f[a+'_full50_root_indices'].copy() for a in ['F_L','F_R','U_L','U_R']}
    for a in ['F_L','F_R','U_L','U_R']:
        rootids=original_bank[a+'_full50_root_indices']
        roots=seed[a+'_actual_full_position_target_seed'][0,rootids];ci=params[a+'_controlled_indices'].ravel()
        bank[a+'_full50_root_indices']=rootids;bank[a+'_full50_issued_target']=np.repeat(roots[None],1024,axis=0)
        bank[a+'_raw_requested_target']=np.repeat(seed[a+'_actual_full_position_target_seed'][0,ci][None],1024,axis=0)
        bank[a+'_full50_requested_velocity']=np.zeros((1024,len(rootids)),np.float32)
    bankfile=K/'NATIVE118_INITIAL_TASK_BANK_V1.npz';np.savez_compressed(bankfile,**bank)
    new=copy.deepcopy(reg);new.update(attempt_version=118,scientific_candidate_version=118,registered_utc=plan['created_utc'],
        maximum_controls=1024,change_scope='Contact-driven dual-holder friction task from native105 terminal seed, admitted source IK route and two destination supports; original motors/solver/gates unchanged',
        independent_capture_auditor=str(auditor),private_asset_settings=dict(reg['private_asset_settings'],private_directory=str(P/'private_kit_user_data')))
    new['contact_reproducer']=dict(reg['contact_reproducer'],fixture=str(K/'PAIR_TASK_NATIVE_SEED_V1.npz'),fixture_sha256=sha(K/'PAIR_TASK_NATIVE_SEED_V1.npz'),source_event=8191,maximum_physics_events=32768,no_task_or_new_random_input=False)
    new['pair_task']={'spec':admission['task'],'IK_endpoints':str(K/'PAIR_TASK_IK_ENDPOINTS_V2.npz'),'destination_support_posts':admission['destination_support_posts'],'original_acceptance_required':True}
    new['motion_test']=dict(reg['motion_test'],command_bank=str(bankfile),random_episode=None,synthetic_operator=True,hands='Adaptive contact pressure and actual opening, no attachment',family='pair_colift')
    new['actual_state_guard']['source']=str(P/'full50_actual_guard_v1.py');new['full50_guard']['source']=str(P/'full50_actual_guard_v1.py')
    sources={}
    for f in reg['source_sha256']:
        old=Path(f)
        if old.is_relative_to(B):dest=P/old.relative_to(B)
        elif old==H/'launch_native_clock_admission_v117.py':dest=launcher
        elif old==H/'audit_motion_capture117_v1.py':dest=auditor
        else:dest=old
        sources[str(dest)]=sha(dest)
    for f in [Path(__file__),planfile,bankfile,K/'PAIR_TASK_NATIVE_SEED_V1.npz',K/'PAIR_TASK_IK_ENDPOINTS_V2.npz',K/'PAIR_TASK_SCENE_ADMISSION_V2.json',K/'test_multitask_phases_v1.py',K/'test_opposed_contact_v1.py',K/'NATIVE117_CAPTURE_CAPACITY_COMPATIBILITY_V1.json']:
        sources[str(f)]=sha(f)
    for name in ['multitask_phases_v1.py','opposed_contact_v1.py','pair_task_operator_v1.py']:sources[str(P/name)]=sha(P/name)
    new['source_sha256']=sources;(P/'REGISTRATION_V1.json').write_text(json.dumps(new,indent=2)+'\n')
    print('NATIVE118_PAIR_TASK_REGISTERED;adaptive friction task;source-admitted route;one launch;no learnedSystem0 or acceptance claim',flush=True)


if __name__=='__main__':main()
