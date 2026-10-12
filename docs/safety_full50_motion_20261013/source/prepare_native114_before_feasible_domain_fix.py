"""One predeclared full50 random motor intent diagnostic with real FIFO6."""
from pathlib import Path
import ast,copy,datetime,hashlib,json,shutil,sys
import numpy as np

E=Path(__file__).resolve().parent
H=Path('/home/liyufeng/safeduo/artifacts/safety_physical_qualified_random_20261009_1613')
B=H/'fourhand_native_probe_v113';P=H/'fourhand_native_probe_v114'
L=Path('/mnt/nas/data/lyf/double_hand/safeduo_long_random_motion_20261012')
ARMS=('F_L','F_R','U_L','U_R')
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def once(s,old,new):
    assert s.count(old)==1,(s.count(old),old[:90])
    return s.replace(old,new)


def main():
    assert not P.exists()
    assert read(E/'FULL74_GEOMETRY_REPLAY_AUDIT_V2.json')['all_recorded_state_accuracy_gates_pass']
    assert read(B/'INDEPENDENT_EMERGENCY_BRAKE_V1.json')['measured_emergency_brake_case_pass']
    reg=read(B/'REGISTRATION_V1.json')
    for f,h in reg['source_sha256'].items():assert sha(f)==h,f
    raw=Path(read(B/'RESOURCE_EXECUTION_V1.json')['out'])
    with np.load(raw/'native_parameters.npz') as z:params={k:z[k] for k in z.files}
    meta=read(reg['coupled_assets'])['assets']
    with np.load(reg['contact_reproducer']['fixture']) as z:seed={k:z[k] for k in z.files}
    sys.path.insert(0,str(L));from long_motion_stream import load_domain,MotionStream
    domain=load_domain();master_seed=202610130114;stream=MotionStream(domain,'iid_every_control',master_seed,128/60.)
    packets=next(stream.chunks(129));assert packets['requested_q'].shape==(129,50)
    fullnames=domain.names.tolist();bank={};roots={};positions={}
    for a in ARMS:
        names=params[a+'_joint_names'].tolist();followers={r['slave'] for r in meta[a]['relations']}
        roots[a]=np.array([i for i,n in enumerate(names) if n not in followers],int)
        positions[a]=np.array([list(roots[a]).index(int(i)) for i in params[a+'_controlled_indices']],int)
        columns=np.array([fullnames.index(a+'/'+names[i]) for i in roots[a]],int)
        np.testing.assert_array_equal(domain.lower[columns],params[a+'_hard_limits'][0,roots[a],0])
        np.testing.assert_array_equal(domain.upper[columns],params[a+'_hard_limits'][0,roots[a],1])
        targets=packets['requested_q'][:128,columns].astype(np.float32)
        targets[:4]=seed[a+'_actual_full_position_target_seed'][0,roots[a]]
        bank[a+'_full50_issued_target']=targets;bank[a+'_full50_requested_velocity']=packets['requested_velocity'][:128,columns]
        bank[a+'_full50_root_indices']=roots[a];bank[a+'_full50_root_names']=np.asarray([names[i] for i in roots[a]])
        bank[a+'_issued_target']=targets[:,positions[a]];bank[a+'_raw_requested_target']=targets[:,positions[a]].copy()
    bankfile=E/'NATIVE114_FULL50_COMMAND_BANK_V1.npz';np.savez_compressed(bankfile,**bank)
    plan={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'candidate':114,'baseline':113,
        'family':'iid_every_control','development_seed':master_seed,'maximum_controls':128,'maximum_microsteps':4096,
        'physics_hz':1920,'control_hz':60,'FIFO_controls':6,'maximum_native_launches':1,'maximum_owned_wall_s':1200,
        'first4_controls_original_common_seed_tape':True,'independent_random_dimensions':50,'arm_dimensions':26,'finger_motor_dimensions':24,
        'intent_range':'Every original per-motor joint interval; proposals are simulation-only, not commands directly safe for hardware',
        'unselected_draws_after_first4_controls':124,'proposal_rate_rad_s':.1,'proposal_acceleration_rad_s2':.5,
        'intervention':'all74 predicted moving-finger geometry after real full50 FIFO6; first UNQUALIFIED_BRAKE latches original113 full50 damping brake; native physics remains active',
        'original_kernel_bridge_masses_effort_gains_and_physical_q_qd_gates_unchanged':True,
        'hand_motors_are_now_random_intent_inputs_not_only_brake_actuators':True,
        'original_stop_observer_all74_speed_cap_rad_s':.02,'original_stop_dwell_s':.3,
        'diagnostic_movement_measurement':'Each50 motor actual range, departure, speed and each4 actual hand mount path; original arm .1rad/.02m movement admission preserved. Report request support separately.',
        'future_tracking_or_braking_envelope_certified':False,'native30_120_600s_completed':0,
        'formal_holdouts':0,'new_learning_training_jobs':0,'genuine_task_acceptance_unchanged':True,'fullSystem0_accepted':False}
    planfile=E/'NATIVE114_FULL50_PROSPECTIVE_PLAN_V1.json';planfile.write_text(json.dumps(plan,indent=2)+'\n')
    P.mkdir()
    for f in B.iterdir():
        if f.suffix in ('.py','.usda') or f.name=='PRIVATE_KIT_USER_CONFIG_FROZEN_V1.json':shutil.copyfile(f,P/f.name)
    shutil.copytree(B/'private_kit_user_data',P/'private_kit_user_data')
    for name in ['full74_geometry_v2.py','hand_kinematic_replay_v1.py','full50_actual_guard_v1.py']:shutil.copyfile(E/name,P/name)
    worker=P/'native_common_beam_v1.py';s=worker.read_text()
    start=s.index('        from actual_state_arm_guard_v1 import ActualStateArmGuard')
    stop=s.index('        from emergency_brake_latch_v1 import EmergencyBrakeLatch',start)
    s=s[:start]+'''        from full74_geometry_v2 import Full74Geometry
        from full50_actual_guard_v1 import Full50ActualGuard
        full50_ids={a:np.asarray(motor_control[a]['rootids'],int) for a in ARMS}
        arm_in_full50={a:np.asarray([list(full50_ids[a]).index(int(i)) for i in parameters[a+'_controlled_indices']],int) for a in ARMS}
        for a in ARMS:np.testing.assert_array_equal(full50_ids[a],motion_bank[a+'_full50_root_indices'])
        full74_geometry=Full74Geometry(parameters,coupled_meta['assets'])
        actual_state_guard=Full50ActualGuard(full74_geometry,parameters,full50_ids,**reg['full50_guard']['settings'])
        pending=deque({a:torch.as_tensor(repro[a+'_actual_full_position_target_seed'][0,full50_ids[a]],device=env.device)[None] for a in ARMS} for _ in range(6))
        guard_previous_target={a:array(env._arms[a].root_physx_view.get_dof_position_targets())[0,full50_ids[a]].copy() for a in ARMS}
        guard_previous_velocity={a:np.zeros_like(guard_previous_target[a],dtype=float) for a in ARMS}
'''+s[stop:]
    s=once(s,"motion_bank[a+'_issued_target'][step],device=env.device,dtype=initial[a].dtype)","motion_bank[a+'_full50_issued_target'][step],device=env.device,dtype=initial[a].dtype)")
    # Full50 source request flows through the existing actual FIFO without flush.
    start=s.index('            nominal_hand_targets={};filtered_targets={}')
    stop=s.index('                effort={};gravity_raw={}',start)
    s=s[:start]+'''            nominal_hand_targets={};filtered_targets={}
            for sub in range(substeps):
                tape_index=min((step*substeps+sub)//reg['execution_time_refinement_factor'],7)
                for a in ARMS:
                    view=env._arms[a].root_physx_view
                    if step<4:
                        full=repro[a+'_actual_full_position_target_tape'][tape_index].copy()
                        full[0,parameters[a+'_controlled_indices']]=array(applied[a])[0,arm_in_full50[a]]
                    elif emergency_active:
                        # The same existing50-root damping brake, never a state write.
                        full=array(view.get_dof_positions()).copy()
                    else:
                        current=array(view.get_dof_positions()).copy()
                        delta=array(applied[a])[0]-current[0,full50_ids[a]]
                        full=current+(delta@parameters[a+'_full_motor_velocity_mapping'].T)[None]
                        full[0,full50_ids[a]]=array(applied[a])[0]
                    env._arms[a].set_joint_position_target(torch.as_tensor(full,device=env.device))
                    if emergency_active:applied[a]=torch.as_tensor(full[0,full50_ids[a]],device=env.device)[None]
                    nominal_hand_targets[a]=full[0,hands.ids[a][0]].copy()
                    filtered_targets[a]=nominal_hand_targets[a].copy()
'''+s[stop:]
    # Keep original26 fields for the existing independent capture observer and
    # add full50 queue/submission fields for a distinct independent queue audit.
    s=once(s,"data[a+'_issued_target']=array(issued[a]);data[a+'_applied_target']=array(applied[a])",
        "data[a+'_issued_target']=array(issued[a])[:,arm_in_full50[a]];data[a+'_applied_target']=array(applied[a])[:,arm_in_full50[a]]")
    s=once(s,"data[a+'_fifo_applied_target']=array(fifo_applied[a])","data[a+'_fifo_applied_target']=array(fifo_applied[a])[:,arm_in_full50[a]]")
    s=once(s,"data[a+'_pending_fifo_targets']=np.stack([packet[a] for packet in guard_pending])","data[a+'_pending_fifo_targets']=np.stack([packet[a][arm_in_full50[a]] for packet in guard_pending])")
    s=once(s,"data[a+'_guard_previous_command_target']=guard_prior_target[a];data[a+'_guard_previous_command_velocity']=guard_prior_velocity[a]",
        """data[a+'_guard_previous_command_target']=guard_prior_target[a][arm_in_full50[a]];data[a+'_guard_previous_command_velocity']=guard_prior_velocity[a][arm_in_full50[a]]
                    data[a+'_full50_issued_target']=array(issued[a]);data[a+'_full50_fifo_applied_target']=array(fifo_applied[a])
                    data[a+'_full50_pending_fifo_targets']=np.stack([packet[a] for packet in guard_pending])
                    data[a+'_full50_previous_command_target']=guard_prior_target[a];data[a+'_full50_previous_command_velocity']=guard_prior_velocity[a]
                    data[a+'_full50_effective_target']=array(view.get_dof_position_targets())[:,full50_ids[a]]
                    data[a+'_full50_root_indices']=full50_ids[a]
                    data[a+'_full50_requested_velocity']=motion_bank[a+'_full50_requested_velocity'][step]""")
    s=once(s,"            if emergency_active:\n                for a in ARMS:\n                    submitted=array(env._arms[a].root_physx_view.get_dof_position_targets())[0,parameters[a+'_controlled_indices']].copy()",
        "            if True:\n                for a in ARMS:\n                    submitted=array(env._arms[a].root_physx_view.get_dof_position_targets())[0,full50_ids[a]].copy()")
    s=s.replace("method='RANDOM26_ACTUAL_STATE_SLEW_POST_FIFO_GUARD'","method='RANDOM_FULL50_ACTUAL74_FINGER_GEOMETRY_FIFO6_GUARD'")
    s=s.replace("Raw frozen random26 intents through actualFIFO6 and online actual-state/velocity/queue guard; original full50 effort and held hand feedback; deterministic candidate, not learnedSystem0",
        "Fresh developmentIID50 motor intents through actual full50FIFO6; moving-finger actual74 geometry and latched native50 brake; original physics and motor effort; not learnedSystem0")
    s=s.replace("np.asarray('RANDOM_ARM_ACTUAL_STATE_GUARD')","np.asarray('RANDOM_FULL50_ACTUAL74_FINGER_GEOMETRY')")
    ast.parse(s);worker.write_text(s)
    launcher=H/'launch_native_clock_admission_v114.py';s=(H/'launch_native_clock_admission_v113.py').read_text().replace('v113','v114').replace('V113','V114').replace('NATIVE113','NATIVE114');ast.parse(s);launcher.write_text(s)
    auditor=H/'audit_motion_capture114_v1.py';s=(H/'audit_motion_capture113_v1.py').read_text().replace('assert version == 113','assert version == 114');ast.parse(s);auditor.write_text(s)
    new=copy.deepcopy(reg);new.update(attempt_version=114,scientific_candidate_version=114,registered_utc=plan['created_utc'],
        execution_only_retry=False,execution_replica_of_version=None,change_scope=plan['intervention'],
        independent_capture_auditor=str(auditor),private_asset_settings=dict(reg['private_asset_settings'],private_directory=str(P/'private_kit_user_data')),
        motion_test=dict(reg['motion_test'],command_bank=str(bankfile),random_episode='full50IIDdev202610130114'),
        full50_guard={'source':str(P/'full50_actual_guard_v1.py'),'settings':{'rate':.1,'acceleration':.5,'reserve_m':.0005,'horizon_s':.12}},full50_input=plan)
    sources={}
    for file in reg['source_sha256']:
        old=Path(file)
        if old.is_relative_to(B):dest=P/old.relative_to(B)
        elif old==H/'launch_native_clock_admission_v113.py':dest=launcher
        elif old==H/'audit_motion_capture113_v1.py':dest=auditor
        else:dest=old
        sources[str(dest)]=sha(dest)
    for f in [Path(__file__),planfile,bankfile,E/'FULL74_GEOMETRY_REPLAY_AUDIT_V2.json',E/'GEOMETRY_ACTUAL_EXITS_V1.json',L/'long_motion_stream.py',*map(lambda n:P/n,['full74_geometry_v2.py','hand_kinematic_replay_v1.py','full50_actual_guard_v1.py'])]:sources[str(f)]=sha(f)
    new['source_sha256']=sources
    assert new['native_sweep_spike']==reg['native_sweep_spike'] and new['full_motor_effort_controller']==reg['full_motor_effort_controller']
    (P/'REGISTRATION_V1.json').write_text(json.dumps(new,indent=2)+'\n')
    print('NATIVE114_REGISTERED;full50IID source;all74 moving geometry;FIFO6;128controls4096steps;one launch;no general safety acceptance',flush=True)


if __name__=='__main__':main()
