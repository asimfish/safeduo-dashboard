"""Real native braking counterfactual after no safe forecast; no simulation freeze."""
from pathlib import Path
import ast,copy,datetime,hashlib,json,shutil

D=Path(__file__).resolve().parent
H=Path('/home/liyufeng/safeduo/artifacts/safety_physical_qualified_random_20261009_1613')
B=H/'fourhand_native_probe_v112';P=H/'fourhand_native_probe_v113'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def once(s,old,new):
    assert s.count(old)==1,old[:80]
    return s.replace(old,new)


def main():
    assert not P.exists()
    resource=read(B/'RESOURCE_EXECUTION_V1.json');raw=Path(resource['out']);protocol=read(raw/'fourhand_protocol.json')
    assert resource['actual_exit']==resource['actual_native_exit']==0 and resource['native_owned_process_reaped_and_absent']
    assert protocol['physics_events']==1472 and protocol['task_status']=='PHYSICAL_GATE_FAILED'
    reg=read(B/'REGISTRATION_V1.json')
    for f,d in reg['source_sha256'].items():assert sha(f)==d,f
    plan={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'candidate':113,'baseline':112,'maximum_native_launches':1,'maximum_native_wall_s':1200,
        'maximum_controls':128,'maximum_microsteps':4096,'physics_hz':1920,'controller_hz':60,'FIFO_controls':6,
        'cause':'112 continued a smooth target after UNQUALIFIED_BRAKE began at control15 and failed proxy gap at event1471. No safe candidate must not be treated as permission to continue that candidate.',
        'intervention':'Latch emergency at first UNQUALIFIED_BRAKE. Every later actual physical microstep commands all50 independent motors to their current measured positions using the unchanged coupled implicit motor damping/gravity controller. Continue original simulation and all point/q/qd captures to the full horizon. Never freeze or write q/qd/object state.',
        'nominal_command_slew_override_during_emergency_explicit':True,
        'native_original_joint_speed_limits_and_all_other_gates_unchanged':True,
        'successful_brake_observer':'Measured all74 native speed<=0.02rad/s continuously0.3s after trigger, every original gate including mimic q AND qd passes; record latency, movement and minimum gap throughout.',
        'no_resume_in_this_calibration':True,'infeasible_intent_safety_and_actual_movement_scored_separately':True,
        'input_random_arm_dimensions':26,'input_random_hand_dimensions':0,'brake_actuated_motor_dimensions':50,
        'hardware_or_general_braking_bound_not_proven_by_one_case':True,
        'old_failed_cases_retained':True,'formal_holdout_count':0,'new_training_jobs':0,
        'learnedSystem0_active':False,'fullSystem0_accepted':False}
    planfile=D/'NATIVE113_BRAKE_PROSPECTIVE_PLAN_V1.json';planfile.write_text(json.dumps(plan,indent=2)+'\n')
    P.mkdir()
    for f in B.iterdir():
        if f.suffix in ('.py','.usda') or f.name=='PRIVATE_KIT_USER_CONFIG_FROZEN_V1.json':shutil.copyfile(f,P/f.name)
    shutil.copytree(B/'private_kit_user_data',P/'private_kit_user_data')
    shutil.copyfile(D/'emergency_brake_latch_v1.py',P/'emergency_brake_latch_v1.py')
    worker=P/'native_common_beam_v1.py';s=worker.read_text()
    old="        capture(0)\n        for step in range(reg['maximum_controls']):"
    s=once(s,old,"        from emergency_brake_latch_v1 import EmergencyBrakeLatch\n        brake_latch=EmergencyBrakeLatch()\n"+old)
    old="            for a in ARMS:\n                submitted=array(applied[a])[0].copy()"
    s=once(s,old,"            emergency_active=brake_latch.update(step,guard_decision.mode if guard_decision else 'COMMON_PREFIX')\n"+old)
    old="                effort={};gravity_raw={};motor_q_before={};motor_v_before={};motor_pd={};native_body_transforms_before={};full_motor_mass_before={};full_motor_target_before={};native_armature_before={};native_coriolis_before={} "
    new="""                if emergency_active:
                    for a in ARMS:
                        view=env._arms[a].root_physx_view
                        braking_target=array(view.get_dof_positions())[0].copy()
                        # Only motor targets change; followers remain unactuated.
                        env._arms[a].set_joint_position_target(torch.as_tensor(braking_target,device=env.device)[None])
                        applied[a]=torch.as_tensor(braking_target[parameters[a+'_controlled_indices']],device=env.device)[None]
                        nominal_hand_targets[a]=braking_target[hands.ids[a][0]].copy()
                        filtered_targets[a]=nominal_hand_targets[a].copy()
"""+old
    s=once(s,old,new)
    s=once(s,"data.update(guard_mode=np.asarray(guard_decision.mode if guard_decision else 'COMMON_PREFIX'),",
        "data.update(emergency_brake_active=np.asarray(emergency_active),\n                    emergency_trigger_control=np.asarray(brake_latch.trigger_control if brake_latch.trigger_control is not None else -1),\n                    guard_mode=np.asarray('EMERGENCY_FULL50_BRAKE' if emergency_active else guard_decision.mode if guard_decision else 'COMMON_PREFIX'),")
    old="            completed=step+1"
    new="""            # Bind next control's previous command to the final actual submission.
            if emergency_active:
                for a in ARMS:
                    submitted=array(env._arms[a].root_physx_view.get_dof_position_targets())[0,parameters[a+'_controlled_indices']].copy()
                    guard_previous_velocity[a]=(submitted.astype(float)-guard_prior_target[a].astype(float))/dt
                    guard_previous_target[a]=submitted
"""+old
    s=once(s,old,new)
    ast.parse(s);worker.write_text(s)
    launcher=H/'launch_native_clock_admission_v113.py'
    s=(H/'launch_native_clock_admission_v112.py').read_text().replace('v112','v113').replace('V112','V113').replace('NATIVE112','NATIVE113')
    ast.parse(s);launcher.write_text(s)
    auditor=H/'audit_motion_capture113_v1.py'
    s=(H/'audit_motion_capture112_v1.py').read_text().replace('assert version == 112','assert version == 113')
    ast.parse(s);auditor.write_text(s)
    new=copy.deepcopy(reg)
    new.update(attempt_version=113,scientific_candidate_version=113,registered_utc=plan['utc'],
        execution_only_retry=False,execution_replica_of_version=None,
        change_scope=plan['intervention'],independent_capture_auditor=str(auditor),
        private_asset_settings=dict(reg['private_asset_settings'],private_directory=str(P/'private_kit_user_data')),
        emergency_brake=plan)
    new['actual_state_guard']['source']=str(P/'actual_state_arm_guard_v1.py')
    sources={}
    for file in reg['source_sha256']:
        old=Path(file)
        if old.is_relative_to(B):dest=P/old.relative_to(B)
        elif old==H/'launch_native_clock_admission_v112.py':dest=launcher
        elif old==H/'audit_motion_capture112_v1.py':dest=auditor
        else:dest=old
        sources[str(dest)]=sha(dest)
    for f in (Path(__file__),planfile,P/'emergency_brake_latch_v1.py',raw/'fourhand_protocol.json'):
        sources[str(f)]=sha(f)
    new['source_sha256']=sources
    assert new['native_sweep_spike']==reg['native_sweep_spike']
    assert new['full_motor_effort_controller']==reg['full_motor_effort_controller']
    (P/'REGISTRATION_V1.json').write_text(json.dumps(new,indent=2)+'\n')
    print('NATIVE113_REGISTERED;actual full50 braking after failed forecast;4096 measured steps;no simulation freeze;not movement acceptance',flush=True)


if __name__=='__main__':main()
