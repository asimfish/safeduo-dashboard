"""Audit original safety AND distinguish incoming random fingers from braking."""
from pathlib import Path
import ast,collections,hashlib,json
import numpy as np
from full74_geometry_v2 import Full74Geometry,ARMS
from hand_kinematic_replay_v1 import quaternion_matrix

H=Path('/home/liyufeng/safeduo/artifacts/safety_physical_qualified_random_20261009_1613')
E=Path(__file__).resolve().parent
D=Path('/mnt/nas/data/lyf/double_hand/safeduo_actual_state_guard_20261012')
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=H/'fourhand_native_probe_v115';output=p/'INDEPENDENT_FULL50_RANDOM_MOTION_V1.json';assert not output.exists()
    reg=read(p/'REGISTRATION_V1.json');resource=read(p/'RESOURCE_EXECUTION_V1.json');raw=Path(resource['out'])
    capture=read(p/'INDEPENDENT_CAPTURE_AUDIT_V1.json');model=read(p/'INDEPENDENT_NATIVE_MIMIC_WINDOW_V6.json')
    motor=read(p/'INDEPENDENT_FULL_MOTOR_EFFORT_V4.json');camera=read(p/'INDEPENDENT_CAMERA_FRAMING_V2.json')
    assert resource['actual_exit']==resource['actual_native_exit']==0 and resource['native_owned_process_reaped_and_absent']
    assert capture['record_integrity_verified'] and sum(r['independent_motors'] for r in motor['rows'])==50
    with np.load(raw/'native_parameters.npz') as z:params={k:z[k] for k in z.files}
    baseline=Path(read(H/'fourhand_native_probe_v113/RESOURCE_EXECUTION_V1.json')['out'])
    with np.load(baseline/'native_parameters.npz') as z:
        assert set(params)==set(z.files) and len(params)==77
        for k in params:np.testing.assert_array_equal(params[k],z[k])
    with np.load(raw/'actual_initial_seed_binding.npz') as z,np.load(baseline/'actual_initial_seed_binding.npz') as b:
        assert set(z.files)==set(b.files) and len(z.files)==13
        for k in z.files:np.testing.assert_array_equal(z[k],b[k])
    with np.load(reg['motion_test']['command_bank']) as z:bank={k:z[k] for k in z.files}
    with np.load(reg['contact_reproducer']['fixture']) as z:seed={k:z[k] for k in z.files}
    roots={a:bank[a+'_full50_root_indices'] for a in ARMS};meta=read(reg['coupled_assets'])['assets'];geometry=Full74Geometry(params,meta)
    pieces=[];sources={};max_gap_error=0.;records=read(raw/'physics_chunks.json')
    needed=['physics_event','state_time_s','guard_mode','guard_prediction_qualified','guard_decision_actual_geometry','all9021_raw_geometry','emergency_brake_active','emergency_trigger_control']
    fields=['_native_q','_native_qd','_guard_decision_q','_guard_decision_qd','_actual_full_position_target','_motor_control_q_before','_body_position_local',
            '_full50_issued_target','_full50_fifo_applied_target','_full50_pending_fifo_targets','_full50_previous_command_target','_full50_previous_command_velocity','_full50_effective_target','_full50_root_indices']
    for c in records:
        f=raw/c['path'];assert sha(f)==c['sha256'];sources[str(f)]=sha(f)
        with np.load(f) as z:
            q={a:z[a+'_native_q'][:,0] for a in ARMS};gap=geometry.gaps(q)
            max_gap_error=max(max_gap_error,float(abs(gap-z['all9021_raw_geometry'][:,0]).max()))
            pieces.append({k:z[k] for k in needed+[a+s for a in ARMS for s in fields]})
            if c['stop']<=128:
                with np.load(baseline/c['path']) as old:
                    # Physical states/force/material/clock fields of the shared
                    # prefix. New method labels and queue dimensionality differ.
                    for k in old.files:
                        if any(s in k for s in ['guard','target','motion_family','emergency']):continue
                        np.testing.assert_array_equal(z[k],old[k])
    data={k:np.concatenate([part[k] for part in pieces]) for k in pieces[0]};n=len(data['physics_event']);substeps=reg['physics_substeps_per_control'];controls=n//substeps
    assert n==capture['physics_events'] and substeps==32 and not data['guard_prediction_qualified'].any()
    pending=[{a:seed[a+'_actual_full_position_target_seed'][0,roots[a]].copy() for a in ARMS} for _ in range(6)]
    for step in range(controls):
        e=step*substeps;pending.append({a:bank[a+'_full50_issued_target'][step] for a in ARMS});applied=pending.pop(0)
        for a in ARMS:
            np.testing.assert_array_equal(data[a+'_full50_root_indices'][e:e+substeps],np.repeat(roots[a][None],substeps,axis=0))
            np.testing.assert_array_equal(data[a+'_full50_issued_target'][e:e+substeps,0],np.repeat(bank[a+'_full50_issued_target'][step][None],substeps,axis=0))
            np.testing.assert_array_equal(data[a+'_full50_fifo_applied_target'][e:e+substeps,0],np.repeat(applied[a][None],substeps,axis=0))
            np.testing.assert_array_equal(data[a+'_full50_pending_fifo_targets'][e:e+substeps],np.repeat(np.stack([packet[a] for packet in pending])[None],substeps,axis=0))
            np.testing.assert_array_equal(data[a+'_full50_effective_target'][e:e+substeps],data[a+'_actual_full_position_target'][e:e+substeps,:,roots[a]])
            if step:
                np.testing.assert_array_equal(data[a+'_guard_decision_q'][e:e+substeps],np.repeat(data[a+'_native_q'][e-1],substeps,axis=0))
                np.testing.assert_array_equal(data[a+'_guard_decision_qd'][e:e+substeps],np.repeat(data[a+'_native_qd'][e-1],substeps,axis=0))
                np.testing.assert_array_equal(data[a+'_full50_previous_command_target'][e],data[a+'_full50_effective_target'][e-1,0])
        if step:np.testing.assert_array_equal(data['guard_decision_actual_geometry'][e:e+substeps],np.repeat(data['all9021_raw_geometry'][e-1,0][None],substeps,axis=0))
    active=data['emergency_brake_active'];trigger=np.flatnonzero(active);brake=None
    if len(trigger):
        event=int(trigger[0]);assert active[event:].all() and not active[:event].any()
        sys_path=str(D)
        import sys
        sys.path.insert(0,sys_path);from brake_measurements_v1 import settling_window
        speeds=np.max(np.concatenate([abs(data[a+'_native_qd'][:,0]) for a in ARMS],axis=1),axis=1)
        settle=settling_window(data['state_time_s'],speeds,event*reg['physics_dt_s'])
        brake={'first_control':event//substeps,'first_event':event,'trigger_time_s':event*reg['physics_dt_s'],'settling':settle,'minimum_gap_after_trigger_m':float(data['all9021_raw_geometry'][event:].min())}
        for a in ARMS:np.testing.assert_array_equal(data[a+'_actual_full_position_target'][active],data[a+'_motor_control_q_before'][active])
    worker=p/'native_common_beam_v1.py';s=worker.read_text();line=s[:s.index('        capture(0)')].count('\n')+1
    for node in ast.walk(ast.parse(s)):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.lineno>line:
            assert not node.func.attr.startswith(('write_joint','write_root')) and node.func.attr not in ('set_joint_positions','set_joint_velocities','set_root_state')
    rows=[];arm_rows=[];modes=data['guard_mode'][::substeps];ordinary=(np.arange(controls)>=10)&(modes!='EMERGENCY_FULL50_BRAKE')
    ordinary_steps=np.flatnonzero(ordinary);normal_events=np.repeat(ordinary,substeps)
    fingers_updated=0
    for a in ARMS:
        names=params[a+'_joint_names'].tolist();controlled=set(params[a+'_controlled_indices'].tolist());initial=params[a+'_initial_q'][0]
        for i in roots[a]:
            j=list(roots[a]).index(int(i));name=names[i];q=data[a+'_native_q'][:,0,i];cmd=data[a+'_full50_effective_target'][:,0,j]
            request=bank[a+'_full50_issued_target'][4:controls,j]
            isfinger=int(i) not in controlled
            change=float(abs(np.diff(cmd[normal_events])).max()) if normal_events.sum()>1 else 0.
            if isfinger and change>1e-7:fingers_updated+=1
            rows.append({'coordinate':a+'/'+name,'finger_motor':isfinger,'incoming_request_span_rad':float(np.ptp(request)) if len(request) else 0.,
                'actual_whole_record_span_rad':float(np.ptp(np.r_[initial[i],q])),
                'maximum_actual_departure_rad':float(abs(q-initial[i]).max()),'maximum_actual_speed_rad_s':float(abs(data[a+'_native_qd'][:,0,i]).max()),
                'ordinary_random_effective_target_span_rad':float(np.ptp(cmd[normal_events])) if normal_events.any() else 0.,
                'actual_span_during_ordinary_random_execution_rad':float(np.ptp(q[normal_events])) if normal_events.any() else 0.,
                'actual_span_is_not_attributed_wholly_to_randomness':True})
        ids=params[a+'_controlled_indices'];mount=('left_hand_base' if a.endswith('L') else 'right_hand_base') if a.startswith('F') else 'wrist_3_link'
        xyz=data[a+'_body_position_local'][:,0,params[a+'_body_names'].tolist().index(mount)]
        path=float(np.linalg.norm(np.diff(xyz.astype(float),axis=0),axis=1).sum());depart=float(abs(data[a+'_native_q'][:,0,ids]-initial[ids]).max())
        arm_rows.append({'arm':a,'mount_path_m':path,'max_actual_departure_rad':depart,'original_arm_movement_admission_pass':path>=.02 and depart>=.1})
    tolerance=model['predeclared_relation_tolerance'];relation=model['maximum_q_residual_rad']<=tolerance and model['maximum_qd_residual_rad_s']<=tolerance
    complete=n==reg['expected_maximum_physics_events'];finite_pass=complete and capture['physical_gates_pass'] and relation
    report={'version':115,'status':'CLOSED_FULL50_FINGER_MOTION_NATIVE_DEVELOPMENT_DIAGNOSTIC','physics_events':n,'duration_s':n*reg['physics_dt_s'],
        'full_preregistered_window_complete':complete,'all_original_physical_gates_pass':capture['physical_gates_pass'],
        'original_q_AND_qd_residuals_pass':relation,'maximum_qd_residual_rad_s':model['maximum_qd_residual_rad_s'],
        'finite_preregistered_physical_and_model_pass':bool(finite_pass),'original_all4_arm_movement_admission_pass':all(r['original_arm_movement_admission_pass'] for r in arm_rows),
        'genuine_full_task_pass':capture['four_hand_task_development_pass'],'native_minimum_gap_m':capture['minimum_all9021_gap_m'],
        'maximum_full74_geometry_pair_error_m_on_new_native_states':max_gap_error,'registered_recorded_state_geometry_accuracy_pass':max_gap_error<=.0001,
        'incoming_independent_motor_dimensions':50,'incoming_finger_motor_dimensions':24,'incoming_new_IID_controls':max(0,controls-4),
        'new_IID_packets_delivered_by_FIFO_controls':max(0,controls-10),'ordinary_controls_after_new_random_FIFO_delivery':int(ordinary.sum()),
        'ordinary_finger_motor_target_dimensions_updated':fingers_updated,'guard_control_counts':dict(collections.Counter(modes.tolist())),
        'all50_FIFO6_and_targets_native_submission_exact':True,'all74_q_qd_and9021_actual_state_custody_verified':True,
        'all77_native_parameters_and13_seed_fields_byte_equal113':True,'first128_physical_prefix_byte_equal113':True,
        'no_post_seed_articulation_state_write':True,'brake':brake,'motor_rows':rows,'arm_rows':arm_rows,
        'whole_record_movement_includes_common_prefix_and_natural_dynamics':True,'braking_stop_does_not_count_as_broad_random_motion':True,
        'full50_reachable_operating_domain_not_qualified':True,'random_workspace_coverage_pass':False,
        'native30_120_600s_completed':0,'independent_initial_states':1,'formal_holdouts':0,'new_training_jobs':0,'fullSystem0_accepted':False,
        'source_sha256':{**sources,str(Path(__file__)):sha(__file__),str(worker):sha(worker),str(E/'full74_geometry_v2.py'):sha(E/'full74_geometry_v2.py')}}
    output.write_text(json.dumps(report,indent=2)+'\n')
    print('FULL50_NATIVE_CLOSED','finite_model_physical',finite_pass,'finger_target_dimensions',fingers_updated,'ordinary_random_controls',int(ordinary.sum()),'geometry_error_m',max_gap_error,'brake',json.dumps(brake),flush=True)


if __name__=='__main__':main()
