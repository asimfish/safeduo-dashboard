"""Independent capture/command custody and measured motion; original gates stay separate."""
from pathlib import Path
import argparse,collections,hashlib,json
import numpy as np

H=Path('/home/liyufeng/safeduo/artifacts/safety_physical_qualified_random_20261009_1613')
ARMS=('F_L','F_R','U_L','U_R')
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(version):
    p=H/f'fourhand_native_probe_v{version}'
    output=p/'INDEPENDENT_ACTUAL_STATE_GUARD_MOTION_V2.json';assert not output.exists()
    reg=read(p/'REGISTRATION_V1.json');r=read(p/'RESOURCE_EXECUTION_V1.json');raw=Path(r['out'])
    clock_refined=reg['physics_dt_s']==1/1920. and reg['physics_substeps_per_control']==32
    capture=read(p/'INDEPENDENT_CAPTURE_AUDIT_V1.json')
    model=read(p/'INDEPENDENT_NATIVE_MIMIC_WINDOW_V6.json')
    effort=read(p/'INDEPENDENT_FULL_MOTOR_EFFORT_V4.json')
    camera=read(p/'INDEPENDENT_CAMERA_FRAMING_V2.json')
    assert r['actual_exit']==r['actual_native_exit']==0 and r['native_owned_process_reaped_and_absent']
    assert capture['record_integrity_verified'] and effort['physics_events']==capture['physics_events']
    baseline=Path(read(H/'fourhand_native_probe_v105/RESOURCE_EXECUTION_V1.json')['out'])
    with np.load(raw/'native_parameters.npz') as z:params={k:z[k] for k in z.files}
    with np.load(baseline/'native_parameters.npz') as z:
        assert len(z.files)==77 and set(z.files)==set(params)
        declared_clock_changes={'actual_physics_dt_s'} if clock_refined else set()
        changed={k for k in params if not np.array_equal(params[k],z[k])}
        assert changed==declared_clock_changes,changed
        for k in params:
            if k not in declared_clock_changes:np.testing.assert_array_equal(params[k],z[k])
    with np.load(raw/'actual_initial_seed_binding.npz') as z,np.load(baseline/'actual_initial_seed_binding.npz') as b:
        assert len(z.files)==13 and set(z.files)==set(b.files)
        for k in z.files:np.testing.assert_array_equal(z[k],b[k])
    parts=[];sources={};records=read(raw/'physics_chunks.json')
    for c in records:
        f=raw/c['path'];assert sha(f)==c['sha256'];sources[str(f)]=sha(f)
        with np.load(f) as z:
            parts.append({k:z[k].copy() for k in z.files if k in ('physics_event','state_time_s','guard_mode','guard_alpha','guard_forecast_minimum_m','guard_decision_actual_geometry','guard_prediction_qualified','all9021_raw_geometry') or
                any(k==a+s for a in ARMS for s in ('_native_q','_native_qd','_applied_target','_issued_target','_fifo_applied_target','_pending_fifo_targets','_guard_decision_q','_guard_decision_qd','_body_position_local','_guard_previous_command_target','_guard_previous_command_velocity','_actual_full_position_target'))})
    data={k:np.concatenate([x[k] for x in parts]) for k in parts[0]}
    substeps=reg['physics_substeps_per_control']
    n=len(data['physics_event']);assert n==capture['physics_events'] and n%substeps==0
    assert not data['guard_prediction_qualified'].any()
    with np.load(reg['contact_reproducer']['fixture']) as z:seed={k:z[k] for k in z.files}
    with np.load(reg['motion_test']['command_bank']) as z:bank={k:z[k] for k in z.files}
    pending=[{a:seed[a+'_actual_full_position_target_seed'][0,params[a+'_controlled_indices']].copy() for a in ARMS} for _ in range(6)]
    for step in range(n//substeps):
        e=step*substeps;pending.append({a:bank[a+'_issued_target'][step] for a in ARMS});applied=pending.pop(0)
        for a in ARMS:
            np.testing.assert_array_equal(data[a+'_issued_target'][e:e+substeps,0],np.repeat(bank[a+'_issued_target'][step][None],substeps,axis=0))
            np.testing.assert_array_equal(data[a+'_fifo_applied_target'][e:e+substeps,0],np.repeat(applied[a][None],substeps,axis=0))
            np.testing.assert_array_equal(data[a+'_pending_fifo_targets'][e:e+substeps],np.repeat(np.stack([row[a] for row in pending])[None],substeps,axis=0))
            if step:
                np.testing.assert_array_equal(data[a+'_guard_decision_q'][e:e+substeps],np.repeat(data[a+'_native_q'][e-1],substeps,axis=0))
                np.testing.assert_array_equal(data[a+'_guard_decision_qd'][e:e+substeps],np.repeat(data[a+'_native_qd'][e-1],substeps,axis=0))
                if version in (110,111,112):
                    np.testing.assert_array_equal(data[a+'_guard_previous_command_target'][e],data[a+'_applied_target'][e-1,0])
        if step:
            np.testing.assert_array_equal(data['guard_decision_actual_geometry'][e:e+substeps],np.repeat(data['all9021_raw_geometry'][e-1],substeps,axis=0))
    # Changed integration clock must not masquerade as byte-equal physics.
    if not clock_refined:
        for c in records:
            if c['stop']>64:break
            with np.load(raw/c['path']) as new,np.load(baseline/c['path']) as old:
                for k in old.files:
                    if k=='motion_family':continue
                    assert new[k].dtype==old[k].dtype and new[k].shape==old[k].shape and new[k].tobytes()==old[k].tobytes(),(c['start'],k)
    else:
        assert reg['physics_dt_s']==1/1920. and substeps==32
        # Check all hand target samples at their original physical times.
        for a in ARMS:
            handids=np.asarray([i for i in range(len(params[a+'_joint_names'])) if i not in params[a+'_controlled_indices']],int)
            expected=seed[a+'_actual_full_position_target_tape'][np.minimum(np.arange(n)//2,7),0][:,handids]
            np.testing.assert_array_equal(data[a+'_actual_full_position_target'][:,0,handids],expected)
    rows=[];admission=[]
    for a in ARMS:
        ids=params[a+'_controlled_indices'];initial=params[a+'_initial_q'][0,ids]
        q=data[a+'_native_q'][:,0,ids];names=params[a+'_body_names'].tolist()
        mount=('left_hand_base' if a.endswith('L') else 'right_hand_base') if a.startswith('F') else 'wrist_3_link'
        position=data[a+'_body_position_local'][:,0,names.index(mount)]
        path=float(np.linalg.norm(np.diff(position.astype(float),axis=0),axis=1).sum())
        depart=float(abs(q-initial).max());accepted=depart>=.1 and path>=.02;admission.append(accepted)
        cmd=data[a+'_applied_target'][::substeps,0].astype(float)
        start=seed[a+'_actual_full_position_target_seed'][0,ids].astype(float)
        velocity=np.diff(np.concatenate([start[None],cmd]),axis=0)*60.
        acceleration=np.diff(np.concatenate([np.zeros((1,len(ids))),velocity]),axis=0)*60.
        rows.append({'arm':a,'maximum_actual_departure_rad':depart,'mount_path_m':path,
                     'mount_xyz_span_m':np.ptp(position,axis=0).tolist(),
                     'actual_arm_joint_span_rad':np.ptp(np.concatenate([initial[None],q]),axis=0).tolist(),
                     'original_actual_movement_admission_pass':accepted,
                     'maximum_guard_command_slew_rad_s':float(abs(velocity).max()),
                     'maximum_guard_command_acceleration_rad_s2':float(abs(acceleration).max())})
    complete=n==reg['expected_maximum_physics_events']
    finite_ok=complete and capture['physical_gates_pass'] and model['model_window_pass']
    report={'status':'CLOSED_NATIVE_ACTUAL_STATE_GUARD_DEVELOPMENT_DIAGNOSTIC',
        'version':version,'native_actual_exit':0,'owned_native_reaped':True,'physics_events':n,
        'duration_s':n*reg['physics_dt_s'],'physics_hz':1/reg['physics_dt_s'],'full_preregistered_window_complete':complete,
        'original_physical_gates_pass':capture['physical_gates_pass'],
        'original_model_q_AND_qd_pass':model['model_window_pass'],
        'finite_preregistered_model_and_physical_pass':bool(finite_ok),
        'all4_actual_movement_admission_pass':bool(all(admission)),
        'operational_motion_diagnostic_pass':bool(finite_ok and all(admission)),
        'all77_physics_and13_initial_seed_fields_byte_equal105':not clock_refined,
        'declared_native_clock_parameter_changes':sorted(declared_clock_changes),
        'all_nonclock_native_properties_and13_initial_seed_fields_byte_equal105':True,
        'common64_original_physical_fields_byte_equal105':not clock_refined,
        'new_clock_hand_target_tape_original_time_binding_verified':clock_refined,
        'actual_q_qd_and9021_gaps_exactly_match_previous_native_event':True,
        'realFIFO6_and_raw_bank_reconstructed':True,'queue_never_flushed_or_relabelled':True,
        'guard_output_not_confused_with_FIFO_output':True,
        'guard_control_counts':dict(collections.Counter(data['guard_mode'][::substeps].tolist())),
        'rows':rows,'original_minimum_proxy_gap_m':capture['minimum_all9021_gap_m'],
        'original_max_speed_excess_rad_s':capture['all74_speed_exceedance_rad_s'],
        'new_learned_training_jobs':0,'random_hand_motor_dimensions':0,
        'independent_initial_states':1,'reused_development_random_intent_sequences':1,
        'new_formal_holdouts':0,'native_long_motion_episodes_completed':0,
        'forecast_tracking_braking_bound_qualified':False,'fullSystem0_accepted':False,
        'source_sha256':{**sources,str(Path(__file__)):sha(__file__),str(p/'REGISTRATION_V1.json'):sha(p/'REGISTRATION_V1.json')}}
    output.write_text(json.dumps(report,indent=2)+'\n')
    print('ACTUAL_STATE_GUARD',version,'duration',n*reg['physics_dt_s'],'model+physical',finite_ok,
          'all4movement',all(admission),'rows',json.dumps(rows),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('version',type=int);main(parser.parse_args().version)
