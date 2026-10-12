"""Independent full74 measured stopping, distinct from random workspace coverage."""
from pathlib import Path
import ast,hashlib,json
import numpy as np
from brake_measurements_v1 import settling_window

H=Path('/home/liyufeng/safeduo/artifacts/safety_physical_qualified_random_20261009_1613')
ARMS=('F_L','F_R','U_L','U_R')
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=H/'fourhand_native_probe_v113';output=p/'INDEPENDENT_EMERGENCY_BRAKE_V1.json'
    assert not output.exists()
    reg=read(p/'REGISTRATION_V1.json');resource=read(p/'RESOURCE_EXECUTION_V1.json')
    raw=Path(resource['out']);capture=read(p/'INDEPENDENT_CAPTURE_AUDIT_V1.json')
    model=read(p/'INDEPENDENT_NATIVE_MIMIC_WINDOW_V6.json')
    motor=read(p/'INDEPENDENT_FULL_MOTOR_EFFORT_V4.json')
    custody=read(p/'INDEPENDENT_ACTUAL_STATE_GUARD_MOTION_V3.json')
    assert capture['record_integrity_verified'] and sum(r['independent_motors'] for r in motor['rows'])==50
    worker=p/'native_common_beam_v1.py';source=worker.read_text()
    first_capture_line=source[:source.index('        capture(0)')].count('\n')+1
    for node in ast.walk(ast.parse(source)):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.lineno>first_capture_line:
            name=node.func.attr
            assert not (name.startswith('write_joint') or name.startswith('write_root') or
                        name in ('set_joint_positions','set_joint_velocities','set_root_state')),(node.lineno,name)
    parts=[];sources={}
    for c in read(raw/'physics_chunks.json'):
        f=raw/c['path'];assert sha(f)==c['sha256'];sources[str(f)]=sha(f)
        with np.load(f) as z:
            fields=['physics_event','state_time_s','emergency_brake_active','emergency_trigger_control','all9021_raw_geometry']
            fields += [a+s for a in ARMS for s in ('_native_q','_native_qd','_motor_control_q_before','_actual_full_position_target','_body_position_local')]
            parts.append({k:z[k] for k in fields})
    data={k:np.concatenate([part[k] for part in parts]) for k in parts[0]}
    n=len(data['physics_event']);dt=reg['physics_dt_s'];nsub=reg['physics_substeps_per_control']
    np.testing.assert_array_equal(data['physics_event'],np.arange(n))
    np.testing.assert_allclose(data['state_time_s'],(np.arange(n)+1)*dt,rtol=0,atol=1e-12)
    active=data['emergency_brake_active'];indices=np.flatnonzero(active)
    assert len(indices) and not active[:indices[0]].any() and active[indices[0]:].all()
    event=int(indices[0]);control=int(data['emergency_trigger_control'][event])
    assert event==control*nsub and (data['emergency_trigger_control'][active]==control).all()
    trigger_time=event*dt
    speeds=np.max(np.concatenate([np.abs(data[a+'_native_qd'][:,0]) for a in ARMS],axis=1),axis=1)
    assert sum(data[a+'_native_qd'].shape[-1] for a in ARMS)==74
    settle=settling_window(data['state_time_s'],speeds,trigger_time)
    with np.load(raw/'native_parameters.npz') as z:params={k:z[k] for k in z.files}
    rows=[]
    for a in ARMS:
        np.testing.assert_array_equal(data[a+'_actual_full_position_target'][active],data[a+'_motor_control_q_before'][active])
        names=params[a+'_body_names'].tolist()
        mount=('left_hand_base' if a.endswith('L') else 'right_hand_base') if a.startswith('F') else 'wrist_3_link'
        pos=data[a+'_body_position_local'][:,0,names.index(mount)]
        start=event-1
        assert start>=0
        stop=settle['confirmation_index'] if settle['settled'] else n-1
        path=float(np.linalg.norm(np.diff(pos[start:stop+1].astype(float),axis=0),axis=1).sum())
        q=data[a+'_native_q'][:,0];initial=data[a+'_motor_control_q_before'][event,0]
        rows.append({'arm':a,'maximum_joint_departure_after_trigger_rad':float(np.abs(q[event:stop+1]-initial).max()),
                     'mount_path_until_stop_confirmation_or_record_end_m':path,
                     'maximum_joint_speed_after_trigger_rad_s':float(np.abs(data[a+'_native_qd'][event:]).max())})
    tolerance=model['predeclared_relation_tolerance']
    relation_pass=model['maximum_q_residual_rad']<=tolerance and model['maximum_qd_residual_rad_s']<=tolerance
    complete=n==reg['expected_maximum_physics_events']
    passed=complete and capture['physical_gates_pass'] and relation_pass and settle['settled']
    report={'version':113,'status':'CLOSED_MEASURED_EMERGENCY_BRAKE_DEVELOPMENT_CASE',
        'physics_events':n,'duration_s':n*dt,'first_brake_control':control,'first_brake_event':event,
        'brake_trigger_time_s':trigger_time,'settling_observation':settle,'rows':rows,
        'full_preregistered_window_complete':complete,'all_original_physical_gates_pass':capture['physical_gates_pass'],
        'original_mimic_q_AND_qd_residuals_pass':relation_pass,
        'minimum_proxy_gap_after_trigger_m':float(data['all9021_raw_geometry'][event:].min()),
        'actual_position_target_equals_current_q_before_each_emergency_microstep':True,
        'native_physics_continues_after_brake':True,'no_post_seed_q_qd_or_object_state_writes':True,
        'all50_original_motor_effort_and_passive_follower_bindings_verified':True,
        'measured_emergency_brake_case_pass':bool(passed),'random_workspace_coverage_pass':False,
        'input_random_arm_dimensions':26,'input_random_finger_dimensions':0,'brake_motor_dimensions':50,
        'braking_envelope_or_hardware_safety_qualified':False,'new_training_jobs':0,'formal_holdouts':0,
        'fullSystem0_accepted':False,'source_sha256':{**sources,str(Path(__file__)):sha(__file__),
            str(worker):sha(worker),
            str(Path(__file__).with_name('brake_measurements_v1.py')):sha(Path(__file__).with_name('brake_measurements_v1.py')),
            **{str(p/f):sha(p/f) for f in ['REGISTRATION_V1.json','INDEPENDENT_CAPTURE_AUDIT_V1.json','INDEPENDENT_NATIVE_MIMIC_WINDOW_V6.json','INDEPENDENT_FULL_MOTOR_EFFORT_V4.json','INDEPENDENT_ACTUAL_STATE_GUARD_MOTION_V3.json']}}}
    output.write_text(json.dumps(report,indent=2)+'\n')
    print('BRAKE_CASE',bool(passed),'settle',json.dumps(settle),'physical',capture['physical_gates_pass'],'q_AND_qd',relation_pass,flush=True)


if __name__=='__main__':main()
