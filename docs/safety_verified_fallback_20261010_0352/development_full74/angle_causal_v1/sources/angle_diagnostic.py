"""Read-only float32 camera-angle causal replay. Never changes an acceptance gate."""
from evidence_io import cpu_limits
cpu_limits()
import argparse
import ast
import itertools
import math
from pathlib import Path
import resource
import numpy as np
from evidence_io import HERE,H,SCREEN,OUTPUT,Evidence,require,exact,write_new,parse_json,sha,contained
from camera_index import read_index
from camera_adapter import CameraAudit,exact_snapshot
from camera_optics import frustum
from science import parameters_layout

FOCUS={(0,'F_L',1),(0,'F_L',4),(3,'U_R',2),(3,'U_R',6),(6,'U_L',3),(6,'U_L',5),(8,'F_R',2),(8,'F_R',6),(9,'F_R',3)}


def angle(v):
    x,y=map(float,np.asarray(v)[:2]);require(math.hypot(x,y)>0,'azimuth undefined')
    return math.degrees(math.atan2(y,x))%360


def wrap(a,b):return (float(a)-float(b)+180)%360-180


def quat_from_row_matrix(row):
    """Derived orientation only; original runtime quaternion was NOT persisted."""
    m=np.asarray(row,dtype=float).T;t=float(np.trace(m))
    if t>0:
        s=math.sqrt(t+1)*2;q=np.array([(m[2,1]-m[1,2])/s,(m[0,2]-m[2,0])/s,(m[1,0]-m[0,1])/s,s/4])
    else:
        i=int(np.argmax(np.diag(m)));j=(i+1)%3;k=(i+2)%3;s=math.sqrt(1+m[i,i]-m[j,j]-m[k,k])*2
        q=np.zeros(4);q[i]=s/4;q[j]=(m[j,i]+m[i,j])/s;q[k]=(m[k,i]+m[i,k])/s;q[3]=(m[k,j]-m[j,k])/s
    q/=np.linalg.norm(q)
    if q[3]<0:q=-q
    return q


def row_from_quat(q):
    x,y,z,w=np.asarray(q,dtype=float)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]).T


def metrics(rec,points):
    world=np.asarray(rec['camera']['actual_camera_to_world_row_matrix'],float)
    target=np.asarray(rec['target_world_m'],float);requested=np.asarray(rec['requested_eye_world_m'],float)
    planned=np.asarray(rec['world_direction'],float);center=(points.min(0)+points.max(0))/2
    require(np.array_equal(center,target),'producer target differs from reconstructed native hand center')
    require(np.array_equal(points,np.asarray(rec['hand_link_positions_world_m'])),'native framing points changed')
    computed=frustum(points,rec['camera']);eye=world[3,:3];d=eye-center;ideal=requested-center
    rounded=requested.astype(np.float32).astype(np.float64);target32=target.astype(np.float32).astype(np.float64)
    rho=math.hypot(d[0],d[1]);distance=float(np.linalg.norm(d));require(rho>0,'nonzero horizontal baseline')
    declared=float(rec['azimuth_deg']);az=angle(d);expected32=angle(rounded-center)
    error=wrap(az,declared);require(abs(wrap(az,computed['azimuth_deg']))<1e-10,'independent atan2 vs frozen reader')
    eye_error=eye-requested;dx,dy=ideal[:2];rho_ideal=math.hypot(dx,dy)
    first_order=math.degrees((-dy*eye_error[0]+dx*eye_error[1])/rho_ideal**2)
    xy_error=float(np.linalg.norm(eye_error[:2]));bound=math.degrees(math.asin(min(1,xy_error/rho_ideal)))
    q=quat_from_row_matrix(world[:3,:3]);roundtrip=row_from_quat(q)
    q32=q.astype(np.float32);rotation32=row_from_quat(q32)
    forward=world[2,:3];forward_angle=angle(forward)
    f32view=(requested.astype(np.float32)-target.astype(np.float32)).astype(float)
    old_pose_position=float(np.max(abs(eye-requested)));old_pose_axis=float(np.max(abs(forward-ideal/np.linalg.norm(ideal))))
    ideal_angle=angle(ideal);requested_dir_angle=angle(planned)
    identity=dict(declared_matches_direction=abs(wrap(declared,requested_dir_angle))<1e-9,
        declared_matches_requested_eye=abs(wrap(declared,ideal_angle))<1e-9,
        actual_eye_equals_float32_requested_eye=bool(np.array_equal(eye,rounded)),
        float32_position_counterfactual_reproduces_actual_angle=abs(wrap(expected32,az))<1e-12,
        producer_pose_component_gate_pass=old_pose_position<=2e-5 and old_pose_axis<=2e-5,
        frozen_reader_angle_gate_pass=abs(error)<1e-4,
        circular_wrap_already_applied=True,
        removing_only_eye_rounding_restores_angle=abs(wrap(ideal_angle,declared))<1e-9)
    return dict(declared_deg=declared,world_direction_deg=requested_dir_angle,requested_eye_minus_center_deg=ideal_angle,
        actual_eye_minus_native_center_deg=az,frozen_reader_deg=computed['azimuth_deg'],signed_error_deg=error,absolute_error_deg=abs(error),
        float32_eye_counterfactual_deg=expected32,float32_eye_and_target_direction_deg=angle(f32view),
        matrix_optical_back_axis_deg=forward_angle,matrix_axis_vs_declared_deg=wrap(forward_angle,declared),
        actual_eye_world_m=eye.tolist(),requested_eye_world_m=requested.tolist(),native_target_world_m=target.tolist(),
        actual_eye_minus_requested_m=eye_error.tolist(),float32_target_minus_declared_m=(target32-target).tolist(),
        horizontal_baseline_m=rho,distance_m=distance,inverse_horizontal_baseline_per_m=1/rho,
        azimuth_condition_relative_to_3D_direction=distance/rho,elevation_deg=math.degrees(math.atan2(d[2],rho)),
        first_order_position_error_prediction_deg=first_order,position_error_angular_bound_deg=bound,
        producer_pose_position_max_error_m=old_pose_position,producer_pose_axis_max_error=old_pose_axis,
        matrix_orthogonality_max_error=float(np.max(abs(world[:3,:3]@world[:3,:3].T-np.eye(3)))),
        matrix_determinant=float(np.linalg.det(world[:3,:3])),derived_quaternion_xyzw=q.tolist(),
        quaternion_reconstruction_residual=float(np.max(abs(roundtrip-world[:3,:3]))),
        derived_quaternion_float32_roundtrip_axis_angle_deg=angle(rotation32[2]),
        original_camera_quaternion_bytes='NOT_PERSISTED; derived quaternion is not native evidence',
        identity=identity)


def cpu_oracles():
    tests=[]
    def check(name,condition):require(condition,'CPU oracle '+name);tests.append(name)
    check('wrap359_to001',abs(wrap(1,359)-2)<1e-12)
    check('wrap001_to359',abs(wrap(359,1)+2)<1e-12)
    check('same_angle_plus360',wrap(10,370)==0)
    check('exact30_separation_unchanged',abs(wrap(30,0))>=30)
    check('below30_rejected',abs(wrap(29.99999,0))<30)
    for q in ([0,0,0,1],[1,0,0,0],[0,1,0,0],[0,0,1,0],[.2,.3,.4,.5]):
        m=row_from_quat(q);check('quaternion_roundtrip_'+str(q),np.allclose(row_from_quat(quat_from_row_matrix(m)),m,atol=2e-15,rtol=0))
    for x in (0.,8.,16.,32.):
        eye=np.array([x+.123456789,.234567891,1.]);center=np.array([x,0,0]);actual=eye.astype(np.float32).astype(float)
        check('float32_eye_'+str(x),math.isfinite(wrap(angle(actual-center),angle(eye-center))))
    return dict(tests_run=len(tests),passed=tests,native_pass=False,safety_acceptance=False)


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False);parser.add_argument('--proof-dir',required=True);args=parser.parse_args()
    out=contained(args.proof_dir,OUTPUT);out.mkdir(parents=True,exist_ok=False);e=Evidence();a=e.js(HERE/'anchors.json')
    tests=cpu_oracles();write_new(out/'CPU_ORACLES.json',tests)
    e.verify(a['analysis_sources'])
    ast_proof={}
    for name in ('camera_optics.py','camera_adapter.py'):
        original=H/'astra/full74_failed_camera_diagnostic_v2'/name
        old=e.raw(original);new=e.raw(HERE/name)
        require(ast.dump(ast.parse(old),include_attributes=False)==ast.dump(ast.parse(new),include_attributes=False),'frozen reader pose AST')
        ast_proof[name]=dict(original=str(original),original_sha256=sha(original),copy_sha256=sha(HERE/name),AST_identical=True)
    root=Path(a['actual_attempt_root']);w=e.js(root/'PARENT_NATIVE_WAIT.json',a['actual_failed_closure_binding']['native_wait_sha256'])
    require(w['actual_wait_exit']==-9 and w['raw_wait_status']==9 and w['waitpid_observed'] is True,'only already closed V9')
    batch=root/'native/batch_00000';native=e.js(root/'native/NATIVE_RECEIPT.json');packet=read_index(e,batch/'initial_camera_receipts.json',native['artifacts'][str(batch/'initial_camera_receipts.json')],kind='worker')
    binding=e.js(SCREEN/'INPUT_BINDING_V1.json',a['files'][str(SCREEN/'INPUT_BINDING_V1.json')])
    params=e.npz(binding['parameter_path'],binding['files'][binding['parameter_path']]);layout=parameters_layout(params)
    body=e.npz(Path(binding['parameter_path']).parent/'body_identity.npz');identity=e.js(binding['contact_identity_path'])
    audit=CameraAudit(batch,e,body,identity,layout);records=[];groups=[];seen_focus=set()
    for receipt in packet['receipts']:
        slot=receipt['env_id'];state=parse_json(audit.ref(dict(path=receipt['state'],sha256=receipt['sha256'])))
        baseline=audit.snapshot(state['native_before_all64']);inventory=audit.load_inventory(state['native_path_mapping']);byrgb={}
        for embedded in state['attempts']:
            rec=parse_json(audit.ref(embedded['attempt_record']));require(rec=={k:v for k,v in embedded.items() if k!='attempt_record'},'original embedded attempt agreement')
            if 'camera' not in rec:continue
            arm=rec['arm'];number=rec['attempt'];rows=[m for m in inventory.values() if m['env_id']==slot and m['arm']==arm and m['hand']]
            points=baseline[arm+'_native_link_transforms_xyzw'][slot,[m['body_index'] for m in rows],:3].astype(float)
            row=dict(env_id=slot,arm=arm,attempt=number,producer_status=rec['status'],attempt_reference=embedded['attempt_record'],metrics=metrics(rec,points))
            if (slot,arm,number) in FOCUS:
                exact_snapshot(baseline,audit.snapshot(rec['native_before_all64']));seen_focus.add((slot,arm,number));row['parent_reported_case']=True
            records.append(row)
            if 'rgb' in rec:byrgb[rec['rgb']['path']]=row
        for arm,group in state['arms'].items():
            refs=group.get('selected_group_images',group.get('qualified_images',[]))
            if len(refs)!=3:continue
            selected=[byrgb[r['path']] for r in refs]
            def separated(key):return min(abs(wrap(x['metrics'][key],y['metrics'][key])) for x,y in itertools.combinations(selected,2))
            planned=separated('declared_deg');actual=separated('actual_eye_minus_native_center_deg')
            groups.append(dict(env_id=slot,arm=arm,producer_status=group['status'],selected_attempts=[r['attempt'] for r in selected],
                minimum_declared_separation_deg=planned,minimum_actual_separation_deg=actual,
                declared_ge30=planned>=30,actual_ge30=actual>=30,threshold_unchanged_deg=30.))
        print('ANGLE_DIAGNOSTIC_READONLY env='+str(slot),flush=True)
    require(seen_focus==FOCUS,'all9 parent reported cases covered')
    failing=[r for r in records if not r['metrics']['identity']['frozen_reader_angle_gate_pass']]
    focus=[r for r in records if r.get('parent_reported_case')]
    require(all(not r['metrics']['identity']['frozen_reader_angle_gate_pass'] for r in focus),'all9 discrepancies reproduced unchanged')
    counter=all(r['metrics']['identity']['actual_eye_equals_float32_requested_eye'] and
        r['metrics']['identity']['float32_position_counterfactual_reproduces_actual_angle'] and
        r['metrics']['identity']['removing_only_eye_rounding_restores_angle'] for r in focus)
    report=dict(schema='astra.camera_angle_diagnostic.v1',status='CLOSED_READONLY_ANGLE_CAUSAL_REPLAY_ONLY',CPU_oracles=tests,
        scope='all32 initial hand camera metadata only; no raw pixel audit or updated acceptance',native_original_actual_exit=-9,
        original_reader_threshold_deg=1e-4,minimum_view_separation_deg=30.,frozen_reader_pose_AST=ast_proof,
        hand_attempts_with_camera=len(records),angle_gate_failure_count=len(failing),parent_reported_cases=focus,
        all_failure_cases=failing,all_selected_group_separations=groups,
        declared_ge30_but_actual_below30=[g for g in groups if g['declared_ge30'] and not g['actual_ge30']],
        largest_angle_error=max(records,key=lambda r:r['metrics']['absolute_error_deg']),
        largest_condition=max(records,key=lambda r:r['metrics']['azimuth_condition_relative_to_3D_direction']),
        parent9_float32_translation_cause_exactly_reproduced=counter,
        original_camera_quaternions='UNAVAILABLE; derived matrix quaternion only, no original-quaternion attribution',
        changes_to_producer_reader_or_thresholds=False,raw_mask_findings='NOT_EVALUATED',native_pass=False,stage_pass=False,safety_acceptance=False,
        max_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,address_space_bytes=1024**3)
    e.recheck();write_new(out/'ALL_HAND_ANGLES.json',records);write_new(out/'EVIDENCE_LEDGER.json',e.ledger)
    write_new(out/'ANGLE_DIAGNOSTIC.json',report)
    print('ANGLE_CLOSED cases='+str(len(records))+' failures='+str(len(failing))+' focus9_float32_cause='+str(counter),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
