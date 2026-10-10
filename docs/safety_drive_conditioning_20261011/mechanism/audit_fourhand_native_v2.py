from microstep_contract_v1 import microsteps,control_fifo_targets,event_indices
"""Independent recorded-point, full-joint, queue and payload-time-window audit.

This assesses one complete development capture. It never promotes that case to
System0, fresh-seed generalization, cooked-mesh clearance or hardware acceptance.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

H = Path(__file__).resolve().parent
ARMS = ('F_L', 'F_R', 'U_L', 'U_R')
CONTACT_PHASES = {'CLOSE', 'LIFT', 'CARRY', 'PLACE', 'RELEASE'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def hand_sensor(path):
    body = path.rsplit('/', 1)[-1]
    return ('/f2_hand/' in path or body in {'fr3_link8', 'wrist_3_link'}
            or any(n in body for n in ['thumb', 'index', 'middle', 'ring', 'little', 'pinky']))


def raw_scalars(chunk, prefix, n):
    """Recompute sums from raw scalars and original ownership, independently."""
    c, starts = chunk[prefix+'counts'], chunk[prefix+'starts']
    offsets = chunk[prefix+'event_offsets']
    fields = ['normal_forces', 'points', 'normals', 'separations']
    assert c.shape[0] == n and starts.shape == c.shape
    assert offsets.shape == (n+1,) and offsets[0] == 0
    assert np.all(np.diff(offsets) >= 0)
    assert c.dtype.kind in 'iu' and starts.dtype.kind in 'iu'
    assert all(np.isfinite(chunk[prefix+k]).all() for k in fields)
    assert offsets[-1] == len(chunk[prefix+'point_indices'])
    result = []
    for t in range(n):
        lo, hi = map(int, offsets[t:t+2])
        si, pi, ix = [chunk[prefix+k][lo:hi] for k in ['sensor_indices', 'partner_indices', 'point_indices']]
        assert np.all((si >= 0) & (si < c.shape[1])) and np.all((pi >= 0) & (pi < c.shape[2]))
        assert len(np.unique(ix)) == len(ix) and len(ix) == sum(map(int, c[t].flat))
        assert (ix >= starts[t, si, pi]).all() and (ix < starts[t, si, pi]+c[t, si, pi]).all()
        owners = si*c.shape[2]+pi
        counts = np.bincount(owners, minlength=c[t].size).reshape(c[t].shape)
        assert np.array_equal(counts, c[t])
        scalar = np.bincount(owners, weights=np.abs(chunk[prefix+'normal_forces'][lo:hi, 0].astype(float)),
                             minlength=c[t].size).reshape(c[t].shape)
        assert np.array_equal(scalar, chunk[prefix+'partner_abs_normal_sum_N'][t])
        result.append(scalar)
    return np.stack(result)


def longest_window(times, mask):
    """Duration between actual first/last true samples; no missing-gap credit."""
    start = None
    maximum = 0.
    for t, flag in zip(times, mask):
        if flag:
            if start is None:
                start = float(t)
            maximum = max(maximum, float(t)-start)
        else:
            start = None
    return maximum


def audit(version):
    p = H/f'fourhand_native_probe_v{version}'
    resource = json.loads((p/'RESOURCE_EXECUTION_V1.json').read_text())
    assert resource['status'] == 'complete_native_probe_closed_independent_audit_pending'
    assert resource['actual_native_exit'] == resource['actual_exit'] == 0
    root = Path(resource['out'])
    protocol = json.loads((root/'fourhand_protocol.json').read_text())
    assert protocol['status'] == 'complete'
    reg = json.loads((p/'REGISTRATION_V1.json').read_text())
    params = load(root/'native_parameters.npz')
    dt, physics_dt = reg['control_dt_s'], reg['physics_dt_s'];substeps=microsteps(reg)
    assert float(params['actual_control_dt_s']) == dt and float(params['actual_physics_dt_s']) == physics_dt
    assert protocol['actual_control_dt_s'] == dt and protocol['actual_physics_dt_s'] == physics_dt
    identity = json.loads((root/'native_contact_identity.json').read_text())
    task = json.loads((root/'native_task_identity.json').read_text())
    count = substeps*protocol['completed_controls']
    assert count == protocol['physics_events'] > 0
    assert sum(len(v['sensors']) for v in identity['views']) == 82
    expected = {r['rigid_owner'] for r in identity['inventory'] if r['collision_enabled'] and r['rigid_owner']
                and any('/'+a+'/' in r['rigid_owner'] for a in ARMS)}
    assert expected <= {s for v in identity['views'] for s in v['sensors']}
    assert sha(p/'REGISTRATION_V1.json') == task['registration_sha256']
    assert all(sha(f) == digest for f, digest in reg['source_sha256'].items())
    pieces = []
    next_event = 0
    for record in json.loads((root/'physics_chunks.json').read_text()):
        assert record['start'] == next_event and sha(root/record['path']) == record['sha256']
        z = load(root/record['path'])
        n = record['stop']-record['start']
        assert np.array_equal(z['physics_event'], np.arange(next_event, next_event+n))
        assert np.allclose(z['state_time_s'], (z['physics_event']+1)*physics_dt, atol=1e-12, rtol=0)
        assert z['all9021_raw_geometry'].shape == (n, 1, 9021)
        assert all(np.isfinite(v).all() for v in z.values() if v.dtype.kind in 'f')
        raw_scalars(z, 'object_normal_', n)
        off = z['object_friction_event_offsets']
        assert off.shape == (n+1,) and off[0] == 0 and np.all(np.diff(off) >= 0)
        assert off[-1] == len(z['object_friction_raw_indices'])
        fc, fs = z['object_friction_counts'], z['object_friction_starts']
        assert fc.shape == fs.shape and fc.shape[0] == n and fc.shape[1] == 1
        for t in range(n):
            lo, hi = map(int, off[t:t+2]);ix=z['object_friction_raw_indices'][lo:hi];pi=z['object_friction_partner_indices'][lo:hi]
            assert hi-lo == sum(map(int, fc[t].flat)) and len(np.unique(ix)) == len(ix)
            assert (ix >= fs[t, 0, pi]).all() and (ix < fs[t, 0, pi]+fc[t, 0, pi]).all()
        pieces.append({k:v for k,v in z.items() if not k.startswith('object_normal_') and not k.startswith('object_friction_')})
        next_event += n
    assert next_event == count
    data = {k:np.concatenate([z[k] for z in pieces]) for k in pieces[0]}
    receipt = json.loads((root/'point_contact_receipts.json').read_text())
    assert receipt['status'] == 'complete' and receipt['physics_events'] == count
    assert receipt['substeps_per_control']==substeps and receipt['control_steps']==protocol['completed_controls']
    assert receipt['identity_sha256'] == sha(root/'native_contact_identity.json')
    peak, details, next_event = [], [], 0
    first_roots = {}
    for record in receipt['chunks']:
        assert record['start'] == next_event and sha(root/record['path']) == record['sha256']
        z=load(root/record['path']);n=record['stop']-record['start'];chunk_peak=np.zeros(n)
        assert np.array_equal(event_indices(z['frame'],z['substep'],substeps), np.arange(next_event, next_event+n))
        assert np.allclose(z['physics_dt'], physics_dt, atol=1e-12, rtol=0)
        for a in ARMS:
            view=next(v for v in identity['views'] if v['arm']==a)
            scalar=raw_scalars(z, a+'_points_', n)
            allowed=np.zeros(scalar.shape[1:], bool)
            for si, sensor in enumerate(view['sensors']):
                if hand_sensor(sensor):
                    allowed[si, view['filters'][si].index(task['object_sensor'])]=True
            for t in range(n):
                active=allowed if str(data['phase'][next_event+t]) in CONTACT_PHASES else np.zeros_like(allowed)
                forbidden=np.where(active, 0., scalar[t]);si,pi=np.unravel_index(forbidden.argmax(),forbidden.shape)
                chunk_peak[t]=max(chunk_peak[t],float(forbidden[si,pi]))
                if forbidden[si,pi]>.1:
                    details.append(dict(event=next_event+t, sensor=view['sensors'][si],partner=view['filters'][si][pi],normal_N=float(forbidden[si,pi])))
            first_roots.setdefault(a,z[a+'_native_root_xyzw'][0,0])
            assert np.array_equal(z[a+'_native_q'],data[a+'_native_q'][next_event:next_event+n])
            assert np.array_equal(z[a+'_native_qd'],data[a+'_native_qd'][next_event:next_event+n])
            assert np.array_equal(z[a+'_native_position_targets'],data[a+'_actual_full_position_target'][next_event:next_event+n])
        peak.extend(chunk_peak);next_event+=n
    assert next_event == count and np.array_equal(np.asarray(peak), data['unintended_peak_normal_N'])
    hard, speed=np.zeros(count),np.zeros(count)
    for a in ARMS:
        q,qd=data[a+'_native_q'],data[a+'_native_qd'];bounds=params[a+'_hard_limits'];vmax=params[a+'_max_velocity']
        hard=np.maximum(hard,np.maximum(np.maximum(bounds[None,...,0]-q,q-bounds[None,...,1]),0).max((1,2)))
        speed=np.maximum(speed,np.maximum(abs(qd)-vmax[None],0).max((1,2)))
        ix=params[a+'_controlled_indices'];applied=data[a+'_applied_target'];issued=data[a+'_issued_target']
        assert np.array_equal(applied,data[a+'_actual_full_position_target'][...,ix])
        control_fifo_targets(applied,issued,params[a+'_initial_q'][:,ix],substeps,reg['position_target_FIFO_controls'])
        assert np.max(abs(np.diff(np.concatenate([params[a+'_initial_q'][:,ix][None],issued[0::substeps]],axis=0),axis=0)))<=.015+1e-6
    assert np.array_equal(hard,data['all74_hard_violation_rad']) and np.array_equal(speed,data['all74_speed_exceedance_rad_s'])
    fixture=load(reg['fixture']);origins=[first_roots[a][:3]-fixture[a+'_native_root_xyzw'][0,:3]+fixture['source_origin'][0] for a in ARMS]
    assert np.max(np.ptp(origins,axis=0))<=1e-6
    origin=np.mean(origins,axis=0);position=data['object_state'][:,:3]-origin
    q=data['object_state'][:,3:7];assert np.max(abs((q*q).sum(-1)-1))<=2e-4
    height=position[:,2]-1.02;four=data['four_hand_normal_N'];support=data['support_normal_N'];time=data['state_time_s']
    # Independent measured predicates. The observer's DONE flag supplies no evidence by itself.
    tilt=np.degrees(np.arccos(np.clip(1-2*(q[:,1]**2+q[:,2]**2),-1,1)))
    rest=(np.linalg.norm(data['object_state'][:,7:10],axis=-1)<=.01)&(np.linalg.norm(data['object_state'][:,10:13],axis=-1)<=.1)&(tilt<=10)
    allfour=four.min(-1)>.1;xy=np.linalg.norm(position[:,:2]-[.20,.04],axis=-1)
    supported=(support>=.8*.45*9.81)&(support<=1.5*.45*9.81)&(height>-.006)&(height<.012)
    phase=data['phase'].astype(str)
    windows=dict(four_hand_close=longest_window(time,allfour&(phase=='CLOSE')),
                 stable_lift=longest_window(time,(height>=.1)&allfour&rest&(support<=.1)&(phase=='LIFT')),
                 stable_carry=longest_window(time,(height>=.1)&allfour&rest&(xy<=.02)&(phase=='CARRY')),
                 supported_place=longest_window(time,supported&rest&(xy<=.02)&(phase=='PLACE')),
                 actual_detached_release=longest_window(time,(four.max(-1)<=.1)&(data['hand_open_error_rad'].max(-1)<=.05)&supported&rest&(xy<=.02)&(phase=='RELEASE')),
                 detached_retreat=longest_window(time,(four.max(-1)<=.1)&supported&rest&(xy<=.02)&(phase=='RETREAT')))
    minimums=dict(four_hand_close=.25,stable_lift=2.,stable_carry=.5,supported_place=.5,actual_detached_release=.5,detached_retreat=1.)
    task_time_ok=all(windows[k]>=v-1e-9 for k,v in minimums.items())
    physical_ok=bool(data['all9021_raw_geometry'].min()>=0 and max(peak)<=.1 and hard.max()<=1e-5 and speed.max()<=1e-5 and four.max()<=30)
    frames=json.loads((root/'camera_frames.json').read_text());controls=protocol['completed_controls'];expected_steps=[0,*range(4,controls+1,4)]
    if controls%4:expected_steps.append(controls)
    for view in reg['camera_views']:
        observed=[f for f in frames if f['view']==view]
        assert [f['control'] for f in observed]==expected_steps
        assert all(sha(root/f['path'])==f['sha256'] and abs(f['state_time_s']-f['control']*dt)<=1e-12 for f in observed)
    result=dict(status='CLOSED_INDEPENDENT_COMPLETE_CAPTURE_AUDIT',version=version,completed_controls=controls,physics_events=count,
        physics_substeps_per_control=substeps,position_target_FIFO_controls=reg['position_target_FIFO_controls'],
        observed_duration_s=count*physics_dt,task_status=protocol['task_status'],physical_gates_pass=physical_ok,
        measured_stage_windows_s=windows,stage_minimums_s=minimums,all_task_stage_time_windows_pass=task_time_ok,
        four_hand_task_development_pass=physical_ok and task_time_ok and protocol['task_status']=='DONE',
        unintended_peak_normal_N=float(max(peak)),minimum_all9021_gap_m=float(data['all9021_raw_geometry'].min()),
        all74_hard_violation_rad=float(hard.max()),all74_speed_exceedance_rad_s=float(speed.max()),
        maximum_height_gain_m=float(height.max()),maximum_hand_payload_normal_N=float(four.max()),
        first_unintended_contact=details[0] if details else None,all_unintended_contacts=details,
        camera_frames=len(frames),all4_native_view_cadence_and_frame_SHA_pass=True,
        record_integrity_verified=True,all_owned_native_children_reaped=True,
        maximum_window_complete=controls==reg['maximum_controls'],early_task_or_gate_stop=controls<reg['maximum_controls'],
        fullSystem0_accepted=False,formal_holdout=False,system0_tested=False,
        nonclaims=['cooked whole-mesh clearance','total implicit PD torque','vision perception','random coverage','selective pause/resume','hardware safety'],
        sources={str(root/'fourhand_protocol.json'):sha(root/'fourhand_protocol.json'),str(Path(__file__)):sha(__file__)})
    (p/'INDEPENDENT_CAPTURE_AUDIT_V1.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(p/'INDEPENDENT_TASK_METRICS_V1.npz',state_time_s=time,phase=phase,height_gain_m=height,
                        tilt_deg=tilt,four_hand_normal_N=four,support_normal_N=support,
                        unintended_peak_N=np.asarray(peak),geometry_min_m=data['all9021_raw_geometry'].min((1,2)),
                        all74_hard_rad=hard,all74_speed_rad_s=speed)
    print('FOURHAND_CAPTURE_AUDIT',version,'physical',physical_ok,'task',result['four_hand_task_development_pass'],windows,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('version',type=int);audit(parser.parse_args().version)
