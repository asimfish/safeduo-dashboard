"""Recompute native friction-capacity proxy and stricter prelift admission.

This is an admission-trace audit. It never treats aggregate capacity as a full
contact-wrench certificate, and it retains every original physical/task result.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
H=Path(__file__).resolve().parent
ARMS=('F_L','F_R','U_L','U_R')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(version):
    P=H/f'fourhand_native_probe_v{version}';audit=json.loads((P/'INDEPENDENT_CAPTURE_AUDIT_V1.json').read_text())
    assert audit['record_integrity_verified']
    reg=json.loads((P/'REGISTRATION_V1.json').read_text());raw=Path(json.loads((P/'RESOURCE_EXECUTION_V1.json').read_text())['out'])
    params=np.load(raw/'native_parameters.npz');identity=json.loads((raw/'native_task_identity.json').read_text())
    mu=min(1.,float(params['object_native_material_properties'][...,1].min()))*min(1.,min(float(params[a+'_native_material_properties'][...,1].min()) for a in ARMS))
    assert abs(mu-float(params['prelift_conservative_dynamic_mu']))<1e-12
    weight_reserve=reg['prelift_load_reserve_fraction']*identity['native_mass_kg']*9.81
    keys=['physics_event','state_time_s','phase','feedback_phase','four_hand_normal_N','feedback_grip_thumb_guard','feedback_grip_finger_guard',
          'prelift_ready','prelift_friction_capacity_proxy_N','prelift_weight_reserve_N','prelift_native_hand_min_margin_rad','prelift_max_hand_speed_rad_s']
    keys += [a+k for a in ARMS for k in ['_native_q','_native_qd']]
    pieces={k:[] for k in keys};paths=[]
    for c in json.loads((raw/'physics_chunks.json').read_text()):
        f=raw/c['path'];assert sha(f)==c['sha256'];paths.append(f)
        with np.load(f) as z:
            for k in keys:pieces[k].append(z[k].copy())
    data={k:np.concatenate(v) for k,v in pieces.items()};n=len(data['physics_event']);margin=np.full(n,np.inf);speed=np.zeros(n)
    for arm in ARMS:
        names=params[arm+'_joint_names'];hand=np.flatnonzero(np.array([any(x in name for x in ['thumb','index','middle','ring','pinky','little']) for name in names]))
        assert len(hand)==12
        q=data[arm+'_native_q'][:,0,hand];v=data[arm+'_native_qd'][:,0,hand];lim=params[arm+'_hard_limits'][0,hand]
        margin=np.minimum(margin,np.minimum(q-lim[:,0],lim[:,1]-q).min(1));speed=np.maximum(speed,abs(v).max(1))
    capacity=data['four_hand_normal_N'].sum(1)*mu
    guarded=data['feedback_grip_thumb_guard'].any(1)|data['feedback_grip_finger_guard'].any(1)
    active=np.isin(data['phase'],['CLOSE','LIFT','CARRY','PLACE','RELEASE','RETREAT'])
    ready=(capacity>=weight_reserve)&(margin>=.02)&(speed<=.05)&~guarded&active
    assert np.allclose(data['prelift_friction_capacity_proxy_N'],capacity,atol=1e-9,rtol=0)
    assert np.allclose(data['prelift_weight_reserve_N'],weight_reserve,atol=1e-9,rtol=0)
    assert np.allclose(data['prelift_native_hand_min_margin_rad'],margin,atol=1e-8,rtol=0)
    assert np.allclose(data['prelift_max_hand_speed_rad_s'],speed,atol=1e-8,rtol=0)
    assert np.array_equal(data['prelift_ready'],ready),'Registered ready predicate differs from actual states'
    admissions=[];start=None
    for i in range(n):
        if str(data['phase'][i])!='CLOSE':continue
        now=float(data['state_time_s'][i]);predicate=bool(ready[i] and min(data['four_hand_normal_N'][i])>.1)
        start=now if predicate and start is None else start if predicate else None
        if str(data['feedback_phase'][i])=='LIFT':
            assert start is not None and now-start>=.25-1e-9,'Unmeasured or insufficient prelift dwell'
            admissions.append(dict(event=int(data['physics_event'][i]),state_time_s=now,independent_ready_dwell_s=now-start,
                capacity_proxy_N=float(capacity[i]),required_weight_reserve_N=float(weight_reserve),min_joint_margin_rad=float(margin[i]),max_hand_speed_rad_s=float(speed[i])))
            break
    protocol=json.loads((raw/'fourhand_protocol.json').read_text());actual=[x for x in protocol['phase_transitions'] if x['from_phase']=='CLOSE' and x['to_phase']=='LIFT']
    assert bool(actual)==bool(admissions)
    if actual:assert 0<=actual[0]['time_s']-admissions[0]['state_time_s']<=reg['physics_dt_s']+1e-9
    sources=[Path(__file__),P/'REGISTRATION_V1.json',P/'INDEPENDENT_CAPTURE_AUDIT_V1.json',raw/'native_parameters.npz',raw/'native_task_identity.json',*paths]
    report=dict(status='PASS_RECONSTRUCTED_STRONGER_PRELIFT_ADMISSION_INTEGRITY',version=version,
        conservative_dynamic_mu=mu,weight_reserve_N=weight_reserve,admissions=admissions,
        maximum_observed_capacity_proxy_N=float(capacity.max()),max_measured_ready_window_required_s=.25,
        necessary_aggregate_proxy_is_not_full_friction_wrench_certificate=True,
        actual_lift_stage_pass=audit['measured_stage_windows_s']['stable_lift']>=2.,
        physical_gates_pass=audit['physical_gates_pass'],complete_task_pass=audit['four_hand_task_development_pass'],
        fullSystem0_accepted=False,source_SHA256={str(p):sha(p) for p in sources})
    target=P/'INDEPENDENT_PRELIFT_ADMISSION_V1.json';assert not target.exists();target.write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'],'admissions',len(admissions),'task',report['complete_task_pass'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('version',type=int);main(parser.parse_args().version)
