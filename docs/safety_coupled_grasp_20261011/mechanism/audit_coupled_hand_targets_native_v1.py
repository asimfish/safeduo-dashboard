"""Independent XML/matrix reconstruction of every applied coupled hand target."""
from pathlib import Path
import argparse, hashlib, json, xml.etree.ElementTree as ET
import numpy as np
H=Path(__file__).resolve().parent
ARMS=('F_L','F_R','U_L','U_R')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def matrix(names,urdf):
    joints={j.get('name'):j for j in ET.parse(urdf).findall('joint')}
    roots=[n for n in names if joints[n].find('mimic') is None]
    cache={}
    def row(n,seen):
        assert n not in seen,'Cycle in original URDF'
        if n in cache:return cache[n]
        m=joints[n].find('mimic')
        if m is None:
            c=np.zeros(len(roots));c[roots.index(n)]=1.;b=0.
        else:
            c,b=row(m.get('joint'),seen|{n});s=float(m.get('multiplier','1'));c=c*s;b=b*s+float(m.get('offset','0'))
        cache[n]=(c,b);return c,b
    entries=[row(n,set()) for n in names]
    return roots,np.stack([c for c,b in entries]),np.array([b for c,b in entries])
def expected_target(names,A,b,rootids,nominal,previous,hard,dt,loaded,spec):
    np.testing.assert_allclose(A@nominal[rootids]+b,nominal,atol=1e-6,rtol=0)
    np.testing.assert_allclose(A@previous[rootids]+b,previous,atol=1e-6,rtol=0)
    assert (previous>=hard[:,0]-1e-6).all() and (previous<=hard[:,1]+1e-6).all()
    bounds=hard.copy()
    if loaded:
        for i,n in enumerate(names):
            if 'thumb_3_joint' in n or 'thumb_4_joint' in n:
                bounds[i,0]+=min(spec['distal_lower_reserve_rad'],.2*(hard[i,1]-hard[i,0]))
    lo=np.full(len(rootids),-np.inf);hi=np.full(len(rootids),np.inf)
    for i in range(len(names)):
        ids=np.flatnonzero(A[i]);assert len(ids)==1
        j=int(ids[0]);ends=(bounds[i]-b[i])/A[i,j]
        lo[j]=max(lo[j],float(ends.min()));hi[j]=min(hi[j],float(ends.max()))
    assert (lo<=hi).all()
    desired=np.clip(nominal[rootids],lo,hi)
    step=spec['max_target_rate_rad_s']*dt/np.max(abs(A),axis=0)
    return A@(previous[rootids]+np.clip(desired-previous[rootids],-step,step))+b
def main(version):
    P=H/f'fourhand_native_probe_v{version}';reg=json.loads((P/'REGISTRATION_V1.json').read_text())
    audit=json.loads((P/'INDEPENDENT_CAPTURE_AUDIT_V1.json').read_text());assert audit['record_integrity_verified']
    raw=Path(json.loads((P/'RESOURCE_EXECUTION_V1.json').read_text())['out'])
    params=np.load(raw/'native_parameters.npz');refs=np.load(reg['coupled_reference_endpoints'])
    meta=json.loads(Path(reg['coupled_assets']).read_text());sources={};pieces=[]
    for chunk in json.loads((raw/'physics_chunks.json').read_text()):
        path=raw/chunk['path'];assert sha(path)==chunk['sha256'];sources[str(path)]=sha(path)
        with np.load(path) as z:pieces.append({k:z[k].copy() for k in z.files if k in ['physics_event','phase','feedback_grip_finger_fraction','feedback_grip_thumb_fraction'] or any(k==a+s for a in ARMS for s in ['_actual_full_position_target','_hand_nominal_position_target','_hand_filtered_position_target'])})
    d={k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]};assert np.array_equal(d['physics_event'],np.arange(audit['physics_events']))
    records=[]
    for ai,a in enumerate(ARMS):
        names=params[a+'_joint_names'].tolist();hn=meta['assets'][a]['hand_joint_names'];ids=[names.index(n) for n in hn]
        hard=params[a+'_hard_limits'][0,ids];opened=params[a+'_initial_q'][0,ids];far=params[a+'_coupled_hand_far_q']
        np.testing.assert_allclose(opened,refs[a+'_hand_open_q'],atol=1e-7,rtol=0);np.testing.assert_array_equal(far,refs[a+'_hand_far_q'])
        urdf=Path(meta['assets'][a]['original_URDF']);roots,A,b=matrix(hn,urdf);rootids=[hn.index(n) for n in roots]
        thumb=np.array(['thumb' in n for n in hn]);max_rate=0.;max_binding=0.;max_relation=0.
        for e in range(0,len(d['physics_event']),2):
            phase=str(d['phase'][e]);assert phase==str(d['phase'][e+1])
            progress=np.where(thumb,float(d['feedback_grip_thumb_fraction'][e,ai]),float(d['feedback_grip_finger_fraction'][e,ai]))
            nominal=opened+progress*(far-opened);previous=opened if e==0 else d[a+'_actual_full_position_target'][e-1,0,ids]
            expected=nominal
            filtered=phase in ['CLOSE','LIFT','CARRY','PLACE','RELEASE','RETREAT']
            if filtered:expected=expected_target(hn,A,b,rootids,nominal,previous,hard,reg['control_dt_s'],phase in ['CLOSE','LIFT','CARRY','PLACE'],reg['interior_hand_target_settings'])
            for micro in [e,e+1]:
                measured=d[a+'_actual_full_position_target'][micro,0,ids]
                np.testing.assert_allclose(nominal,d[a+'_hand_nominal_position_target'][micro],atol=1e-6,rtol=0)
                for target in [d[a+'_hand_filtered_position_target'][micro],measured]:np.testing.assert_allclose(expected,target,atol=1e-6,rtol=0)
                residual=float(abs(A@measured[rootids]+b-measured).max());assert residual<1e-6
                assert (measured>=hard[:,0]-1e-6).all() and (measured<=hard[:,1]+1e-6).all()
                max_relation=max(max_relation,residual);max_binding=max(max_binding,float(abs(expected-measured).max()))
            if filtered:
                rate=float(abs(d[a+'_actual_full_position_target'][e,0,ids]-previous).max()/reg['control_dt_s']);max_rate=max(max_rate,rate)
                assert rate<=reg['interior_hand_target_settings']['max_target_rate_rad_s']+1e-5
        records.append(dict(arm=a,controls=audit['completed_controls'],independent_motors=len(roots),followers=len(hn)-len(roots),max_filtered_target_rate_rad_s=max_rate,max_native_binding_error_rad=max_binding,max_native_target_relation_error_rad=max_relation))
        sources[str(urdf)]=sha(urdf)
    for f in [Path(__file__),P/'REGISTRATION_V1.json',P/'INDEPENDENT_CAPTURE_AUDIT_V1.json',raw/'native_parameters.npz',Path(reg['coupled_reference_endpoints']),Path(reg['coupled_assets'])]:sources[str(f)]=sha(f)
    report=dict(status='PASS_INDEPENDENT_ALL_PHASE_COUPLED_NATIVE_TARGET_BINDING',controls=audit['completed_controls'],physics_events=audit['physics_events'],records=records,source_sha256=sources,physical_gates_pass=audit['physical_gates_pass'],task_pass=audit['four_hand_task_development_pass'],fullSystem0_accepted=False)
    path=P/'INDEPENDENT_COUPLED_HAND_TARGETS_V1.json';assert not path.exists();path.write_text(json.dumps(report,indent=2)+'\n');print(report['status'],report['controls'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('version',type=int);main(p.parse_args().version)
