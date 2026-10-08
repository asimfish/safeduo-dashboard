"""All64 native hand contact and native state audit at both physics substeps."""
from pathlib import Path
import json,hashlib,sys,datetime
import numpy as np
H=Path(__file__).resolve().parent
ARMS=['F_L','F_R','U_L','U_R'];NORMAL_DIAGNOSTIC_LIMIT_N=.1
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def audit(root,tag):
 root=Path(root);r=json.loads((root/'native_contact_receipts.json').read_text());identity=json.loads((root/'native_contact_identity.json').read_text());assert sha(root/'native_contact_identity.json')==r['identity_sha256'] and r['status']=='complete';T=r['control_steps'];cell=load(root/'cell_001.npz');assert len(cell['q'])==T and r['physics_events']==T*2
 metrics=np.zeros((T,2,64,4),np.float64);netmetrics=np.zeros_like(metrics);peaks=[None]*4;capacity_peak=0;accounting=0.;seen=0
 ids={v['arm']:v for v in identity['views']}
 native=load(root/'native_initial.npz')
 for chunk in r['chunks']:
  p=root/chunk['path'];assert sha(p)==chunk['sha256'];z=load(p);assert chunk['start']==seen
  for j,event in enumerate(range(chunk['start'],chunk['stop'])):
   t,sub=divmod(event,2);assert int(z['frame'][j])==t and int(z['substep'][j])==sub
   assert float(z['physics_dt'][j])==identity['physics_dt_s']
   for ai,a in enumerate(ARMS):
    view=ids[a];envs=np.asarray(view['env_ids']);matrix=z[a+'_partner_normal'][j];net=z[a+'_net'][j];counts=z[a+'_normal_counts'][j];starts=z[a+'_normal_starts'][j]
    assert matrix.shape==(len(envs),len(view['filters'][0]),3) and net.shape==(len(envs),3) and counts.shape==matrix.shape[:2] and starts.shape==counts.shape
    assert np.isfinite(matrix).all() and np.isfinite(net).all() and (counts>=0).all() and (starts>=0).all() and (starts+counts<=r['capacity']).all() and int(counts.sum())<r['capacity']
    capacity_peak=max(capacity_peak,int(counts.sum()));norm=np.linalg.norm(matrix.astype(np.float64),axis=-1);body=np.linalg.norm(net.astype(np.float64),axis=-1);accounting=max(accounting,float(np.abs(matrix.sum(1)-net).max()))
    for e in range(64):
     chosen=np.flatnonzero(envs==e);assert len(chosen)>0;metrics[t,sub,e,ai]=norm[chosen].max();netmetrics[t,sub,e,ai]=body[chosen].max()
    peakindex=np.unravel_index(norm.argmax(),norm.shape);value=float(norm[peakindex]);sensor,partner=peakindex
    if peaks[ai] is None or value>peaks[ai]['normal_N']:peaks[ai]=dict(arm=a,normal_N=value,frame=t,substep=sub,env=int(envs[sensor]),sensor=view['sensors'][sensor],partner=view['filters'][sensor][partner])
    for field in ['native_q','native_qd','native_position_targets','native_root_xyzw','native_root_velocity']:assert np.isfinite(z[a+'_'+field][j]).all()
    idx=native[a+'_controlled_joint_indices'];offset=sum([7,7,6,6][:ai]);w=[7,7,6,6][ai]
    assert np.array_equal(z[a+'_native_position_targets'][j][:,idx],cell['actuator_target'][t,:,offset:offset+w]),'actual native actuator targets vs FIFO6'
    if sub==1:
     # Full native hand states are saved here; compact controlled indices bind cell.
     assert np.array_equal(z[a+'_native_q'][j][:,idx],cell['q'][t,:,offset:offset+w])
   seen+=1
  assert sha(p)==chunk['sha256']
 assert seen==T*2
 over=(metrics>NORMAL_DIAGNOSTIC_LIMIT_N).any((1,3));peak=metrics.max((0,1,3));p=H/(tag+'_native_contact_metrics.npz')
 camera=json.loads((root/'camera_receipts.json').read_text());labels=load(root/'bank_assignment.npz')['risk_pair_index'];captured={}
 for cap in camera['receipts']:
  if cap['capture_kind']=='first_native_contact':
   e,t=cap['env_id'],cap['step'];label=int(labels[e]);assert label not in captured
   hit=over[:,labels==label].any(-1);assert hit.any() and t==int(hit.argmax()) and e==int(np.flatnonzero(over[t]&(labels==label))[0]);captured[label]=(e,t)
 assert set(captured)=={int(l) for l in np.unique(labels) if over[:,labels==l].any()}
 with p.open('xb') as f:np.savez_compressed(f,partner_normal_max_N=metrics,body_net_max_N=netmetrics)
 return dict(status='PASS_NATIVE_HAND_CONTACT_OBSERVATION_NOT_SAFETY',root=str(root),windows=64,control_steps=T,physics_events=T*2,raw_normal_diagnostic_threshold_N=NORMAL_DIAGNOSTIC_LIMIT_N,same_hand_contacts_included_in_raw=True,exemption_adjusted_contact_gate=False,windows_exceeding_normal_diagnostic=int(over.any(0).sum()),window_peak_normal_N=peak.tolist(),arm_peaks=peaks,peak_contact_count=capacity_peak,capacity=r['capacity'],maximum_net_minus_filtered_normal_abs_N=accounting,full_macro_post_controlled_q_exact=True,all_contact_micro_native_states_finite=True,friction_fully_observed=False,constructor_contacts_observed=False,whole_mesh_safety_certified=False,metric_file=str(p),metric_sha256=sha(p),utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
if __name__=='__main__':
 result=audit(sys.argv[1],sys.argv[2]);p=H/(sys.argv[2]+'.json')
 with p.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(result['status'],'overlimit',result['windows_exceeding_normal_diagnostic'],'peak',max(result['window_peak_normal_N']),flush=True)
