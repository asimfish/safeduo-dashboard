import argparse,json,hashlib
from pathlib import Path
import numpy as np
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--registration',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
r=json.loads(a.registration.read_text());raw=a.raw;rec=json.loads((raw/'recording_receipt.json').read_text())
assert rec['status']=='PASS_COMPLETE_STATIC_DIAGNOSTIC' and not (raw/'failure.txt').exists()
assert rec['registration_sha256']==sha(a.registration)
for file,key in [('native_metadata.json','metadata_sha256'),('contact_identities.json','identities_sha256'),('sparse_contact_points.json','sparse_contact_points_sha256')]:assert sha(raw/file)==rec[key]
rows={};next_step=0
for chunk in rec['chunks']:
 assert sha(raw/chunk['file'])==chunk['sha256'] and chunk['first_step']==next_step
 with np.load(raw/chunk['file'],allow_pickle=False) as d:
  assert np.array_equal(d['step'],np.arange(chunk['first_step'],chunk['last_step']+1))
  for key in d.files:rows.setdefault(key,[]).append(d[key])
 next_step=chunk['last_step']+1
assert next_step==r['steps']==rec['steps']
for img in rec['images']:assert sha(raw/img['file'])==img['sha256']
d={key:np.concatenate(v) for key,v in rows.items()};meta=json.loads((raw/'native_metadata.json').read_text());identity=json.loads((raw/'contact_identities.json').read_text())
criteria=r['candidate_criteria'];start,end=criteria['window_s'];idx=(d['time_s']>=start)&(d['time_s']<end);closed=(d['time_s']>=5)&(d['time_s']<6)
assert np.allclose(d['time_s'],(d['step']+1)*r['physics_dt_s'],atol=1e-10,rtol=0)
cases=[];max_accounting=0.
for c in r['cases']:
 e=c['env'];hands=[]
 for arm,m in meta['arms'].items():
  hid=m['hand_ids'];openq=np.asarray(m['hand_default_rad'])[e].copy()
  for j,jid in enumerate(hid):
   if arm.startswith('U') and 'thumb_1_joint' in m['joint_names'][jid]:openq[j]=c['u_thumb_open_rad']
  actual=d[arm+':q'][idx,e][:,hid];target=d[arm+':target'][idx,e][:,hid]
  native_limits=np.asarray(m['native_limits_rad'])[e,hid]
  err=float(np.max(np.abs(actual-openq)));terr=float(np.max(np.abs(target-openq)))
  net=d[f'e{e}:{arm}:hand_net'][idx];pairs=d[f'e{e}:{arm}:hand_partner_normal'][idx]
  account=float(np.max(np.abs(net-pairs.sum(-2))));max_accounting=max(max_accounting,account)
  force=float(np.linalg.norm(net,axis=-1).max());pairforce=float(np.linalg.norm(pairs,axis=-1).max())
  excursion=float(np.max(np.abs(d[arm+':q'][closed,e][:,hid]-openq)))
  valid=bool(np.all(openq>=native_limits[:,0]) and np.all(openq<=native_limits[:,1]))
  hident=next(x for x in identity['hands'] if x['env']==e and x['arm']==arm)
  shape_max=np.linalg.norm(pairs,axis=-1).max(0);order=np.dstack(np.unravel_index(np.argsort(shape_max.ravel())[-4:][::-1],shape_max.shape))[0]
  contacts=[dict(sensor=hident['sensors'][int(i)],partner=hident['filters'][int(i)][int(j)],max_normal_n=float(shape_max[int(i),int(j)])) for i,j in order if shape_max[int(i),int(j)]>.001]
  accepted=err<=criteria['max_error_to_proposed_open_rad'] and terr<=1e-6 and force<=criteria['max_each_body_net_n'] and pairforce<=criteria['max_each_pair_normal_n'] and account<=criteria['net_filter_reconstruction_max_abs_n'] and excursion>=criteria['closed_excursion_min_rad'] and valid
  hands.append(dict(arm=arm,reference_open_rad=openq.tolist(),hand_joint_names=[m['joint_names'][i] for i in hid],
   max_error_to_reference_rad=err,target_max_error_to_reference_rad=terr,max_each_body_net_n=force,max_each_pair_normal_n=pairforce,
   net_filter_accounting_max_abs_n=account,closed_excursion_max_rad=excursion,reference_inside_native_limits=valid,
   accepted_static_opening=bool(accepted),largest_open_contacts=contacts,states=int(idx.sum())))
 cases.append(dict(**c,hands=hands,accepted_static_opening=all(h['accepted_static_opening'] for h in hands)))
candidate=[c for c in cases if c['method']=='u_thumb_neutral'];baseline=[c for c in cases if c['method']=='original_zero']
result=dict(status='COMPLETE_OPENING_POSE_DIAGNOSTIC_NOT_TASK_ACCEPTANCE',registration_sha256=sha(a.registration),raw_receipt_sha256=sha(raw/'recording_receipt.json'),
 criteria=criteria,cases=cases,candidate_accepted=sum(c['accepted_static_opening'] for c in candidate),candidate_cases=len(candidate),
 baseline_accepted=sum(c['accepted_static_opening'] for c in baseline),baseline_cases=len(baseline),max_hand_net_filter_accounting_abs_n=max_accounting,
 verified_images=len(rec['images']),native_render_checks=len(rec['render_checks']),new_task_trials=0,new_final_trials=0,
 claim_boundary='Only contact-free opening-target attainment and removal of measured hand self contacts at these four static task poses after one close/open cycle. Not aperture, stable grasp, scene safety, policy performance, hardware, random workspace or held-out task qualification.')
a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
