"""Independent CPU scorer. Reconstruct expected targets; never import joint_paths."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--registration',type=Path,required=True);p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
sha=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest()
r=json.loads(a.registration.read_text());receipt=json.loads((a.raw/'recording_receipt.json').read_text());meta=json.loads((a.raw/'native_metadata.json').read_text());ident=json.loads((a.raw/'contact_identities.json').read_text())
assert receipt['registration_sha256']==sha(a.registration)==meta['registration_sha256']
for fn,key in [('native_metadata.json','metadata_sha256'),('contact_identities.json','identities_sha256'),('initial_states.json','initial_states_sha256'),('sparse_contact_points.json','sparse_contact_points_sha256')]:assert sha(a.raw/fn)==receipt[key]
assert receipt['steps']==r['steps'] and receipt['cases']==len(r['cases']) and not (a.raw/'failure.txt').exists()
for v in receipt['images']:assert sha(a.raw/v['file'])==v['sha256']
assert len(receipt['images'])==len(r['capture_steps'])*len(r['capture_envs'])*len(r['views'])
assert all(all(v==0 for v in check['max_abs_diff'].values()) for check in receipt['render_checks'])
rows={};last=-1;recorded=0;max_target_error=0.;max_accounting=0.;max_cache_error=0.;max_bounds=0.;worst=None
for c in receipt['chunks']:
 assert sha(a.raw/c['file'])==c['sha256']
 with np.load(a.raw/c['file']) as z:
  steps=z['step'];assert np.array_equal(steps,np.arange(last+1,c['last_step']+1));last=int(steps[-1]);recorded+=len(steps)*len(r['cases'])
  assert all(np.isfinite(z[k]).all() for k in z.files)
  for k,e in enumerate(r['cases']):
   for cyc in np.unique(z['cycle']):
    mask=z['cycle']==cyc;ph=(steps[mask]%r['cycle_steps'])*r['physics_dt_s'];profile_id=int(cyc)*r['profiles_per_cycle']+e['profile_slot'];profile=r['profiles'][profile_id]
    assert (z['profile_index'][mask,k]==profile_id).all()
    for arm in ['F_L','F_R','U_L','U_R']:
     v=meta['arms'][arm];ids=v['hand_ids'];names=[v['joint_names'][j] for j in ids];limits=np.asarray(v['soft_limits_rad'][k])[ids];op=np.zeros(len(ids),np.float32)
     if arm.startswith('U'):op[names.index(('left' if arm=='U_L' else 'right')+'_thumb_1_joint')]=.35
     goal=op.copy()
     if arm.startswith('U') and profile['kind']!='open_control':
      frac=profile['finger_fraction'];goal=(limits[:,1]*frac).astype(np.float32)
      if profile['kind']=='original_task':
       for j,name in enumerate(names):goal[j]=limits[j,1]*(.7569 if arm=='U_L' else .7546) if 'thumb' in name else limits[j,1]*(.4769 if arm=='U_L' else .4746)
      else:
       for axis in [1,2]:j=names.index(('left' if arm=='U_L' else 'right')+f'_thumb_{axis}_joint');goal[j]=limits[j,1]*profile[f'thumb{axis}_fraction']
     expected=np.tile(op,(len(ph),1)).astype(np.float64)
     closing=(ph>=1)&(ph<2);hold=(ph>=2)&(ph<3.5);opening=(ph>=3.5)&(ph<4.5)
     expected[closing]=op+np.minimum(1,(ph[closing]-1+r['physics_dt_s'])/r['ramp_s'])[:,None]*(goal-op)
     expected[hold]=goal
     expected[opening]=goal+np.minimum(1,(ph[opening]-3.5+r['physics_dt_s'])/r['ramp_s'])[:,None]*(op-goal)
     target=z[arm+':target'][mask,k][:,ids];q=z[arm+':q'][mask,k][:,ids];qd=z[arm+':qd'][mask,k][:,ids]
     tgt_err=float(np.max(abs(target-expected)));max_target_error=max(max_target_error,tgt_err)
     assert float(np.max(abs(z[arm+':goal'][mask,k]-goal)))<1e-6
     bound=max(0.,float(np.max(limits[:,0]-target)),float(np.max(target-limits[:,1])));max_bounds=max(max_bounds,bound)
     max_cache_error=max(max_cache_error,float(np.max(abs(q-z[arm+':cache_q'][mask,k][:,ids]))))
     cv=z[f'e{k}:{arm}:hand_partner_normal'][mask];net=z[f'e{k}:{arm}:hand_net'][mask]
     ac=float(np.max(abs(cv.sum(axis=2)-net)));max_accounting=max(max_accounting,ac)
     paths=next(x for x in ident['hands'] if x['env']==k and x['arm']==arm)
     # Only hand internal force for qualification; full external force reported separately.
     filters=paths['filters'];internal=np.asarray([[f'/env_{k}/{arm}/' in f and f in paths['sensors'] for f in fs] for fs in filters])
     norms=np.linalg.norm(cv,axis=-1);self_force=np.where(internal[None],norms,0).max(axis=(1,2));full_force=norms.max(axis=(1,2))
     key=(int(cyc),k,arm)
     row=rows.setdefault(key,dict(cycle=int(cyc),env=k,arm=arm,profile_index=profile_id,reference_time_s=e['reference_time_s'],kind=profile['kind'],profile=profile,states=0,path_max_self_normal_n=0.,path_max_all_partner_normal_n=0.,startup_error_rad=0.,startup_self_normal_n=0.,closed_error_rad=0.,return_error_rad=0.,return_self_normal_n=0.,peak_speed_rad_s=0.,max_target_error_rad=0.,max_accounting_error_n=0.,bounds_violation_rad=0.))
     row['states']+=int(mask.sum())
     for field,value in [('path_max_self_normal_n',float(self_force.max())),('path_max_all_partner_normal_n',float(full_force.max())),('peak_speed_rad_s',float(abs(qd).max())),('max_target_error_rad',tgt_err),('max_accounting_error_n',ac),('bounds_violation_rad',bound)]:row[field]=max(row[field],value)
     for lo,hi,field,g in [(0,1,'startup_error_rad',op),(2.5,3.5,'closed_error_rad',goal),(5,6,'return_error_rad',op)]:
      m=(ph>=lo)&(ph<hi)
      if m.any():row[field]=max(row[field],float(abs(q[m]-g).max()))
     for lo,hi,field in [(0,1,'startup_self_normal_n'),(5,6,'return_self_normal_n')]:
      m=(ph>=lo)&(ph<hi)
      if m.any():row[field]=max(row[field],float(self_force[m].max()))
     if worst is None or float(self_force.max())>worst['n']:
      ix=int(np.argmax(self_force));worst=dict(n=float(self_force[ix]),step=int(steps[mask][ix]),cycle=int(cyc),env=k,arm=arm,profile_index=profile_id)
assert last==r['steps']-1
for row in rows.values():
 assert row['states']==r['cycle_steps']
 row['startup_qualified']=row['startup_self_normal_n']<=.1 and row['startup_error_rad']<=.02
 row['closed_qualified']=row['closed_error_rad']<=.02
 row['return_qualified']=row['return_error_rad']<=.02 and row['return_self_normal_n']<=.1
 row['path_qualified']=row['path_max_self_normal_n']<=.1
 row['qualified']=row['startup_qualified'] and row['closed_qualified'] and row['return_qualified'] and row['path_qualified'] and row['max_target_error_rad']<=1e-6 and row['bounds_violation_rad']<=1e-6 and row['max_accounting_error_n']<=.001
u=[v for v in rows.values() if v['arm'].startswith('U')];qualified=[]
for i,profile in enumerate(r['profiles']):
 cases=[v for v in u if v['profile_index']==i]
 assert len(cases)==6
 if all(v['qualified'] for v in cases):qualified.append(i)
summary=dict(status='PASS_COMPLETENESS_NOT_ALL_PHYSICAL_PATHS',registration_sha256=sha(a.registration),recording_receipt_sha256=sha(a.raw/'recording_receipt.json'),native_cycles=len(r['profiles'])*3,unique_profiles=len(r['profiles']),u_hand_paths=len(u),u_paths_qualified=sum(v['qualified'] for v in u),profiles_qualified_at_all_three_references=qualified,profile_count_qualified=len(qualified),env_states=recorded,original_images=len(receipt['images']),max_target_error_rad=max_target_error,max_accounting_error_n=max_accounting,max_cache_error_rad=max_cache_error,max_bounds_violation_rad=max_bounds,worst_self_contact=worst,constructor_contacts_qualified=False,new_task_trials=0,system0_trials=0,rows=list(rows.values()))
a.out.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))
