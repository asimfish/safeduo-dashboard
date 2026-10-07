"""Independent CPU measurement and command oracle; imports no controller module."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from collections import Counter
p=Path(__file__).resolve().parent;pa=argparse.ArgumentParser();pa.add_argument('--name',required=True);a=pa.parse_args();name=a.name.upper();raw=Path('/mnt/nas/data/lyf/double_hand/safety_hand_closure_qualification_20261007')/a.name
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
r=json.loads((p/f'REGISTRATION_{name}.json').read_text());receipt=json.loads((raw/'recording_receipt.json').read_text());m=json.loads((raw/'native_metadata.json').read_text());ident=json.loads((raw/'contact_identities.json').read_text());events=json.loads((raw/'admission_records.json').read_text());reqs=[x for x in events if 'decision' in x];aborts=[x for x in events if 'abort_step' in x];dt=r['physics_dt_s']
assert receipt['steps']==r['steps'] and not (raw/'failure.txt').exists()
assert receipt['registration_sha256']==sha(p/f'REGISTRATION_{name}.json')==m['registration_sha256']
for fn,key in [('native_metadata.json','metadata_sha256'),('contact_identities.json','identities_sha256'),('initial_states.json','initial_states_sha256'),('sparse_contact_points.json','sparse_contact_points_sha256')]:assert sha(raw/fn)==receipt[key]
for im in receipt['images']:assert sha(raw/im['file'])==im['sha256']
assert len(receipt['images'])==sum(len(r.get('capture_envs_by_step',{}).get(str(s),r['capture_envs']))*len(r.get('capture_views_by_step',{}).get(str(s),list(r['views']))) for s in r['capture_steps'])
assert all(all(v==0 for v in x['max_abs_diff'].values()) for x in receipt['render_checks'])
rows={};traces={};frames={};last=-1;point_error=0.;triangle_error=0.;countmax=0;accounting=0.;req_by_step={(x['env'],x['arm'],x['step']-1):x for x in reqs}
def params(e,arm,cycle):
 v=m['arms'][arm];h=v['hand_ids'];names=[v['joint_names'][i] for i in h];lim=np.asarray(v['soft_limits_rad'][e],np.float32)[h];op=np.zeros(len(h),np.float32);profile=r['profiles'][cycle*r['profiles_per_cycle']+r['cases'][e]['profile_slot']]
 if arm.startswith('U'):op[names.index(('left' if arm=='U_L' else 'right')+'_thumb_1_joint')]=.35
 g=op.copy()
 if arm.startswith('U') and profile['kind']!='open_control':
  g=lim[:,1]*np.float32(profile['finger_fraction'])
  if profile['kind']=='original_task':
   for j,n in enumerate(names):g[j]=lim[j,1]*np.float32((.7569 if arm=='U_L' else .7546) if 'thumb' in n else (.4769 if arm=='U_L' else .4746))
  else:
   for axis in [1,2]:j=names.index(('left' if arm=='U_L' else 'right')+f'_thumb_{axis}_joint');g[j]=lim[j,1]*np.float32(profile[f'thumb{axis}_fraction'])
 return h,lim,op,g,profile
def nominal(e,arm,steps):
 out=[]
 for step in steps:
  cycle=int(step)//720;h,lim,op,g,profile=params(e,arm,cycle);ph=(int(step)%720)*dt
  if ph<1:want=op
  elif ph<2:want=op+min(1.,ph-1+dt)*(g-op)
  elif ph<3.5:want=g
  elif ph<4.5:want=g+min(1.,ph-3.5+dt)*(op-g)
  else:want=op
  if any(x['env']==e and x['arm']==arm and x.get('cycle',0)==cycle and not x['decision']['admitted'] for x in reqs):want=op
  out.append(want)
 return np.asarray(out)
def expected(e,arm,steps):
 want=nominal(e,arm,steps);bs=[x for x in aborts if x['env']==e and x['arm']==arm]
 if bs:
  start=min(x['abort_step'] for x in bs);prior=nominal(e,arm,[start-1])[0];op=params(e,arm,0)[2];mask=steps>=start
  if r['cases'][e].get('method')=='fault_hold':want[mask]=prior
  else:want[mask]=prior+np.minimum(1.,(steps[mask]-start+1)*dt)[:,None]*(op-prior)
 return want
for ch in receipt['chunks']:
 assert sha(raw/ch['file'])==ch['sha256']
 with np.load(raw/ch['file']) as z:
  steps=z['step'];assert np.array_equal(steps,np.arange(last+1,ch['last_step']+1));last=int(steps[-1]);assert all(np.isfinite(z[k]).all() for k in z.files)
  cycle=int(z['cycle'][0]);assert (z['cycle']==cycle).all();ph=(steps%720)*dt
  for e,c in enumerate(r['cases']):
   for arm in ['F_L','F_R','U_L','U_R']:
    h,lim,op,g,profile=params(e,arm,cycle);q=z[arm+':q'][:,e,h];qd=z[arm+':qd'][:,e,h];target=z[arm+':target'][:,e,h];assert np.max(abs(z[arm+':goal'][:,e]-g))<1e-6
    pid=cycle*r['profiles_per_cycle']+c['profile_slot'];assert (z['profile_index'][:,e]==pid).all()
    cv=z[f'e{e}:{arm}:hand_partner_normal'];net=z[f'e{e}:{arm}:hand_net'];ac=float(abs(cv.sum(axis=2)-net).max());accounting=max(accounting,ac)
    paths=next(x for x in ident['hands'] if x['env']==e and x['arm']==arm);internal=np.asarray([[f in paths['sensors'] for f in fs] for fs in paths['filters']]);scalar=z[f'e{e}:{arm}:hand_partner_scalar_normal'];vector=np.linalg.norm(cv,axis=-1)
    if arm.startswith('U'):
     counts=z[f'e{e}:{arm}:hand_normal_count'];starts=z[f'e{e}:{arm}:hand_point_start'];points=z[f'e{e}:{arm}:hand_point_force'];assert (counts>=0).all() and np.all((counts==0)|((starts>=0)&(starts+counts<=r['contact_capacity'])))
     ss=np.where(counts>0,starts,0).astype(int);end=ss+counts.astype(int);prefix=np.pad(np.abs(points).astype(np.float64).cumsum(axis=1),((0,0),(1,0)));ix=np.arange(len(points))[:,None,None];oracle=prefix[ix,end]-prefix[ix,ss]
     err=abs(oracle-scalar);point_error=max(point_error,float(err.max()));assert np.all(err<=.001+2e-6*oracle);scalar=oracle
     countmax=max(countmax,int(counts.sum(axis=(1,2)).max()));assert np.all(vector<=scalar+.001+2e-6*scalar)
     triangle_error=max(triangle_error,float((vector-scalar).max()))
    selfn=np.where(internal[None],scalar,0).max(axis=(1,2));alln=scalar.max(axis=(1,2));extn=np.where(~internal[None],scalar,0).max(axis=(1,2));ter=float(abs(target-expected(e,arm,steps)).max());qerr=abs(q-op).max(axis=1);speed=abs(qd).max(axis=1)
    key=(cycle,e,arm);row=rows.setdefault(key,dict(cycle=cycle,env=e,arm=arm,profile_index=pid,profile=profile,kind=profile['kind'],reference_time_s=c['reference_time_s'],method=c.get('method','validation'),states=0,path_max_self_normal_n=0.,path_max_scalar_self_normal_n=0.,path_max_all_partner_normal_n=0.,path_max_scalar_all_partner_normal_n=0.,path_max_unintended_normal_n=0.,closed_error_rad=0.,startup_error_rad=0.,return_error_rad=0.,neutral_error_rad=0.,return_all_partner_normal_n=0.,peak_speed_rad_s=0.,max_target_error_rad=0.,max_accounting_error_n=0.,bounds_violation_rad=0.,state_bounds_violation_rad=0.,violation_steps=0))
    row['states']+=len(steps);row['violation_steps']+=int(((alln>np.float32(.1))|(speed>3)).sum())
    for field,val in [('path_max_self_normal_n',float(np.where(internal[None],vector,0).max())),('path_max_scalar_self_normal_n',float(selfn.max())),('path_max_all_partner_normal_n',float(vector.max())),('path_max_scalar_all_partner_normal_n',float(alln.max())),('path_max_unintended_normal_n',float(extn.max())),('neutral_error_rad',float(qerr.max())),('peak_speed_rad_s',float(speed.max())),('max_target_error_rad',ter),('max_accounting_error_n',ac),('bounds_violation_rad',max(0.,float((lim[:,0]-target).max()),float((target-lim[:,1]).max()))),('state_bounds_violation_rad',max(0.,float((lim[:,0]-.02-q).max()),float((q-lim[:,1]-.02).max())))]:row[field]=max(row[field],val)
    for lo,hi,field,goal in [(0,1,'startup_error_rad',op),(2.5,3.5,'closed_error_rad',g),(5,6,'return_error_rad',op)]:
     mask=(ph>=lo)&(ph<hi)
     if mask.any():row[field]=max(row[field],float(abs(q[mask]-goal).max()))
    mask=(ph>=5)&(ph<6)
    if mask.any():row['return_all_partner_normal_n']=max(row['return_all_partner_normal_n'],float(alln[mask].max()))
    if arm.startswith('U'):
     traces.setdefault((e,arm),[]).append(dict(steps=steps.copy(),qerr=qerr,speed=speed,alln=alln,selfn=selfn,external=extn))
     for ix,st in enumerate(steps):
      if (e,arm,int(st)) in req_by_step:frames[e,arm,int(st)]=dict(q=q[ix],qd=qd[ix],max_n=alln[ix])
assert last==r['steps']-1
for row in rows.values():
 assert row['states']==720
 req=next((x for x in reqs if x['env']==row['env'] and x['arm']==row['arm'] and x.get('cycle',0)==row['cycle']),None);row['requested']=req is not None;row['admitted']=bool(req and req['decision']['admitted']);row['refusal_reason']=req['decision']['reason'] if req and not req['decision']['admitted'] else None
 row['qualified']=bool(row['closed_error_rad']<=.02 and row['startup_error_rad']<=.02 and row['return_error_rad']<=.02 and row['path_max_scalar_all_partner_normal_n']<=np.float32(.1) and row['peak_speed_rad_s']<=3 and row['max_target_error_rad']<=1e-6 and row['max_accounting_error_n']<=.001 and row['bounds_violation_rad']<=1e-6 and row['state_bounds_violation_rad']<=0)
 row['executed_and_qualified']=row['admitted'] and row['qualified'];row['neutral_safe']=bool(row['neutral_error_rad']<=.02 and row['path_max_scalar_all_partner_normal_n']<=np.float32(.1) and row['peak_speed_rad_s']<=3 and row['max_target_error_rad']<=1e-6)
for req in reqs:
 e,arm,prev=req['env'],req['arm'],req['step']-1;st=frames[e,arm,prev];assert req['native_state_step']==prev;op=np.asarray(r['passport']['open_rad'][arm],np.float32);goal=np.asarray(req['request_goal_rad'],np.float32);allow=any(np.array_equal(goal,np.asarray(x,np.float32)) for x in r['passport']['allowed_goals_rad'][arm]);excluded=any(x['arm']==arm and x['reference_time_s']==r['cases'][e]['reference_time_s'] and np.array_equal(goal,np.asarray(x['goal_rad'],np.float32)) for x in r['passport']['context_exclusions']);eligible=allow and not excluded and st['max_n']<=np.float32(.1) and abs(st['qd']).max()<=3 and abs(st['q']-op).max()<=.02
 if req['decision']['admitted']:assert eligible,req
 if req['decision']['reason']=='known_unsafe_context':assert excluded,req
 assert all(not x['decision']['admitted'] for x in req['unknown_decisions']);assert req['original_task_rejection']!='qualified_exact_path'
recoveries=[]
for ab in aborts:
 e,arm,start=ab['env'],ab['arm'],ab['abort_step'];t=traces[e,arm];steps=np.concatenate([x['steps'] for x in t]);qerr=np.concatenate([x['qerr'] for x in t]);speed=np.concatenate([x['speed'] for x in t]);alln=np.concatenate([x['alln'] for x in t]);safe=(qerr<=.02)&(speed<=3)&(alln<=np.float32(.1))&(steps>=start);consecutive=np.convolve(safe.astype(int),np.ones(6,dtype=int),'valid');index=np.flatnonzero(consecutive==6);first=int(steps[index[0]+5]) if len(index) else None;post=steps>=start
 recoveries.append(dict(**ab,method=r['cases'][e].get('method','validation'),first_six_safe_frames_end_step=first,recovery_s=None if first is None else (first-start+1)*dt,post_abort_max_n=float(alln[post].max()),last_second_max_n=float(alln[steps>=r['steps']-120].max()),last_second_neutral_error_rad=float(qerr[steps>=r['steps']-120].max()),post_abort_contact_impulse_proxy_n_s=float(alln[post].sum()*dt),contact_proxy_note='sum over time of maximum measured sensor/partner magnitude; not total physical wrench impulse',latch_cleared=False))
u=[x for x in rows.values() if x['arm'].startswith('U')];admitted=[x for x in u if x['admitted']];guard=[x for x in u if x['method']=='guard'];bypass=[x for x in u if x['method']=='bypass'];denied=[x for x in u if x['requested'] and not x['admitted']]
init_q_diff=0.;init_obj_diff=0.;init_vel_diff=0.
if a.name.startswith('paired'):
 initial=json.loads((raw/'initial_states.json').read_text())[-1]
 for left,right in [(0,1),(2,3),(4,5),(6,7)]:
  for arm in ['F_L','F_R','U_L','U_R']:
   for field in ['q','qd','target']:
    values=np.asarray(initial['arms'][arm][field]);diff=float(abs(values[left]-values[right]).max());init_q_diff=max(init_q_diff,diff);assert diff==0
  for obj in ['beam700','beam300']:
   poses=np.asarray(initial['objects'][obj]['pose']);origins=np.asarray(m['origins']);diff=float(abs((poses[left,:3]-origins[left])-(poses[right,:3]-origins[right])).max());init_obj_diff=max(init_obj_diff,diff);assert diff<=1e-6;vel=np.asarray(initial['objects'][obj]['velocity']);diff=float(abs(vel[left]-vel[right]).max());init_vel_diff=max(init_vel_diff,diff);assert diff==0
status='PASS_CONDITIONAL_EXECUTIONS_WITH_REFUSALS' if all(x['executed_and_qualified'] for x in admitted) and all(x['neutral_safe'] for x in denied) and not aborts else 'REJECT_CONDITIONAL_HAND_CANDIDATE'
if guard:status='PASS_MATCHED_HAND_ADMISSION_COUNTERFACTUAL' if all(x['neutral_safe'] for x in guard) and len(guard)==6 else 'REJECT_MATCHED_HAND_ADMISSION_COUNTERFACTUAL'
result=dict(status=status,run=a.name,registration_sha256=sha(p/f'REGISTRATION_{name}.json'),recording_receipt_sha256=sha(raw/'recording_receipt.json'),admission_records_sha256=sha(raw/'admission_records.json'),env_states=len(r['cases'])*r['steps'],native_cycles=len(r['cases'])*(r['steps']//720),unique_profiles=len(r['profiles']),u_hand_paths=len(u),u_paths_qualified=sum(x['qualified'] for x in u),executed_requests=len(admitted),executed_paths_qualified=sum(x['executed_and_qualified'] for x in admitted),denied_requests=len(denied),denial_reasons=dict(Counter(x['refusal_reason'] for x in denied)),denied_neutral_safe=sum(x['neutral_safe'] for x in denied),original_images=len(receipt['images']),max_target_error_rad=max(x['max_target_error_rad'] for x in rows.values()),max_accounting_error_n=accounting,max_point_reconstruction_error_n=point_error,max_vector_excess_n=triangle_error,max_normal_count=countmax,max_scalar_all_partner_normal_n=max(x['path_max_scalar_all_partner_normal_n'] for x in u),max_scalar_self_normal_n=max(x['path_max_scalar_self_normal_n'] for x in u),abort_count=len(aborts),recoveries=recoveries,point_scalar_oracle='independent float64 prefix sums of absolute valid native point magnitudes',first_post_init_physics_step_exempted=False,constructor_contacts_qualified=False,new_task_trials=0,system0_trials=0,unknown_rejections=sum(x['unknown_rejections'] for x in reqs),unique_unknown_profiles=32,new_unknown_physical_paths=0,guard_paths=len(guard),guard_safe_paths=sum(x['neutral_safe'] for x in guard),guard_max_scalar_self_normal_n=max([x['path_max_scalar_self_normal_n'] for x in guard],default=0),guard_max_scalar_all_partner_normal_n=max([x['path_max_scalar_all_partner_normal_n'] for x in guard],default=0),bypass_max_scalar_self_normal_n=max([x['path_max_scalar_self_normal_n'] for x in bypass],default=0),guard_grasp_successes=0,matched_init_joint_max_abs_rad=init_q_diff,matched_object_local_max_abs_m=init_obj_diff,matched_object_velocity_max_abs=init_vel_diff,rows=list(rows.values()))
(p/f'{name}_FULL_RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
