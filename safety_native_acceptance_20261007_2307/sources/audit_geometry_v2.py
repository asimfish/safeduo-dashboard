"""Score every post-step from full9021 raw geometry, not producer summaries."""
from pathlib import Path
import json,sys,hashlib,datetime
import numpy as np
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def audit(root):
 root=Path(root);cell=load(root/'cell_001.npz');identity=json.loads((root/'full_row_identity.json').read_text());classes=np.asarray(identity['class_id']);pairs=np.asarray(identity['pair_sphere_idx']);arms=np.asarray(identity['sphere_arm_id']);labels=classes.copy();labels[(classes==1)&(arms[pairs[:,0]]>=2)]=2;labels[classes==2]=3
 T=len(cell['q']);computed=np.zeros((T,64,4),np.float32);covered=np.zeros(T,bool);receipt=json.loads((root/'forecast_receipts.json').read_text());assert receipt['steps']==T and receipt['rows']==9021
 def margins(d,ex):
  assert d.shape==(64,9021) and ex.shape==d.shape and d.dtype==np.float32 and np.isfinite(d).all()
  return np.stack([np.where(ex|(labels[None,:]!=k),np.inf,d).min(-1) for k in range(4)],-1)
 seen=0;initial_margins=None
 for chunk in receipt['chunks']:
  p=root/chunk['path'];assert sha(p)==chunk['sha256'];z=load(p);assert chunk['start']==seen
  for i,t in enumerate(range(chunk['start'],chunk['stop'])):
   d=z['measured_d'][i];ex=np.unpackbits(z['exempt'][i],axis=-1,count=9021).astype(bool);m=margins(d,ex)
   if t==0:initial_margins=m.copy()
   if t>0:computed[t-1]=m;covered[t-1]=True
   seen+=1
  assert sha(p)==chunk['sha256']
 assert seen==T
 z=load(root/'post_geometry_final.npz');computed[T-1]=margins(z['d'],z['exempt']);covered[T-1]=True
 assert covered.all() and np.array_equal(computed,cell['official_margins']),'full raw vs official native float32 mismatch'
 strict=(computed<0).any(-1);deep=(computed<-.005).any(-1);dt=json.loads(str(cell['meta_json']))['dt'];q=cell['q'].astype(np.float64);increments=np.diff(np.concatenate([cell['q_initial'][None].astype(np.float64),q]),axis=0);path=np.linalg.norm(increments,axis=-1).sum(0)
 ranges=np.ptp(np.concatenate([cell['q_initial'][None],cell['q']]),axis=0);splits=[slice(0,7),slice(7,14),slice(14,20),slice(20,26)];fourmoving=np.stack([np.linalg.norm(increments[...,s],axis=-1)>.0005 for s in splits],-1).all(-1)
 first=[int(np.flatnonzero(strict[:,e])[0]) if strict[:,e].any() else None for e in range(64)];assign=load(root/'bank_assignment.npz')['risk_pair_index']
 initial_bad=(initial_margins<0).any(-1)
 equal_initial=np.array_equal(initial_bad,cell['initial_violation'])
 result=dict(initial_negative_envs=int(initial_bad.sum()),initial_negative_ids=np.flatnonzero(initial_bad).tolist(),initial_raw_minimum_by_class_m=initial_margins.min(0).tolist(),initial_producer_flag_exact=equal_initial,status='PASS_ALL_FULL_RAW_FLOAT32_GEOMETRY_SCORING',root=str(root),windows=64,frames=T,geometry_rows=9021,raw_env_frames=T*64,strict_windows=int(strict.any(0).sum()),deep_windows=int(deep.any(0).sum()),strict_env_frames=int(strict.sum()),deep_env_frames=int(deep.sum()),strict_env_s=float(strict.sum()*dt),deep_env_s=float(deep.sum()*dt),strict_ids=np.flatnonzero(strict.any(0)).tolist(),deep_ids=np.flatnonzero(deep.any(0)).tolist(),first_failure_steps=first,minimum_by_class_m=computed.min((0,1)).tolist(),q_l2_path_by_env=path.tolist(),mean_q_l2_path=float(path.mean()),four_arms_moving_fraction=float(fourmoving.mean()),joint_range_by_env=ranges.tolist(),command_abs_max=float(np.abs(cell['external_unscaled_cmd']).max()),stratum_metrics=[dict(label=int(l),windows=int((assign==l).sum()),strict=int(strict[:,assign==l].any(0).sum()),deep=int(deep[:,assign==l].any(0).sum()),path_mean=float(path[assign==l].mean())) for l in np.unique(assign)],cell_sha256=sha(root/'cell_001.npz'),future_safety_certified=False,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 # A threshold-relaxation mutation must change outcomes when any violation exists.
 if strict.any():assert not np.array_equal(strict,(computed<computed.min()-1).any(-1))
 return result
if __name__=='__main__':
 r=audit(sys.argv[1]);p=H/(sys.argv[2]+'.json')
 with p.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
 print(r['status'],'strict',r['strict_windows'],'deep',r['deep_windows'],flush=True)
