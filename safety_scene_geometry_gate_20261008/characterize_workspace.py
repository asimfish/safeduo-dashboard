"""Describe observed coverage; never infer a reachable volume from marginals."""
import json,hashlib,itertools
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parent;raw=Path('/mnt/nas/data/lyf/double_hand/safety_scene_geometry_gate_20261008/fresh_scene_batch');sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
r=json.loads((p/'REGISTRATION_FRESH_SCENE_BATCH.json').read_text());m=json.loads((raw/'native_metadata.json').read_text());schema=json.loads((raw/'geometry_schema.json').read_text());traj=np.load(r['trajectory']);bank=np.array(r['fresh_reference_frames']);origins=np.array(m['origins']);arms=list(m['arms']);records=[]
with np.load(raw/'dense_0000_0119.npz') as z:poses={arm:z[arm+':link_pose'][0].astype(np.float64) for arm in arms}
def rotation(q):
 x,y,z,w=q;assert abs(np.linalg.norm(q)-1)<1e-5
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
boxes={};wrist={}
for arm in arms:
 v=m['arms'][arm];ids=v['arm_ids'];lim=np.array(v['soft_limits_rad'])[0,ids,:];q=traj['q_'+arm];selected=q[bank];norm=(q-lim[:,0])/(lim[:,1]-lim[:,0]);selnorm=norm[bank]
 for j,jid in enumerate(ids):
  records.append(dict(arm=arm,joint=v['joint_names'][jid],limits_rad=lim[j].tolist(),trajectory_minmax_rad=[float(q[:,j].min()),float(q[:,j].max())],bank_minmax_rad=[float(selected[:,j].min()),float(selected[:,j].max())],trajectory_span_fraction=float(np.ptp(norm[:,j])),bank_span_fraction=float(np.ptp(selnorm[:,j])),bank_bins_occupied_of_20=int(len(np.unique(np.clip(np.floor(selnorm[:,j]*20),0,19)))),trajectory_bins_occupied_of_20=int(len(np.unique(np.clip(np.floor(norm[:,j]*20),0,19))))))
 name='fr3_link8' if arm.startswith('F') else 'wrist_3_link';idx=v['body_names'].index(name);wrist[arm]=dict(body=name,sample_positions_m=(poses[arm][:,idx,:3]-origins).tolist(),xyz_span_m=np.ptp(poses[arm][:,idx,:3]-origins,axis=0).tolist())
 for e in range(len(bank)):
  group=[]
  for c in schema['colliders']:
   if c['arm']!=arm:continue
   pose=poses[arm][e,v['body_names'].index(c['body'])];points=np.array(c['corners'])@rotation(pose[3:]).T+pose[:3]-origins[e];group.append((c['path'],points.min(0),points.max(0)))
  boxes[e,arm]=group
pairs=[]
for a,b in itertools.combinations(arms,2):
 rows=[]
 for e in range(len(bank)):
  best=None
  for ap,al,ah in boxes[e,a]:
   for bp,bl,bh in boxes[e,b]:
    d=float(np.linalg.norm(np.maximum(0,np.maximum(al-bh,bl-ah))))
    if best is None or d<best['aabb_distance_lower_bound_m']:best=dict(aabb_distance_lower_bound_m=d,collider_a=ap,collider_b=bp)
  rows.append(dict(env=e,reference_time_s=r['cases'][e]['reference_time_s'],**best))
 pairs.append(dict(arms=[a,b],minimum_aabb_distance_lower_bound_m=min(x['aabb_distance_lower_bound_m'] for x in rows),zero_aabb_distance_samples=sum(x['aabb_distance_lower_bound_m']==0 for x in rows),samples=rows))
out=dict(status='DESCRIPTIVE_OBSERVED_COVERAGE_NOT_GLOBAL_QUALIFICATION',trajectory_sha256=sha(r['trajectory']),registration_sha256=sha(p/'REGISTRATION_FRESH_SCENE_BATCH.json'),first_native_chunk_sha256=sha(raw/'dense_0000_0119.npz'),geometry_schema_sha256=sha(raw/'geometry_schema.json'),reference_frames=bank.tolist(),joint_dimensions=len(records),joints=records,wrist_positions=wrist,six_arm_pairs=pairs,new_physics_trials=0,full_joint_space_support_fraction=0,scope='26 marginal joint intervals and 24 observed neutral-hand configurations only. Twenty-bin occupancies are per joint, not a 26-dimensional joint-space coverage fraction. Robot AABB distance is a conservative geometric lower bound; zero can be false overlap and is not a contact or safety finding. No motion, constructor, grasp or full System0 acceptance.')
assert len(records)==26 and len(pairs)==6
(p/'WORKSPACE_CHARACTERIZATION.json').write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k not in ['joints','wrist_positions','six_arm_pairs','reference_frames']})
