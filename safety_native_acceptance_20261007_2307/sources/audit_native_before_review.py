"""Actual same-run two-boundary/native/geometry/camera acceptance oracle."""
from pathlib import Path
import json,hashlib,sys,copy,datetime
import numpy as np
from PIL import Image
H=Path(__file__).resolve().parent
ARMS=('F_L','F_R','U_L','U_R');WIDTHS=(7,7,6,6)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def equal(a,b,label):
 if not np.array_equal(a,b):raise AssertionError(label)
def packet_check(v,cell,t):
 for b,expected in [('pre','pre_control_pre_physics'),('physics','pre_physics'),('post','post_physics_pre_render')]:
  if int(v[b+'_frame'])!=t or str(v[b+'_boundary'])!=expected:raise AssertionError('frame/boundary')
  for k,val in v.items():
   if k.startswith(b+'_') and np.issubdtype(val.dtype,np.number) and not np.isfinite(val).all():raise AssertionError('native finite '+k)
  qs=[];qds=[];issued=[];applied=[];pending=[];history=[]
  for a,w in zip(ARMS,WIDTHS):
   p=b+'_'+a+'_';q=v[p+'native_q'];qd=v[p+'native_qd'];idx=v[p+'controlled_joint_indices'].astype(int)
   if q.shape!=qd.shape or q.shape[0]!=64 or len(idx)!=w or len(v[p+'native_joint_names'])!=q.shape[1]:raise AssertionError('native full identity')
   if v[p+'native_root_xyzw'].shape!=(64,7) or v[p+'native_root_velocity'].shape!=(64,6):raise AssertionError('native root identity')
   qs.append(q[:,idx]);qds.append(qd[:,idx]);issued.append(v[p+'issued_controlled_target']);applied.append(v[p+'applied_controlled_target']);pending.append(v[p+'pending_controlled_targets']);history.append(v[p+'pending_project_targets'])
  q=np.concatenate(qs,-1);qd=np.concatenate(qds,-1);targets=np.concatenate(issued,-1);act=np.concatenate(applied,-1);fifo=np.concatenate(pending,-1);hist=np.concatenate(history,-1)
  equal(q,cell['q'][t] if b=='post' else cell['q_initial'] if t==0 else cell['q'][t-1],'native controlled q vs cell')
  if b=='physics':
   equal(qd,cell['pre_qd_compact'][t],'actual pre-physics qd')
   for a in ARMS:
    for key in ['native_q','native_qd','native_root_xyzw','native_root_velocity']:
     equal(v['physics_'+a+'_'+key],v['pre_'+a+'_'+key],'control must not advance native physics')
  if b=='pre':
   equal(qd,cell['pre_qd_compact'][t],'native pre qd vs scored');equal(fifo,cell['pre_pending_actuator_targets'][t],'native pre FIFO');equal(hist,cell['pre_pending_project_history'][t],'native pre projection history')
   equal(targets,cell['q_initial'] if t==0 else cell['controller_target'][t-1],'native pre issued target')
  else:
   equal(targets,cell['controller_target'][t],'native post issued target');equal(act,cell['actuator_target'][t],'native post applied target')
   expected=np.stack([cell['q_initial'] if t-5+i<0 else cell['controller_target'][t-5+i] for i in range(6)])
   equal(fifo,expected,'native post FIFO6 truth')
   equal(act,cell['q_initial'] if t-6<0 else cell['controller_target'][t-6],'actual applied FIFO6 truth')
   if b=='post' and t+1<len(cell['q']):equal(qd,cell['pre_qd_compact'][t+1],'native post qd vs next pre')
def camera_check(root,cap,cell,packet,identity):
 sp=root/cap['state'];assert sha(sp)==cap['sha256'];s=json.loads(sp.read_text());t,e=s['step'],s['env_id']
 assert s['numeric_binding'].startswith('same actual physics run') and len(s['views'])==9 and len(s['images'])==9
 bf=root/s['fresh_native_snapshot'];af=root/s['fresh_native_after_snapshot'];assert sha(bf)==s['fresh_native_before_sha256'] and sha(af)==s['fresh_native_after_sha256'];before,after=load(bf),load(af)
 assert set(before)==set(after) and len(before)==16
 compact=[]
 for a in ARMS:
  idx=s['fresh_native_controlled_joint_indices'][a];compact.extend(s['arms'][a]['q'])
  for field,key in [('q','native_q'),('qd','native_qd'),('root','native_root_xyzw'),('root_vel','native_root_velocity')]:
   equal(before[a+'_'+field],after[a+'_'+field],'render full native unchanged')
   equal(before[a+'_'+field],packet['post_'+a+'_'+key],'render native vs exact post boundary')
  equal(before[a+'_q'][e,idx],np.asarray(s['arms'][a]['q'],np.float32),'camera controlled q');equal(before[a+'_qd'][e,idx],np.asarray(s['arms'][a]['qd'],np.float32),'camera controlled qd')
 equal(np.asarray(compact,np.float32),cell['q'][t,e],'image-state vs same numeric trajectory')
 centers=np.asarray(s['sphere_centers_world_m']);radii=np.asarray(s['sphere_radii_m']);arms=np.asarray(identity['sphere_arm_id']);classes=np.asarray(identity['class_id']);pairs=np.asarray(identity['pair_sphere_idx']);cls=classes.copy();cls[(classes==1)&(arms[pairs[:,0]]>=2)]=2;cls[classes==2]=3
 d=np.asarray(s['full_distances_m'],np.float32);ex=np.asarray(s['full_exempt'],bool);m=np.asarray([np.where(ex|(cls!=k),np.inf,d).min() for k in range(4)],np.float32)
 equal(m,cell['official_margins'][t,e],'same-image raw geometry vs original score')
 minimum=np.inf
 for name,v in s['views'].items():
  W=np.asarray(v['actual_camera_to_world_row_matrix']);K=np.asarray(v['actual_intrinsic_matrix']);clip=np.asarray(v['actual_clipping_range_m']);assert np.isfinite(W).all() and np.isfinite(K).all();assert np.allclose(W[:3,:3]@W[:3,:3].T,np.eye(3),atol=1e-6,rtol=0)
  mask=arms<2 if name.startswith('f_') else arms>=2 if name.startswith('u_') else np.ones(len(arms),bool)
  pc=np.c_[centers[mask],np.ones(mask.sum())]@np.linalg.inv(W);x,y,z=pc[:,0],pc[:,1],-pc[:,2];rad=radii[mask];fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2]
  plane=np.stack([(fx*x+cx*z)/np.hypot(fx,cx)-rad,((1280-cx)*z-fx*x)/np.hypot(fx,1280-cx)-rad,(cy*z-fy*y)/np.hypot(fy,cy)-rad,((720-cy)*z+fy*y)/np.hypot(fy,720-cy)-rad,z-clip[0]-rad,clip[1]-z-rad],-1)
  assert plane.min()>=.049999 and np.allclose(plane.min(0),v['observed_sphere_frustum']['minimum_by_plane_m'],rtol=0,atol=1e-7);minimum=min(minimum,float(plane.min()))
  item=next(a for a in s['images'] if Path(a['path']).stem.endswith('_'+name));p=root/item['path'];assert sha(p)==item['sha256']
  with Image.open(p) as im:
   im.load();assert im.size==(1280,720);assert np.asarray(im).std()>1,'blank camera image'
 return minimum

def audit(root):
 root=Path(root);proto=json.loads((root/'visual_protocol.json').read_text());assert proto['status']=='complete';cell=load(root/'cell_001.npz');T=len(cell['q']);assert T==proto['steps'] and cell['q'].shape==(T,64,26)
 nr=json.loads((root/'native_receipts.json').read_text());cr=json.loads((root/'camera_receipts.json').read_text());assert nr['steps']==T;identity=json.loads((root/'full_row_identity.json').read_text());assert identity['rows']==9021
 bystep={}
 for cap in cr['receipts']:bystep.setdefault(cap['step'],[]).append(cap)
 seen=0;groups=0;minimum=np.inf;sample=None;previous=None
 initial=load(root/'native_initial.npz')
 for r in nr['chunks']:
  p=root/r['path'];assert sha(p)==r['sha256'];z=load(p);assert r['start']==seen and len(z['pre_frame'])==r['stop']-r['start']
  for i,t in enumerate(range(r['start'],r['stop'])):
   v={k:x[i] for k,x in z.items()};packet_check(v,cell,t)
   for a in ARMS:
    for k in ['native_joint_names','controlled_joint_indices']:
     for b in ['pre','physics','post']:equal(v[b+'_'+a+'_'+k],initial[a+'_'+k],'full native joint identity unchanged')
    for k in ['native_q','native_qd','native_root_xyzw','native_root_velocity']:
     equal(v['pre_'+a+'_'+k],initial[a+'_'+k] if previous is None else previous['post_'+a+'_'+k],'full native continuation between scored frames')
   previous=v
   if sample is None:sample=(v,t)
   for cap in bystep.get(t,[]):minimum=min(minimum,camera_check(root,cap,cell,v,identity));groups+=1
   seen+=1
  assert sha(p)==r['sha256']
 assert seen==T and groups==cr['groups'] and cr['PNG']==groups*9
 assignments=load(root/'bank_assignment.npz')['risk_pair_index'];bad=(cell['official_margins']<0).any(-1)
 scheduled={(r['env_id'],r['step']) for r in cr['receipts'] if r['capture_kind']=='scheduled'}
 assert scheduled=={(e,t) for e in proto['slots'] for t in proto['capture_steps']}
 actual={}
 for cap in cr['receipts']:
  if cap['capture_kind']=='first_failure':
   e,t=cap['env_id'],cap['step'];label=int(assignments[e]);assert label not in actual;hit=bad[:,assignments==label].any(-1);assert hit.any() and t==int(hit.argmax()) and e==int(np.flatnonzero(bad[t]&(assignments==label))[0]);actual[label]=(e,t)
 assert set(actual)=={int(x) for x in np.unique(assignments) if bad[:,assignments==x].any()}
 # Mutate real captured packet in memory; each error must be rejected by same oracle.
 caught=[];packet,t=sample
 for key in ['post_F_L_native_q','pre_F_L_native_qd','physics_F_L_native_root_xyzw','physics_U_R_native_root_velocity','post_U_R_issued_controlled_target','post_U_L_applied_controlled_target','pre_F_R_pending_controlled_targets','post_U_R_pending_controlled_targets','pre_F_L_pending_project_targets','post_frame']:
  changed={k:x.copy() for k,x in packet.items()};arr=changed[key]
  if arr.ndim:arr.flat[0]+=.125
  else:changed[key]=arr+1
  try:packet_check(changed,cell,t)
  except AssertionError:caught.append(key)
  else:raise AssertionError('negative control escaped '+key)
 return dict(status='PASS_SAME_RUN_NATIVE_AND_NINE_VIEW_BINDING',root=str(root),cell_sha256=sha(root/'cell_001.npz'),frames=T,windows=64,boundary_packets=3*T,PNG=groups*9,groups=groups,minimum_frustum_clearance_m=minimum,negative_controls_caught=caught,native_controlled_q_qd_exact=True,actual_FIFO6_exact=True,all64_render_native_unchanged=True,original_full_geometry_score_exact=True,exact_restore_certified=False,occlusion_certified=False,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
if __name__=='__main__':
 result=audit(sys.argv[1]);p=H/(sys.argv[2]+'.json')
 with p.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result),flush=True)
