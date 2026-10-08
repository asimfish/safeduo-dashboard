"""Passive all-arm rigid-owner point normal contacts; no friction certificate."""
from pathlib import Path
import json,hashlib
import numpy as np
from native_state_capture import array,ARM_KEYS
CAPACITY=262144

def hand_path(path,arm):
 return '/'+arm+'/' in path and ('f2_hand/' in path or any(x in path.rsplit('/',1)[-1] for x in ('thumb','index','middle','ring','little','pinky','wrist_3_link')))

class NativeContacts:
 def __init__(self,env,out):
  from pxr import Usd,UsdPhysics
  import omni.usd
  import omni.physics.tensors as tensors
  self.env=env;self.out=Path(out);self.rows=[];self.chunks=[];self.event_count=0;self.views=[];self.identities=[]
  self.last_env_normal_max=np.zeros(64)
  stage=omni.usd.get_context().get_stage();rigids=[];inventory=[]
  for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
   path=str(prim.GetPath())
   if not (path.startswith('/World/envs/env_0/') or path.startswith('/World/ground')):continue
   if prim.HasAPI(UsdPhysics.RigidBodyAPI):rigids.append(path)
   if prim.HasAPI(UsdPhysics.CollisionAPI):
    enabled=bool(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get());ancestor=prim
    while ancestor and not ancestor.HasAPI(UsdPhysics.RigidBodyAPI):ancestor=ancestor.GetParent()
    owner=str(ancestor.GetPath()) if ancestor else None
    inventory.append(dict(path=path,collision_enabled=enabled,rigid_owner=owner))
  partners=sorted(set(rigids+[p['path'] for p in inventory if p['collision_enabled'] and not p['rigid_owner']]))
  if not partners:raise ValueError('no native contact partners')
  self.simview=tensors.create_simulation_view('torch');self.simview.set_subspace_roots('/')
  for arm in ARM_KEYS:
   paths=[];filters=[];env_ids=[]
   for e in range(env.num_envs):
    per_env=[p.replace('/env_0/',f'/env_{e}/') for p in partners]
    sensors=[p for p in per_env if '/'+arm+'/' in p and p.replace(f'/env_{e}/','/env_0/') in rigids]
    if not sensors:raise ValueError('all-arm rigid sensors empty '+arm)
    paths.extend(sensors);filters.extend([per_env for _ in sensors]);env_ids.extend([e]*len(sensors))
   cv=self.simview.create_rigid_contact_view(paths,filter_patterns=filters,max_contact_data_count=CAPACITY)
   actual=list(cv.sensor_paths);fp=[list(p) for p in cv.filter_paths]
   expected=dict(zip(paths,filters))
   if not cv.check() or len(actual)!=len(paths) or set(actual)!=set(paths) or any(partners!=expected[sensor] for sensor,partners in zip(actual,fp)):
    raise ValueError('batched native exact contact identity mismatch '+arm)
   env_ids=[int(path.split('/envs/env_',1)[1].split('/',1)[0]) for path in actual]
   self.views.append((arm,cv));self.identities.append(dict(arm=arm,sensors=actual,filters=fp,env_ids=env_ids,capacity=CAPACITY))
  p=self.out/'native_contact_identity.json';p.write_text(json.dumps(dict(schema='safeduo.native_all_arm_contact.v1',environment_count=64,partners_env0=partners,inventory=inventory,views=self.identities,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),physics_dt_s=float(env.cfg.sim.dt),constructor_contacts_observed=False,friction_force_matrix_observed=False,scope='all explicit physics substeps after reset; all arm rigid-owner sensors and exact own-env rigid/static/ground normal-force partners; no changes to targets/actor/assets/selfcollision/exemptions'),indent=2)+'\n')
 def capture(self,frame,substep,dt):
  d=dict(frame=np.asarray(frame),substep=np.asarray(substep),physics_dt=np.asarray(dt))
  self.last_env_normal_max.fill(0)
  for arm,cv in self.views:
   net=array(cv.get_net_contact_forces(dt));matrix=array(cv.get_contact_force_matrix(dt));data=cv.get_contact_data(dt);counts=array(data[-2]);starts=array(data[-1]);total=int(counts.sum())
   if total>=CAPACITY or (counts<0).any() or (starts<0).any() or (starts+counts>CAPACITY).any():raise ValueError('native normal contact capacity reached; partial data not accepted')
   d[arm+'_net']=net;d[arm+'_partner_normal']=matrix;d[arm+'_normal_counts']=counts;d[arm+'_normal_starts']=starts
   identity=next(x for x in self.identities if x['arm']==arm)
   sensor_max=np.linalg.norm(matrix.astype(np.float64),axis=-1).max(-1)
   np.maximum.at(self.last_env_normal_max,np.asarray(identity['env_ids']),sensor_max)
   for field,getter in [('native_q',self.env._arms[arm].root_physx_view.get_dof_positions),('native_qd',self.env._arms[arm].root_physx_view.get_dof_velocities),('native_position_targets',self.env._arms[arm].root_physx_view.get_dof_position_targets),('native_root_xyzw',self.env._arms[arm].root_physx_view.get_root_transforms),('native_root_velocity',self.env._arms[arm].root_physx_view.get_root_velocities)]:d[arm+'_'+field]=array(getter())
  self.rows.append(d)
  if len(self.rows)==32:self.flush()
 def flush(self):
  if not self.rows:return
  root=self.out/'native_contacts';root.mkdir(exist_ok=True);start=self.event_count;stop=start+len(self.rows);path=root/f'physics_events_{start:04d}_{stop:04d}.npz';np.savez_compressed(path,**{k:np.stack([r[k] for r in self.rows]) for k in self.rows[0]});self.chunks.append(dict(path=str(path.relative_to(self.out)),start=start,stop=stop,sha256=hashlib.sha256(path.read_bytes()).hexdigest()));self.event_count=stop;self.rows=[]
 def close(self,control_steps):
  self.flush()
  if self.event_count!=2*control_steps:raise ValueError('missing physics-substep contact events')
  (self.out/'native_contact_receipts.json').write_text(json.dumps(dict(status='complete',control_steps=control_steps,physics_events=self.event_count,substeps_per_control=2,all64=True,chunks=self.chunks,identity_sha256=hashlib.sha256((self.out/'native_contact_identity.json').read_bytes()).hexdigest(),capacity=CAPACITY,nonclaims=['constructor contacts','full friction contact data','whole mesh collision freedom','hardware safety']),indent=2)+'\n')
