"""Supplementary native point recorder; frozen full-window vector stream unchanged."""
from pathlib import Path
import json,hashlib
import numpy as np
from native_all_body_contacts import NativeContacts,CAPACITY
from native_state_capture import array,ARM_KEYS
from point_contact_data import pack_contact_event,pack_contact_events

def copy_raw(value):
 return value.detach().clone().cpu().numpy().copy()

class PointContacts(NativeContacts):
 def __init__(self,env,out):
  super().__init__(env,out);self.point_rows=[];self.last_env_scalar_max=np.zeros(64)
 def capture(self,frame,substep,dt):
  d=dict(frame=np.asarray(frame),substep=np.asarray(substep),physics_dt=np.asarray(dt));packets={}
  self.last_env_normal_max.fill(0);self.last_env_scalar_max.fill(0)
  for arm,cv in self.views:
   net=array(cv.get_net_contact_forces(dt));matrix=array(cv.get_contact_force_matrix(dt))
   raw=[copy_raw(v) for v in cv.get_contact_data(dt)]
   packet=pack_contact_event(*raw,capacity=CAPACITY);packets[arm]=packet
   d[arm+'_net']=net;d[arm+'_partner_normal']=matrix;d[arm+'_normal_counts']=packet['counts'];d[arm+'_normal_starts']=packet['starts'];d[arm+'_partner_scalar_abs_N']=packet['partner_abs_normal_sum_N']
   identity=next(x for x in self.identities if x['arm']==arm);ids=np.asarray(identity['env_ids'])
   np.maximum.at(self.last_env_normal_max,ids,np.linalg.norm(matrix.astype(np.float64),axis=-1).max(-1))
   np.maximum.at(self.last_env_scalar_max,ids,packet['partner_abs_normal_sum_N'].max(-1))
   view=self.env._arms[arm].root_physx_view
   for field,getter in [('native_q',view.get_dof_positions),('native_qd',view.get_dof_velocities),('native_position_targets',view.get_dof_position_targets),('native_root_xyzw',view.get_root_transforms),('native_root_velocity',view.get_root_velocities)]:d[arm+'_'+field]=array(getter())
  self.rows.append(d);self.point_rows.append(packets)
  if len(self.rows)==32:self.flush()
 def flush(self):
  if not self.rows:return
  assert len(self.rows)==len(self.point_rows)
  root=self.out/'native_contacts';root.mkdir(exist_ok=True);start=self.event_count;stop=start+len(self.rows);p=root/f'physics_events_{start:04d}_{stop:04d}.npz'
  values={k:np.stack([r[k] for r in self.rows]) for k in self.rows[0]}
  for a in ARM_KEYS:values.update({a+'_points_'+k:v for k,v in pack_contact_events([r[a] for r in self.point_rows]).items()})
  np.savez_compressed(p,**values);self.chunks.append(dict(path=str(p.relative_to(self.out)),start=start,stop=stop,sha256=hashlib.sha256(p.read_bytes()).hexdigest()));self.event_count=stop;self.rows=[];self.point_rows=[]
 def close(self,steps):
  super().close(steps)
  r=json.loads((self.out/'native_contact_receipts.json').read_text());r.update(schema='safeduo.native_all_arm_point_contact.v1',point_recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),point_helper_sha256=hashlib.sha256((Path(__file__).parent/'point_contact_data.py').read_bytes()).hexdigest(),point_scalar_gate_N=.1,all_valid_raw_point_fields=True,original_point_indices_preserved=True,unused_backend_slots_not_observed=True,old_full128_point_scalar_status='UNKNOWN_UNRECORDED',safety_acceptance=False)
  (self.out/'point_contact_receipts.json').write_text(json.dumps(r,indent=2)+'\n')
