"""Checks registered target publishing against independently calculated endpoints.

CPU adapter checks commands only, not physics or contact safety.
"""
from types import SimpleNamespace
from pathlib import Path
import json,numpy as np,torch,hashlib
from joint_paths import JointPaths
p=Path(__file__).resolve().parent;r=json.loads((p/'REGISTRATION_DEVELOPMENT.json').read_text());meta=json.loads((p/'legacy_native_metadata.json').read_text());published={}
class Art:
 def __init__(self,arm):
  self.arm=arm;v=meta['arms'][arm];self.joint_names=v['joint_names'];self.data=SimpleNamespace(soft_joint_pos_limits=torch.tensor(v['soft_limits_rad'][0],dtype=torch.float32)[None])
 def set_joint_position_target(self,q,joint_ids):published[self.arm]=q.detach().numpy().copy()
e=SimpleNamespace(_arms={a:Art(a) for a in r['hand_ids']},device='cpu',num_envs=12)
x=JointPaths(e,r);count=0
for cyc in range(r['steps']//r['cycle_steps']):
 for ph in [0,120,239,240,359,420,539,540,719]:
  x.step(cyc*720+ph)
  for arm in ['U_L','U_R']:
   ids=r['hand_ids'][arm];names=r['hand_names'][arm];lim=np.asarray(meta['arms'][arm]['soft_limits_rad'][0])[ids,1]
   for en,case in enumerate(r['cases']):
    spec=r['profiles'][cyc*4+case['profile_slot']];expected=lim*spec['finger_fraction'];op=np.zeros(len(ids));op[names.index(('left' if arm=='U_L' else 'right')+'_thumb_1_joint')]=.35
    if spec['kind']=='original_task':expected=np.array([lim[j]*(.7569 if arm=='U_L' else .7546) if 'thumb' in n else lim[j]*(.4769 if arm=='U_L' else .4746) for j,n in enumerate(names)])
    elif spec['kind']=='open_control':expected=op.copy()
    else:
     for a in [1,2]:j=names.index(('left' if arm=='U_L' else 'right')+f'_thumb_{a}_joint');expected[j]=lim[j]*spec[f'thumb{a}_fraction']
    if ph in [240,359,420]:assert np.max(abs(published[arm][en]-expected))<1e-6
    if ph in [0,540,719]:assert np.max(abs(published[arm][en]-op))<1e-6
    assert np.min(published[arm][en])>=0 and np.max(published[arm][en]-lim)<1e-6
    count+=1
print(json.dumps(dict(status='PASS_REGISTERED_TARGET_CPU_CHECK_ONLY',checks=count,physics_checks=0,registration_sha256=hashlib.sha256((p/'REGISTRATION_DEVELOPMENT.json').read_bytes()).hexdigest()),indent=2))
