from pathlib import Path
import json,numpy as np
from itertools import combinations
p=Path(__file__).resolve().parent;raw=Path('/mnt/nas/data/lyf/double_hand/safety_hand_closure_qualification_20261007/validation_fresh_v3');m=json.loads((raw/'native_metadata.json').read_text());identity=json.loads((raw/'contact_identities.json').read_text());receipt=json.loads((raw/'recording_receipt.json').read_text());origins=np.asarray(m['origins']);bounds={a:[np.full(3,np.inf),np.full(3,-np.inf)] for a in m['arms']};bodyids={};points={a:0 for a in m['arms']}
for arm,v in m['arms'].items():
 sensors=next(x for x in identity['hands'] if x['env']==0 and x['arm']==arm)['sensors'];bodyids[arm]=[v['body_names'].index(x.rsplit('/',1)[-1]) for x in sensors]
for ch in receipt['chunks']:
 with np.load(raw/ch['file']) as z:
  for arm in bounds:
   poses=z[arm+':link_pose'][:,:,:,:3];local=poses[:,:,bodyids[arm],:]-origins[None,:,None,:];bounds[arm][0]=np.minimum(bounds[arm][0],local.min(axis=(0,1,2)));bounds[arm][1]=np.maximum(bounds[arm][1],local.max(axis=(0,1,2)));points[arm]+=int(np.prod(local.shape[:-1]))
boxes=[dict(arm=arm,minimum_m=lo.tolist(),maximum_m=hi.tolist(),span_m=(hi-lo).tolist(),hand_body_count=len(bodyids[arm]),center_observations=points[arm]) for arm,(lo,hi) in bounds.items()];pairs=[]
for a,b in combinations(bounds,2):
 span=np.minimum(bounds[a][1],bounds[b][1])-np.maximum(bounds[a][0],bounds[b][0]);pairs.append(dict(arms=[a,b],center_envelope_aabb_overlap_m3=float(np.maximum(span,0).prod()),envelopes_overlap=bool((span>0).all()),collision_qualified=False))
out=dict(status='OBSERVED_HAND_BODY_CENTER_ENVELOPES_ONLY',reference_count=12,measured_states=receipt['steps']*len(m['cases']),boxes=boxes,pairs=pairs,units='environment-local meters',not_reachable_workspace_volume=True,not_mesh_extent=True,not_six_pair_collision_test=True,interpretation='Observed center envelopes on12 static trajectory references and12 sequential goals. AABB overlap is a descriptive indication, not a simultaneous collision or reachable-volume measurement. AllF hands are open. The full26-DOF reachable and safe operation spaces remain unqualified.')
(p/'OBSERVED_HAND_ENVELOPES.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
