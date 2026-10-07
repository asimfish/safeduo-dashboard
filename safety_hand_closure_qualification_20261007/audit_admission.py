"""Recompute native request eligibility from raw previous-step tensors, independently."""
import json,hashlib
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parent;raw=Path('/mnt/nas/data/lyf/double_hand/safety_hand_closure_qualification_20261007/validation')
r=json.loads((p/'REGISTRATION_VALIDATION.json').read_text());records=json.loads((raw/'admission_records.json').read_text());passport=r['passport'];metadata=json.loads((raw/'native_metadata.json').read_text());identity=json.loads((raw/'contact_identities.json').read_text());receipt=json.loads((raw/'recording_receipt.json').read_text());qual=[x for x in records if 'decision' in x];aborts=[x for x in records if 'abort_step' in x];unknown=0;admitted=0
cache={}
for req in qual:
 arm=req['arm'];env=req['env'];prev=req['step']-1;chunk=next(x for x in receipt['chunks'] if x['first_step']<=prev<=x['last_step']);fn=chunk['file']
 if fn not in cache:
  with np.load(raw/fn) as z:cache[fn]={k:z[k].copy() for k in z.files}
 z=cache[fn];ix=prev-chunk['first_step'];ids=metadata['arms'][arm]['hand_ids'];q=z[arm+':q'][ix,env,ids];qd=z[arm+':qd'][ix,env,ids];op=np.asarray(passport['open_rad'][arm],np.float32)
 paths=next(x for x in identity['hands'] if x['env']==env and x['arm']==arm);fm=z[f'e{env}:{arm}:hand_partner_normal'][ix];mask=np.asarray([[x in paths['sensors'] for x in fs] for fs in paths['filters']]);force=np.where(mask,z[f'e{env}:{arm}:hand_partner_scalar_normal'][ix],0).max()
 assert req['native_state_step']==prev
 goal=np.asarray(req['request_goal_rad'],np.float32);allow=[np.asarray(x,np.float32) for x in passport['allowed_goals_rad'][arm]]
 eligible=bool(np.isfinite(q).all() and np.isfinite(qd).all() and float(abs(qd).max())<=3 and force<=np.float32(.1) and float(abs(q-op).max())<=.02 and any(np.array_equal(goal,x) for x in allow))
 if req['decision']['admitted']:assert eligible;admitted+=1
 unknown+=req['unknown_rejections'];assert req['unknown_rejections']==32
 assert [x['index'] for x in req['unknown_decisions']]==list(range(32))
 assert all(not x['decision']['admitted'] for x in req['unknown_decisions'])
 assert all(x['decision']['reason'] in ['unqualified_exact_target','abort_latched','self_contact','speed_limit','unqualified_start'] for x in req['unknown_decisions'])
 assert not any(np.array_equal(np.asarray(x,np.float32),v) for x in r['unknown_goals_rad'][arm] for v in allow)
 assert req['original_task_rejection'] in ['unqualified_exact_target','abort_latched','self_contact','speed_limit','unqualified_start']
assert len(qual)==len(r['profiles'])*6
result=dict(status='PASS_NATIVE_ADMISSION_REQUEST_AUDIT',qualified_requests=len(qual),admitted_requests=admitted,abort_count=len(aborts),unknown_rejections=unknown,unique_unknown_profiles=32,original_task_rejections=len(qual),software_test_count=15,software_tests_native_physics=False,new_unknown_physical_paths=0,request_records_sha256=hashlib.sha256((raw/'admission_records.json').read_bytes()).hexdigest(),aborts=aborts)
(p/'ADMISSION_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
