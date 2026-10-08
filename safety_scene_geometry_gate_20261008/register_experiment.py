import json,hashlib,random,secrets,argparse
from pathlib import Path
from datetime import datetime,timezone
p=Path(__file__).resolve().parent;old=Path('/home/liyufeng/safeduo/artifacts/safety_hand_closure_qualification_20261007')
pa=argparse.ArgumentParser();pa.add_argument('--name',choices=['matched_scene','fresh_scene'],required=True);a=pa.parse_args()
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
r=json.loads((old/'REGISTRATION_VALIDATION_FRESH_V3.json').read_text())
r['run']=a.name;r['created_utc']=datetime.now(timezone.utc).isoformat();r['status']='FROZEN_BEFORE_NATIVE'
r['old_context_exclusions']=r['passport']['context_exclusions'];r['passport']['context_exclusions']=[]
r['geometry']=dict(samples=121,reserve_m=.020,max_fk_position_error_m=.0005,max_fk_quaternion_component_error=.002,absent_contact_offset_reserve_m=.020)
r['views']['detail']['eye_offset_m']=[0,-.4,.35]
r['mechanism_contract']=str(p/'DIAGNOSIS_AND_CONTRACT.md')
if a.name=='matched_scene':
    times=[19.633333333333333,19.85,2.,20.]
    r['cases']=[dict(r['cases'][0],env=2*i+j,reference_time_s=t,method=method,profile_slot=0,label=f'{method}_ref{t}') for i,t in enumerate(times) for j,method in enumerate(['old_v3','scene_v4'])]
    r['scope']='Development matched comparison: four fixed references, 12 frozen goals, two U hands, old V3 versus prospective scene geometry. Not heldout and not manipulation tasks.'
else:
    r['views']['detail']['eye_offset_m']=[0,-.05,.5]
    seed=secrets.randbits(63);g=random.Random(seed)
    prior={round(t*60) for t in [0,4.5,15.2,2,8,20]+[c['reference_time_s'] for c in r['cases']]}
    frames=[i for lo,hi in [(1,480),(480,960),(960,1453)] for i in g.sample([k for k in range(lo,hi) if k not in prior],8)]
    times=[i/60 for i in frames];r['fresh_reference_frames']=frames;r['os_random_seeds']['scene_fresh_references']=seed
    r['cases']=[dict(r['cases'][0],env=i,reference_time_s=t,method='scene_v4',profile_slot=0,label=f'fresh_scene_ref{t}') for i,t in enumerate(times)]
    r['scope']='24 untouched OS-stratified trajectory frames x 12 frozen goals x 2 U hands = 576 requests. 288 native cycles, 207360 environment states. Not independent tasks or full 26-DOF random support.'
r['passport']['registered_reference_times_s']=times;r['profiles_per_cycle']=1;r['steps']=8640
steps=sorted(set(range(0,720,12))|{c*720+s for c in range(12) for s in [359,659]})
r['capture_steps']=steps;r['capture_envs']=[0,1] if a.name=='matched_scene' else [0]
r['capture_envs_by_step']={str(s):(list(range(len(r['cases']))) if s//720==11 and s%720 in [359,659] else r['capture_envs']) for s in steps}
r['capture_views_by_step']={str(s):(list(r['views']) if s%720 in [359,659] else ['detail']) for s in steps}
sources={k:h for k,h in r['source_sha256'].items() if '/safety_hand_closure_qualification_20261007/' not in k}
for f in [r['env_yaml'],r['trajectory'],r['legacy_metadata'],old/'REGISTRATION_VALIDATION_FRESH_V3.json',p/'DIAGNOSIS_AND_CONTRACT.md']:
    sources[str(f)]=sha(f)
for f in (p/'runtime').glob('*.py'):sources[str(f)]=sha(f)
r['source_sha256']=sources
fn=p/f'REGISTRATION_{a.name.upper()}.json';assert not fn.exists();fn.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(run=a.name,cases=len(r['cases']),requests=len(r['cases'])*24,registration_sha256=sha(fn),references=times,source_entries=len(sources)),indent=2))
