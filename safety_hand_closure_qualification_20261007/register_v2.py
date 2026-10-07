import json,hashlib,argparse,secrets,random
from pathlib import Path
from datetime import datetime,timezone
p=Path(__file__).resolve().parent;parser=argparse.ArgumentParser();parser.add_argument('--name',choices=['validation_v2','validation_fresh','paired_v2'],required=True);a=parser.parse_args();base=json.loads((p/'REGISTRATION_VALIDATION.json').read_text());sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest();name=a.name;r=json.loads(json.dumps(base));r['run']=name;r['created_utc']=datetime.now(timezone.utc).isoformat();r['status']='FROZEN_BEFORE_NATIVE_'+name.upper();r['contact_epoch_rule']='Initialization writes invalidate contact cache. First physics step holds neutral, all forces recorded. Observe only after a completed post-init physics step.'
if name=='validation_fresh':
 seed=secrets.randbits(63);g=random.Random(seed);exclude={round(t*60) for t in [0,4.5,15.2,2,8,20]};frames=[g.choice([i for i in range(lo,hi) if i not in exclude]) for lo,hi in [(1,480),(480,960),(960,1453)]];times=[i/60 for i in frames];r['os_random_seeds']['fresh_references']=seed;r['fresh_reference_frames']=frames
 for case in r['cases']:case['reference_time_s']=times[case['env']//4];case['label']='fresh_ref_'+str(case['reference_time_s'])
 r['passport']['registered_reference_times_s']=times;r['scope']='Fresh OS-stratified trajectory-derived arm reference bank after epoch fix. Same 12 frozen exact goals. Not IID tasks or random 26-DOF workspace coverage.'
elif name=='paired_v2':
 dev=json.loads((p/'REGISTRATION_DEVELOPMENT.json').read_text());cases=[]
 for t in [2.,8.,20.]:
  for method in ['bypass','guard']:cases.append({**dev['cases'][0],'env':len(cases),'reference_time_s':t,'method':method,'profile_slot':0,'label':f'{method}_ref{t}'})
 steps=sorted(set(range(0,720,12))|{359,659});r.update(profiles=[dev['profiles'][2]],profiles_per_cycle=1,steps=720,cases=cases,capture_steps=steps,capture_envs=list(range(6)),capture_envs_by_step={str(s):([0,1] if s%12==0 else list(range(6))) for s in steps},capture_views_by_step={str(s):(['detail'] if s%12==0 else list(r['views'])) for s in steps},scope='Matched q/qd/target initial states and same original hand goal at 3 references; bypass executes, guard denies after valid contact epoch. Rejection is not grasp success.')
else:r['scope']='Regression after epoch-initialization fix on the same frozen 12-goal/3-reference bank; not fresh heldout.'
sources={k:h for k,h in base['source_sha256'].items() if '/safety_hand_closure_qualification_20261007/' not in k}
for root in [p,p/'runtime_v2']:
 for f in root.iterdir():
  if f.is_file() and f.suffix in ['.py','.json','.md']:sources[str(f)]=sha(f)
r['source_sha256']=sources;reg=p/f'REGISTRATION_{name.upper()}.json';assert not reg.exists();reg.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');(p/f'{name.upper()}_SEAL.json').write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),registration_sha256=sha(reg),native_sha256=sha(p/'runtime_v2'/('native_paired_v2.py' if name=='paired_v2' else 'native_validation_v2.py')),source_entries=len(sources)),indent=2)+'\n');print(json.dumps(dict(name=name,cases=len(r['cases']),profiles=len(r['profiles']),steps=r['steps'],references=sorted({c['reference_time_s'] for c in r['cases']}),seal=sha(reg)),indent=2))
