import json,hashlib,argparse,secrets,random
from pathlib import Path
from datetime import datetime,timezone
p=Path(__file__).resolve().parent;parser=argparse.ArgumentParser();parser.add_argument('--name',choices=['validation_fresh_v3'],required=True);a=parser.parse_args();base=json.loads((p/'REGISTRATION_VALIDATION.json').read_text());sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest();name=a.name;r=json.loads(json.dumps(base));r['run']=name;r['created_utc']=datetime.now(timezone.utc).isoformat();r['status']='FROZEN_BEFORE_NATIVE_'+name.upper();r['contact_epoch_rule']='Initialization writes invalidate contact cache. First physics step holds neutral, all forces recorded. Observe only after a completed post-init physics step.'
if name=='validation_fresh_v3':
 seed=secrets.randbits(63);g=random.Random(seed);exclude={round(t*60) for t in [0,4.5,15.2,2,8,20]};frames=[frame for lo,hi in [(1,480),(480,960),(960,1453)] for frame in g.sample([i for i in range(lo,hi) if i not in exclude],4)];times=[i/60 for i in frames];r['os_random_seeds']['fresh_references']=seed;r['fresh_reference_frames']=frames
 for case in r['cases']:case['reference_time_s']=times[case['env']];case['profile_slot']=0;case['label']='fresh_ref_'+str(case['reference_time_s'])
 steps=sorted(set(range(0,720,12))|{c*720+s for c in range(12) for s in [359,659]});r.update(profiles_per_cycle=1,steps=12*720,capture_steps=steps,capture_envs=[0],capture_envs_by_step={str(s):(list(range(12)) if s//720==11 and s%720 in [359,659] else [0]) for s in steps},capture_views_by_step={str(s):list(r['views']) for s in steps})
 r['passport']['registered_reference_times_s']=times;r['scope']='12 fresh OS-stratified trajectory-derived references x all 12 frozen exact goals x 2 U hands = 288 U paths; 144 native cycles, 103680 environment states. Not IID tasks or random 26-DOF workspace coverage.'
 r['fresh_expansion_contract']='FRESH_EXTENDED_CONTRACT.md'
elif name=='paired_v3':
 dev=json.loads((p/'REGISTRATION_DEVELOPMENT.json').read_text());cases=[]
 for t in [2.,8.,20.]:
  for method in ['bypass','guard']:cases.append({**dev['cases'][0],'env':len(cases),'reference_time_s':t,'method':method,'profile_slot':0,'label':f'{method}_ref{t}'})
 for method in ['fault_hold','fault_recover']:cases.append({**dev['cases'][0],'env':len(cases),'reference_time_s':20.,'method':method,'profile_slot':1,'label':method+'_ref20'})
 steps=sorted(set(range(0,720,12))|{359,659});r.update(profiles=[dev['profiles'][2],dev['profiles'][13]],profiles_per_cycle=2,steps=720,cases=cases,capture_steps=steps,capture_envs=list(range(8)),capture_envs_by_step={str(s):([0,1,6,7] if s%12==0 else list(range(8))) for s in steps},capture_views_by_step={str(s):(['detail'] if s%12==0 else list(r['views'])) for s in steps},scope='Matched q/qd/target initial states and same original hand goal at 3 references; bypass executes, guard denies after valid contact epoch. Rejection is not grasp success.')
else:r['scope']='Regression after scene-contact/recovery fix on the same frozen 12-goal/3-reference bank; not fresh heldout.'
# Exact context exclusions derive solely from V2, before this run.
dev=json.loads((p/'REGISTRATION_DEVELOPMENT.json').read_text());r['passport']['context_exclusions']=[]
for arm,index in [('U_L',48),('U_R',48),('U_L',13),('U_L',100),('U_R',100)]:
 slot=next(i for i,x in enumerate(base['profiles']) if x['development_profile_index']==index);g=list(base['passport']['allowed_goals_rad'][arm][slot])
 r['passport']['context_exclusions'].append(dict(arm=arm,reference_time_s=20.,development_profile_index=index,goal_rad=g,evidence='VALIDATION_V2_EXTERNAL_DIAGNOSTIC.json'))
r['mechanism_contract']='SCENE_CONTACT_V3_CONTRACT.md'
sources={k:h for k,h in base['source_sha256'].items() if '/safety_hand_closure_qualification_20261007/' not in k}
for root in [p,p/'runtime_v3']:
 for f in root.iterdir():
  if f.is_file() and f.suffix in ['.py','.json','.md']:sources[str(f)]=sha(f)
r['source_sha256']=sources;reg=p/f'REGISTRATION_{name.upper()}.json';assert not reg.exists();reg.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');(p/f'{name.upper()}_SEAL.json').write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),registration_sha256=sha(reg),native_sha256=sha(p/'runtime_v3'/('native_fresh_extended_v3.py')),source_entries=len(sources)),indent=2)+'\n');print(json.dumps(dict(name=name,cases=len(r['cases']),profiles=len(r['profiles']),steps=r['steps'],references=sorted({c['reference_time_s'] for c in r['cases']}),seal=sha(reg)),indent=2))
