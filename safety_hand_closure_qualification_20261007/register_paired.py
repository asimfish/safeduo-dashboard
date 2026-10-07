from pathlib import Path
import json,hashlib
from datetime import datetime,timezone
p=Path(__file__).resolve().parent;v=json.loads((p/'REGISTRATION_VALIDATION.json').read_text());dev=json.loads((p/'REGISTRATION_DEVELOPMENT.json').read_text());sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest();cases=[]
for t in [2.,8.,20.]:
 for method in ['bypass','guard']:cases.append({**dev['cases'][0], 'env':len(cases),'reference_time_s':t,'method':method,'profile_slot':0,'label':f'{method}_ref{t}'})
steps=sorted(set(range(0,720,12))|{359,659})
sources={k:h for k,h in v['source_sha256'].items() if '/safety_hand_closure_qualification_20261007/' not in k}
for f in p.iterdir():
 if f.is_file() and f.suffix in ['.py','.json','.md']:sources[str(f)]=sha(f)
r={**v, 'created_utc':datetime.now(timezone.utc).isoformat(),'status':'FROZEN_BEFORE_MATCHED_NATIVE_EXECUTION','run':'paired','profiles':[dev['profiles'][2]],'profiles_per_cycle':1,'steps':720,'cases':cases,'source_sha256':sources,'capture_steps':steps,'capture_envs':list(range(6)),'capture_envs_by_step':{str(s):([0,1] if s%12==0 else list(range(6))) for s in steps},'capture_views_by_step':{str(s):(['detail'] if s%12==0 else list(v['views'])) for s in steps},'scope':'Same initial state and old requested hand goal. 3 static references x bypass/guard methods, empty hands, no arm System0. Rejection is not grasp success.'}
fn=p/'REGISTRATION_PAIRED.json';assert not fn.exists();fn.write_text(json.dumps(r,indent=2)+'\n');(p/'PAIRED_SEAL.json').write_text(json.dumps(dict(registration_sha256=sha(fn),harness_sha256=sha(p/'native_paired.py'),paths_sha256=sha(p/'paired_paths.py'),scorer_sha256=sha(p/'analyze_paired.py')),indent=2)+'\n');print(json.dumps(dict(cases=6,steps=720,images=sum(len(r['capture_envs_by_step'][str(s)])*len(r['capture_views_by_step'][str(s)]) for s in steps),registration_sha256=sha(fn)),indent=2))
