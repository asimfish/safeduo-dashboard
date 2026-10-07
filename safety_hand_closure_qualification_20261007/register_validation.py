"""Select frozen qualified goals with the preregistered OS seed; new reference bank."""
from pathlib import Path
import json,hashlib,numpy as np,torch
from datetime import datetime,timezone
p=Path(__file__).resolve().parent;dev=json.loads((p/'REGISTRATION_DEVELOPMENT.json').read_text());res=json.loads((p/'DEVELOPMENT_RESULTS.json').read_text())
meta=json.loads(Path('/mnt/nas/data/lyf/double_hand/safety_hand_closure_qualification_20261007/development/native_metadata.json').read_text())
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
assert res['registration_sha256']==sha(p/'REGISTRATION_DEVELOPMENT.json')
unique=[];seen=set()
for i in res['profiles_qualified_at_all_three_references']:
 s=dev['profiles'][i];key=tuple(s[k] for k in ['thumb1_fraction','thumb2_fraction','finger_fraction'])
 if key not in seen:unique.append(i);seen.add(key)
g=np.random.default_rng(dev['os_random_seeds']['validation']);selected=g.permutation(unique)[:12].tolist();assert len(selected)>=4
# Only whole four-slot cycles; if fewer than 12 available, no padding inferred as new coverage.
selected=selected[:len(selected)//4*4];profiles=[{**dev['profiles'][i],'development_profile_index':i} for i in selected]
limits={};opened={};allowed={};original={};unknown={};unknown_profiles=g.random((32,3))
for arm in ['U_L','U_R']:
 v=meta['arms'][arm];ids=v['hand_ids'];names=dev['hand_names'][arm];lim=np.asarray(v['soft_limits_rad'][0],np.float32)[ids];far=torch.tensor(lim[:,1]);op=torch.zeros_like(far);op[names.index(('left' if arm=='U_L' else 'right')+'_thumb_1_joint')]=.35
 def goal(s):
  a=far*s['finger_fraction']
  for axis in [1,2]:j=names.index(('left' if arm=='U_L' else 'right')+f'_thumb_{axis}_joint');a[j]=far[j]*s[f'thumb{axis}_fraction']
  if s['kind']=='open_control':a=op.clone()
  return a.numpy().tolist()
 limits[arm]=lim.tolist();opened[arm]=op.numpy().tolist();allowed[arm]=[goal(s) for s in profiles]
 original[arm]=[float(np.float32(lim[j,1]*(.7569 if arm=='U_L' else .7546) if 'thumb' in n else lim[j,1]*(.4769 if arm=='U_L' else .4746))) for j,n in enumerate(names)]
 unknown[arm]=[goal(dict(kind='uniform_random',thumb1_fraction=float(a),thumb2_fraction=float(b),finger_fraction=float(c))) for a,b,c in unknown_profiles]
 assert not any(x==y for x in unknown[arm] for y in allowed[arm])
 assert original[arm] not in allowed[arm]
passport=dict(status='FROZEN_EXPERIMENTAL_EMPTY_HAND_CANDIDATE_NOT_GLOBAL_CERTIFICATE',open_rad=opened,limits_rad=limits,allowed_goals_rad=allowed,ramp_s=1.,max_speed_rad_s=3.,max_self_normal_n=.1,max_start_error_rad=.02,registered_reference_times_s=[2.,8.,20.],development_selected_indices=selected,development_results_sha256=sha(p/'DEVELOPMENT_RESULTS.json'),production_deployed=False,normal_oracle='maximum over internal sensor/partner pairs of sum(abs(native point normal forces)); no vector cancellation')
(p/'PASSPORT.json').write_text(json.dumps(passport,indent=2)+'\n')
cases=[]
for ref in [2.,8.,20.]:
 for slot in range(4):cases.append({**dev['cases'][slot], 'env':len(cases),'reference_time_s':ref,'label':f'validation_ref{ref}_slot{slot}'})
sources={k:v for k,v in dev['source_sha256'].items() if '/safety_hand_closure_qualification_20261007/' not in k}
for f in p.iterdir():
 if f.is_file() and f.suffix in ['.py','.json','.md']:sources[str(f)]=sha(f)
r={k:v for k,v in dev.items() if k not in ['source_sha256','profiles','cases','steps','capture_steps','run','created_utc']}
r.update(created_utc=datetime.now(timezone.utc).isoformat(),status='FROZEN_BEFORE_NATIVE_VALIDATION',run='validation',scope='Exact development-qualified goals at 3 new static references, fresh native admission every step. 32 unknown target profiles rejected and not physically executed. No System0 or payload tasks.',source_sha256=sources,profiles=profiles,cases=cases,steps=len(profiles)//4*720,capture_envs=[0],capture_steps=sorted(set(range(0,720,12))|{c*720+s for c in range(len(profiles)//4) for s in [359,659]}),passport=passport,unknown_goals_rad=unknown,unknown_profiles_normalized=unknown_profiles.tolist(),original_task_goals_rad=original)
r['views']={**dev['views'],'detail':dict(hand_focus='U_L',eye_offset_m=[.32,-.30,.28])}
fn=p/'REGISTRATION_VALIDATION.json';assert not fn.exists();fn.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
(p/'VALIDATION_SEAL.json').write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),registration_sha256=sha(fn),passport_sha256=sha(p/'PASSPORT.json'),analyzer_sha256=sha(p/'analyze.py'),admission_scorer_sha256=sha(p/'audit_admission.py'),harness_sha256=sha(p/'native_validation.py'),gate_sha256=sha(p/'hand_admission.py'),validation_paths_sha256=sha(p/'validation_paths.py')),indent=2)+'\n')
print(json.dumps(dict(selected=selected,profiles=len(profiles),native_cycles=len(profiles)*3,u_hand_paths=len(profiles)*6,unknown_profiles=32,original_images=len(r['capture_steps'])*4),indent=2))
