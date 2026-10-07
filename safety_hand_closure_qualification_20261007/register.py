from pathlib import Path
import json,hashlib,secrets,numpy as np
from datetime import datetime,timezone
p=Path(__file__).resolve().parent
base=json.loads((p.parent/'safety_hand_initialization_20261007/REGISTRATION_neutral_default.json').read_text());meta=json.loads((p/'legacy_native_metadata.json').read_text())
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
seed_uniform=secrets.randbits(63);seed_lhs=secrets.randbits(63);holdout_seed=secrets.randbits(63)
g=np.random.default_rng(seed_uniform);lhs=np.random.default_rng(seed_lhs)
profiles=[dict(kind='open_control',thumb1_fraction=.35/1.310383915901184,thumb2_fraction=0,finger_fraction=0),dict(kind='moderate_control',thumb1_fraction=.45,thumb2_fraction=.45,finger_fraction=.45),dict(kind='original_task',thumb1_fraction=.7569,thumb2_fraction=.7569,finger_fraction=.4769)]
for f in [0.,.475,1.]:
 for t1 in np.linspace(0,1,5):
  for t2 in np.linspace(0,1,5):profiles.append(dict(kind='full_grid',thumb1_fraction=float(t1),thumb2_fraction=float(t2),finger_fraction=f))
for t1,t2,f in g.random((20,3)):profiles.append(dict(kind='uniform_random',thumb1_fraction=float(t1),thumb2_fraction=float(t2),finger_fraction=float(f)))
x=np.column_stack([(lhs.permutation(20)+lhs.random(20))/20 for _ in range(3)])
for t1,t2,f in x:profiles.append(dict(kind='latin_hypercube',thumb1_fraction=float(t1),thumb2_fraction=float(t2),finger_fraction=float(f)))
while len(profiles)%4:profiles.append(dict(profiles[0],kind='open_control'))
cases=[]
for ref in [0.,4.5,15.2]:
 for slot in range(4):cases.append(dict(env=len(cases),kind='isolated_hand',label=f'ref{ref}_slot{slot}',reference_time_s=ref,post_write_thumb_rad=.35,profile_slot=slot,centers_m=[[.5,1.3,1.5],[-.4,-1.3,1.5]]))
sources={str(f):sha(f) for f in Path('/home/liyufeng/safeduo/src').rglob('*') if f.is_file() and f.suffix in ['.py','.yaml','.json']}
for f in base['source_sha256']:
 if 'assets_real' in f or f in [base['trajectory'],base['env_yaml']]:sources[f]=sha(f)
for f in p.iterdir():
 if f.is_file():sources[str(f)]=sha(f)
r={k:base[k] for k in ['env_yaml','trajectory','physics_dt_s','disable_self_collision','object_yaws_rad','contact_capacity','views','criteria','constructor_thumb_rad','target_open_thumb_rad']}
r.update(created_utc=datetime.now(timezone.utc).isoformat(),status='FROZEN_BEFORE_NATIVE_EXECUTION',run='development',seed=90610761,source_sha256=sources,cases=cases,profiles=profiles,profiles_per_cycle=4,cycle_steps=720,ramp_s=1.,steps=len(profiles)//4*720,legacy_metadata=str(p/'legacy_native_metadata.json'),hand_ids={a:v['hand_ids'] for a,v in meta['arms'].items()},hand_names={a:[v['joint_names'][j] for j in v['hand_ids']] for a,v in meta['arms'].items()},os_random_seeds=dict(uniform=seed_uniform,latin_hypercube=seed_lhs,validation=holdout_seed),scope='Empty-hand development map. 75 grid,20 full-range uniform,20 LHS plus controls; both U hands, 3 static references. No native resets in loop.',capture_steps=[c*720+s for c in [0,1,7,18,25,29] for s in [359,659]],capture_envs=[0,1,2,3],validation_contract=dict(new_reference_times_s=[2.,8.,20.],random_unknown_count=32,max_qualified_profiles=12,ramp_s=1.,no_interpolation=True))
fn=p/'REGISTRATION_DEVELOPMENT.json';assert not fn.exists();fn.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');(p/'REGISTRATION_SEAL.json').write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),sha256=sha(fn),analyzer_sha256=sha(p/'analyze.py'),native_probe_sha256=sha(p/'native_probe.py'),joint_paths_sha256=sha(p/'joint_paths.py')),indent=2)+'\n')
print(json.dumps(dict(profiles=len(profiles),cycles=r['steps']//720,envs=len(cases),env_states=r['steps']*len(cases),original_images=len(r['capture_steps'])*len(r['capture_envs'])*3,seal=sha(fn)),indent=2))
