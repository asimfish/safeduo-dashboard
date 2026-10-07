"""Independent no-cancellation gate over every native U hand state."""
from pathlib import Path
import json,hashlib,numpy as np
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);a=parser.parse_args()
p=Path(__file__).resolve().parent;root=Path('/mnt/nas/data/lyf/double_hand/safety_hand_closure_qualification_20261007');sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
for run,name in [(a.name,a.name.upper())]:
 r=json.loads((p/f'REGISTRATION_{name}.json').read_text());base=json.loads((p/f'{name}_RESULTS.json').read_text());raw=root/run;receipt=json.loads((raw/'recording_receipt.json').read_text());identity=json.loads((raw/'contact_identities.json').read_text());maxima={};violations={};gapmax=0.;count_max=0
 for ch in receipt['chunks']:
  assert sha(raw/ch['file'])==ch['sha256']
  with np.load(raw/ch['file']) as z:
   cyc=int(z['cycle'][0]);assert (z['cycle']==cyc).all()
   for e,case in enumerate(r['cases']):
    for arm in ['U_L','U_R']:
     ids=next(v for v in identity['hands'] if v['env']==e and v['arm']==arm);mask=np.asarray([[x in ids['sensors'] for x in fs] for fs in ids['filters']]);scalar=z[f'e{e}:{arm}:hand_partner_scalar_normal'];vector=np.linalg.norm(z[f'e{e}:{arm}:hand_partner_normal'],axis=-1)
     assert np.isfinite(scalar).all() and (scalar>=0).all()
     # Independent oracle from unmodified valid native point values/count/start.
     counts=z[f'e{e}:{arm}:hand_normal_count'];starts=z[f'e{e}:{arm}:hand_point_start'];forces=z[f'e{e}:{arm}:hand_point_force']
     assert (counts>=0).all() and np.all((counts==0)|((starts>=0)&(starts+counts<=r['contact_capacity'])))
     safe_start=np.where(counts>0,starts,0).astype(int);ends=safe_start+counts.astype(int);prefix=np.pad(np.abs(forces).astype(np.float64).cumsum(axis=1),((0,0),(1,0)))
     ri=np.arange(len(forces))[:,None,None];reconstructed=prefix[ri,ends]-prefix[ri,safe_start]
     assert np.all(np.abs(reconstructed-scalar)<=.001+2e-6*reconstructed),('independent point sum',run,e,arm)
     scalar=reconstructed
     assert np.all(vector<=scalar+.001+2e-6*scalar),('force triangle accounting',run,e,arm)
     pointcounts=z[f'e{e}:{arm}:hand_normal_count'];count_max=max(count_max,int(pointcounts.sum(axis=(1,2)).max()))
     value=np.where(mask[None],scalar,0).max(axis=(1,2));key=(cyc,e,arm);maxima[key]=max(maxima.get(key,0),float(value.max()));violations[key]=violations.get(key,0)+int((value>np.float32(.1)).sum());gapmax=max(gapmax,float((scalar-vector).max()))
 full=json.loads(json.dumps(base));rows=full['rows']
 for row in rows:
  if row['arm'].startswith('U'):
   key=(row['cycle'],row['env'],row['arm']);row['qualified_vector_tracking']=row['qualified'];row['path_max_scalar_self_normal_n']=maxima[key];row['scalar_violation_steps']=violations[key];row['qualified']=row['qualified'] and maxima[key]<=float(np.float32(.1))
 u=[x for x in rows if x['arm'].startswith('U')];full['u_paths_qualified']=sum(x['qualified'] for x in u);full['profile_count_qualified']=sum(all(x['qualified'] for x in u if x['profile_index']==i) for i in range(len(r['profiles'])));full['vector_results_sha256']=sha(p/f'{name}_RESULTS.json');full['normal_measurement']='Per sensor/filter sum of absolute native point normal magnitudes; no vector cancellation.';full['max_scalar_minus_vector_n']=gapmax
 if run.startswith('paired'):
  guard=[x for x in u if x['method']=='guard'];bypass=[x for x in u if x['method']=='bypass'];full['guard_safe_paths']=sum(x['path_max_scalar_self_normal_n']<=.1 and x['neutral_error_rad']<=.02 and x['max_target_error_rad']<=1e-6 for x in guard);full['guard_max_scalar_self_normal_n']=max(x['path_max_scalar_self_normal_n'] for x in guard);full['bypass_max_scalar_self_normal_n']=max(x['path_max_scalar_self_normal_n'] for x in bypass);full['status']='PASS_MATCHED_HAND_ADMISSION_COUNTERFACTUAL' if full['guard_safe_paths']==len(guard) and base['status']=='PASS_MATCHED_HAND_ADMISSION_COUNTERFACTUAL' else 'REJECT_MATCHED_SCALAR_CONTACT'
 else:full['status']='PASS_COMPLETE_SCALAR_VALIDATION_RECORDING' if full['u_paths_qualified']==len(u) else 'REJECT_EXACT_PROFILE_CANDIDATE'
 (p/f'{name}_FULL_RESULTS.json').write_text(json.dumps(full,indent=2,allow_nan=False)+'\n')
 audit=dict(status=full['status'],run=run,u_paths=len(u),paths_with_scalar_contact=sum(x['path_max_scalar_self_normal_n']>.1 for x in u),max_scalar_self_normal_n=max(x['path_max_scalar_self_normal_n'] for x in u),max_scalar_minus_vector_n=gapmax,contact_count_max=count_max,point_vector_triangle_check=True,full_results_sha256=sha(p/f'{name}_FULL_RESULTS.json'))
 (p/f'{name}_SCALAR_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))
