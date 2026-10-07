from pathlib import Path
import numpy as np,json,hashlib
p=Path(__file__).resolve().parent;root=Path('/mnt/nas/data/lyf/double_hand/safety_hand_closure_qualification_20261007');sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest();out=[]
for run in ['development','validation','validation_v2','validation_v3','validation_fresh_v3','paired_v3']:
 r=json.loads((p/('REGISTRATION_'+run.upper()+'.json')).read_text());raw=root/run;m=json.loads((raw/'native_metadata.json').read_text());receipt=json.loads((raw/'recording_receipt.json').read_text());err=0.;cache=0.;actual_motion=0.
 assert not r['disable_self_collision'] and m['no_policy_or_system0_step'] and m['new_task_trials']==0 if 'new_task_trials' in m else (not r['disable_self_collision'] and m['no_policy_or_system0_step'])
 for ch in receipt['chunks']:
  with np.load(raw/ch['file']) as z:
   for arm,v in m['arms'].items():
    ids=v['arm_ids'];reference=np.asarray(v['arm_reference_rad'],np.float32);q=z[arm+':q'][:,:,ids];target=z[arm+':target'][:,:,ids];err=max(err,float(abs(target-reference[None]).max()));cache=max(cache,float(abs(z[arm+':q']-z[arm+':cache_q']).max()));actual_motion=max(actual_motion,float(abs(q-reference[None]).max()))
 assert err<=1e-6 and cache<=1e-6
 missing=[f for f in r['source_sha256'] if not Path(f).exists()];drift=[f for f,h in r['source_sha256'].items() if Path(f).exists() and sha(f)!=h]
 # Earlier experiment registrations captured the historical metadata key order; final bytes were restored.
 assert not missing and not drift,(run,missing,drift)
 out.append(dict(run=run,constant_arm_target_max_error_rad=err,native_vs_cache_max_error_rad=cache,actual_arm_deviation_from_reference_rad=actual_motion,arm_motion_present=actual_motion>0,source_entries=len(r['source_sha256']),final_source_drift=[],metadata_drift_incident='SOURCE_METADATA_DRIFT.json' if run in ['development','validation'] else None,receipt_sha256=sha(raw/'recording_receipt.json')))
raw=root/'paired_v3';init=json.loads((raw/'initial_states.json').read_text())[-1];orientation=0.;masses=0.;m=json.loads((raw/'native_metadata.json').read_text())
for left,right in [(0,1),(2,3),(4,5),(6,7)]:
 for obj,v in init['objects'].items():
  pose=np.asarray(v['pose']);orientation=max(orientation,float(abs(pose[left,3:]-pose[right,3:]).max()));mass=np.asarray(m['masses_kg'][obj]);masses=max(masses,float(abs(mass[left]-mass[right]).max()))
assert orientation==0 and masses==0
result=dict(status='PASS_NATIVE_INPUT_COMMAND_CACHE_AND_PAIRED_ORIENTATION_MASS_GATE',runs=out,paired_orientation_difference=orientation,paired_mass_difference_kg=masses,constructor_contacts_measured=False,full_system0_task_trials=0,all_6_arm_pairs_qualified=False)
(p/'FINAL_NATIVE_GATE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
