"""Preregistered full-window decisions; scientific rejection is a completed result."""
from pathlib import Path
import json,numpy as np,hashlib,datetime
H=Path(__file__).resolve().parent
ARMS=['F_L','F_R','U_L','U_R'];WIDTHS=[7,7,6,6]
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def j(p):return json.loads(Path(p).read_text())
def main():
 reg=j(H/'ACCEPTANCE_REGISTRATION.json');criteria=j(H/'ACCEPTANCE_CRITERIA.json');pairreg=j(H/'PAIRING_REGISTRATION.json');execution=j(H/(reg['tag']+'_execution.json'));assert execution['status']=='complete';assert len(execution['jobs'])==8 and all(x['actual_exit']==0 and x['status']=='complete' for x in execution['jobs'])
 modes=criteria['modes'];by={m:[] for m in modes};cases=[];pairing=[];all_raw_bindings=[];stratum={m:{l:[] for l in [-1,0,1,2,3,4,5]} for m in modes}
 for block,row in enumerate(criteria['initial_bank_rows']):
  jobs={m:next(v for v in reg['jobs'] if v['id']==f'b{block}_{m}') for m in modes};baseline=load(Path(jobs['zero_inclusive']['out'])/'input_recipe.npz');base_native=load(Path(jobs['zero_inclusive']['out'])/'native_initial.npz');base_bank=load(Path(jobs['zero_inclusive']['out'])/'bank_assignment.npz')['risk_pair_index']
  for mode in modes:
   job=jobs[mode];id=job['id'];root=Path(job['out']);g=j(H/(id+'_geometry.json'));n=j(H/(id+'_native.json'));c=j(H/(id+'_contacts.json'));hand=j(H/(id+'_hands.json'));peer=j(H/('ASTRA_NATIVE_REAL_'+id+'.json'))
   assert g['status']=='PASS_ALL_FULL_RAW_FLOAT32_GEOMETRY_SCORING' and n['status']=='PASS_SAME_RUN_NATIVE_AND_NINE_VIEW_BINDING' and c['status']=='PASS_NATIVE_HAND_CONTACT_OBSERVATION_NOT_SAFETY' and peer['status']=='PASS_RAW_BINDING_AND_ORIGINAL_SCORING_ONLY'
   assert hand['status']=='PASS_CLOSED_CELL_NATIVE_HAND_CAMERA_BINDING' and hand['frames']==960 and hand['windows']==64 and hand['groups']==n['groups'] and hand['hand_PNG']==12*n['groups'] and hand['total_PNG']==21*n['groups']
   assert g['windows']==64 and g['frames']==960 and c['physics_events']==1920 and n['frames']==960
   assert peer['strict_windows']==g['strict_windows'] and peer['strict_env_frames']==g['strict_env_frames'] and peer['deep_windows']==g['deep_windows'] and peer['deep_env_frames']==g['deep_env_frames']
   assert peer['contacts']['raw_windows_over_threshold']==c['windows_exceeding_normal_diagnostic']
   current=load(root/'input_recipe.npz');native=load(root/'native_initial.npz');assign=load(root/'bank_assignment.npz')['risk_pair_index'];assert np.array_equal(assign,base_bank)
   for key in ['tape','sampled_initial','joint_soft_limits']:assert np.array_equal(current[key],baseline[key]),'requested input pairing '+key
   assert np.array_equal(current['q_initial'],baseline['q_initial']),'controlled actual initial pairing'
   differences=[]
   for a in ARMS:
    for field in pairreg['full_native_actual_initial_fields']:
     k=a+'_'+field
     if not np.array_equal(native[k],base_native[k]):
      delta=np.abs(native[k].astype(np.float64)-base_native[k].astype(np.float64));differences.append(dict(field=k,max_abs_difference=float(delta.max()),different_elements=int((delta!=0).sum())))
   pairing.append(dict(block=block,mode=mode,requested_initial_and_tape_exact=True,controlled_actual_initial_exact=True,full_native_actual_initial_exact=not differences,differences=differences))
   by[mode].append(dict(block=block,job_id=id,geometry=g,native=n,contacts=c,hands=hand,peer=peer));all_raw_bindings.append(dict(job_id=id,cell_sha256=g['cell_sha256'],peer_raw_sha256=peer['raw_sha256']))
   for e in range(64):
    entry=dict(block=block,env=e,mode=mode,risk_label=int(assign[e]),strict=e in g['strict_ids'],deep=e in g['deep_ids'],first_failure_step=g['first_failure_steps'][e],q_l2_path=g['q_l2_path_by_env'][e],native_hand_peak_N=c['window_peak_normal_N'][e],native_hand_raw_above_diagnostic=c['window_peak_normal_N'][e]>.1,joint_range=g['joint_range_by_env'][e])
    cases.append(entry);stratum[mode][int(assign[e])].append(entry)
 aggregated=[]
 for mode in modes:
  r=by[mode];selected=[x for x in cases if x['mode']==mode]
  aggregated.append(dict(mode=mode,windows=128,strict_windows=sum(x['geometry']['strict_windows'] for x in r),deep_windows=sum(x['geometry']['deep_windows'] for x in r),strict_env_frames=sum(x['geometry']['strict_env_frames'] for x in r),deep_env_frames=sum(x['geometry']['deep_env_frames'] for x in r),native_hand_raw_over_windows=sum(x['contacts']['windows_exceeding_normal_diagnostic'] for x in r),native_hand_raw_peak_N=max(max(x['contacts']['window_peak_normal_N']) for x in r),mean_q_l2_path=float(np.mean([x['q_l2_path'] for x in selected])),four_arms_moving_fraction=float(np.mean([x['geometry']['four_arms_moving_fraction'] for x in r])),groups=sum(x['native']['groups'] for x in r),PNG=sum(x['hands']['total_PNG'] for x in r),hand_PNG=sum(x['hands']['hand_PNG'] for x in r),minimum_frustum_m=min(x['native']['minimum_frustum_clearance_m'] for x in r),initial_negative_envs=sum(x['geometry']['initial_negative_envs'] for x in r),mean_joint_ranges=np.mean([x['joint_range'] for x in selected],axis=0).tolist()))
 metrics={x['mode']:x for x in aggregated};comparisons=[]
 for first,second in [criteria['primary_comparison'],criteria['admission_ablation'],criteria['width_ablation'],criteria['legacy_reference']]:
  a={(x['block'],x['env']):x for x in cases if x['mode']==first};b={(x['block'],x['env']):x for x in cases if x['mode']==second};counts={name:[] for name in ['rescued','new_failure','both_failure','neither_failure']}
  for key in a:
   name='both_failure' if a[key]['strict'] and b[key]['strict'] else 'rescued' if a[key]['strict'] else 'new_failure' if b[key]['strict'] else 'neither_failure';counts[name].append(dict(block=key[0],env=key[1],risk_label=a[key]['risk_label'],first_failure_step_first=a[key]['first_failure_step'],first_failure_step_second=b[key]['first_failure_step']))
  comparisons.append(dict(first=first,second=second,counts={k:len(v) for k,v in counts.items()},cases=counts))
 candidate=metrics['adaptive_joint'];base=metrics['zero_inclusive'];spec=criteria['represented_sphere_acceptance'];reasons=[]
 if candidate['strict_windows']!=0:reasons.append('candidate strict represented-sphere violations')
 if candidate['deep_windows']!=0:reasons.append('candidate deep represented-sphere violations')
 if comparisons[0]['counts']['new_failure']!=0:reasons.append('new paired strict failure vs zero_inclusive')
 pathratio=candidate['mean_q_l2_path']/base['mean_q_l2_path'] if base['mean_q_l2_path']>0 else None
 motionratio=candidate['four_arms_moving_fraction']/base['four_arms_moving_fraction'] if base['four_arms_moving_fraction']>0 else None
 if pathratio is None or pathratio<.9:reasons.append('overall measured motion path below .9 baseline or undefined')
 if motionratio is None or motionratio<.9:reasons.append('four-arm moving fraction below .9 baseline or undefined')
 strata=[]
 for l in [-1,0,1,2,3,4,5]:
  a=float(np.mean([x['q_l2_path'] for x in stratum['zero_inclusive'][l]]));b=float(np.mean([x['q_l2_path'] for x in stratum['adaptive_joint'][l]]));ratio=b/a if a>0 else None
  if ratio is None or ratio<.9:reasons.append('stratum measured motion floor '+str(l))
  strata.append(dict(label=l,cases=len(stratum['zero_inclusive'][l]),path_ratio_candidate_vs_zero=ratio,by_mode=[dict(mode=m,strict=sum(x['strict'] for x in stratum[m][l]),deep=sum(x['deep'] for x in stratum[m][l]),native_hand_raw_over=sum(x['native_hand_raw_above_diagnostic'] for x in stratum[m][l]),mean_path=float(np.mean([x['q_l2_path'] for x in stratum[m][l]]))) for m in modes]))
 if min(candidate['mean_joint_ranges'])<.001:reasons.append('at least1 controlled joint mean measured range below .001rad')
 if any(not x['full_native_actual_initial_exact'] for x in pairing):reasons.append('paired actual full-native initial mismatch')
 sphere_reasons=list(reasons)
 if candidate['initial_negative_envs']!=0:reasons.append('candidate actual initialized geometry negative before first physics step')
 if candidate['native_hand_raw_over_windows']:reasons.append('raw native hand normal contact exceeds registered diagnostic .1N')
 result=dict(status='COMPLETE_PREREGISTERED_FINITE_ACCEPTANCE',initial_state_decision='REJECTED' if candidate['initial_negative_envs'] else 'PASS_OBSERVED_INITIAL_REPRESENTED_GEOMETRY_ONLY',candidate_decision='REJECTED' if reasons else 'PASS_BOUNDED_SIMULATION_ONLY',represented_sphere_decision='REJECTED' if sphere_reasons else 'PASS_BOUNDED_REPRESENTED_SPHERES',rejection_reasons=reasons,unique_paired_cases=128,method_windows=512,total_control_env_frames=512*960,total_physics_env_substeps=512*1920,independent_seed_banks=2,aggregated=aggregated,comparisons=comparisons,strata=strata,native_pairing=pairing,actual_motion_path_ratio=pathratio,actual_four_arm_movement_ratio=motionratio,all_full_window_scoring_and_bindings_checked=True,actual_windows_complete=512,total_PNG=sum(m['PNG'] for m in aggregated),queued_future_status='UNKNOWN',whole_machine_safety_certified=False,constructor_contacts_observed=False,full_friction_observed=False,hardware_approved=False,production_promoted=False,criteria_sha256=sha(H/'ACCEPTANCE_CRITERIA.json'),pairing_registration_sha256=sha(H/'PAIRING_REGISTRATION.json'),acceptance_registration_sha256=sha(H/'ACCEPTANCE_REGISTRATION.json'),camera_registration_sha256=sha(H/'CAMERA_COVERAGE_REGISTRATION.json'),initial_state_registration_sha256=sha(H/'INITIAL_STATE_ACCEPTANCE_REGISTRATION.json'),common_hand_registration_sha256=sha(H/'COMMON_HAND_CONTEXT_REGISTRATION.json'),raw_bindings=all_raw_bindings,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 for name,value in [('ACCEPTANCE_RESULT.json',result),('CASE_TABLE.json',cases)]:
  with (H/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
 print(result['status'],result['candidate_decision'],[(m['mode'],m['strict_windows'],m['deep_windows']) for m in aggregated],flush=True)
if __name__=='__main__':main()
