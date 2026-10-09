"""Matched long-window validation of hand-margin development physics and disposition."""
import json
from pathlib import Path
import sys

import numpy as np

import analyse_paired128_v1 as independent

H = Path(__file__).resolve().parent
REG = json.loads((H / 'FIXEDHAND_TRIPLET_DEV_REG_V1.json').read_text())
OUT = Path(REG['analysis_out'])
METHODS = ('raw', 'multirow', 'multirow_hold_fallback')


def read(path):
    return json.loads(path.read_text())


def load(path):
    with np.load(path) as z:
        return {k:z[k] for k in z.files}


def audit(method):
    independent.H = OUT
    independent.ROOT = Path(REG['native_namespace'])
    independent.audit(0, method)
    leaf = independent.ROOT / f'paired_batch0_{method}_v8'
    p = load(leaf / 'resolved_native_parameters.npz')
    roots = read(leaf / 'actual_solver_roots.json')
    assert len(roots) == 4 and all(x['position_iterations'] == 64 and x['velocity_iterations'] == 0 for x in roots)
    for a in independent.ARMS:
        idx = p[a+'_controlled_joint_indices']
        hand = np.ones(len(p[a+'_joint_names']), bool); hand[idx] = False
        assert np.array_equal(p[a+'_initial_q'][:,idx], p[a+'_original_initial_q'][:,idx])
        assert np.array_equal(p[a+'_full_hold_target'][:,idx], p[a+'_original_full_hold_target'][:,idx])
        bounds = p[a+'_hard_limits']
        for field, original in [('initial_q','original_initial_q'),('full_hold_target','original_full_hold_target')]:
            expected = np.clip(p[a+'_'+original][:,hand],bounds[:,hand,0]+np.float32(.03),bounds[:,hand,1]-np.float32(.03))
            assert np.array_equal(p[a+'_'+field][:,hand],expected)
    if method == 'multirow_hold_fallback':
        stream = load(leaf / 'response_stream.npz')
        initial = np.concatenate([p[a+'_initial_q'][:,p[a+'_controlled_joint_indices']] for a in independent.ARMS],-1)
        blocked = stream['fallback_from_model_unsatisfied']
        assert np.array_equal(blocked, ~stream['multi_model_constraints_satisfied'])
        assert np.array_equal(stream['issued_target'], np.where(blocked[...,None],initial[None],stream['nominal_issued_target']))


def aggregate():
    execution = read(H / 'fixedhand_triplet64_v1_execution.json')
    assert execution['status']=='complete' and len(execution['jobs'])==3
    assert all(j['status']=='complete' and j['actual_exit']==0 for j in execution['jobs'])
    data = {}
    for method in METHODS:
        leaf = Path(REG['native_namespace'])/f'paired_batch0_{method}_v8'
        data[method] = dict(stream=load(leaf/'response_stream.npz'),p=load(leaf/'resolved_native_parameters.npz'),
            oracle=read(OUT/f'PAIRED_ORACLE_batch0_{method}_V1.json'))
    base = data['raw']
    for method in METHODS:
        value = data[method]
        assert np.array_equal(value['stream']['reference_target'],base['stream']['reference_target'])
        for k in ['applied_target']+['post_'+a+'_'+f for a in independent.ARMS for f in ('q','qd')]:
            assert np.array_equal(value['stream'][k][:6],base['stream'][k][:6]),'Matched prefix '+method+' '+k
        for k in base['p']:
            assert np.array_equal(value['p'][k],base['p'][k]),'Matched parameters '+method+' '+k
    counts={}
    states={m:v['oracle']['states'] for m,v in data.items()}
    for method, rows in states.items():
        assert len(rows)==64 and [r['input_id'] for r in rows]==[r['input_id'] for r in states['raw']]
        counts[method]=dict(states=64,replay_prefix_qualified=sum(r['replay_prefix_qualified'] for r in rows),
            primary_failures=sum(r['primary_failure'] for r in rows),geometry_failures=sum(r['geometry_failure'] for r in rows),
            force_failures=sum(r['force_failure'] for r in rows),all74_hard_bad=sum(r['supplementary_max_all74_hard_violation_rad']>1e-5 for r in rows),
            all74_speed_bad=sum(r['supplementary_max_all74_velocity_exceedance_rad_s']>1e-5 for r in rows),
            joint_and_primary_pass=sum(not r['primary_failure'] and r['supplementary_max_all74_hard_violation_rad']<=1e-5 and r['supplementary_max_all74_velocity_exceedance_rad_s']<=1e-5 for r in rows),
            all4_arms_moved=sum(r['four_arms_moved'] for r in rows),total_controlled_path_rad=sum(r['controlled_path_rad'] for r in rows),
            peak_normal_N=max(r['peak_all_arm_scalar_N'] for r in rows),max_hard_breach_rad=max(r['supplementary_max_all74_hard_violation_rad'] for r in rows),
            max_speed_exceedance_rad_s=max(r['supplementary_max_all74_velocity_exceedance_rad_s'] for r in rows))
    blocked=data['multirow_hold_fallback']['stream']['fallback_from_model_unsatisfied']
    ratio=[r['controlled_path_rad']/b['controlled_path_rad'] for r,b in zip(states['multirow_hold_fallback'],states['raw'])]
    result=dict(status='CLOSED_FIXED_HAND_64_NATIVE_MATCHED480STEP_DEVELOPMENT_TRIPLET',counts=counts,states=states,
        all3_native_actual_exit0=True,all3_full960_independent_oracles_pass=True,all74_commonprefix_and_parameters_exact=True,
        all480_random_reference_targets_exact=True,all64_states_retained=True,hand_margin_rad=.03,position_iterations=64,
        fallback_lane_controls=int(blocked.sum()),total_lane_controls=int(blocked.size),
        hold_path_ratio_to_new_raw=dict(min=float(min(ratio)),median=float(np.median(ratio)),max=float(max(ratio))),
        registration_sha256=independent.sha(H/'FIXEDHAND_TRIPLET_DEV_REG_V1.json'),analysis_wrapper_sha256=independent.sha(Path(__file__)),
        safety_scope='Recorded960discrete substeps only on64known conditioned development inputs; holding/suppression is not task completion',
        trained_actor_or_grasp_task=False,formal_holdout=False,production_configuration_adopted=False,fullSystem0_accepted=False)
    (OUT/'FIXEDHAND_TRIPLET64_RESULT_V1.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(result['status'],counts,flush=True)


if __name__=='__main__':
    audit(sys.argv[1]) if len(sys.argv)>1 else aggregate()
