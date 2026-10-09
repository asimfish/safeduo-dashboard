"""Independent native evidence and prospective matched 128-input closure."""
import json
from pathlib import Path
import sys
import numpy as np

P = Path(__file__).resolve().parent
H = P.parent
sys.path.insert(0, str(H))
import analyse_paired128_v1 as independent

REG = json.loads((P/'REGISTRATION_V1.json').read_text())
ROOT = Path(REG['bank']).parent
METHODS = REG['methods']
independent.H = P
independent.ROOT = ROOT
load = independent.load
sha = independent.sha

def read(path): return json.loads(path.read_text())
def write(path, value): path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
def fullpass(row):
    return (not row['primary_failure'] and row['supplementary_max_all74_hard_violation_rad']<=1e-5
            and row['supplementary_max_all74_velocity_exceedance_rad_s']<=1e-5)

def audit(batch, method):
    independent.audit(batch, method)
    leaf = ROOT/f'paired_batch{batch}_{method}_v8'
    p, s, bank = [load(x) for x in [leaf/'resolved_native_parameters.npz',leaf/'response_stream.npz',Path(REG['bank'])]]
    selected = np.asarray(REG['batch_bank_positions'][batch])
    roots = read(leaf/'actual_solver_roots.json')
    assert len(roots)==4 and all(r['position_iterations']==64 and r['velocity_iterations']==0 for r in roots)
    assert np.array_equal(p['global_input_id'],bank['selected_input_id'][selected])
    for key in ['cell_id','velocity_fraction','command_mode','refresh_controls']:
        assert np.array_equal(p[key],bank[key][selected])
    for a in independent.ARMS:
        for key in ['initial_q','initial_qd','full_hold_target']:
            assert np.array_equal(p[a+'_'+key],bank[a+'_'+key][selected]),a+' '+key
        assert np.array_equal(p[a+'_hard_limits'],bank[a+'_hard_limits'][selected])
        assert np.array_equal(p[a+'_max_velocity'],bank[a+'_vmax'][selected])
    assert np.array_equal(s['reference_target'],bank['reference_target'][:,selected])
    tape=load(leaf/'actual_random_issue_tape.npz')
    assert np.array_equal(np.concatenate([tape[a] for a in independent.ARMS],-1),s['reference_target'])
    protocol=read(leaf/'response_protocol.json')
    assert protocol['method']==method and protocol['paired_controller']==method
    initial=np.concatenate([p[a+'_initial_q'][:,p[a+'_controlled_joint_indices']] for a in independent.ARMS],-1)
    if method=='raw':
        assert np.array_equal(s['issued_target'],s['reference_target'])
        assert 'fallback_from_model_unsatisfied' not in s
    else:
        expected=((s['multi_geometry_velocity_residual']<=1e-4)&(s['multi_controlled_velocity_limit_residual']<=1e-3)
            &(s['multi_full_drive_effort_residual']<=1e-3)&~s['multi_missing_direction']&~s['multi_invalid_box']
            &~s['multi_uncontrolled_nearby_raw_negative'])
        assert np.array_equal(expected,s['multi_model_constraints_satisfied'])
        blocked=s['fallback_from_model_unsatisfied']
        if method=='multirow':
            assert not blocked.any() and np.array_equal(s['issued_target'],s['nominal_issued_target'])
        else:
            assert np.array_equal(blocked,~expected)
            assert np.array_equal(s['issued_target'],np.where(blocked[...,None],initial[None],s['nominal_issued_target']))
    metrics=load(P/f'PAIRED_METRICS_batch{batch}_{method}_V1.npz')
    oracle=read(P/f'PAIRED_ORACLE_batch{batch}_{method}_V1.json')
    qualified=((metrics['gap_m'][:12].min(0)>=0)&(metrics['force_N'][:12].max(0)<=.1)
        &(metrics['hard_violation_rad'][:12].max(0)<=1e-5)&(metrics['velocity_exceedance_rad_s'][:12].max(0)<=1e-5))
    for row, accepted in zip(oracle['states'],qualified): row['replay_full74_joint_and_primary_qualified']=bool(accepted)
    oracle.update(strong_input_and_actual_issue_exact=True,all4_actual_roots64_velocity0=True,
        all128_denominator_retained=True,strong_analysis_source_sha256=sha(Path(__file__)))
    write(P/f'STRONG_ORACLE_batch{batch}_{method}_V1.json',oracle)
    print('STRONG_AUDIT_PASS',batch,method,'full74 replay prefix',int(qualified.sum()),flush=True)

def summarize(rows):
    return dict(inputs=len(rows),primary_failures=sum(r['primary_failure'] for r in rows),
        geometry_failures=sum(r['geometry_failure'] for r in rows),force_failures=sum(r['force_failure'] for r in rows),
        all74_hard_bad=sum(r['supplementary_max_all74_hard_violation_rad']>1e-5 for r in rows),
        all74_speed_bad=sum(r['supplementary_max_all74_velocity_exceedance_rad_s']>1e-5 for r in rows),
        joint_and_primary_pass=sum(fullpass(r) for r in rows),
        full74_replay_prefix_qualified=sum(r['replay_full74_joint_and_primary_qualified'] for r in rows),
        all4_arms_moved=sum(r['four_arms_moved'] for r in rows),total_controlled_path_rad=sum(r['controlled_path_rad'] for r in rows),
        peak_normal_N=max(r['peak_all_arm_scalar_N'] for r in rows),
        max_hard_breach_rad=max(r['supplementary_max_all74_hard_violation_rad'] for r in rows),
        max_speed_exceedance_rad_s=max(r['supplementary_max_all74_velocity_exceedance_rad_s'] for r in rows))

def aggregate():
    execution=read(H/'strong_long128_v1_execution.json')
    assert execution['status']=='complete' and len(execution['jobs'])==6
    assert all(j['status']=='complete' and j['actual_exit']==0 for j in execution['jobs'])
    states={m:[] for m in METHODS}
    blocked_count=valid_count=0
    for batch in range(2):
        data={m:(load(ROOT/f'paired_batch{batch}_{m}_v8/response_stream.npz'),
            load(ROOT/f'paired_batch{batch}_{m}_v8/resolved_native_parameters.npz')) for m in METHODS}
        base, params=data['raw']
        for method,(stream,p) in data.items():
            assert np.array_equal(stream['reference_target'],base['reference_target'])
            for key in ['applied_target']+['post_'+a+'_'+f for a in independent.ARMS for f in ('q','qd')]:
                assert np.array_equal(stream[key][:6],base[key][:6]),'matched full74 prefix '+method+' '+key
            assert set(params)==set(p)
            for key in params: assert np.array_equal(params[key],p[key]),'matched parameters '+key
            oracle=read(P/f'STRONG_ORACLE_batch{batch}_{method}_V1.json')
            assert oracle['strong_input_and_actual_issue_exact'] and len(oracle['states'])==64
            states[method].extend(oracle['states'])
        blocked_count+=int(data['multirow_hold_fallback'][0]['fallback_from_model_unsatisfied'].sum())
        valid_count+=int(data['multirow_hold_fallback'][0]['multi_model_constraints_satisfied'].sum())
    for rows in states.values(): rows.sort(key=lambda r:r['input_id'])
    raw=states['raw']
    bank=load(Path(REG['bank']))
    assert [r['input_id'] for r in raw]==sorted(bank['selected_input_id'].tolist())
    for method,rows in states.items():
        assert [r['input_id'] for r in rows]==[r['input_id'] for r in raw] and len(rows)==128
        assert [r['replay_full74_joint_and_primary_qualified'] for r in rows]==[r['replay_full74_joint_and_primary_qualified'] for r in raw]
    cells=[dict(cell=c,methods={m:summarize([r for r in rows if r['cell']==c]) for m,rows in states.items()}) for c in range(16)]
    assert all(v['inputs']==8 for cell in cells for v in cell['methods'].values())
    pairs={}
    for method in METHODS[1:]:
        rows=states[method]
        pairs[method]=dict(primary_rescues=[r['input_id'] for r,b in zip(rows,raw) if b['primary_failure'] and not r['primary_failure']],
            new_primary_failures=[r['input_id'] for r,b in zip(rows,raw) if not b['primary_failure'] and r['primary_failure']],
            full74_rescues=[r['input_id'] for r,b in zip(rows,raw) if not fullpass(b) and fullpass(r)],
            new_full74_failures=[r['input_id'] for r,b in zip(rows,raw) if fullpass(b) and not fullpass(r)])
    ratios=[r['controlled_path_rad']/b['controlled_path_rad'] for r,b in zip(states['multirow_hold_fallback'],raw) if b['controlled_path_rad']>0]
    result=dict(status='CLOSED_STRONG_RANDOM128_MATCHED_NATIVE_TRIPLET480',counts={m:summarize(rows) for m,rows in states.items()},
        cells=cells,states=states,paired_changes=pairs,all6_native_and_independent960_oracles_pass=True,
        all3_same_parameters_all480_reference_and_full74_first6_exact=True,all128_retained_in_denominator=True,
        fallback_lane_controls=blocked_count,model_satisfied_lane_controls=valid_count,total_lane_controls=61440,
        hold_path_ratio_to_raw=dict(min=float(min(ratios)),median=float(np.median(ratios)),max=float(max(ratios)),denominator=len(ratios)),
        registration_sha256=sha(P/'REGISTRATION_V1.json'),analysis_sha256=sha(Path(__file__)),
        position_iterations=64,velocity_iterations=0,hand_margin_rad=.03,
        initial_velocity_support_fractions=[0,.5,.75,1],command_modes=REG['command_modes'],
        safety_scope='960 recorded discrete native substeps per input; conditioned development cohort. Joint motion/hold is not manipulation success.',
        trained_actor_or_grasp_task=False,objects=False,formal_holdout=False,production_adoption=False,fullSystem0_accepted=False)
    write(P/'STRONG_LONG128_RESULT_V1.json',result)
    print(result['status'],result['counts'],flush=True)

if __name__=='__main__':
    audit(int(sys.argv[1]),sys.argv[2]) if len(sys.argv)>1 else aggregate()
