"""Terminal source and result consistency gates; no parent arithmetic rerun."""
from pathlib import Path
import ast
from datetime import datetime, timezone
import hashlib
import json

H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(n):return json.loads((H/n).read_text())


def main():
    names=['REPORT.md','REPORT_BUILD.json','REPRODUCE.md','NEXT.md','panel_template.html','build_report.py','build_delivery.py','update_root.py','check_panel.py',
        'analyze.py','dense_audit.py','execute_analysis.py','failure_audit.py','audit_fixed_feasibility.py','audit_zero_intent.py','audit_adaptive_saturation.py',
        'ASTRA_TRACKING_REVIEW.json','ASTRA_TRACKING_FINAL_SCORE.json','ASTRA_TRACKING_FINAL_TRAJECTORY_IDENTITY.json','ASTRA_TRACKING_FINAL_RECONCILIATION_REREVIEW.json','ASTRA_TRACKING_FINAL_CAMERA_REVIEW.json',
        'holdout_results.json','first_failure_audit.json','FIXED_FEASIBILITY_AUDIT.json','FIXED_LP_REGISTRATION.json','LP_BACKEND_REGISTRATION.json','LP_BACKEND_PRECHECK.json',
        'ADAPTIVE_SATURATION_AUDIT.json','SATURATION_AUDIT_AMENDMENT.json','ZERO_INTENT_AUDIT.json','POSTHOC_EXECUTION.json','OUTER_SESSION_EXIT_AUDIT.json','NUMERIC_EXECUTION.json','ALL_ANALYSIS_EXECUTION.json','ANALYSIS_EXECUTION.json',
        'visual_execution.json','visual_verification.json','CAMERA_STATE_AUDIT.json','bank_audit.json','command_audit.json',Path(__file__).name]
    before={n:sha(H/n) for n in names}
    plan=load('astra_tracking_raw_plan.json');bound={str(H/n):s for n,s in plan['source_sha256'].items()}
    bound.update(plan['source_review_policy_hashes_current']);bound.update(plan['plans'])
    registered={}
    campaigns=[]
    for p,digest in plan['plans'].items():
        assert sha(p)==digest
        p=json.loads(Path(p).read_text())
        registered.update({str(Path(p['cwd'])/n):s for n,s in p['source_sha256'].items()})
        registered.update(p['research_source_sha256'])
        campaign=Path(p['output_root'])/'campaign.json';c=json.loads(campaign.read_text())
        assert c['status']=='complete' and len(c['jobs'])==3
        assert all(j['status']=='complete' and j['exit_code']==0 for j in c['jobs'])
        campaigns.append(dict(path=str(campaign),sha256=sha(campaign),jobs=[dict(id=j['id'],pid=j['pid'],exit_code=j['exit_code']) for j in c['jobs']]))
    bound.update(registered)
    assert all(sha(n)==s for n,s in bound.items()),'registered source drift'
    own=load('ASTRA_TRACKING_FINAL_SCORE.json');parent=load('holdout_results.json')
    assert own['status']=='PASS_COMPLETE_INDEPENDENT_TRACKING_NUMERIC' and own['raw_hash_before_after_pass']
    assert load('ASTRA_TRACKING_FINAL_RECONCILIATION_REREVIEW.json')['status'].startswith('PASS')
    camera=load('ASTRA_TRACKING_FINAL_CAMERA_REVIEW.json')
    assert camera['actual_original_pngs_viewed']==72 and camera['all_group_math_native_state_count']==54
    allanalysis=load('ALL_ANALYSIS_EXECUTION.json')
    assert allanalysis['status']=='PASS_ALL6_FRESH_ANALYSIS_JOBS'
    assert all(j['exit_code']==0 and sha(H/j['name'])==j['source_sha256'] for j in allanalysis['jobs'])
    lp=load('first_failure_audit.json');fixed=load('FIXED_FEASIBILITY_AUDIT.json')
    assert lp['source_sha256']==before['failure_audit.py']==load('LP_BACKEND_REGISTRATION.json')['first_failure_source_sha256']
    assert fixed['source_sha256']==before['audit_fixed_feasibility.py']==load('FIXED_LP_REGISTRATION.json')['source_sha256']
    expected={(c['command_seed'],m,c['env']):c['windows'][m]['metrics']['strict']['first_step'] for c in own['cases'] for m in own['modes'] if c['windows'][m]['metrics']['strict']['failed']}
    observed={(r['seed'],r['mode'],r['env']):r['first_failure_step'] for r in lp['records']}
    assert observed==expected and len(observed)==len(lp['records'])==lp['cases']==382
    declared_snapshots={}
    for condition in own['conditions']:
        p=Path(condition['root'])/'first_failure_receipts.json'
        assert sha(p)==own['input_sha256'][str(p)]
        for r in json.loads(p.read_text())['receipts']:
            declared_snapshots[str(p.parent/r['path'])]=r['sha256']
    assert all(declared_snapshots[r['snapshot_path']]==r['snapshot_sha256'] for r in lp['records'])
    selected=[s for c in own['conditions'] for s in c['geometry']['selected_first_failures']]
    assert len(selected)==18 and all(s['status']=='PASS_SELECTED_FIRST_FAILURE_BINDINGS' for s in selected)
    lp_summary=[]
    for m in own['modes']:
        rows=[r for r in lp['records'] if r['mode']==m]
        assert all(r['nearest_selected_previous'] for r in rows)
        lp_summary.append(dict(mode=m,cases=len(rows),F_infeasible=sum(not r['joint_set_checks']['F']['hard_set_feasible'] for r in rows),U_infeasible=sum(not r['joint_set_checks']['U']['hard_set_feasible'] for r in rows)))
    reg=load('FIXED_LP_REGISTRATION.json');expected_fixed={(m,c['command_seed'],c['env'],t) for m in own['modes'] for c in own['cases'] for t in reg['pre_steps']}
    observed_fixed={(r['mode'],r['seed'],r['env'],r['pre_step']) for r in fixed['records']}
    assert observed_fixed==expected_fixed and len(fixed['records'])==fixed['states']==5184 and fixed['robot_sets']==10368
    for row in fixed['summary']:
        group=[r for r in fixed['records'] if r['mode']==row['mode'] and r['pre_step']==row['pre_step']]
        assert len(group)==row['cases']==192
        for robot in ['F','U']:assert sum(not r['checks'][robot]['feasible'] for r in group)==row[robot+'_infeasible']
    zero=load('ZERO_INTENT_AUDIT.json');prefix=[]
    for m in own['modes']:
        n=sum(c['windows'][m]['metrics']['strict']['first_step'] is not None and c['windows'][m]['metrics']['strict']['first_step']<60 for c in own['cases'])
        assert n==next(r['zero_prefix_strict_windows'] for r in zero['summary'] if r['mode']==m)
        prefix.append(dict(mode=m,independent_zero_prefix_failed_windows=n))
    assert all(w['initial_target_sha256']==w['q0_sha256'] for c in own['cases'] for w in c['windows'].values())
    saturation=load('ADAPTIVE_SATURATION_AUDIT.json')
    floor=sum(r['audit']['tracking_gap_tight_env_steps'] for r in parent['rows'] if r['mode']=='delay_reserve')
    assert floor==saturation['adaptive_tight_floor_env_steps']==saturation['total_env_steps']==184320
    assert sum(r['selected_J_exact_zero'] for r in saturation['initial_min_rows'])==saturation['initial_zero_gradient_min_rows']==0
    assert sum(r['physical_distance_positive'] for r in saturation['initial_min_rows'])==192
    post=load('POSTHOC_EXECUTION.json')
    assert all(sha(H/n)==s for n,s in post['receipts'].items())
    assert sha(H/'audit_adaptive_saturation.py')==load('SATURATION_AUDIT_AMENDMENT.json')['source_sha256']
    outer=load('OUTER_SESSION_EXIT_AUDIT.json')
    assert [r['observed_tool_exit_code'] for r in outer['outer_tool_sessions']]==[143,143]
    canonical={j['id']:j for c in campaigns for j in c['jobs']}
    assert len(canonical)==9
    assert all(canonical[r['id']]['pid']==r['pid'] and canonical[r['id']]['exit_code']==r['actual_exit_code']==0 for r in outer['numeric'])
    visual=load('visual_execution.json')
    assert visual['status']=='complete' and all(j['exit_code']==0 for j in visual['jobs'])
    assert {r['mode']:r['pid'] for r in outer['cameras']}=={r['mode']:r['pid'] for r in visual['jobs']}
    report=(H/'REPORT.md').read_text()
    assert before['REPORT.md']==load('REPORT_BUILD.json')['report_sha256']=='f4906f51ae79e16fa67fa4d42f1991b0a06ead64463b0d36f085f86827c031e4'
    for row in parent['totals']:
        assert f"| {row['mode']} | {row['violations']} | {row['deep']} | {row['strict_env_steps']} | {row['deep_env_steps']} |" in report
    assert '候选拒绝采用' in report and '不增加192初态或576数值窗口分母' in report
    assert 'FAIL_EXACT_REPLAY' in report and '探索性' in report and '唯一物理原因' in report
    for n in ['build_report.py','build_delivery.py','update_root.py','check_panel.py']:ast.parse((H/n).read_text())
    builder=(H/'build_report.py').read_text();delivery=(H/'build_delivery.py').read_text()
    assert "load('ASTRA_TRACKING_FINAL_REVIEW.json')" not in builder
    assert "review['status'].startswith('PASS')" in delivery
    assert "HERE.glob('astra_tracking_*')" in delivery and "HERE.glob('ASTRA_TRACKING_*')" in delivery
    assert 'actuation:s.actuation' in (H/'panel_template.html').read_text()
    assert "actuation=s['actuation']" in (H/'check_panel.py').read_text()
    assert before=={n:sha(H/n) for n in names}
    assert all(sha(n)==s for n,s in bound.items())
    receipt=dict(status='PASS_SCOPED_TERMINAL_SOURCE_REPORT_GATES',utc=datetime.now(timezone.utc).isoformat(),input_sha256=before,registered_source_sha256=registered,registered_source_count=len(registered),independent_frozen_bindings_sha256=bound,inputs_before_after_equal=True,
        canonical_campaigns=campaigns,first_failure_cases_matched_to_independent_score=382,first_failure_snapshot_hashes_matched_to_raw_receipts=True,independently_raw_bound_selected_first_failures=18,first_failure_lp_parent_summary=lp_summary,
        fixed_lp_parent_record_identity_count=5184,fixed_robot_sets=10368,lp_independently_resolved_again=False,lp_scope='Source, registered backend, actual exit receipts, record inventory and report aggregates checked; no independent rerun of parent LP solves.',
        independent_zero_prefix_window_reconciliation=prefix,posthoc_floor_count_crosschecked_between_two_parent_readbacks=184320,posthoc_initial_zeroJ_parent_count=0,
        floor_all_frames_independently_recounted_here=False,posthoc_forced_bound_counts_independently_recomputed=False,
        presentation_findings_resolved=['PRESENT-1: REPORT generation no longer waits on its own final review; publication still gates on final review.','PRESENT-2: dynamic download inventory includes all registered Astra plans, source dependencies, tests, failed attempts and actual closure/pixel receipts.'],
        outer_tool_exit_codes_retained=[143,143],outer143_cause_established=False,physical_safety_certified=False,actual_deployment_browser_verified=False)
    with (H/'astra_tracking_final_evidence_gate.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps({k:receipt[k] for k in ['status','registered_source_count','first_failure_cases_matched_to_independent_score','independently_raw_bound_selected_first_failures','fixed_lp_parent_record_identity_count']}))


if __name__=='__main__':main()
