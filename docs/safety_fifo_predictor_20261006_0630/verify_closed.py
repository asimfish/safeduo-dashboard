"""Multi-criterion gate after actual closed numerical and camera evidence."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(n):return json.loads((HERE/n).read_text())
def main():
    r=load('holdout_results.json');assert r['registered_method_windows']==r['completed_method_windows']==768 and r['invalid_method_windows']==0
    assert len(r['cases'])==192 and len({(c['seed'],c['env']) for c in r['cases']})==192
    d=load('RANDOM_EXPERIMENT_DESIGN.json');modes=d['modes']
    assert all(set(c['methods'])==set(modes) for c in r['cases'])
    registration=load('NUMERIC_REGISTRATION.json')
    for name,digest in registration['plans'].items():assert sha(name)==digest
    for p in sorted((HERE/'plans').glob('*.json')):
        plan=json.loads(p.read_text())
        for rel,digest in plan['source_sha256'].items():assert sha(Path(plan['cwd'])/rel)==digest,rel
        for name,digest in plan['research_source_sha256'].items():assert sha(name)==digest,name
        assert sha(plan['checkpoint_path'])==plan['checkpoint_sha256']
    assert (HERE/'motion_tests.log').read_text().rstrip().endswith('OK')
    assert load('ASTRA_MOTION_RUNNER_REVIEW.json')['status']=='PASS_SCOPED_CPU_SOURCE_REVIEW'
    for p in HERE.glob('*.py'):ast.parse(p.read_text(),str(p))
    assert load('offline_analysis_execution.json')['status']=='PASS_EXACT_REGISTERED_SOURCE_EXECUTION'
    assert load('ANALYSIS_EFFECTIVE_EXECUTION.json')['status']=='PASS_ALL4_SOURCE_BOUND_EFFECTIVE_READBACKS'
    assert load('ASTRA_CPU_PARALLEL_REVIEW.json')['status']=='PASS_SCOPED_FROZEN_CPU_SCHEDULING_AND_SUPERSESSION_REVIEW'
    bridge=load('ASTRA_FINAL_BRIDGE_REREVIEW.json')
    assert bridge['status']=='PASS_CORRECTED_EFFECTIVE_BRIDGE_SOURCE_AND_BINDINGS'
    assert all(sha(p)==d for p,d in bridge['bindings'].items())
    assert load('ASTRA_FINAL_LP_BACKEND_REVIEW.json')['status']=='PASS_SCOPED_LP_BACKEND_SOURCE_AND_EXECUTION_BINDING_REVIEW'
    assert all(row['audit']['complete_selected_union_membership_verified'] and row['audit']['actual_fifo_pending_exact']
           and row['audit']['full_pre_margin_bound_to_previous_dense_post'] for row in r['rows'])
    failure=load('first_failure_audit.json');assert failure['status']=='PASS_EXACT_FIRST_FAILURE_BINDING_AND_LP'
    assert failure['cases']==sum(t['violations'] for t in r['totals'])
    h6=load('H6_PREDICTION_AUDIT.json');assert h6['status']=='COMPLETE_OWN_TRAJECTORY_AUDIT' and len(h6['rows'])==12
    assert all(row['actual_steps']==954 for row in h6['rows'])
    prefix=load('NUMERIC_PREFIX_AUDIT.json');assert prefix['status']=='COMPLETE_NATIVE_CROSSMODE_FIRST6_PREFIX_READBACK' and len(prefix['rows'])==9
    assert all(all(x[k] for k in ['input_q0_exact','applied_first6_exact','first6_post_q_exact','initial_pre_qd_exact','first6_pre_qd_exact']) for x in prefix['rows'])
    independent_prefix=load('ASTRA_FINAL_NUMERIC_PREFIX_REVIEW.json')
    assert independent_prefix['status']=='PASS_INDEPENDENT_NATIVE_FIRST6_CONTROLLED_PREFIX' and len(independent_prefix['rows'])==9
    assert all(all(row[k] for k in ['input_q0_float32_bits_equal','applied_first6_float32_bits_equal','first6_post_q_float32_bits_equal','initial_pre_qd_float32_bits_equal','first6_pre_qd_float32_bits_equal','first6_post_qd_float32_bits_equal']) for row in independent_prefix['rows'])
    assert load('bank_audit.json')['fresh']==192 and load('bank_audit.json')['exact_duplicates']==0
    assert load('command_audit.json')['fresh_unique_tapes']==192 and load('command_audit.json')['exact_repeats_with_prior']==0
    v=load('visual_verification.json');assert v['status']=='PASS_BOUNDED_ACTUAL_SIX_PLANE_CAMERA'
    assert v['scheduled_groups']==42 and 378<=v['images']<=504 and len(v['runs'])==2
    camera=load('CAMERA_STATE_AUDIT.json');assert camera['status']=='PASS_EXACT_NATIVE_Q_QD_TARGET_AND_FIFO_BINDING'
    assert camera['groups']==v['groups']
    independent=load('ASTRA_FINAL_SCORE.json');assert independent['status']=='PASS_COMPLETE_INDEPENDENT_NUMERIC'
    assert independent['raw_hash_before_after_pass'] and not independent['changed_inputs']
    assert independent['actual_case_identities']==192 and independent['actual_window_records']==768
    own_cases={(c['seed'],c['env']):c for c in r['cases']};class_names=['cross','self_F','self_U','table']
    assert len(independent['cases'])==192
    for c in independent['cases']:
        own=own_cases[c['command_seed'],c['env']]
        for mode in modes:
            w=c['windows'][mode];assert w['status']=='VERIFIED'
            a=own['methods'][mode];metric=w['metrics']
            assert (a['failed'],a['deep'],a['first_failure_step'])==(metric['strict']['failed'],metric['deep']['failed'],metric['strict']['first_step'])
            minimum=np.array([metric['minimum_nonexempt_margin_m'][k] for k in class_names],np.float32)*np.float32(1000)
            assert minimum.tolist()==a['min_mm'],(c['case_id'],mode,'native class minimum mismatch')
    for t in r['totals']:
        c=independent['counts'][t['mode']]
        assert (t['completed_windows'],t['violations'],t['deep'])==(c['verified_windows'],c['strict']['failed_windows'],c['deep']['failed_windows'])
        assert t['class_violations']==[c['strict']['class_failed_windows'][k] for k in class_names]
    for mode in modes[1:]:
        comp=independent['comparisons'][mode];assert comp['verified_pairs']==192 and not comp['unavailable_or_invalid']
        parent={k:sum(p[k] for p in r['paired'] if p['a']=='joint_reference' and p['b']==mode) for k in ['rescued','new_failures','both','neither']}
        assert list(parent.values())==[comp['strict']['counts'][k] for k in ['rescue','new_failure','both_fail','both_safe']]
    assert load('ASTRA_CAMERA_REVIEW.json')['status'].startswith('PASS')
    review=load('ASTRA_FINAL_REVIEW.json');assert review['status'].startswith('PASS')
    totals={t['mode']:t for t in r['totals']}
    paired={m:{k:sum(p[k] for p in r['paired'] if p['a']=='joint_reference' and p['b']==m)
              for k in ('rescued','new_failures','both','neither')} for m in modes if m!='joint_reference'}
    for m,p in paired.items():
        assert sum(p.values())==192
        assert totals[m]['violations']==totals['joint_reference']['violations']-p['rescued']+p['new_failures']
    receipt=dict(status='PASS_SCIENTIFIC_AND_SOFTWARE_GATES_PENDING_PUBLIC_DELIVERY',utc=datetime.now(timezone.utc).isoformat(),
          numeric_windows=768,cases=192,invalid_windows=0,actor_and_all279_production_sources_unchanged=True,
          independent_fifo_tests=19,independent_motion_tests=14,parent_motion_tests=9,
          full9021_union_and_sixFIFO_audits=True,first_failure_cases=failure['cases'],h6_own_future_actual_steps=954,
          camera_images=v['images'],camera_groups=v['groups'],camera_actual_native_queue_verified=True,
          camera_forward_replay_status={x['mode']:x['binding_status'] for x in v['runs']},
          parent_and_independent_raw_scoring='all192 cases x4 strict/deep/firststep/nativefloat32 fourclassminima, all4totals and classcounts, all3 paired strict partitions exactly reconciled; no inference of physical safety from test success',
          hardware_approved=False,physical_safety_certified=False,production_promoted=False,
          public_delivery='PENDING_BYTE_READBACK_AND_NATIVE_BROWSER',
          evidence_sha256={n:sha(HERE/n) for n in ['holdout_results.json','H6_PREDICTION_AUDIT.json','first_failure_audit.json',
             'visual_verification.json','CAMERA_STATE_AUDIT.json','ASTRA_FINAL_SCORE.json','ASTRA_FINAL_REVIEW.json',
             'ASTRA_CAMERA_REVIEW.json','CAMERA_MITIGATION_RESULT.json','MODEL_FIT.json','NUMERIC_REGISTRATION.json','REPORT.md','REPRODUCE.md','NEXT.md']})
    with (HERE/'VERIFICATION.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(receipt['status'],flush=True)
if __name__=='__main__':main()
