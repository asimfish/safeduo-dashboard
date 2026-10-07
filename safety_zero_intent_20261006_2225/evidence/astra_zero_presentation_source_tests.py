"""Pre-data review fixtures; source functions run with in-memory IO seams only."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

H=Path(__file__).resolve().parent
SNAPSHOT=json.loads((H/'astra_zero_presentation_source_snapshot_initial.json').read_text())
SOURCES={n:row['text'] for n,row in SNAPSHOT['sources'].items()}


def extract(name,function,scope):
    tree=ast.parse(SOURCES[name])
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function)
    namespace=dict(scope)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<frozen presentation source '+name+'>','exec'),namespace)
    return namespace[function]


DECIDE=extract('candidate_decision.py','decide',{})


def result(reference=2,candidate=1,reference_deep=1,candidate_deep=1,reference_duration=10,candidate_duration=10,new=0):
    totals=[]
    for mode,strict,deep,duration in [
        ('joint_reference',reference,reference_deep,reference_duration),
        ('tight_reference',3,1,15),('zero_inclusive',candidate,candidate_deep,candidate_duration)]:
        totals.append(dict(mode=mode,violations=strict,deep=deep,strict_env_steps=20,
            deep_env_steps=duration,min_nonexempt_mm=-1.,joint_path_l2_mean_rad=3.,four_arms_moving_fraction=.5))
    paired=[dict(seed=i,a='joint_reference',b='zero_inclusive',new_failures=new if i==0 else 0,
                 rescued=1,both=1,neither=62) for i in range(3)]
    return dict(totals=totals,paired=paired,completed_method_windows=576,invalid_method_windows=0)


def report_docs():
    zero=[]
    for mode in ('joint_reference','tight_reference','zero_inclusive'):
        zero.append(dict(mode=mode,zero_prefix_strict_windows=0,zero_prefix_strict_env_steps=0,
            bounds_excluding_zero_env_steps=0,reference_prelimit_injection_env_steps=0,
            original_projector_nonzero_return_env_steps=1,nonzero_effective_target_env_steps=1,maximum_target_drift_rad=.125))
    return {'ALL_ANALYSIS_EXECUTION.json':dict(status='PASS_ALL6_FRESH_ANALYSIS_JOBS'),
        'ASTRA_ZERO_FINAL_SCORE.json':dict(status='PASS_COMPLETE_INDEPENDENT_ZERO_NUMERIC'),
        'ZERO_INTENT_AUDIT.json':dict(status='PASS_SYNTHETIC_ZERO_AUDIT',summary=zero),
        'holdout_results.json':result(),
        'visual_verification.json':dict(groups=42,images=378,runs=[dict(mode=m,binding_status='FAIL_EXACT_FORWARD',q_max_difference=.01) for m in ['joint_reference','zero_inclusive']]),
        'bank_audit.json':dict(prior_compared=1536,minimum_l2_distance_rad=1.68),
        'command_audit.json':dict(prior_unique_tapes=1536)}


def run_report(docs):
    from datetime import datetime,timezone
    writes={}
    def write(name,text):
        if name in writes:raise FileExistsError(name)
        writes[name]=text
    def sha(path):return hashlib.sha256(writes[Path(path).name].encode()).hexdigest()
    main=extract('build_report.py','main',dict(load=lambda n:docs[n],decide=DECIDE,write=write,
        HERE=H,sha=sha,json=json,datetime=datetime,timezone=timezone))
    with patch('builtins.print'):main()
    return writes


class DecisionRuleTests(unittest.TestCase):
    def test_each_registered_adverse_endpoint_rejects(self):
        for kwargs in [dict(new=1),dict(candidate=3),dict(candidate_deep=2),dict(candidate_duration=11)]:
            with self.subTest(kwargs=kwargs):
                self.assertEqual(DECIDE(result(**kwargs))['code'],'REJECTED_ADVERSE_ENDPOINTS')

    def test_all_adverse_reasons_retained(self):
        value=DECIDE(result(new=1,candidate=3,candidate_deep=2,candidate_duration=11))
        self.assertEqual(len(value['reasons']),4)

    def test_residual_failure_and_zero_observed_are_limited_scope(self):
        self.assertEqual(DECIDE(result())['code'],'RESEARCH_ONLY_RESIDUAL_FAILURES')
        self.assertEqual(DECIDE(result(candidate=0,candidate_deep=0,candidate_duration=0))['code'],
                         'OBSERVED_ZERO_FAILURES_LIMITED_SCOPE_ONLY')

    def test_primary_requires_three_block_rows(self):
        data=result();data['paired'].pop()
        with self.assertRaises(AssertionError):DECIDE(data)


class ReportGateTests(unittest.TestCase):
    def test_report_build_does_not_need_final_review_or_raw_seal(self):
        docs=report_docs()
        self.assertNotIn('ASTRA_ZERO_FINAL_REVIEW.json',docs)
        self.assertNotIn('raw_evidence_manifest.json',docs)
        output=run_report(docs)
        report=output['REPORT.md']
        self.assertIn('初始GPU空闲内存门禁实际失败',report)
        self.assertIn('原失败不改为PASS',report)
        self.assertIn('第t帧新目标不能越过前六个待执行目标',report)
        self.assertIn('FAIL_EXACT_FORWARD',report)
        self.assertIn('不声称独立复算每帧全J',report)
        self.assertFalse(json.loads(output['CANDIDATE_DECISION.json'])['physical_safety_certified'])
        self.assertEqual(json.loads(output['REPORT_BUILD.json'])['report_sha256'],hashlib.sha256(report.encode()).hexdigest())

    def test_required_analysis_independent_score_and_zero_audit_gates(self):
        for gate in ['ALL_ANALYSIS_EXECUTION.json','ASTRA_ZERO_FINAL_SCORE.json','ZERO_INTENT_AUDIT.json']:
            docs=report_docs();docs[gate]['status']='RUNNING'
            with self.subTest(gate=gate),self.assertRaises(AssertionError):run_report(docs)

    def test_incomplete_or_invalid_windows_reject_before_report_write(self):
        for key,value in [('completed_method_windows',575),('invalid_method_windows',1)]:
            docs=report_docs();docs['holdout_results.json'][key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):run_report(docs)


class AssetGateTests(unittest.TestCase):
    def test_final_review_and_double_raw_read_gate_precede_asset_directory_creation(self):
        class ReachedDirectory(Exception):pass
        class StopAssets:
            def mkdir(self,**kwargs):raise ReachedDirectory('all gating predicates traversed; no filesystem mutation')
        docs={'holdout_results.json':dict(completed_method_windows=576,invalid_method_windows=0),
            'visual_verification.json':dict(status='PASS_BOUNDED_ACTUAL_SIX_PLANE_CAMERA'),
            'ASTRA_ZERO_FINAL_REVIEW.json':dict(status='PASS_SCOPED'),
            'raw_evidence_manifest.json':dict(status='PASS_TWO_CLOSED_RAW_READS'),
            'ALL_ANALYSIS_EXECUTION.json':dict(status='PASS_ALL6_FRESH_ANALYSIS_JOBS'),
            'ZERO_INTENT_AUDIT.json':dict(status='PASS_SYNTHETIC')}
        fn=extract('build_delivery.py','assets',dict(HERE=H,ASSETS=StopAssets(),load=lambda p:docs[Path(p).name]))
        for name in ['ASTRA_ZERO_FINAL_REVIEW.json','raw_evidence_manifest.json']:
            saved=docs[name];docs[name]=dict(status='RUNNING')
            with self.subTest(name=name),self.assertRaises(AssertionError):fn()
            docs[name]=saved
        with self.assertRaises(ReachedDirectory):fn()

    def test_fixed_asset_commit_original_copy_and_nine_local_files_are_explicit(self):
        source=SOURCES['build_delivery.py']
        self.assertIn("publication['commit']",source)
        self.assertIn('assert sha(src)==sha(dest)==s',source)
        self.assertIn('pixel_transformation=False',source)
        self.assertIn("p.suffix in ['.py','.json','.md','.html','.log','.png','.pdf','.svg']",source)
        self.assertIn("new_initial_Pages_budget_bytes",source)


class ScopeRegressionTests(unittest.TestCase):
    def test_frozen_prefix_metric_is_measured_q0_relative_not_issued_target_drift(self):
        source=SOURCES['audit_zero_intent.py']
        self.assertIn("data['controller_target'][:60]-data['q_initial']",source)
        q0=0.;initial_issued=.125;later_issued=.125
        self.assertEqual(abs(later_issued-q0),.125)
        self.assertEqual(abs(later_issued-initial_issued),0.)
        self.assertIn('最大目标漂移',SOURCES['build_report.py'])
        self.assertIn('最大目标漂移',SOURCES['panel_template.html'])

    def test_root_edit_guard_preserves_unowned_cards(self):
        import re
        fn=extract('update_root.py','strip_owned',dict(re=re))
        before='<main>\n<section id="trackingReserveEvidence">old</section>\n<p>keep</p></main>'
        after='<main>\n<section id="zeroIntentEvidence">new</section>\n<section id="trackingReserveEvidence">older</section>\n<p>keep</p></main>'
        self.assertEqual(fn(before),fn(after))
        self.assertNotEqual(fn(before),fn(after.replace('<p>keep</p>','<p>changed</p>')))


if __name__=='__main__':unittest.main(verbosity=2)
