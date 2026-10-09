"""Close delivery provenance and the explicitly requested legacy479 subset."""
import json
from pathlib import Path
import resource
import sys

from run_native_calibration_v1 import sha, write_json

D=Path(__file__).resolve().parent
H=D.parent.parent


def main():
    frozen=json.loads((D/'two_witness_v1/frozen493_before_after.json').read_text())
    material=json.loads((D/'two_witness_v1/input_source_sha256.json').read_text())['entries']
    legacy={}
    for name in ('QUEUE74_PREFLIGHT_PLAN_V1.json','QUEUE74_FULL_PLAN_V1.json'):
        path=H/name
        source=json.loads(path.read_text())['sources']
        assert len(source)==479
        assert all(p in frozen['sources'] and expected==frozen['sources'][p]['expected_sha256'] for p,expected in source.items())
        legacy[name]=dict(path=str(path),sha256=sha(path),count=479,exact_subset_of493=True,
                          all_original479_unchanged=all(sha(p)==expected for p,expected in source.items()))
        assert legacy[name]['all_original479_unchanged']
    for name in ('demo_original_snapshot_v1.py','frozen_cpu_providers_v2.py'):
        p=H/'astra/next_mechanism'/name
        material[str(p)]=dict(sha256=sha(p),role='additional read-only source interpretation')
    p=H/'astra/adaptation32/analyze_short_frontier_v4.py'
    material[str(p)]=dict(sha256=sha(p),role='original witness extraction semantics')
    for p,e in material.items():
        assert sha(p)==e['sha256'],p
    waits=[]
    for label in ('two_witness_attempt01','positive_negative_attempt01'):
        path=D/(label+'_actual_wait.json')
        r=json.loads(path.read_text())
        assert r['actual_exit']==0 and r['wait_returned'] and r['owned_pid']==r['waited_pid']
        assert sha(r['log'])==r['log_sha256']
        assert r['script_sha256_before']==r['script_sha256_after']
        assert (r['child_max_rss_KiB']+r['supervisor_max_rss_KiB'])*1024<1024**3
        waits.append(dict(path=str(path),sha256=sha(path),actual_exit=0,
            child_max_rss_KiB=r['child_max_rss_KiB'],supervisor_max_rss_KiB=r['supervisor_max_rss_KiB']))
    tests=json.loads((D/'positive_negative_tests_v1.json').read_text())
    assert tests['passed'] and tests['tests_run']==28 and tests['failures']==tests['errors']==0
    artifacts={}
    for p in sorted(D.rglob('*')):
        if p.is_file() and not p.name.startswith('delivery_closure_attempt'):
            artifacts[str(p)]=dict(sha256=sha(p),bytes=p.stat().st_size)
    for p,expected in json.loads((D/'two_witness_v1/output_sha256.json').read_text()).items():
        assert sha(p)==expected
    assert not list(D.rglob('__pycache__'))
    assert not any(k=='torch' or k.startswith(('isaac','omni.')) for k in sys.modules)
    report=json.loads((D/'two_witness_v1/calibration_report.json').read_text())
    baseline=[x for x in report['summaries'] if x['variant']['name']=='implicit_add_armature']
    output=dict(schema='safeduo.astra.native_calibration_delivery.v1',status='COMPLETED_REQUESTED_MINIMUM_CPU_CALIBRATION',
        helper=str(D/'native_teacherforced_cpu_v1.py'),callable='teacher_forced_two_micro(Macro74, Variant)',
        measurement_report=str(D/'two_witness_v1/calibration_report.json'),
        known_witness_report=str(D/'two_witness_v1/known_witness_report.json'),
        per74_table=str(D/'two_witness_v1/per74_error_summary.csv'),
        per74_micro_table=str(D/'two_witness_v1/per74_micro_errors.csv'),
        unique_native_macros=16,unique_native_microsteps=32,lanes=[18,19],variants=8,
        baseline_summaries=baseline,tests=tests,actual_wait_receipts=waits,
        all493_unchanged=frozen['all493_unchanged'],original479_manifests=legacy,
        material_inputs_and_sources=material,artifacts=artifacts,
        CONTACT='UNKNOWN',ACTUAL_DRIVE_TORQUE='UNKNOWN',TERMINAL_SET='UNKNOWN',GLOBAL_MODEL_ERROR_BOUND='UNKNOWN',
        full36_direct_comparison_enabled=False,future_plan_actual_target_first_mismatch_micro=14,
        full480_960_evaluated=False,global_bound_claimed=False,physical_safety_certified=False,
        gpu_started=False,isaac_or_applauncher_started=False,other_processes_stopped=False,
        additional_astra_xhigh_review='UNAVAILABLE_SESSION_BINDING_ERROR',
        closure_wait_receipt=str(D/'delivery_closure_attempt01_actual_wait.json'),
        closure_wait_receipt_status='separate supervisor writes this only after real wait4 returns',
        closure_self_hash_excluded=True,closure_running_log_and_wait_excluded_from_artifact_hashes=True)
    write_json(D/'NATIVE_CALIBRATION_DELIVERY_V1.json',output)
    print(json.dumps(dict(status=output['status'],tests_passed=28,all493_unchanged=True,
        original479_unchanged=True,report=str(D/'NATIVE_CALIBRATION_DELIVERY_V1.json'),
        max_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)),flush=True)


if __name__=='__main__':main()
