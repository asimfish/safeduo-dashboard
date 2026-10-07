"""One-shot reviewer freeze after final registration, before any policy launch.

Read-only parent metadata; writes only the independent owned registration.
Never starts a worker, watches files, or loads outcome arrays.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from astra_zero_raw_score import H,RAW,Ledger,registered_inputs,require,sha

SOURCE_NAMES=(
    'astra_zero_score_core.py','astra_zero_raw_score.py','astra_zero_raw_tests.py',
    'astra_zero_raw_supervise.py','astra_zero_contract_tests.py','astra_zero_guard_fixture.py',
    'astra_zero_oracle.py','astra_zero_cpu_supervise.py','astra_zero_freeze.py',
    'astra_zero_raw_contract.md')


def freeze_sources():
    name='ASTRA_ZERO_RAW_SOURCE_REGISTRATION.json'
    require(not (H/name).exists(),'source registration already exists; immutable')
    receipt_names=['astra_zero_raw_attempt02_receipt.json','astra_zero_contract_attempt01_receipt.json']
    receipts={n:json.loads((H/n).read_text()) for n in receipt_names}
    for n,r in receipts.items():
        require(r['actual_child_exit_code']==0 and r['child_reaped'] and r['source_stable'],'CPU attempt not successful: '+n)
        require(not Path('/proc',str(r['child_pid'])).exists(),'CPU child still present: '+n)
    raw=receipts[receipt_names[0]]
    for n in ('astra_zero_score_core.py','astra_zero_raw_score.py','astra_zero_raw_tests.py','astra_zero_raw_supervise.py'):
        require(sha(H/n)==raw['own_source_sha256'][n],'raw test does not bind current source: '+n)
    review=json.loads((H/'ASTRA_ZERO_SOURCE_REVIEW.json').read_text())
    for path,digest in review['source_sha256'].items():
        require(sha(path)==digest,'reviewed policy source changed')
    for root in (RAW/f'holdout_{b}' for b in range(3)):
        require(not root.exists(),'policy output directory exists before source freeze; investigate')
    import numpy._core._multiarray_umath as arithmetic
    doc=dict(schema='astra.zero.raw_source_registration.v1',
        status='FROZEN_INDEPENDENT_RAW_SCORER_BEFORE_ALL_NEW_POLICY_OUTCOMES',
        registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256={n:sha(H/n) for n in SOURCE_NAMES},
        own_imported_helpers=['astra_zero_score_core.py','astra_zero_oracle.py','astra_zero_guard_fixture.py'],
        runtime=dict(interpreter=sys.executable,python=sys.version,numpy=np.__version__,
            numpy_module_sha256=sha(np.__file__),numpy_native_arithmetic_sha256=sha(arithmetic.__file__),
            executable_sha256=sha(Path(sys.executable).resolve()),
            native_arithmetic_path=arithmetic.__file__),
        cpu_execution_receipts={n:dict(sha256=sha(H/n),actual_child_exit_code=r['actual_child_exit_code'],
            child_pid=r['child_pid'],child_reaped=r['child_reaped'],log_sha256=sha(r['log']),
            actual_argv=r['actual_argv']) for n,r in receipts.items()},
        tests=dict(raw_synthetic=32,candidate_cpu_contracts=20,meaningful_candidate_mutants_detected=5),
        provenance=dict(derivation=str(H/'astra_zero_derivation.json'),derivation_sha256=sha(H/'astra_zero_derivation.json'),
            prior_readonly_namespace=str(H.parent/'safety_tracking_reserve_20261006_1535'),
            raw_math='Native float32 comparison before any JSON conversion; <0 and <-float32(.005), no epsilon.',
            reused='Prior independent reviewer core/raw scorer/raw tests/supervisor and scalar oracle/full-row fixture, adapted in new owned namespace.',
            parent_analysis_or_scores_imported=False),
        reviewed_policy_sha256=review['source_sha256'],source_review_sha256=sha(H/'ASTRA_ZERO_SOURCE_REVIEW.json'),
        modes=['joint_reference','tight_reference','zero_inclusive'],primary='joint_reference__vs__zero_inclusive',
        expected_cases=192,expected_windows=576,steps=960,zero_prefix_steps=60,
        math_frozen=True,algorithm_changes_to_reconcile_forbidden=True,
        final_plans_binding_pending=True,plan_binding_file='astra_zero_raw_plan.json',
        new_policy_outcome_reads=0,parent_scores_used=False,watcher_running=False,workers_running=0,
        no_gpu_or_simulation_launched=True,no_old_namespace_writes=True,
        limits=['No independent full-J or all-LP every-frame claim.','Camera state/pixels and final reconciliation remain pending.',
                'Model family GPT-6 supplied by session; Astra/xhigh runtime setting not independently visible.'])
    with (H/name).open('x') as f: json.dump(doc,f,indent=2,allow_nan=False);f.write('\n')
    print(doc['status'],str(H/name),sha(H/name),flush=True)


def freeze_plans():
    source_reg=json.loads((H/'ASTRA_ZERO_RAW_SOURCE_REGISTRATION.json').read_text())
    require(source_reg['math_frozen'],'source registration not frozen')
    for name,digest in source_reg['source_sha256'].items():
        require(sha(H/name)==digest,'frozen scorer source changed: '+name)
    ledger=Ledger()
    design,plans=registered_inputs(ledger)
    # No scoring or campaign watcher can start until this succeeds. A launched
    # campaign also blocks this stronger prelaunch freeze, even with no arrays.
    for plan in plans:
        root=Path(plan['output_root'])
        require(not (root/'campaign.json').exists(),'campaign already launched; prelaunch freeze unavailable')
        for job in plan['jobs']:
            require(not (root/job['id']).exists(),'policy job directory already exists; preserve evidence and investigate')
    require(not (H/'astra_zero_raw_plan.json').exists(),'existing scorer freeze must not be overwritten')
    review=ledger.json(H/'ASTRA_ZERO_SOURCE_REVIEW.json')
    for name,digest in review['source_sha256'].items():ledger.bind(name,digest)
    for name,digest in review['supporting_readonly_source_sha256'].items():ledger.bind(name,digest)
    receipt=ledger.json(H/'astra_zero_raw_attempt02_receipt.json')
    require(receipt['actual_child_exit_code']==0 and receipt['child_reaped'],'raw CPU child did not close successfully')
    for name in ('astra_zero_score_core.py','astra_zero_raw_score.py','astra_zero_raw_tests.py','astra_zero_raw_supervise.py'):
        ledger.bind(H/name,receipt['own_source_sha256'][name])
    require(not ledger.recheck(),'registration/review/test input changed before freeze')
    names=['NUMERIC_REGISTRATION.json','RANDOM_EXPERIMENT_DESIGN.json','ASTRA_ZERO_SOURCE_REVIEW.json',
           'ASTRA_ZERO_RAW_SOURCE_REGISTRATION.json']
    names += [f'plans/holdout_{b}_plan.json' for b in range(3)]
    names += ['astra_zero_raw_attempt02_receipt.json','astra_zero_raw_attempt02.log',
              'astra_zero_contract_attempt01_receipt.json','astra_zero_contract_attempt01.log']
    specification=dict(schema='astra.zero.independent_raw_plan.v1',
        status='REGISTERED_INDEPENDENT_TWO_WORKER_SCORER_BEFORE_RAW_OUTCOME_READS',
        registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        before_all_policy_launches=True,own_new_policy_outcome_reads=0,
        workers=2,minimum_mem_available_bytes=25*1024**3,
        expected_cases=192,expected_windows=576,frames=960,rows=9021,
        conditions=['joint_reference','tight_reference','zero_inclusive'],
        primary_comparison='joint_reference__vs__zero_inclusive',
        source_sha256={n:sha(H/n) for n in SOURCE_NAMES},
        registration_sha256={n:sha(H/n) for n in names},
        scorer_argv=[sys.executable,'-u',str(H/'astra_zero_raw_score.py')],
        supervisor_argv=[sys.executable,'-u',str(H/'astra_zero_raw_supervise.py')],
        math_frozen=True,scoring_algorithm_changes_to_reconcile_forbidden=True,
        native_thresholds=dict(strict='float32 <0',deep='float32 <-float32(.005)',epsilon=None),
        prefix_frames_in_primary=True,zero_prefix_steps=60,
        allframe_reference_bounds_and_prelimit_independently_recomputed=True,
        camera_and_report_review_pending=True,all_LP_or_all_J_every_frame_claim=False,
        parent_scores_are_inputs=False,indefinite_watcher=False,
        canonical_gate='All three campaigns terminal before worker creation; complete zero-exit actual child/argv/protocol required per counted condition.',
        termination='Two threads join, scorer exits, single-use supervisor records actual child return code and exits; no automatic retry.',
        first_failure_selection='First two distinct failing environments by (step,env) per condition.',
        consumed_before_freeze_sha256=ledger.hashes)
    with (H/'astra_zero_raw_plan.json').open('x') as file:
        json.dump(specification,file,indent=2,allow_nan=False);file.write('\n')
    print('FROZEN_BEFORE_ALL_NEW_POLICY_OUTCOMES',sha(H/'astra_zero_raw_plan.json'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--sources-only',action='store_true')
    args=parser.parse_args()
    freeze_sources() if args.sources_only else freeze_plans()
