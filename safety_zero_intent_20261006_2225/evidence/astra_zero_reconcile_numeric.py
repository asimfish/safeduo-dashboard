"""Post-outcome schema comparison only; never recompute or modify raw scores."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

H = Path(__file__).resolve().parent
OUTPUT = H / 'ASTRA_ZERO_NUMERIC_RECONCILIATION_01.json'
SCORE_SHA = '979de60872246e070fe569521106faeecb852ae7d6205254a3c0653e0a533bde'
RAW_REG_SHA = '4f5cd77d28f3a0b5f6707096807c10795b840ca544528ad64ed964e4960c70fa'
CLASSES = ['cross', 'self_F', 'self_U', 'table']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not OUTPUT.exists(), 'Preserve existing reconciliation attempts'
    names = ['holdout_results.json', 'ANALYSIS_EXECUTION.json',
             'ASTRA_ZERO_FINAL_SCORE.json', 'ASTRA_ZERO_RAW_SOURCE_REGISTRATION.json',
             'astra_zero_raw_plan.json', 'astra_zero_raw_process.json',
             'astra_zero_raw_process_exit.json', 'astra_zero_raw_supervisor_attempt01_receipt.json',
             'analyze.py', 'dense_audit.py', 'execute_analysis.py', 'follow_analysis.py',
             'CANDIDATE_DECISION_REGISTRATION.json', 'ALL_ANALYSIS_EXECUTION.json',
             'recovery_follower_closed_execution.json', Path(__file__).name]
    before = {name: sha(H / name) for name in names}
    assert before['ASTRA_ZERO_FINAL_SCORE.json'] == SCORE_SHA
    assert before['ASTRA_ZERO_RAW_SOURCE_REGISTRATION.json'] == RAW_REG_SHA
    own = json.loads((H / 'ASTRA_ZERO_FINAL_SCORE.json').read_text())
    parent = json.loads((H / 'holdout_results.json').read_text())
    receipt = json.loads((H / 'ANALYSIS_EXECUTION.json').read_text())
    all6 = json.loads((H / 'ALL_ANALYSIS_EXECUTION.json').read_text())
    follower = json.loads((H / 'recovery_follower_closed_execution.json').read_text())
    assert all6['status'] == 'PASS_ALL6_FRESH_ANALYSIS_JOBS' and len(all6['jobs']) == 6
    assert all(j['exit_code'] == 0 for j in all6['jobs'])
    assert follower['actual_exit_code'] == 0 and follower['child_reaped']
    launch = json.loads((H / 'astra_zero_raw_process.json').read_text())
    assert all(sha(H / name) == digest for name, digest in launch['source_sha256'].items())
    assert own['status'] == 'PASS_COMPLETE_INDEPENDENT_ZERO_NUMERIC'
    assert receipt['status'] == 'PASS_FROZEN_FRESH_FULL_READBACK'
    assert receipt['source_after_verified'] is True and receipt['persisted_cache'] is False
    assert receipt['result_sha256'] == before['holdout_results.json']
    assert parent['analysis_source_sha256'] == before['analyze.py']
    assert parent['dense_audit_source_sha256'] == before['dense_audit.py']
    assert own['class_order'] == CLASSES

    counts = Counter()
    mismatches = []

    def equal(group, key, actual, expected):
        counts[group] += 1
        if actual != expected:
            mismatches.append(dict(group=group, key=key, parent=actual, independent=expected))

    def mm(value):
        # Exactly reproduce the declared float32 DISPLAY unit conversion only.
        # Strict/deep classification remains the original frozen native-metre result.
        return float(np.float32(value) * np.float32(1000))

    def minimum(metrics):
        return min(v for v in metrics['minimum_nonexempt_margin_m'].values() if v is not None)

    equal('inventory', 'registered_windows', parent['registered_method_windows'], 576)
    equal('inventory', 'completed_windows', parent['completed_method_windows'], 576)
    equal('inventory', 'invalid_current_windows', parent['invalid_method_windows'], 0)
    equal('inventory', 'parent_rows', len(parent['rows']), 9)
    equal('inventory', 'parent_cases', len(parent['cases']), 192)
    equal('inventory', 'parent_pairs', len(parent['paired']), 9)
    own_cases = {(c['command_seed'], c['env']): c for c in own['cases']}
    parent_cases = {(c['seed'], c['env']): c for c in parent['cases']}
    equal('inventory', 'unique_parent_case_keys', len(parent_cases), 192)
    equal('inventory', 'case_keys', sorted(parent_cases), sorted(own_cases))
    own_rows = {c['id']: c for c in own['conditions']}
    equal('inventory', 'row_keys', sorted(r['id'] for r in parent['rows']), sorted(own_rows))
    equal('inventory', 'mode_keys', sorted(r['mode'] for r in parent['totals']), sorted(own['totals']))

    for total in parent['totals']:
        mode = total['mode']
        expected = own['totals'][mode]
        fields = {
            'completed_windows': expected['verified_windows'],
            'violations': expected['strict']['failed_windows'],
            'deep': expected['deep']['failed_windows'],
            'strict_env_steps': expected['strict']['env_steps'],
            'deep_env_steps': expected['deep']['env_steps'],
            'class_violations': [expected['strict']['class_failed_windows'][c] for c in CLASSES],
            'min_nonexempt_mm': mm(minimum(expected)),
        }
        for key, value in fields.items():
            equal('mode_totals', mode + '/' + key, total[key], value)

    input_bindings = []
    parent_only_hashes = []
    for row in parent['rows']:
        expected = own_rows[row['id']]
        mode, seed = row['mode'], row['seed']
        subset = [c['windows'][mode] for c in own['cases'] if c['command_seed'] == seed]
        equal('condition_rows', row['id'] + '/status', row['status'], 'complete')
        fields = {
            'completed_windows': expected['counts']['verified_windows'],
            'invalid_windows': expected['counts']['unavailable_or_invalid_windows'],
            'violations': expected['counts']['strict']['failed_windows'],
            'deep': expected['counts']['deep']['failed_windows'],
            'class_violations': [expected['counts']['strict']['class_failed_windows'][c] for c in CLASSES],
            'min_nonexempt_mm': mm(minimum(expected['counts'])),
            'zero_prefix_violations': sum(w['prefix_metrics']['strict']['failed'] for w in subset),
        }
        for key, value in fields.items():
            equal('condition_rows', row['id'] + '/' + key, row[key], value)
        equal('condition_rows', row['id'] + '/root', row['path'], expected['root'])
        geometric = expected['geometry']
        audit_fields = {
            'rows_per_step': geometric['full_rows'],
            'frames': geometric['full_pre_frames'],
            'values_checked': geometric['full_rows'] * geometric['full_pre_frames'] * 64,
            'selected_ids_count': geometric['selected_row_instances'],
            'motion_only_added_rows_at_own_state': geometric['non_target_added_row_instances_at_own_state'],
            'motion_only_added_env_steps': geometric['non_target_added_env_steps_at_own_state'],
            'negative_full_pre_geometric_margin_env_steps': geometric['full_geometry_pre_strict_env_steps'],
        }
        for key, value in audit_fields.items():
            equal('shared_geometry_metadata', row['id'] + '/' + key, row['audit'][key], value)
        for relative, digest in row['input_sha256'].items():
            path = str(Path(row['path']) / relative)
            if path in own['input_sha256']:
                equal('shared_raw_hashes', path, digest, own['input_sha256'][path])
                input_bindings.append(path)
            else:
                parent_only_hashes.append(path)

    for key, record in parent_cases.items():
        case = own_cases[key]
        equal('case_identity', case['case_id'] + '/methods', sorted(record['methods']), sorted(case['windows']))
        for mode, window in case['windows'].items():
            actual = record['methods'][mode]
            metrics = window['metrics']
            name = case['case_id'] + '/' + mode
            equal('case_identity', name + '/stratum', record['stratum'], window['risk_pair_index'])
            equal('window_endpoints', name + '/failed', actual['failed'], metrics['strict']['failed'])
            equal('window_endpoints', name + '/deep', actual['deep'], metrics['deep']['failed'])
            equal('window_endpoints', name + '/first_failure_step', actual['first_failure_step'], metrics['strict']['first_step'])
            equal('window_minimum_mm', name, actual['min_mm'], [mm(metrics['minimum_nonexempt_margin_m'][c]) for c in CLASSES])

    pair_keys = [(r['seed'], r['a'], r['b']) for r in parent['paired']]
    equal('inventory', 'unique_pair_keys', len(set(pair_keys)), 9)
    expected_pair_keys = [(seed, v['reference'], v['candidate'])
                          for seed in sorted({c['command_seed'] for c in own['cases']})
                          for v in own['paired'].values()]
    equal('inventory', 'pair_keys', sorted(pair_keys), sorted(expected_pair_keys))
    for row in parent['paired']:
        actual_counts = dict(rescued=0, new_failures=0, both=0, neither=0)
        for case in own['cases']:
            if case['command_seed'] != row['seed']:
                continue
            a = case['windows'][row['a']]['metrics']['strict']['failed']
            b = case['windows'][row['b']]['metrics']['strict']['failed']
            key = ('both' if b else 'rescued') if a else ('new_failures' if b else 'neither')
            actual_counts[key] += 1
        for key, value in actual_counts.items():
            equal('per_seed_pair_partitions', f"{row['seed']}/{row['a']}/{row['b']}/{key}", row[key], value)

    unique_tapes = len({c['windows']['joint_reference']['tape_sha256'] for c in own['cases']})
    equal('tape_uniqueness', 'actual_960_frame_tapes', parent['unique_actual_tapes'], unique_tapes)
    parent_new = [own_cases[key]['case_id'] for key, c in parent_cases.items()
                  if not c['methods']['joint_reference']['failed'] and c['methods']['zero_inclusive']['failed']]
    independent_new = own['paired']['joint_reference__vs__zero_inclusive']['strict']['new_failure']
    equal('primary_new_failure_identities', 'case_ids', sorted(parent_new), sorted(independent_new))

    # The live composite observer log can still append camera records. Preserve only
    # the observed numeric closure lines, without claiming terminal ALL6 closure.
    log_path = H / 'recovered_analysis_child.log'
    observed = log_path.read_bytes()
    numeric_names = ['execute_analysis.py', 'failure_audit.py', 'audit_fixed_feasibility.py', 'audit_commands.py']
    numeric_lines = [line for line in observed.decode().splitlines()
                     if any(line.startswith('READBACK_CLOSED ' + name + ' ') for name in numeric_names)]
    equal('analysis_closure_evidence', 'execute_analysis_closed0_line',
          'READBACK_CLOSED execute_analysis.py 0' in numeric_lines, True)
    changed = [name for name, digest in before.items() if sha(H / name) != digest]
    equal('input_stability', 'closed_inputs_unchanged', changed, [])
    equal('input_stability', 'frozen_raw_sources_unchanged',
          all(sha(H / name) == digest for name, digest in launch['source_sha256'].items()), True)
    result = dict(
        schema='astra.zero.numeric_reconciliation.v1',
        utc=datetime.now(timezone.utc).isoformat(),
        status='PASS_SCOPED_PARENT_NUMERIC_RECONCILIATION' if not mismatches else 'FAIL_NUMERIC_RECONCILIATION',
        source_and_input_sha256=before,
        independent_score_completed_before_parent_reconciliation=True,
        frozen_scoring_math_changed=False,
        parent_scores_used_to_compute_independent_results=False,
        purpose='Compare existing independently scored results with closed parent output; no rescoring, parent module imports or new simulations.',
        expected_windows=576, checked_windows=576, checked_cases=192,
        checked_condition_rows=9, checked_mode_totals=3, checked_per_seed_pair_rows=9,
        checked_window_class_minima=576 * 4,
        checks_by_category=dict(counts), checks_total=sum(counts.values()),
        mismatch_count=len(mismatches), mismatches=mismatches,
        shared_raw_input_hashes_compared=len(input_bindings),
        parent_only_raw_input_hashes_not_compared=parent_only_hashes,
        parent_fresh_readback=dict(status=receipt['status'], seconds=receipt['seconds'],
                                  result_sha256=receipt['result_sha256'], persisted_cache=receipt['persisted_cache']),
        parent_all6_actual_closed=dict(status=all6['status'], exit_codes=[j['exit_code'] for j in all6['jobs']],
                                      follower_actual_exit=follower['actual_exit_code'], follower_reaped=follower['child_reaped'],
                                      scope='Execution receipts only, not independent reconstruction of every analysis arithmetic.'),
        observed_numeric_closure_log=dict(path=str(log_path), bytes_observed=len(observed),
            observed_prefix_sha256=hashlib.sha256(observed).hexdigest(), numeric_lines=numeric_lines,
            scope='Captured numeric wait-return log lines; observer may append and this is not a final ALL6 structured receipt.'),
        minimum_display_comparison='Exact float32 metres multiplied by float32(1000), then JSON float; no epsilon and no changes to original native float32 strict/deep classification.',
        totals=own['totals'], zero_prefix_totals=own['zero_prefix_totals'],
        primary_partition={kind: own['paired']['joint_reference__vs__zero_inclusive'][kind]['counts']
                           for kind in ['strict', 'deep']},
        new_primary_strict_cases=independent_new,
        candidate_decision='REJECTED_ADVERSE_ENDPOINTS',
        decision_reason='Two new primary paired strict failures trigger the unchanged pre-outcome rejection rule.',
        explicitly_not_reconciled=[
            'Parent per-frame curves/actuation exports; not available in the frozen score output schema.',
            'Parent L2 movement, coverage volumes/bin counts, six-pair exposure and solver diagnostics. Frozen score has different L1 movement summaries; these are not interchangeable.',
            'Deep per-class and per-case frame durations are retained independently but are not exported as comparable fields in the parent aggregate.',
            'No independent all-LP or full-J everyframe reconstruction.',
            'Camera groups, native state, PNG inventory and actual pixels remain pending explicit BOTH-camera closure.',
            'Actual REPORT independent review and final delivery remain pending.'
        ],
        retained_deviations=['First attempt448complete/128unstartedinvalid; original erroneous rc-only PASS retained.',
            'Initial GPU gate FAIL and baseline-only early SIGSTOP/SIGCONT pause retained.',
            'Actual failed remote-asset and unsupportedGPU1 camera attempts retained; old unknown childwait remainsUNRECORDED.',
            'No cross-attempt remoteUSD dependency byte-equivalence or clean one-attempt preregistered-execution claim.'],
        camera_audit_started=False, actual_camera_pixels_viewed=0,
        final_review_pass=False, physical_safety_certified=False,
        hardware_approved=False, production_promoted=False,
        runtime=dict(python=sys.version, numpy=np.__version__))
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ['status', 'checks_total', 'mismatch_count',
        'shared_raw_input_hashes_compared', 'parent_only_raw_input_hashes_not_compared', 'candidate_decision']}))
    if mismatches:
        print(json.dumps(mismatches[:20], ensure_ascii=False))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
