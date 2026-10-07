"""Finite read-only verification of closed reports; never recomputes raw scores."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import ast
import hashlib
import json
import os
import sys

H = Path(__file__).resolve().parent
RAW = Path('/mnt/nas/data/lyf/double_hand') / H.name
checks = Counter()
bindings = {}


def sha(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    value = digest.hexdigest()
    bindings[str(path)] = value
    return value


def read(name):
    path = H / name
    sha(path)
    return json.loads(path.read_text())


def check(value, category, detail):
    if not value:
        raise AssertionError((category, detail))
    checks[category] += 1


def verify_map(values, base=H, category='registered_hash'):
    for name, expected in values.items():
        check(sha(base / name) == expected, category, name)


def main():
    own = read('ASTRA_ZERO_FINAL_SCORE.json')
    raw_reg = read('ASTRA_ZERO_RAW_SOURCE_REGISTRATION.json')
    camera_reg = read('ASTRA_ZERO_CAMERA_SOURCE_REGISTRATION.json')
    raw_plan = read('astra_zero_raw_plan.json')
    reconcile = read('ASTRA_ZERO_NUMERIC_RECONCILIATION_01.json')
    camera = read('ASTRA_ZERO_FINAL_CAMERA_REVIEW.json')
    verify_map(raw_reg['source_sha256'])
    verify_map(raw_reg['reviewed_policy_sha256'], category='six_policy_hash')
    verify_map(camera_reg['helper_source_sha256'])
    verify_map(camera_reg['producer_source_sha256'])
    verify_map(raw_plan['registration_sha256'])
    verify_map(reconcile['source_and_input_sha256'], category='reconciliation_binding')
    verify_map(camera['source_and_evidence_sha256'], category='camera_review_binding')
    verify_map(read('ASTRA_ZERO_SOURCE_REVIEW.json')['supporting_readonly_source_sha256'])
    numeric_reg = read('NUMERIC_REGISTRATION.json')
    verify_map(numeric_reg['plans'])
    numeric_jobs = []
    for name in numeric_reg['plans']:
        plan = read(name)
        verify_map(plan['source_sha256'], Path(plan['cwd']), 'plan_production_source')
        verify_map(plan['research_source_sha256'], category='plan_research_source')
        check(sha(plan['checkpoint_path']) == plan['checkpoint_sha256'], 'actor_checkpoint', name)
        campaign = json.loads((Path(plan['output_root']) / 'campaign.json').read_text())
        sha(Path(plan['output_root']) / 'campaign.json')
        check(campaign['status'] == 'complete' and len(campaign['jobs']) == 3, 'numeric_campaign', name)
        for planned, actual in zip(plan['jobs'], campaign['jobs']):
            root = Path(plan['output_root']) / actual['id']
            protocol = json.loads((root / 'protocol.json').read_text())
            expected_argv = planned['argv'] + ['--out', str(root)]
            check(actual['id'] == planned['id'] and actual['argv'] == expected_argv, 'numeric_job_plan', actual['id'])
            check(actual['status'] == 'complete' and actual['exit_code'] == 0, 'numeric_child_exit', actual['id'])
            check(protocol['status'] == 'complete' and protocol['completed_cells'] == 1, 'numeric_protocol', actual['id'])
            numeric_jobs.append(dict(id=actual['id'], actual_exit_code=actual['exit_code'], protocol_sha256=sha(root/'protocol.json')))

    parent = read('holdout_results.json')
    zero = read('ZERO_INTENT_AUDIT.json')
    report = (H / 'REPORT.md').read_text()
    build = read('REPORT_BUILD.json')
    check(sha(H / 'REPORT.md') == build['report_sha256'], 'report_binding', 'REPORT.md')
    check(own['actual_case_identities'] == 192 and own['actual_window_records'] == 576, 'scope', 'independent576')
    check(parent['completed_method_windows'] == 576 and parent['invalid_method_windows'] == 0, 'scope', 'parent576')
    check(reconcile['mismatch_count'] == 0 and reconcile['checked_windows'] == 576, 'scope', 'reconciliation')
    lines = report.splitlines()
    prefix_reconciliation = []
    for row in parent['totals']:
        mode = row['mode']
        independent = own['totals'][mode]
        expected_counts = [independent['strict']['failed_windows'], independent['deep']['failed_windows'], independent['strict']['env_steps'], independent['deep']['env_steps']]
        table_row = next(line for line in lines if line.startswith('| '+mode+' |'))
        actual = [cell.strip() for cell in table_row.strip('|').split('|')]
        check(list(map(int, actual[1:5])) == expected_counts, 'report_independent_counts', mode)
        check(actual[5:] == [f"{row['min_nonexempt_mm']:.6f}", f"{row['joint_path_l2_mean_rad']:.9f}", f"{row['four_arms_moving_fraction']:.9f}"], 'report_parent_presentation_only', mode)
        z = next(x for x in zero['summary'] if x['mode'] == mode)
        for own_key, parent_key in [('prefix_prelimit_injected_env_steps','reference_prelimit_injection_env_steps'), ('prefix_nonzero_projector_output_env_steps','original_projector_nonzero_return_env_steps'), ('prefix_nonzero_effective_target_delta_env_steps','nonzero_effective_target_env_steps')]:
            total = sum(c['windows'][mode]['reference_contract'][own_key] for c in own['cases'])
            check(total == z[parent_key], 'independent_saved_prefix_sum', mode+':'+own_key)
            prefix_reconciliation.append(dict(mode=mode, independent_field=own_key, total=total, parent_field=parent_key))
        prefix_line = f"| {mode} | {z['zero_prefix_strict_windows']} | {z['zero_prefix_strict_env_steps']} | {z['bounds_excluding_zero_env_steps']} | {z['reference_prelimit_injection_env_steps']} | {z['original_projector_nonzero_return_env_steps']} | {z['nonzero_effective_target_env_steps']} | {z['maximum_target_drift_rad']:.9f} |"
        check(prefix_line in lines, 'report_prefix_table', mode)
    for pair in parent['paired']:
        expected = f"| {pair['seed']} | {pair['a']}→{pair['b']} | {pair['rescued']} | {pair['new_failures']} | {pair['both']} | {pair['neither']} |"
        check(expected in lines, 'report_reconciled_pair', expected)
    paired = own['paired'][own['primary_comparison']]
    decision = 'REJECTED_ADVERSE_ENDPOINTS' if paired['strict']['new_failure'] else None
    check(decision == build['decision']['code'], 'independent_decision', decision)
    check(len(paired['strict']['new_failure']) == 2, 'independent_decision', 'new2')
    for label in ['拒绝采用', '首尝试实际448完成、128', '无计分epsilon', '不声称独立复算每帧全J', '相对初始测量q的最大已发目标位移rad', '原可达区间本身排除零', '不强制执行零，也不清空FIFO', '第t帧新目标不能越过前六个待执行目标', 'FAIL_EXACT_REPLAY', '实际usdrt不支持GPU1', 'SIGKILL', 'UNRECORDED', '最终以原CUDA0计划', '未对首尝与补跑所有传递USD依赖做逐字节等价证明', 'SIGSTOP', 'SIGCONT', '运动减少或暴露不足不被称为安全改善']:
        check(label in report, 'report_required_disclosure', label)
    for run in camera['camera_runs']:
        check(str(run['maximum_q_difference_rad']) in report, 'report_camera_replay', run['mode'])
    check(camera['actual_original_pngs_viewed'] == 72 and camera['all_native_groups_checked'] == 47 and camera['all_png_hashes_and_headers_checked'] == 423, 'camera_scope', '72/47/423')

    source_names = ['build_report.py','build_delivery.py','panel_template.html','check_panel.py','update_root.py','candidate_decision.py','plot_results.py','finalize_certificate.py','seal_evidence.py','recover_unstarted.py','follow_recovery_analysis.py','run_closed_step.py','follow_analysis.py','audit_zero_intent.py','finalize_recovery_termination.py','recover_original_camera.py','close_recovered_study.py']
    current_sources = {name: sha(H/name) for name in source_names}
    snapshot = read('ASTRA_ZERO_RECOVERY_PRESENTATION_SNAPSHOT_01.json')['sources']
    source_deltas = {name: 'unchanged' if (H/name).read_text() == saved['text'] else 'reviewed_final_disclosure_or_plot_change' for name,saved in snapshot.items() if name in source_names}
    for name in source_names:
        if name.endswith('.py'):
            ast.parse((H/name).read_text(), filename=name)
            checks['presentation_operational_ast_parse'] += 1
    check('相对初始测量q的最大已发目标位移rad' in (H/'panel_template.html').read_text(), 'presentation_label', 'panel')
    check(current_sources['candidate_decision.py'] == snapshot['candidate_decision.py']['sha256'], 'decision_source_unchanged', 'candidate_decision')
    check(sha(H/'CANDIDATE_DECISION_REGISTRATION.json') == snapshot['CANDIDATE_DECISION_REGISTRATION.json']['sha256'], 'decision_registration_unchanged', 'registration')

    parent_receipts = {}
    for name in ['original_camera_closed_execution.json','composed_recovery_closed_execution.json','recovery_follower_closed_execution.json','recovered_analysis_closed_execution.json','zero_intent_audit_closed_execution.json','closed_report_closed_execution.json','scientific_plots_v3_closed_execution.json']:
        receipt = read(name)
        check(receipt['actual_exit_code'] == 0 and receipt['child_reaped'], 'actual_parent_wait', name)
        source = next(arg for arg in receipt['actual_argv'] if arg.endswith('.py'))
        check(receipt['source_sha256_before'] == receipt['source_sha256_after'] == sha(source), 'actual_parent_source', name)
        parent_receipts[name] = receipt
    analysis = read('ALL_ANALYSIS_EXECUTION.json')
    check(len(analysis['jobs']) == 6 and all(j['exit_code'] == 0 for j in analysis['jobs']), 'analysis_actual_exits', 'all6')
    for job in analysis['jobs']:
        check(sha(H/job['name']) == job['source_sha256'], 'analysis_actual_sources', job['name'])
    visual = read('visual_execution.json')
    vplan = read('visual_plan.json')
    check(visual['plan_sha256'] == camera_reg['visual_plan_sha256'] == sha(H/'visual_plan.json'), 'original_camera_plan', 'sha')
    for actual, planned in zip(visual['jobs'], vplan['jobs']):
        check(actual['argv'] == planned['argv'] and actual['exit_code'] == 0 and actual['child_reaped'], 'camera_actual_wait_and_argv', actual['mode'])
        check(sha(Path(actual['out'])/'visual_protocol.json') == actual['visual_protocol_sha256'], 'camera_producer_protocol_binding', actual['mode'])
    for name in ['RECOVERY_EXECUTION.json','RECOVERY_TERMINAL_CLOSURE.json','CAMERA_AMENDMENT_EXECUTION_BINDING.json']:
        check(read(name)['status'].startswith('PASS'), 'actual_composed_closure', name)
    check(read('RECOVERY_TERMINAL_CLOSURE.json')['unpersisted_failed_GPU1_child_wait'] == 'UNRECORDED', 'failure_retention', 'unknownwait')
    first = read('initial_attempts/holdout_results.json')
    check(first['completed_method_windows'] == 448 and first['invalid_method_windows'] == 128, 'failure_retention', '448/128')
    retained = {}
    for name in ['initial_attempts/NUMERIC_EXECUTION.json','initial_attempts/manifest.json','PLANS_RECOVERY_AMENDMENT.json','NUMERIC_INITIAL_RESOURCE_GATE.json','INITIAL_RESOURCE_GATE_CONTAINMENT.json','RESOURCE_RESUME_REGISTRATION.json','RESOURCE_PAUSE_RESUME_GATE.json','camera_gpu1_failed_attempt/RECOVERY_TERMINAL_CLOSURE.json','camera_gpu1_failed_attempt/visual_execution.json','recovery_closed_execution.json','recovery_termination_closed_execution.json','scientific_plots_v2_closed_execution.json','scientific_plots_v2_child.log','initial_plot_export/PLOT_EXECUTION.json','initial_plot_export/comparison.png']:
        retained[name] = sha(H/name)
    for name in ['recovery_closed_execution.json','recovery_termination_closed_execution.json','scientific_plots_v2_closed_execution.json']:
        receipt = read(name)
        check(receipt['actual_exit_code'] == 1 and receipt['child_reaped'], 'retained_failed_actual_wait', name)
    check(read('PLOT_EXECUTION.json')['result_sha256'] == sha(H/'holdout_results.json'), 'plot_data_binding', 'result')
    for name in ['comparison.png','comparison.pdf','comparison.svg','DELIVERY_VERIFICATION_CONTRACT.md']:
        sha(H/name)

    own_receipts = {}
    for path in sorted(H.glob('astra_zero*receipt*.json')):
        value = read(path.name)
        own_receipts[path.name] = {key: value[key] for key in ['actual_child_exit_code','actual_exit_code','child_reaped','child_pid','supervisor_pid'] if key in value}
    check(read('astra_zero_raw_process_exit.json')['actual_child_exit_code'] == 0, 'own_scoring_wait', 'raw')
    for name in ['astra_zero_raw_supervisor_attempt01_receipt.json','astra_zero_camera_audit-closed_attempt01_receipt.json','astra_zero_camera_supervisor_attempt01_receipt.json','astra_zero_reconcile_numeric_attempt01_receipt.json']:
        receipt = read(name)
        check(receipt['actual_child_exit_code'] == 0 and receipt['child_reaped'], 'own_actual_wait', name)
    own_pid_observations = {}
    for receipt in own_receipts.values():
        for key in ['child_pid','supervisor_pid']:
            if key in receipt:
                pid = receipt[key]
                path = Path('/proc')/str(pid)/'cmdline'
                try: argv = path.read_bytes().split(b'\0')
                except FileNotFoundError: argv = []
                still_ours = any(str(H).encode() in arg and b'astra_zero_' in arg for arg in argv)
                check(not still_ours, 'own_recorded_worker_closed', str(pid))
                own_pid_observations[str(pid)] = dict(proc_exists=path.exists(), same_owned_worker=still_ours)
    live = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try: argv = path.read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError): continue
        if int(path.parent.name) == os.getpid(): continue
        if any(arg.startswith(str(H/'astra_zero_').encode()) and arg.endswith(b'.py') for arg in argv):
            live.append(dict(pid=int(path.parent.name),argv=[a.decode(errors='replace') for a in argv if a]))
    check(not live, 'own_live_worker_scan', live)
    result = dict(schema='astra.zero.final_report_source_check.v1', status='PASS_SCOPED_ACTUAL_REPORT_SOURCE_AND_CLOSURE_CHECK', utc=datetime.now(timezone.utc).isoformat(), checks=dict(checks), checks_total=sum(checks.values()), report_sha256=sha(H/'REPORT.md'), report_bytes=(H/'REPORT.md').stat().st_size, source_sha256=current_sources, source_delta_review=source_deltas, evidence_sha256=bindings, numeric_jobs=numeric_jobs, parent_actual_wait_receipts=parent_receipts, own_prior_wait_receipts=own_receipts, own_recorded_pid_observations=own_pid_observations, other_own_workers_running=live, this_finite_check_pid=os.getpid(), own_prefix_summary_reconciliation=prefix_reconciliation, retained_history_sha256=retained, independent_candidate_decision=decision, new_primary_strict_cases=paired['strict']['new_failure'], raw_scoring_math_modified=False, parent_scores_used_to_compute_independent_raw_results=False, presentation_L2_motion_exposure_LP_values_independently_recomputed=False, current_plot_actual_view_image=True, plot_view_counted_toward72=False, final_publication_verified=False, raw_entire_directory_twice_read_verified=False, runtime=dict(executable=sys.executable,version=sys.version,bytecode_disabled=sys.dont_write_bytecode,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
    with (H/'ASTRA_ZERO_FINAL_REPORT_SOURCE_REVIEW_01.json').open('x') as handle:
        json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.write('\n')
    print(result['status'], result['checks_total'], result['report_sha256'], flush=True)


if __name__ == '__main__':
    main()
