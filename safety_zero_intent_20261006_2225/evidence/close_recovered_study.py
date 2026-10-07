"""Compose actual closed numerical and original-camera attempts, retaining failures."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

H = Path(__file__).resolve().parent
RAW = Path('/mnt/nas/data/lyf/double_hand') / H.name


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = path.with_suffix('.combined.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    original_failed = read(H / 'recovery_closed_execution.json')
    camera_closed = read(H / 'original_camera_closed_execution.json')
    assert original_failed['actual_exit_code'] == 1 and original_failed['child_reaped']
    assert camera_closed['actual_exit_code'] == 0 and camera_closed['child_reaped']
    registration = read(H / 'CAMERA_ORIGINAL_RECOVERY_REGISTRATION.json')
    assert camera_closed['source_sha256_before'] == camera_closed['source_sha256_after'] == registration['driver_sha256']
    numeric = read(H / 'NUMERIC_EXECUTION.json')
    assert numeric['status'] == 'PASS_ALL_NUMERIC_CLOSED' and len(numeric['recovery_jobs']) == 2
    for block in range(3):
        plan = read(H / 'plans' / f'holdout_{block}_plan.json')
        campaign = read(RAW / f'holdout_{block}' / 'campaign.json')
        assert campaign['status'] == 'complete' and len(campaign['jobs']) == 3
        for job in campaign['jobs']:
            assert job['status'] == 'complete' and job['exit_code'] == 0
            protocol = read(Path(plan['output_root']) / job['id'] / 'protocol.json')
            assert protocol['status'] == 'complete' and protocol['completed_cells'] == 1
    first = read(H / 'initial_attempts/holdout_results.json')
    assert first['completed_method_windows'] == 448 and first['invalid_method_windows'] == 128
    plan = read(H / 'visual_plan.json')
    visual = read(H / 'visual_execution.json')
    assert visual['status'] == 'complete' and len(visual['jobs']) == 2
    assert visual['plan_sha256'] == sha(H / 'visual_plan.json') == registration['original_plan_sha256']
    for expected, actual in zip(plan['jobs'], visual['jobs']):
        assert expected['argv'] == actual['argv'] and expected['mode'] == actual['mode']
        assert actual['status'] == 'complete' and actual['exit_code'] == 0 and actual['child_reaped']
    binding = dict(status='PASS_ACTUAL_ORIGINAL_CAMERA_PLAN_RESTORED_AND_BOUND', original_plan_sha256=sha(H / 'visual_plan.json'), actual_final_executed_plan_sha256=sha(H / 'visual_plan.json'), actual_camera_execution_sha256=sha(H / 'visual_execution.json'), final_device='cuda:0', registered_GPU1_amendment_sha256=sha(H / 'PLANS_RECOVERY_AMENDMENT.json'), unsupported_GPU1_attempt_zero_saved_products=True, unsupported_GPU1_attempt_retained='camera_gpu1_failed_attempt/visual_execution.json', original72_selection_and_camera_math_unchanged=True, no_GPU1_execution_relabelled_as_GPU0=True)
    write(H / 'CAMERA_AMENDMENT_EXECUTION_BINDING.json', binding)
    write(H / 'RECOVERY_EXECUTION.json', dict(status='PASS_COMBINED_RECOVERY_WITH_RETAINED_FAILURES', completed_registered_numerical_conditions=9, completed_camera_conditions=2, initial_completed_windows=448, initial_invalid_unstarted_windows=128, recovered_numerical_jobs=numeric['recovery_jobs'], final_camera_outer_actual_exit_code=0, earlier_combined_recovery_outer_actual_exit_code=1, earlier_failure_receipt='recovery_closed_execution.json', original_camera_plan_restored=True, unknown_failed_camera_wait_not_guessed=True, no_completed_numerical_trajectory_restarted_or_discarded=True, post_outcome_operational_recovery=True, policy_math_actor_thresholds_exemptions_FIFO_unchanged=True, utc=datetime.now(timezone.utc).isoformat()))
    write(H / 'RECOVERY_TERMINAL_CLOSURE.json', dict(status='PASS_ACTUAL_COMPOSED_RECOVERY_CLOSURE_WITH_RETAINED_FAILURE_HISTORY', final_camera_actual_outer_exit_code=0, all_success_child_waits_persisted_before_acceptance=True, retained_original_terminal_failure='camera_gpu1_failed_attempt/RECOVERY_TERMINAL_CLOSURE.json', unpersisted_failed_GPU1_child_wait='UNRECORDED', camera_binding_sha256=sha(H / 'CAMERA_AMENDMENT_EXECUTION_BINDING.json'), utc=datetime.now(timezone.utc).isoformat()))
    write(H / 'CAMERA_SUPERVISOR_EXIT.json', dict(pid=camera_closed['child_pid'], exit_code=0, source='original_camera_closed_execution.json', original_supervisor_failure='initial_attempts/CAMERA_SUPERVISOR_EXIT.json', intervening_recovery_failure='recovery_closed_execution.json'))
    print('COMPOSED_RECOVERY_CLOSED_WITH_ALL_FAILURES_RETAINED', flush=True)


if __name__ == '__main__':
    main()
