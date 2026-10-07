"""Finite outer-exit observer; unknown unpersisted child waits remain unknown."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import time

H = Path(__file__).resolve().parent
RAW = Path('/mnt/nas/data/lyf/double_hand') / H.name


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = path.with_suffix('.termination.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def failed_record(original, outer_code):
    value = copy.deepcopy(original)
    for row in value.get('jobs', []):
        if row.get('status') in ['running', 'recovery_running']:
            row.update(status='failed', validation_stage='child_exit_or_artifact_validation', observed_child_wait_code=row.get('exit_code', 'UNRECORDED'), unpersisted_wait_not_guessed=True)
    value.update(status='failed', outer_driver_actual_exit_code=outer_code, error='Recovery driver closed unsuccessfully; see retained original metadata and recovery_child.log', terminal_status_observed=True)
    return value


def camera_binding(original, amended, execution, amendment):
    assert len(original['jobs']) == len(amended['jobs']) == len(execution['jobs']) == 2
    reconstructed = copy.deepcopy(original)
    deltas = []
    for before, after, actual, expected in zip(original['jobs'], amended['jobs'], execution['jobs'], reconstructed['jobs']):
        position = before['argv'].index('--device') + 1
        assert before['argv'][position] == 'cuda:0' and after['argv'][position] == 'cuda:1'
        expected['argv'][position] = 'cuda:1'
        assert actual['argv'] == after['argv'] and actual['mode'] == after['mode']
        assert actual['status'] == 'complete' and actual['exit_code'] == 0
        deltas.append(dict(mode=actual['mode'], argument_index=position, before='cuda:0', after='cuda:1'))
    assert reconstructed == amended, 'Any change beyond the device-only argv delta rejects binding'
    assert execution['status'] == 'complete'
    return dict(status='PASS_ACTUAL_CAMERA_DEVICE_AMENDMENT_BINDING', device_only_deltas=deltas, original_plan_sha256=sha(H / 'visual_plan.json'), executed_recovery_plan_sha256=sha(H / 'visual_recovery_plan.json'), amendment_sha256=sha(H / 'PLANS_RECOVERY_AMENDMENT.json'), actual_camera_execution_sha256=sha(H / 'visual_execution.json'), original_receipt_not_modified=True, original_plan_not_relabelled_as_executed=True, original_selection_and_math_unchanged=True, registered_recovery_plan_sha256=amendment['visual_recovery_plan_sha256'])


def main():
    deadline = time.monotonic() + 7200
    target = H / 'recovery_closed_execution.json'
    while not target.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError('No actual recovery outer wait receipt within the finite bound')
        time.sleep(5)
    outer = read(target)
    assert outer['child_reaped'] and isinstance(outer['actual_exit_code'], int)
    code = outer['actual_exit_code']
    if code:
        preserved = H / 'recovery_terminal_failure_originals'
        preserved.mkdir(exist_ok=False)
        values = [H / 'visual_execution.json'] + [RAW / f'holdout_{b}' / 'campaign.json' for b in range(3)]
        for index, path in enumerate(values):
            if not path.exists():
                continue
            value = read(path)
            if value['status'] not in ['complete', 'complete_with_failures', 'failed', 'FAIL']:
                (preserved / (str(index) + '_' + path.name)).write_bytes(path.read_bytes())
                write(path, failed_record(value, code))
        value = dict(status='FAIL_RECOVERY_TERMINALIZED_WITHOUT_PASS', actual_outer_exit_code=code, known_wait_values_retained=True, missing_wait_values='UNRECORDED; not inferred from protocol or Kit logs', finite_follower_failure_signal=True)
        write(H / 'RECOVERY_TERMINAL_CLOSURE.json', value)
        raise RuntimeError(value['status'])
    assert read(H / 'RECOVERY_EXECUTION.json')['status'] == 'PASS_REGISTERED_UNSTARTED_RECOVERY_CLOSED'
    for block in range(3):
        campaign = read(RAW / f'holdout_{block}' / 'campaign.json')
        assert campaign['status'] == 'complete' and len(campaign['jobs']) == 3
        assert all(j['status'] == 'complete' and j['exit_code'] == 0 for j in campaign['jobs'])
    original = read(H / 'visual_plan.json')
    amended = read(H / 'visual_recovery_plan.json')
    amendment = read(H / 'PLANS_RECOVERY_AMENDMENT.json')
    assert sha(H / 'visual_recovery_plan.json') == amendment['visual_recovery_plan_sha256']
    binding = camera_binding(original, amended, read(H / 'visual_execution.json'), amendment)
    binding['utc'] = datetime.now(timezone.utc).isoformat()
    write(H / 'CAMERA_AMENDMENT_EXECUTION_BINDING.json', binding)
    write(H / 'RECOVERY_TERMINAL_CLOSURE.json', dict(status='PASS_ACTUAL_RECOVERY_OUTER_AND_ALL_PERSISTED_CHILD_CLOSURES', actual_outer_exit_code=code, complete_numerical_conditions=9, complete_camera_conditions=2, no_unrecorded_success_child_waits=True, original_failure_observation_limit_retained=True, camera_binding_sha256=sha(H / 'CAMERA_AMENDMENT_EXECUTION_BINDING.json'), utc=datetime.now(timezone.utc).isoformat()))
    print('RECOVERY_TERMINAL_AND_CAMERA_BINDING_CLOSED', flush=True)


if __name__ == '__main__':
    main()
