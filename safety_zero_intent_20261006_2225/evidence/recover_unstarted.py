"""Registered recovery of initialization failures; keep all completed trajectories."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import copy
import hashlib
import json
import os
import subprocess
import sys

H = Path(__file__).resolve().parent
RAW = Path('/mnt/nas/data/lyf/double_hand') / H.name
sys.dont_write_bytecode = True
sys.path.insert(0, str(H.parent / 'safety_random_space_20261004'))
from isolated_campaign_v2 import verify_child


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    temp = path.with_suffix('.recovery.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def frozen(plan):
    for relative, digest in plan['source_sha256'].items():
        assert sha(Path(plan['cwd']) / relative) == digest, relative
    for path, digest in plan['research_source_sha256'].items():
        assert sha(path) == digest, path
    assert sha(plan['checkpoint_path']) == plan['checkpoint_sha256']


def capacity(tag):
    child = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    rows = {int(a): int(b) for a, b in (line.split(',') for line in child.stdout.splitlines())}
    value = dict(status='PASS_ACTUAL_GPU1_25GiB_PRELAUNCH' if rows[1] >= 25600 else 'FAIL_ACTUAL_GPU1_PRELAUNCH', gpu_free_MiB=rows, participation='recovery numerical and camera CUDA1; camera graphics backend may enumerate CUDA0 separately', foreign_processes_untouched=True, utc=datetime.now(timezone.utc).isoformat())
    write(H / (tag + '_capacity.json'), value)
    assert rows[1] >= 25600, value


def numeric(block):
    plan = read(H / 'plans' / f'holdout_{block}_plan.json')
    frozen(plan)
    root = Path(plan['output_root'])
    campaign = read(root / 'campaign.json')
    job = plan['jobs'][2]
    old = copy.deepcopy(campaign['jobs'][2])
    assert old['id'] == job['id'] and old['status'] == 'failed'
    out = root / job['id']
    protocol = read(out / 'protocol.json')
    assert protocol['completed_cells'] == 0 and list(out.iterdir()) == [out / 'protocol.json']
    saved = RAW / 'initialization_failures' / f'block_{block}'
    saved.mkdir(parents=True, exist_ok=False)
    (saved / 'campaign_original.json').write_bytes((root / 'campaign.json').read_bytes())
    out.rename(saved / job['id'])
    (root / (job['id'] + '.log')).rename(saved / (job['id'] + '.log'))
    argv = job['argv'] + ['--out', str(out)]
    row = dict(id=job['id'], status='running', argv=argv, retained_initial_attempt=old, initialization_only_retry=True)
    campaign['jobs'][2] = row
    campaign['status'] = 'recovery_running'
    write(root / 'campaign.json', campaign)
    with (root / (job['id'] + '.log')).open('x') as log:
        child = subprocess.Popen(argv, cwd=plan['cwd'], env={**os.environ, **plan.get('env', {}), **job['env'], 'PYTHONDONTWRITEBYTECODE': '1'}, stdout=log, stderr=subprocess.STDOUT)
        row['pid'] = child.pid
        write(root / 'campaign.json', campaign)
        row['exit_code'] = child.wait()
    assert row['exit_code'] == 0
    row.update(verify_child(out, job['expected']))
    protocol = read(out / 'protocol.json')
    assert protocol['source_sha256'] == plan['source_sha256'] and protocol['checkpoint_sha256'] == plan['checkpoint_sha256']
    for key, value in job['expected_args'].items():
        assert protocol['args'][key] == value
    assert (out / 'cell_001.npz').is_file() and (out / 'project_diagnostics.npz').is_file()
    frozen(plan)
    row['status'] = 'complete'
    campaign.update(status='complete', recovery_finished_utc=datetime.now(timezone.utc).isoformat(), original_initialization_failure_retained=True)
    assert all(j['status'] == 'complete' and j['exit_code'] == 0 for j in campaign['jobs'])
    write(root / 'campaign.json', campaign)
    print('RECOVERED_NUMERIC', block, flush=True)
    return row


def camera():
    original = read(H / 'visual_plan.json')
    plan = read(H / 'visual_recovery_plan.json')
    assert plan['actual_visual_root'] == original['actual_visual_root']
    rows = []
    receipt = dict(status='running', jobs=rows, original_attempt_retained='initial_attempts/visual_execution.json', amended_execution='PLANS_RECOVERY_AMENDMENT.json')
    write(H / 'visual_execution.json', receipt)
    for job in plan['jobs']:
        for path, digest in plan['sources'].items():
            assert sha(path) == digest, path
        for relative, digest in plan['original_source_sha256'].items():
            assert sha(Path(plan['cwd']) / relative) == digest, relative
        out = Path(job['argv'][job['argv'].index('--out') + 1])
        assert not out.exists()
        row = dict(mode=job['mode'], status='running', argv=job['argv'], out=str(out))
        rows.append(row)
        with (H / ('visual_recovery_' + job['mode'] + '.log')).open('x') as log:
            child = subprocess.Popen(job['argv'], cwd=plan['cwd'], env={**os.environ, **job['env'], 'PYTHONDONTWRITEBYTECODE': '1'}, stdout=log, stderr=subprocess.STDOUT)
            row['pid'] = child.pid
            write(H / 'visual_execution.json', receipt)
            row['exit_code'] = child.wait()
        protocol = read(out / 'visual_protocol.json')
        assert row['exit_code'] == 0 and protocol['status'] == 'complete' and protocol['steps'] == 960 and protocol['completed_windows'] == 64
        assert sha(job['argv'][job['argv'].index('--ckpt') + 1]) == protocol['actor_sha256'] == plan['actor_sha256']
        for path, digest in plan['sources'].items():
            assert sha(path) == digest, path
        for relative, digest in plan['original_source_sha256'].items():
            assert sha(Path(plan['cwd']) / relative) == digest, relative
        row['status'] = 'complete'
        write(H / 'visual_execution.json', receipt)
        print('RECOVERED_CAMERA', job['mode'], flush=True)
    receipt.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat())
    write(H / 'visual_execution.json', receipt)


def main():
    amendment = read(H / 'PLANS_RECOVERY_AMENDMENT.json')
    assert sha(__file__) == amendment['recovery_driver_sha256']
    assert not (H / 'RECOVERY_EXECUTION.json').exists()
    capacity('recovery_numeric_initial')
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(numeric, [1, 2]))
    original = read(H / 'NUMERIC_EXECUTION.json')
    write(H / 'NUMERIC_EXECUTION.json', dict(status='PASS_ALL_NUMERIC_CLOSED', original_exit_only_summary='initial_attempts/NUMERIC_EXECUTION.json', completed_registered_conditions=9, complete_method_windows_expected=576, recovery_jobs=rows, original_summary=original, attempt_validity_not_erased=True))
    capacity('recovery_camera_initial')
    camera()
    write(H / 'CAMERA_SUPERVISOR_EXIT.json', dict(pid=os.getpid(), exit_code=0, original_supervisor_failure_retained='initial_attempts/CAMERA_SUPERVISOR_EXIT.json', recovery=True))
    write(H / 'RECOVERY_EXECUTION.json', dict(status='PASS_REGISTERED_UNSTARTED_RECOVERY_CLOSED', numerical_jobs=rows, camera_execution_sha256=sha(H / 'visual_execution.json'), all_original_initialization_attempts_retained=True, scientific_policy_source_unchanged=True, initial_448_complete_128_invalid_still_reported=True, initial_resource_gate_still_failed=True, post_outcome_operational_amendment=True, utc=datetime.now(timezone.utc).isoformat()))
    print('RECOVERY_ALL_CLOSED', flush=True)


if __name__ == '__main__':
    main()
