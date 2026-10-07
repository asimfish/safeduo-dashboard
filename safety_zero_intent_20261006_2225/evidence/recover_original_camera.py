"""Use the original CUDA0 camera plan; persist waits before validating products."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess

H = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = path.with_suffix('.camera.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def verify_sources(plan):
    for path, digest in plan['sources'].items():
        assert sha(path) == digest, path
    for relative, digest in plan['original_source_sha256'].items():
        assert sha(Path(plan['cwd']) / relative) == digest, relative


def actual_capacity(mode):
    output = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout
    free = {int(a): int(b) for a, b in (line.split(',') for line in output.splitlines())}
    value = dict(status='PASS_ACTUAL_GPU0_PRELAUNCH' if free[0] >= 25600 else 'FAIL_GPU0_CAPACITY', gpu_free_MiB=free, minimum_required_MiB=25600, foreign_processes_untouched=True, utc=datetime.now(timezone.utc).isoformat())
    write(H / ('original_camera_' + mode + '_capacity.json'), value)
    assert free[0] >= 25600


def main():
    registration = read(H / 'CAMERA_ORIGINAL_RECOVERY_REGISTRATION.json')
    assert sha(__file__) == registration['driver_sha256']
    plan_digest = sha(H / 'visual_plan.json')
    assert plan_digest == registration['original_plan_sha256']
    plan = read(H / 'visual_plan.json')
    receipt = dict(status='running', plan_sha256=plan_digest, registered_original_plan_executed_exactly=True, camera_recovery_registration_sha256=sha(H / 'CAMERA_ORIGINAL_RECOVERY_REGISTRATION.json'), superseded_failed_GPU1_attempt='camera_gpu1_failed_attempt/visual_execution.json', jobs=[], started_utc=datetime.now(timezone.utc).isoformat())
    target = H / 'visual_execution.json'
    write(target, receipt)
    try:
        for job in plan['jobs']:
            row = dict(mode=job['mode'], status='running', argv=job['argv'], out=job['argv'][job['argv'].index('--out') + 1], stage='prelaunch')
            receipt['jobs'].append(row)
            write(target, receipt)
            verify_sources(plan)
            assert job['argv'][job['argv'].index('--device') + 1] == 'cuda:0'
            checkpoint = job['argv'][job['argv'].index('--ckpt') + 1]
            assert sha(checkpoint) == plan['actor_sha256']
            out = Path(row['out'])
            assert not out.exists()
            actual_capacity(job['mode'])
            with (H / ('visual_original_' + job['mode'] + '.log')).open('x') as log:
                child = subprocess.Popen(job['argv'], cwd=plan['cwd'], env={**os.environ, **job['env'], 'PYTHONDONTWRITEBYTECODE': '1'}, stdout=log, stderr=subprocess.STDOUT)
                row.update(pid=child.pid, process_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21], stage='child_running')
                write(target, receipt)
                row.update(exit_code=child.wait(), child_reaped=True, stage='closed_product_validation')
                write(target, receipt)
            assert row['exit_code'] == 0, 'Nonzero actual camera child exit'
            protocol = read(out / 'visual_protocol.json')
            assert protocol['status'] == 'complete' and protocol['steps'] == 960 and protocol['completed_windows'] == 64
            assert protocol['actor_sha256'] == plan['actor_sha256'] == sha(checkpoint)
            assert (out / 'cell_001.npz').is_file() and (out / 'camera_receipts.json').is_file()
            captures = read(out / 'camera_receipts.json')
            assert captures['scheduled_groups'] == 21 and captures['PNG'] == 9 * captures['groups']
            verify_sources(plan)
            row.update(status='complete', stage='complete', visual_protocol_sha256=sha(out / 'visual_protocol.json'))
            write(target, receipt)
            print('ORIGINAL_CAMERA_CLOSED', job['mode'], captures['groups'], captures['PNG'], flush=True)
        receipt['status'] = 'complete'
    except BaseException as error:
        receipt.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        if receipt['jobs'] and receipt['jobs'][-1]['status'] == 'running':
            receipt['jobs'][-1].update(status='failed', failed_validation_stage=receipt['jobs'][-1]['stage'])
        raise
    finally:
        receipt['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write(target, receipt)


if __name__ == '__main__':
    main()
