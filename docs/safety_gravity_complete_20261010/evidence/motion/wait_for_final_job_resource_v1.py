"""Bounded scheduling wait; launch the last job once, without relaxing any gate."""
import datetime
import json
from pathlib import Path
import subprocess
import time

P = Path(__file__).resolve().parent
RECEIPT = P / 'RESOURCE_WAIT_EXECUTION_V1.json'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(data):
    tmp = RECEIPT.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(RECEIPT)


def main():
    assert not RECEIPT.exists()
    assert not (P / 'FINISH_EXECUTION_V6.json').exists()
    assert not (P.parent / 'queue_response128_v7_execution.json').exists()
    data = dict(status='waiting_resource', started_utc=now(), max_wait_seconds=900,
                gpu_index=1, original_required_initial_free_MiB=20480,
                scheduling_consecutive_samples=2, native_launches=0, samples=[])
    save(data)
    deadline = time.monotonic() + 900
    consecutive = 0
    try:
        while True:
            assert time.monotonic() < deadline, 'bounded resource scheduling deadline exceeded; no new child'
            probe = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                                   capture_output=True, text=True, timeout=10)
            free = None
            if probe.returncode == 0:
                rows = dict((int(a.strip()), int(b.strip())) for a, b in
                            (line.split(',') for line in probe.stdout.splitlines()))
                free = rows[1]
            data['samples'].append(dict(utc=now(), actual_exit=probe.returncode, free_MiB=free))
            consecutive = consecutive + 1 if free is not None and free >= 20480 else 0
            save(data)
            if consecutive >= 2:
                break
            time.sleep(20)
        data.update(status='running_finite_completion', launch_utc=now(), native_launches=1)
        save(data)
        with (P / 'resource_wait_v1_finish.log').open('xb') as log:
            child = subprocess.Popen(['/usr/bin/python3', '-B', str(P / 'finish_completed_jobs_v6.py')],
                                     cwd=P, stdout=log, stderr=subprocess.STDOUT)
            data['finite_child_pid'] = child.pid
            save(data)
            actual = child.wait()
        data.update(actual_exit=actual, status='complete' if actual == 0 else 'failed')
        save(data)
        assert actual == 0, 'finite completion failed; immutable attempt preserved'
        print('ACTUAL_WAIT_FINAL_JOB_COMPLETION', actual, flush=True)
    except BaseException as error:
        data.update(status='failed', error=repr(error))
        raise
    finally:
        data['closed_utc'] = now()
        save(data)


if __name__ == '__main__':
    main()
