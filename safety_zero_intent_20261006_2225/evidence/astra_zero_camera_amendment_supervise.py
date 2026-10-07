"""Finite CPU-only amendment fixture/check/audit children; no automatic retry."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

H = Path(__file__).resolve().parent
NAMES = ['astra_zero_camera_amendment.py', 'astra_zero_camera_amendment_tests.py',
         'astra_zero_camera_amendment_supervise.py']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['tests', 'check', 'audit-closed'])
    args = parser.parse_args()
    before = {name: sha(H / name) for name in NAMES}
    attempt = 1
    while (H / ('astra_zero_camera_amendment_' + args.action + '_attempt%02d.log' % attempt)).exists():
        attempt += 1
    stem = 'astra_zero_camera_amendment_' + args.action + '_attempt%02d' % attempt
    script = 'astra_zero_camera_amendment_tests.py' if args.action == 'tests' else 'astra_zero_camera_amendment.py'
    argv = [sys.executable, '-B', '-u', str(H / script)]
    if args.action == 'audit-closed':
        argv.append('--audit-closed')
    environment = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': '',
                   'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    started = datetime.now(timezone.utc).isoformat()
    timed_out = False
    with (H / (stem + '.log')).open('x') as log:
        child = subprocess.Popen(argv, cwd=H, env=environment, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = child.wait(timeout=60 if args.action != 'audit-closed' else 3600)
        except subprocess.TimeoutExpired:
            timed_out = True
            child.kill()  # Only the exact CPU child created by this supervisor.
            code = child.wait()
    after = {name: sha(H / name) for name in NAMES}
    log = H / (stem + '.log')
    matched = re.search(r'Ran (\d+) tests', log.read_text())
    receipt = dict(action=args.action, started_utc=started, closed_utc=datetime.now(timezone.utc).isoformat(),
        actual_argv=argv, child_pid=child.pid, supervisor_pid=os.getpid(), actual_child_exit_code=code,
        child_reaped=True, timed_out=timed_out, source_before=before, source_after=after,
        source_stable=before == after, tests=int(matched.group(1)) if matched else None,
        log_sha256=sha(log), automatic_retries=0, gpu_or_simulation_launched=False,
        actual_original_images_viewed=0)
    with (H / (stem + '_receipt.json')).open('x') as file:
        json.dump(receipt, file, indent=2)
        file.write('\n')
    print(json.dumps(dict(receipt=stem + '_receipt.json', actual_child_exit_code=code,
                         child_reaped=True, tests=receipt['tests'], source_stable=before == after)), flush=True)
    if before != after:
        return 3
    if code == 0 and args.action == 'audit-closed' and not (H / 'astra_zero_camera_metadata.json').exists():
        return 2
    return code if code >= 0 else 128 - code


if __name__ == '__main__':
    raise SystemExit(main())
