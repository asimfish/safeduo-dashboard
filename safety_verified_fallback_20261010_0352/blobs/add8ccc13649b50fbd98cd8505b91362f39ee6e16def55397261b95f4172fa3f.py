"""Fork only our CPU job, perform real wait4, persist exit/RSS/log/source SHA."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

D = Path(__file__).resolve().parent
R = Path('/mnt/nas/data/lyf/double_hand/safety_verified_fallback_20261010_0352/native_calibration_cpu_tmp')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    label, script, *arguments = sys.argv[1:]
    if not label.replace('_', '').isalnum() or Path(script).name != script or not (D / script).is_file():
        raise ValueError('owned local script and new label required')
    R.mkdir(exist_ok=True)
    logfile = D / (label + '.log')
    receipt = D / (label + '_actual_wait.json')
    if receipt.exists():
        raise FileExistsError(receipt)
    command = [sys.executable, '-B', str(D / script), *arguments]
    environment = dict(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
                       CUDA_VISIBLE_DEVICES='', TMPDIR=str(R))
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    start = time.monotonic()
    source_before = sha(D / script)
    with logfile.open('xb') as log:
        pid = os.fork()
        if pid == 0:
            resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            os.environ.update(environment)
            os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
            os.chdir(D)
            os.execv(sys.executable, command)
        print(json.dumps(dict(owned_pid=pid, command=command, waiting=True)), flush=True)
        waited_pid, status, usage = os.wait4(pid, 0)
    exitcode = os.waitstatus_to_exitcode(status)
    result = dict(schema='native_calibration.real_wait4.v1', command=command, environment=environment,
        started_utc=started, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        elapsed_s=time.monotonic()-start, owned_pid=pid, waited_pid=waited_pid, raw_wait_status=status,
        actual_exit=exitcode, wait_returned=True, wait_mechanism='os.wait4(owned_pid, 0)',
        child_max_rss_KiB=usage.ru_maxrss, child_address_space_limit_bytes=1024**3,
        supervisor_max_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        user_cpu_s=usage.ru_utime, system_cpu_s=usage.ru_stime,
        log=str(logfile), log_sha256=sha(logfile), script_sha256_before=source_before,
        script_sha256_after=sha(D / script), supervisor_sha256=sha(__file__),
        gpu_started=False, other_processes_stopped=False)
    with receipt.open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result), flush=True)
    sys.exit(0 if exitcode == 0 else 1)


if __name__ == '__main__':
    main()
