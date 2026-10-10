"""Finite native jobs followed by actual CPU waits; preserve every failure."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

P = Path(__file__).resolve().parent
H = P.parent
CPU = '/home/liyufeng/miniforge3/envs/safeduo/bin/python'
RECEIPT = P / 'FINISH_EXECUTION_V1.json'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(value):
    tmp = RECEIPT.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(RECEIPT)


def main():
    assert not RECEIPT.exists(), 'immutable execution attempt already exists'
    reg = json.loads((P / 'REGISTRATION_V1.json').read_text())
    record = dict(status='running', started_utc=now(), steps=[])
    write(record)
    tasks = [('native', ['/usr/bin/python3', '-B', str(H / 'execute_physical_plan_v1.py'), str(H / 'GRAVITY_LONG128_PLAN_V1.json')])]
    tasks += [(f'audit_batch{batch}_{factor}', [CPU, '-B', str(P / 'analyse_gravity_v1.py'), str(batch), factor]) for batch in range(2) for factor in reg['gravity_modes']]
    tasks += [('aggregate', [CPU, '-B', str(P / 'analyse_gravity_v1.py')])]
    try:
        for label, argv in tasks:
            row = dict(label=label, argv=argv, status='running', started_utc=now())
            record['steps'].append(row)
            write(record)
            logpath = P / (label + '.log')
            with logpath.open('xb') as log:
                child = subprocess.Popen(argv, cwd=P, stdout=log, stderr=subprocess.STDOUT)
                actual = child.wait()
            row.update(actual_exit=actual, closed_utc=now(), status='complete' if actual == 0 else 'failed',
                       log_sha256=hashlib.sha256(logpath.read_bytes()).hexdigest())
            write(record)
            print('ACTUAL_WAIT', label, actual, flush=True)
            assert actual == 0, 'failed finite step ' + label
        result = P / 'GRAVITY_LONG128_RESULT_V1.json'
        record.update(status='complete', result_sha256=hashlib.sha256(result.read_bytes()).hexdigest())
    except BaseException as error:
        record.update(status='failed', error=repr(error))
        raise
    finally:
        record['closed_utc'] = now()
        write(record)


if __name__ == '__main__':
    main()
