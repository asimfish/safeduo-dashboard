"""Finite native jobs followed by actual CPU waits; preserve every failure."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

P = Path(__file__).resolve().parent
H = P.parent
CPU = '/home/liyufeng/miniforge3/envs/safeduo/bin/python'
RECEIPT = P / 'FINISH_EXECUTION_V3.json'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(value):
    tmp = RECEIPT.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(RECEIPT)


def main():
    assert not RECEIPT.exists(), 'immutable execution attempt already exists'
    reg = json.loads((P / 'REGISTRATION_V1.json').read_text())
    long_finish = json.loads((H / 'gravity_long_completion_v3/FINISH_EXECUTION_V1.json').read_text())
    long_result = json.loads((H / 'gravity_long128_v1/GRAVITY_LONG128_RESULT_V1.json').read_text())
    assert long_finish['status'] == 'complete' and all(s['actual_exit'] == 0 for s in long_finish['steps'])
    assert long_result['long_hold_candidate_pass'], 'long holding failed: no new action probe launched'
    record = dict(status='running', started_utc=now(), steps=[])
    write(record)
    tasks = [('native', ['/usr/bin/python3', '-B', str(H / 'execute_physical_plan_v1.py'), str(H / 'QUEUE_RESPONSE128_PLAN_V4.json')])]
    tasks += [(f'audit_batch{batch}_{factor}', [CPU, '-B', str(P / 'analyse_completion_dependency_v3.py'), str(batch), factor]) for batch in range(2) for factor in reg['action_modes']]
    tasks += [('aggregate', [CPU, '-B', str(P / 'analyse_completion_dependency_v3.py')])]
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
        result = P / 'QUEUE_RESPONSE128_RESULT_V1.json'
        record.update(status='complete', result_sha256=hashlib.sha256(result.read_bytes()).hexdigest())
    except BaseException as error:
        record.update(status='failed', error=repr(error))
        raise
    finally:
        record['closed_utc'] = now()
        write(record)


if __name__ == '__main__':
    main()
