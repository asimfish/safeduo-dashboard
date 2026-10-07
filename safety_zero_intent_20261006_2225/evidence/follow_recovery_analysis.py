"""Fresh original audits after artifact-verified recovered numerical closure."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
import time

H = Path(__file__).resolve().parent


def wait_recovered():
    deadline = time.monotonic() + 7200
    while time.monotonic() < deadline:
        file = H / 'NUMERIC_EXECUTION.json'
        if file.exists():
            data = json.loads(file.read_text())
            if len(data.get('recovery_jobs', [])) == 2:
                assert data['status'] == 'PASS_ALL_NUMERIC_CLOSED'
                for block in range(3):
                    plan = json.loads((H / 'plans' / f'holdout_{block}_plan.json').read_text())
                    campaign = json.loads((Path(plan['output_root']) / 'campaign.json').read_text())
                    assert campaign['status'] == 'complete' and len(campaign['jobs']) == 3
                    for job in campaign['jobs']:
                        assert job['status'] == 'complete' and job['exit_code'] == 0
                return
        failure = H / 'recovery_closed_execution.json'
        if failure.exists():
            assert json.loads(failure.read_text())['actual_exit_code'] == 0, 'Recovery failed; no analysis PASS fabricated'
        time.sleep(5)
    raise TimeoutError('Recovered numerical closure not available')


def run(task):
    label, argv = task
    child = subprocess.Popen(['python3', str(H / 'run_closed_step.py'), label, *argv], cwd=H, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    code = child.wait()
    assert code == 0, (label, code)


if __name__ == '__main__':
    wait_recovered()
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, [
            ('recovered_analysis', ['python3', str(H / 'follow_analysis.py')]),
            ('zero_intent_audit', ['/home/liyufeng/miniforge3/envs/safeduo/bin/python', str(H / 'audit_zero_intent.py')])
        ]))
    print('ALL_RECOVERED_ANALYSIS_CLOSED', flush=True)
