"""Read-only check of the exact delivered Pages commit, suitable for bounded retries."""
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
API = 'https://api.github.com/repos/asimfish/safeduo-dashboard'


def main():
    delivery = json.loads((HERE / 'DELIVERY.json').read_text())
    sha = delivery['main_commit']
    response = requests.get(API + '/actions/runs', params={'branch': 'main', 'per_page': 30}, timeout=30)
    response.raise_for_status()
    candidates = [run for run in response.json()['workflow_runs']
                  if run['head_sha'] == sha and run['name'] == 'pages build and deployment']
    if not candidates:
        print(json.dumps({'status': 'pending', 'main_commit': sha, 'reason': 'exact Pages run not listed yet'}))
        return
    run = max(candidates, key=lambda item: (item['run_number'], item.get('run_attempt', 1)))
    if run['status'] != 'completed':
        print(json.dumps({'status': 'pending', 'main_commit': sha, 'workflow_status': run['status'],
                          'workflow_url': run['html_url']}))
        return
    evidence = {'status': 'pass' if run['conclusion'] == 'success' else 'failed',
                'checked_utc': datetime.now(timezone.utc).isoformat(),
                'main_commit': sha, 'data_commit': delivery['data_commit'],
                'workflow_id': run['id'], 'workflow_name': run['name'],
                'workflow_status': run['status'], 'conclusion': run['conclusion'],
                'workflow_url': run['html_url'], 'run_attempt': run.get('run_attempt', 1)}
    (HERE / 'REMOTE_CI.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence))
    assert evidence['status'] == 'pass', evidence


if __name__ == '__main__':
    main()
