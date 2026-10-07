"""Request the already authorized Pages build for one exact main commit."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import subprocess
import urllib.request

H = Path(__file__).resolve().parent
API = 'https://api.github.com/repos/asimfish/safeduo-dashboard/'


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError('Unexpected GitHub API redirect')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit', required=True)
    parser.add_argument('--tag', choices=['initial', 'final'], required=True)
    args = parser.parse_args()
    assert len(args.commit) == 40 and all(c in '0123456789abcdef' for c in args.commit)
    assert not (H / 'SEALED.json').exists()
    output = H / ('MANUAL_PAGES_BUILD_' + args.tag + '.json')
    assert not output.exists()
    credential = subprocess.run(['git', 'credential', 'fill'], input='protocol=https\nhost=github.com\n\n', capture_output=True, text=True)
    if credential.returncode:
        raise RuntimeError('Existing GitHub credential unavailable')
    values = dict(line.split('=', 1) for line in credential.stdout.splitlines() if '=' in line)
    token = values.get('password')
    if not token:
        raise RuntimeError('Existing GitHub credential has no password')
    opener = urllib.request.build_opener(NoRedirect())

    def request(relative, method='GET'):
        req = urllib.request.Request(API + relative, method=method, data=b'{}' if method == 'POST' else None, headers={'User-Agent': 'SafeDuo-authorized-Pages-build', 'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28', 'Content-Type': 'application/json'})
        with opener.open(req, timeout=30) as response:
            return response.status, json.load(response)

    _, branch = request('branches/main')
    assert branch['commit']['sha'] == args.commit and not branch['protected']
    _, page = request('pages')
    assert page['build_type'] == 'legacy' and page['source'] == {'branch': 'main', 'path': '/'}
    _, runs = request('actions/runs?head_sha=' + args.commit + '&per_page=30')
    exact = [r for r in runs['workflow_runs'] if 'pages' in r['name'].lower() and r['head_sha'] == args.commit]
    if exact:
        value = dict(status='OBSERVED_EXACT_PAGES_RUN_NO_MANUAL_POST', commit=args.commit, actual_run_ids=[r['id'] for r in exact], manual_post=False)
    else:
        code, result = request('pages/builds', 'POST')
        assert code == 201 and result['status'] in ['queued', 'building', 'built']
        value = dict(status='REQUESTED_EXISTING_LEGACY_PAGES_BUILD_NOT_DEPLOYMENT_PROOF', commit=args.commit, request_HTTP_status=code, build_status=result['status'], manual_post=True, source={'branch': 'main', 'path': '/'}, automatic_trigger_absence_cause='UNDETERMINED', no_configuration_or_access_control_changes=True)
    value['utc'] = datetime.now(timezone.utc).isoformat()
    with output.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    main()
