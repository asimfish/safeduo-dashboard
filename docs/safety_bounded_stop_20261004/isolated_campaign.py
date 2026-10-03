"""Sequential fresh-process evaluation; preserve failed and incomplete evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def verify_child(out, expected):
    p = json.loads((out / 'protocol.json').read_text())
    if p['status'] != 'complete' or p['completed_cells'] != 1 or len(p['design']) != 1:
        raise ValueError('each fresh process must complete exactly one cell')
    for key, value in expected.items():
        if p['design'][0][key] != value:
            raise ValueError(f'child condition differs: {key}')
    summary = json.loads((out / 'summary.json').read_text())
    if len(summary['windows']) != 1:
        raise ValueError('summary must contain one window')
    return dict(protocol_sha256=hashlib.sha256((out / 'protocol.json').read_bytes()).hexdigest(),
                condition=p['design'][0], completed_cells=1)


def run_campaign(plan, out):
    out = Path(out)
    ids = [job['id'] for job in plan['jobs']]
    if len(set(ids)) != len(ids) or any(not i or Path(i).name != i for i in ids):
        raise ValueError('unique directory-safe job ids required')
    if not ids:
        raise ValueError('at least one job required')
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'campaign.json'
    if manifest_path.exists() or any((out / i).exists() for i in ids):
        raise ValueError('existing campaign or cell output must not be overwritten')
    manifest = dict(schema='safeduo.isolated_campaign.v1', status='running',
                    started_utc=datetime.now(timezone.utc).isoformat(), plan=plan,
                    isolation='one fresh OS process per single-cell condition; sequential GPU use',
                    statistics='matched/correlated conditions; no IID aggregate confidence claim', jobs=[])
    atomic_json(manifest_path, manifest)
    try:
        for job in plan['jobs']:
            for name, sha in plan.get('source_sha256', {}).items():
                if hashlib.sha256((Path(plan['cwd']) / name).read_bytes()).hexdigest() != sha:
                    raise ValueError(f'frozen source changed before child launch: {name}')
            cell_out = out / job['id']
            argv = job['argv'] + ['--out', str(cell_out)]
            if '--out' in job['argv']:
                raise ValueError('output is owned by campaign driver')
            record = dict(id=job['id'], status='running', argv=argv)
            manifest['jobs'].append(record)
            print('START', job['id'], flush=True)
            with (out / f'{job["id"]}.log').open('x') as log:
                child = subprocess.Popen(argv, cwd=plan['cwd'],
                                         env={**os.environ, **plan.get('env', {}), **job.get('env', {})},
                                         stdout=log, stderr=subprocess.STDOUT)
                record['pid'] = child.pid
                atomic_json(manifest_path, manifest)
                try:
                    record['exit_code'] = child.wait()
                except BaseException:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    raise
            if record['exit_code']:
                raise RuntimeError(f'{job["id"]} exited {record["exit_code"]}; log and protocol retained')
            record.update(verify_child(cell_out, job['expected']))
            protocol = json.loads((cell_out / 'protocol.json').read_text())
            if 'source_sha256' in plan and protocol['source_sha256'] != plan['source_sha256']:
                raise ValueError('child source snapshot differs from registered campaign')
            if 'checkpoint_sha256' in plan and protocol['checkpoint_sha256'] != plan['checkpoint_sha256']:
                raise ValueError('child actor differs from registered campaign')
            for key, value in job.get('expected_args', {}).items():
                if protocol['args'][key] != value:
                    raise ValueError(f'child argument differs: {key}')
            record['status'] = 'complete'
            atomic_json(manifest_path, manifest)
            print('DONE', job['id'], flush=True)
        manifest['status'] = 'complete'
    except BaseException as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        if manifest['jobs'] and manifest['jobs'][-1]['status'] == 'running':
            manifest['jobs'][-1]['status'] = 'failed'
        raise
    finally:
        manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
        atomic_json(manifest_path, manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run_campaign(json.loads(args.plan.read_text()), args.out)
