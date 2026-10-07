"""Capture a finite child execution without interpreting its scientific verdict."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('label')
    parser.add_argument('argv', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert re.fullmatch('[a-z][a-z0-9_]*', args.label)
    assert args.argv and not (HERE / 'SEALED.json').exists()
    receipt = HERE / (args.label + '_closed_execution.json')
    launch = HERE / (args.label + '_child_launch.json')
    log = HERE / (args.label + '_child.log')
    assert not any(p.exists() for p in [receipt, launch, log])
    source = Path(args.argv[1]) if len(args.argv) > 1 else None
    before = hashlib.sha256(source.read_bytes()).hexdigest() if source and source.is_file() else None
    with log.open('x') as output:
        child = subprocess.Popen(args.argv, cwd=HERE, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, stdout=output, stderr=subprocess.STDOUT)
        write(launch, dict(actual_argv=args.argv, child_pid=child.pid, source_sha256=before, utc=datetime.now(timezone.utc).isoformat()))
        code = child.wait()
    after = hashlib.sha256(source.read_bytes()).hexdigest() if before else None
    value = dict(status='PASS_ACTUAL_CHILD_CLOSED' if code == 0 and before == after else 'FAIL_ACTUAL_CHILD_EXECUTION', actual_argv=args.argv, child_pid=child.pid, child_reaped=True, actual_exit_code=code, source_sha256_before=before, source_sha256_after=after, scientific_verdict_from_child_not_overridden=True, utc=datetime.now(timezone.utc).isoformat())
    write(receipt, value)
    print(json.dumps(value, ensure_ascii=False), flush=True)
    return code if code else (0 if before == after else 1)


if __name__ == '__main__':
    sys.exit(main())
