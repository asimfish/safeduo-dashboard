"""Persist the real process exit before interpreting products."""
from pathlib import Path
import datetime, hashlib, json, subprocess, sys
HERE=Path(__file__).resolve().parent

if __name__=='__main__':
    tag=sys.argv[1]; argv=sys.argv[2:]
    if not tag.replace('_','').isalnum() or not argv: raise ValueError('tag/argv')
    start=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (HERE/(tag+'.log')).open('xb') as log:
        child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,cwd=HERE)
        rc=child.wait()
    receipt=dict(tag=tag,argv=argv,pid=child.pid,actual_exit=rc,started_utc=start,
                 closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 log_sha256=hashlib.sha256((HERE/(tag+'.log')).read_bytes()).hexdigest())
    with (HERE/(tag+'_execution.json')).open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps(receipt),flush=True)
    sys.exit(rc)
