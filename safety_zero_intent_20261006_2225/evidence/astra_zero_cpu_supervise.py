"""Bounded fresh CPU test child, immutable attempts, actual exit receipts."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

H=Path(__file__).resolve().parent
NAMES=('reference_envelope.py','zero_intent.py','guard_runner.py',
       'target_forecast.py','projection_diagnostics.py','tracking_reserve.py')

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write_once(path,data):
    assert path.parent==H and path.name.startswith('astra_zero_')
    with path.open('x') as f: json.dump(data,f,indent=2,allow_nan=False);f.write('\n')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('suite',choices=['raw','contract'])
    args=parser.parse_args()
    attempt=1
    while (H/f'astra_zero_{args.suite}_attempt{attempt:02d}.log').exists(): attempt+=1
    stem=f'astra_zero_{args.suite}_attempt{attempt:02d}'
    owned={p.name:sha(p) for p in H.glob('astra_zero_*.py')}
    before={n:sha(H/n) for n in NAMES}
    if args.suite=='contract':
        write_once(H/(stem+'_source_snapshot.json'),dict(utc=now(),sources={
            str(H/n):dict(sha256=before[n],text=(H/n).read_text()) for n in NAMES}))
    argv=[sys.executable,'-B','-u',str(H/f'astra_zero_{args.suite}_tests.py')]
    if args.suite=='contract': argv+=['--mutations']
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':'',
         'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
    started=now();clock=time.monotonic();timed_out=False
    with (H/(stem+'.log')).open('x') as log:
        child=subprocess.Popen(argv,cwd=H,env=env,stdout=log,stderr=subprocess.STDOUT)
        try: rc=child.wait(timeout=180)
        except subprocess.TimeoutExpired:
            timed_out=True;child.kill();rc=child.wait()
    after={n:sha(H/n) for n in NAMES}
    stable=before==after
    receipt=dict(schema='astra.zero.cpu_attempt.v1',suite=args.suite,attempt=attempt,
        started_utc=started,ended_utc=now(),seconds=time.monotonic()-clock,actual_argv=argv,
        supervisor_pid=os.getpid(),child_pid=child.pid,actual_child_exit_code=rc,
        timed_out=timed_out,source_before=before,source_after=after,source_stable=stable,
        own_source_sha256=owned,log=str(H/(stem+'.log')),log_sha256=sha(H/(stem+'.log')),
        child_reaped=True,automatic_retries=0,gpu_launched=False,simulation_launched=False,
        parent_scores_loaded=False,new_policy_outcomes_loaded=False)
    write_once(H/(stem+'_receipt.json'),receipt)
    print(json.dumps(dict(receipt=str(H/(stem+'_receipt.json')),actual_child_exit_code=rc,
                         source_stable=stable,child_reaped=True)),flush=True)
    return (rc if rc>=0 else 128-rc) if stable else 3

if __name__=='__main__': raise SystemExit(main())
