"""Single-use CPU camera test/audit child with real exit receipts; no watcher."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

H=Path(__file__).resolve().parent
NAMES=('astra_zero_camera_core.py','astra_zero_camera_prepare.py',
       'astra_zero_camera_tests.py','astra_zero_camera_supervise.py',
       'astra_zero_score_core.py','astra_zero_raw_score.py')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['tests','audit-closed'])
    args=parser.parse_args()
    before={n:sha(H/n) for n in NAMES}
    if args.action=='audit-closed':
        plan=json.loads((H/'astra_zero_camera_plan.json').read_text())
        for n,digest in plan['helper_source_sha256'].items():
            if sha(H/n)!=digest:raise ValueError('camera audit source changed: '+n)
    attempt=1
    while (H/f'astra_zero_camera_{args.action}_attempt{attempt:02d}.log').exists():attempt+=1
    stem=f'astra_zero_camera_{args.action}_attempt{attempt:02d}'
    script='astra_zero_camera_tests.py' if args.action=='tests' else 'astra_zero_camera_prepare.py'
    argv=[sys.executable,'-B','-u',str(H/script)]
    if args.action=='audit-closed':argv+=['--audit-closed']
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':'',
         'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
    started=now();timed_out=False
    with (H/(stem+'.log')).open('x') as log:
        child=subprocess.Popen(argv,cwd=H,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:rc=child.wait(timeout=120 if args.action=='tests' else 3600)
        except subprocess.TimeoutExpired:
            timed_out=True;child.kill();rc=child.wait()
    after={n:sha(H/n) for n in NAMES}
    log=(H/(stem+'.log')).read_text()
    match=re.search(r'Ran (\d+) tests',log)
    result=dict(schema='astra.zero.camera_cpu_exit.v1',action=args.action,attempt=attempt,
        started_utc=started,closed_utc=now(),actual_argv=argv,actual_child_exit_code=rc,
        supervisor_pid=os.getpid(),child_pid=child.pid,child_reaped=True,
        timed_out=timed_out,source_before=before,source_after=after,source_stable=before==after,
        test_count=int(match.group(1)) if match else None,
        log=str(H/(stem+'.log')),log_sha256=sha(H/(stem+'.log')),
        actual_images_viewed=0,automatic_retries=0,simulation_launched=False,gpu_launched=False)
    with (H/(stem+'_receipt.json')).open('x') as file:json.dump(result,file,indent=2);file.write('\n')
    print(json.dumps(dict(receipt=str(H/(stem+'_receipt.json')),actual_child_exit_code=rc,
        source_stable=before==after,tests=result['test_count'],child_reaped=True)),flush=True)
    if before!=after:return 3
    if rc==0 and args.action=='audit-closed' and not (H/'astra_zero_camera_metadata.json').exists():return 2
    return rc if rc>=0 else 128-rc

if __name__=='__main__':raise SystemExit(main())
