"""Single-use CPU audit supervisor; preserve actual child exit, never retry."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

H=Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_once(name, value):
    assert Path(name).name==name and name.startswith('astra_')
    with (H/name).open('x') as file:
        json.dump(value,file,indent=2,allow_nan=False)
        file.write('\n')

def main():
    registered=H/'astra_tracking_raw_plan.json'
    plan=json.loads(registered.read_text())
    assert plan['workers']==2
    assert plan['source_sha256'][Path(__file__).name]==sha(__file__)
    for name,digest in plan['source_sha256'].items():
        assert sha(H/name)==digest, name
    assert not (H/'astra_tracking_raw_process.json').exists(), 'previous attempt exists'
    assert not (H/'astra_tracking_raw_writer.lock').exists(), 'previous writer exists'
    argv=[sys.executable,'-u',str(H/'astra_tracking_raw_score.py'),'--watch']
    assert argv==plan['scorer_argv']
    child=subprocess.Popen(argv,cwd=H,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1',
        'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
        'NUMEXPR_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''})
    write_once('astra_tracking_raw_process.json',dict(status='CHILD_STARTED_NOT_COMPLETE',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),supervisor_pid=os.getpid(),
        child_pid=child.pid,actual_argv=argv,cwd=str(H),registration_sha256=sha(registered),
        source_sha256=plan['source_sha256'],numeric_gpu_launch=False,workers=2))
    rc=child.wait()
    score=H/'ASTRA_TRACKING_FINAL_SCORE.json'
    final=json.loads(score.read_text()) if score.exists() else None
    write_once('astra_tracking_raw_process_exit.json',dict(
        status='CLOSED_ACTUAL_CHILD_EXIT',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        supervisor_pid=os.getpid(),child_pid=child.pid,actual_child_exit_code=rc,
        final_score_exists=score.exists(),final_score_sha256=sha(score) if score.exists() else None,
        final_score_status=final['status'] if final else None,
        registration_sha256=sha(registered),automatic_retries=0,
        final_closure_action='Supervisor returns immediately after writing this receipt; inspect PID absence for observed writer closure.'))
    return rc if rc>=0 else 128-rc

if __name__=='__main__':
    raise SystemExit(main())
