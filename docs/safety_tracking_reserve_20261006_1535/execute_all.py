"""Registered three independent blocks; camera follows closed CUDA0 numeric block."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib,json,os,subprocess,time

HERE=Path(__file__).resolve().parent


def write(p,x):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)


def run(block):
    argv=['python3',str(HERE/'execute_numeric.py'),str(block)]
    with (HERE/f'numeric_block_{block}.log').open('x') as f:
        child=subprocess.Popen(argv,cwd=HERE,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdout=f,stderr=subprocess.STDOUT)
        row=dict(block=block,pid=child.pid,argv=argv)
        write(HERE/f'numeric_block_{block}_execution.json',row)
        row['exit_code']=child.wait()
    row['completed_utc']=datetime.now(timezone.utc).isoformat();write(HERE/f'numeric_block_{block}_execution.json',row)
    print('BLOCK_CLOSED',block,row['exit_code'],flush=True)
    return row


def main():
    registered=json.loads((HERE/'SCHEDULING_REGISTRATION.json').read_text())
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==registered['driver_sha256']
    assert not (HERE/'NUMERIC_EXECUTION.json').exists()
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(run,b) for b in range(3)]
        first=futures[0].result()
        if first['exit_code']==0:
            with (HERE/'camera_driver.log').open('x') as f:
                camera=subprocess.Popen(['python3',str(HERE/'execute_visual.py')],cwd=HERE,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdout=f,stderr=subprocess.STDOUT)
            write(HERE/'CAMERA_SUPERVISOR.json',dict(pid=camera.pid,argv=['python3',str(HERE/'execute_visual.py')]))
        else:camera=None
        rows=[future.result() for future in futures]
        write(HERE/'NUMERIC_EXECUTION.json',dict(status='PASS_ALL_NUMERIC_CLOSED' if all(r['exit_code']==0 for r in rows) else 'FAIL',jobs=rows,actual_driver_pid=os.getpid()))
        if camera is not None:
            rc=camera.wait();write(HERE/'CAMERA_SUPERVISOR_EXIT.json',dict(pid=camera.pid,exit_code=rc))
        assert all(r['exit_code']==0 for r in rows)
        assert camera is not None and rc==0
    print('ALL_NUMERIC_AND_CAMERA_CLOSED',flush=True)


if __name__=='__main__':main()
