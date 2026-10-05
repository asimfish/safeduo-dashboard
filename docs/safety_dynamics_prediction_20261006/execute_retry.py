"""Registered observer seam correction; abort scheduling on any invalid job."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json,os,subprocess,threading
from execute import sha,check,now,write
HERE=Path(__file__).resolve().parent
STOP=threading.Event()
def run(file):
    plan=json.loads(file.read_text());root=Path(plan['output_root']);root.mkdir(exist_ok=False)
    state=dict(status='running',registered_plan_sha256=sha(file),jobs=[],started_utc=now(),driver_pid=os.getpid())
    for j in plan['jobs']:
        if STOP.is_set():break
        check(plan)
        record=dict(id=j['id'],status='running',started_utc=now(),planned_windows=64)
        state['jobs'].append(record);write(root/'campaign.json',state)
        with (root/(j['id']+'.log')).open('x') as log:
            child=subprocess.Popen(j['argv'],cwd=plan['cwd'],env={**os.environ,**plan['env'],**j['env']},stdout=log,stderr=subprocess.STDOUT)
            record['pid']=child.pid;write(root/'campaign.json',state);print('START',j['id'],child.pid,flush=True)
            code=child.wait()
        protocol=root/j['id']/'protocol.json';p=json.loads(protocol.read_text()) if protocol.exists() else {}
        complete=code==0 and p.get('status')=='complete' and p.get('steps')==960 and p.get('completed_cells')==1
        record.update(exit_code=code,finished_utc=now(),status='complete' if complete else 'invalid',
            completed_windows=64 if complete else 0,invalid_windows=0 if complete else 64,
            error=p.get('error'),protocol_sha256=sha(protocol) if protocol.exists() else None)
        check(plan);write(root/'campaign.json',state);print('END',j['id'],record['status'],flush=True)
        if not complete:STOP.set();break
    state.update(status='complete' if len(state['jobs'])==2 and all(j['status']=='complete' for j in state['jobs']) else 'contains_invalid_or_pending',finished_utc=now())
    write(root/'campaign.json',state);return state
def worker(files):
    results=[]
    for file in files:
        if STOP.is_set():break
        results.append(run(file))
    return results
if __name__=='__main__':
    files=sorted((HERE/'registered_retry').glob('block*.json'));assert len(files)==3
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,files[::2]),pool.submit(worker,files[1::2])]
        results=[f.result() for f in futures]
    write(HERE/'retry_completion.json',dict(status='complete' if not STOP.is_set() and sum(map(len,results))==3 else 'contains_invalid_or_pending',blocks=[r for rs in results for r in rs],finished_utc=now()))
