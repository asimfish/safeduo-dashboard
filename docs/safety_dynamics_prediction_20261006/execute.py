"""Two isolated GPU workers; preserve failures and completed denominators."""
from pathlib import Path
import json,hashlib,os,subprocess,time
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def write(p,x):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(x,indent=2)+'\n');temp.replace(p)
def check(p):
    for path,h in p['research_source_sha256'].items():assert sha(path)==h,path
    for rel,h in p['source_sha256'].items():assert sha(Path(p['cwd'])/rel)==h,rel
    assert sha(p['checkpoint_path'])==p['checkpoint_sha256']
def run(file):
    plan=json.loads(file.read_text());root=Path(plan['output_root']);root.mkdir(exist_ok=False)
    state=dict(status='running',registered_plan_sha256=sha(file),jobs=[],started_utc=now(),driver_pid=os.getpid())
    for j in plan['jobs']:
        check(plan)
        record=dict(id=j['id'],status='running',started_utc=now(),planned_windows=64)
        state['jobs'].append(record);write(root/'campaign.json',state)
        with (root/(j['id']+'.log')).open('x') as log:
            child=subprocess.Popen(j['argv'],cwd=plan['cwd'],env={**os.environ,**plan['env'],**j['env']},stdout=log,stderr=subprocess.STDOUT)
            record['pid']=child.pid;write(root/'campaign.json',state)
            print('START',j['id'],'pid',child.pid,flush=True);code=child.wait()
        record.update(exit_code=code,finished_utc=now())
        protocol=root/j['id']/'protocol.json'
        p=json.loads(protocol.read_text()) if protocol.exists() else {}
        complete=(code==0 and p.get('status')=='complete' and p.get('steps')==960 and p.get('completed_cells')==1)
        record.update(status='complete' if complete else 'invalid',completed_windows=64 if complete else 0,
                      invalid_windows=0 if complete else 64,protocol_sha256=sha(protocol) if protocol.exists() else None)
        check(plan);write(root/'campaign.json',state);print('END',j['id'],record['status'],flush=True)
    state.update(status='complete' if all(j['status']=='complete' for j in state['jobs']) else 'contains_invalid',finished_utc=now())
    write(root/'campaign.json',state)
    return state
def worker(paths):return [run(path) for path in paths]
if __name__=='__main__':
    files=sorted((HERE/'registered').glob('block*.json'));assert len(files)==3
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,files[::2]),pool.submit(worker,files[1::2])]
        results=[f.result() for f in futures]
    write(HERE/'campaign_completion.json',dict(status='complete' if all(r['status']=='complete' for rs in results for r in rs) else 'contains_invalid',blocks=[r for rs in results for r in rs],finished_utc=now()))
