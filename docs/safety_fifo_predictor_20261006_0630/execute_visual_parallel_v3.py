"""Registered camera scheduling replacement, one owned process per GPU."""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,os,subprocess,threading,time
HERE=Path(__file__).resolve().parent
RAW=Path('/mnt/nas/data/lyf/double_hand')/HERE.name
LOCK=threading.Lock()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def verify(plan,job):
    for p,d in plan['sources'].items():assert sha(p)==d,p
    for p,d in plan['original_source_sha256'].items():assert sha(Path(plan['cwd'])/p)==d,p
    ckpt=job['argv'][job['argv'].index('--ckpt')+1];assert sha(ckpt)==plan['actor_sha256']
def main():
    plan_path=HERE/'visual_plan_v3.json';plan= json.loads(plan_path.read_text())
    registration=sha(plan_path)
    assert sha(plan_path)==registration
    target=HERE/'visual_execution.json';assert not target.exists()
    receipt=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=registration,
          scheduling='parallel primary methods on separate GPUs concurrent with ongoing registered numerical blocks; exact same camera args and source',jobs=[])
    write(target,receipt)
    def run(job):
        verify(plan,job)
        out=Path(job['argv'][job['argv'].index('--out')+1]);assert not out.exists()
        row=dict(mode=job['mode'],status='running',argv=job['argv'],out=str(out))
        with (HERE/f'visual_{job["mode"]}.log').open('x') as log:
            child=subprocess.Popen(job['argv'],cwd=plan['cwd'],env={**os.environ,**job['env']},stdout=log,stderr=subprocess.STDOUT)
            with LOCK:row['pid']=child.pid;receipt['jobs'].append(row);write(target,receipt)
            rc=child.wait()
        verify(plan,job)
        p=json.loads((out/'visual_protocol.json').read_text())
        complete=rc==0 and p['status']=='complete' and p.get('completed_windows')==64 and p['actor_sha256']==plan['actor_sha256']
        with LOCK:
            row.update(exit_code=rc,status='complete' if complete else 'failed',protocol_sha256=sha(out/'visual_protocol.json'))
            if not complete:row['error']=p.get('error','incomplete camera job')
            write(target,receipt)
        print('CLOSED_CAMERA',job['mode'],row['status'],flush=True)
        return complete
    with ThreadPoolExecutor(max_workers=2) as pool:complete=all(list(pool.map(run,plan['jobs'])))
    receipt.update(status='complete' if complete else 'complete_with_failures',finished_utc=datetime.now(timezone.utc).isoformat())
    write(target,receipt)
    assert complete,'camera failure retained; no incomplete job scored complete'
if __name__=='__main__':main()
