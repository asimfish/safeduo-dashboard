"""Run immutable camera jobs; check actual protocol rather than Kit exit alone."""
from pathlib import Path
import json,hashlib,subprocess,os
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
if __name__=='__main__':
    plan=json.loads((HERE/'visual_plan.json').read_text());jobs=[]
    receipt=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(HERE/'visual_plan.json'),jobs=jobs)
    target=HERE/'visual_execution.json';assert not target.exists();write(target,receipt)
    try:
        for j in plan['jobs']:
            for p,digest in plan['sources'].items():assert sha(p)==digest,p
            for rel,digest in plan['original_source_sha256'].items():assert sha(Path(plan['cwd'])/rel)==digest,rel
            checkpoint=j['argv'][j['argv'].index('--ckpt')+1]
            assert sha(checkpoint)==plan['actor_sha256'], 'frozen camera actor changed before launch'
            out=Path(j['argv'][j['argv'].index('--out')+1]);assert not out.exists()
            log=HERE/f'visual_{j["mode"]}.log'
            row=dict(mode=j['mode'],status='running',argv=j['argv'],out=str(out));jobs.append(row)
            with log.open('x') as f:
                child=subprocess.Popen(j['argv'],cwd=plan['cwd'],env={**os.environ,**j['env']},stdout=f,stderr=subprocess.STDOUT)
                row['pid']=child.pid;write(target,receipt);row['exit_code']=child.wait()
            p=json.loads((out/'visual_protocol.json').read_text())
            assert sha(checkpoint)==plan['actor_sha256']==p['actor_sha256'], 'camera actor binding changed'
            for source,digest in plan['sources'].items():assert sha(source)==digest,source
            for rel,digest in plan['original_source_sha256'].items():assert sha(Path(plan['cwd'])/rel)==digest,rel
            if row['exit_code'] or p['status']!='complete' or p.get('completed_windows')!=64:
                row.update(status='failed',error=p.get('error','incomplete'));receipt['status']='failed';write(target,receipt);raise ValueError('camera job incomplete; never treat failed capture as complete')
            row['status']='complete';write(target,receipt);print('DONE_CAMERA',j['mode'],flush=True)
        receipt['status']='complete'
    finally:receipt['finished_utc']=datetime.now(timezone.utc).isoformat();write(target,receipt)
